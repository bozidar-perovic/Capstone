"""Validation boundaries, reproducibility, selection, and final refitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.AdaptiveAnalyzer import AdaptiveAnalyzer, DEFAULT_RULE_WEIGHTS
from src.BaseAnalyzer import BaseAnalyzer
from src.MLAnalyzer import MLAnalyzer
from src.RandomForestAnalyzer import RandomForestAnalyzer
from src.RuleBasedAnalyzer import RuleBasedAnalyzer
from src import validation as workflow


def sample_data(prefix="train", projects=9):
    rows = []
    for project in range(projects):
        for row in range(6):
            rows.append({
                "id": f"{prefix}_{project}_{row}",
                "project": f"{prefix}_project_{project}",
                "test_name": f"test_{row}",
                "category": 0 if row == 0 else 5,
                "full_code": (
                    "Thread.sleep(10);" if row == 0 else "assertTrue(true);"
                ),
            })
    return pd.DataFrame(rows)


def test_splits_keep_projects_together_and_cover_every_row_once():
    data = sample_data()
    first = workflow.make_splits(data)
    second = workflow.make_splits(data)
    visits = np.zeros(len(data), dtype=int)
    for (train, valid), (train_again, valid_again) in zip(first, second):
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(valid, valid_again)
        assert set(train).isdisjoint(valid)
        assert len(train) + len(valid) == len(data)
        assert set(data.iloc[train].project).isdisjoint(
            data.iloc[valid].project
        )
        assert (data.iloc[train].category != 5).nunique() == 2
        assert (data.iloc[valid].category != 5).nunique() == 2
        visits[valid] += 1
    assert (visits == 1).all()


@pytest.mark.parametrize("classes", ["one_class", "one_positive_project"])
def test_insufficient_classes_fail_before_fitting(classes):
    data = sample_data()
    data["category"] = 5
    if classes == "one_positive_project":
        data.loc[0, "category"] = 0
    with pytest.raises(ValueError, match="both classes"):
        workflow.make_splits(data)


def test_default_candidates_match_existing_analyzers_exactly():
    data = sample_data()
    original = [
        RuleBasedAnalyzer(), MLAnalyzer(),
        RandomForestAnalyzer(), AdaptiveAnalyzer(),
    ]
    for candidate, default in zip(workflow.parse_candidates(), original):
        configured = candidate.build()
        if default.requires_training:
            default.train(data)
            configured.train(data)
        pd.testing.assert_frame_equal(
            configured.analyze(data), default.analyze(data)
        )


def test_adaptive_weight_overrides_keep_other_defaults():
    candidate = workflow.parse_candidates([{
        "candidate_id": "sleep", "analyzer": "Adaptive Hybrid",
        "parameters": {
            "rule_weights": {"sleep_count": 0.4}, "ml_max_count": None,
        },
    }])[0]
    expected = {**DEFAULT_RULE_WEIGHTS, "sleep_count": 0.4}
    assert candidate.parameters["rule_weights"] == expected
    assert candidate.build().rule_weights == expected
    assert candidate.build().ml_analyzer.max_count is None


@pytest.mark.parametrize("records", [
    [],
    [{"candidate_id": "x", "analyzer": "unknown"}],
    [{"candidate_id": "x", "analyzer": "Rule-Based",
      "parameters": {"cap": 5}}],
    [{"candidate_id": "x", "analyzer": "Adaptive Hybrid",
      "parameters": {"low_threshold": 0.5, "high_threshold": 0.5}}],
    [{"candidate_id": "x", "analyzer": "Logistic Reg"}] * 2,
])
def test_invalid_candidate_configuration_is_rejected(records):
    with pytest.raises(ValueError):
        workflow.parse_candidates(records)


class RecordingAnalyzer(BaseAnalyzer):
    requires_training = True

    def __init__(self, cap, records):
        self.cap = cap
        self.records = records

    def train(self, data):
        self.training = data.copy()
        return self

    def analyze(self, data):
        self.records.append((self.training.copy(), data.copy()))
        assert set(self.training.id).isdisjoint(data.id)
        assert set(self.training.project).isdisjoint(data.project)
        scores = data.full_code.str.contains("sleep").astype(float)
        result = data.copy()
        result["flaky_score"] = scores if self.cap == 5 else 1 - scores
        return result


@pytest.fixture
def prepared_run(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for fold in (1, 2):
        sample_data(f"train{fold}").to_csv(
            data_dir / f"train_set_{fold}.csv", index=False
        )
    records = []
    monkeypatch.setattr(workflow.Candidate, "build", lambda candidate: (
        RecordingAnalyzer(candidate.parameters["max_count"], records)
    ))
    candidates = [
        {"candidate_id": "bad", "analyzer": "Logistic Reg",
         "parameters": {"max_count": 1}},
        {"candidate_id": "z_first_best", "analyzer": "Logistic Reg"},
        {"candidate_id": "a_second_best", "analyzer": "Logistic Reg"},
    ]
    loaded = []
    original_reader = workflow.read_dataset

    def read(path):
        loaded.append(Path(path).name)
        return original_reader(path)

    monkeypatch.setattr(workflow, "read_dataset", read)
    run = workflow.validate(
        data_dir, tmp_path / "run", candidates, folds=(1, 2)
    )
    return data_dir, run, records, loaded


def test_validation_reads_training_only_and_selects_by_mean_ap(prepared_run):
    data_dir, run, records, loaded = prepared_run
    assert loaded == ["train_set_1.csv", "train_set_2.csv"]
    assert len(records) == 18  # 2 outer folds x 3 inner x 3 candidates
    selections = json.loads((run / "selected_settings.json").read_text())
    for fold in (1, 2):
        assert selections[str(fold)][0]["candidate_id"] == "z_first_best"
        assignments = pd.read_csv(run / f"fold_{fold}_assignments.csv")
        assert len(assignments) == len(sample_data())
        assert assignments.groupby("project").inner_fold.nunique().eq(1).all()
        assert assignments.row_index.tolist() == list(range(len(assignments)))
    metrics = pd.read_csv(run / "validation_metrics.csv")
    assert len(metrics) == 18
    assert "recall@10%" in metrics
    assert "recall@20%" in metrics
    assert not (run / "evaluation").exists()
    manifest = json.loads((run / "experiment.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["selection_metric"] == "pr_auc"
    with pytest.raises(FileExistsError):
        workflow.validate(data_dir, run, folds=(1,))


def test_evaluation_refits_on_full_outer_training(prepared_run):
    data_dir, run, records, loaded = prepared_run
    for fold in (1, 2):
        sample_data(f"test{fold}", projects=2).to_csv(
            data_dir / f"test_set_{fold}.csv", index=False
        )
    records.clear()
    before = {path: workflow.checksum(path) for path in run.glob("*.*")}
    output = workflow.evaluate(run)
    assert len(records) == 2
    assert loaded[-4:] == [
        "train_set_1.csv", "test_set_1.csv",
        "train_set_2.csv", "test_set_2.csv",
    ]
    for fold, (training, testing) in enumerate(records, start=1):
        pd.testing.assert_frame_equal(
            training, pd.read_csv(data_dir / f"train_set_{fold}.csv")
        )
        assert len(testing) == 12
    assert before == {path: workflow.checksum(path) for path in before}
    result = pd.read_csv(output / "outer_metrics.csv")
    assert result.candidate_id.tolist() == ["z_first_best"] * 2
    assert result.pr_auc.tolist() == [1.0] * 2
    assert (output / "evaluation.json").exists()
    with pytest.raises(FileExistsError):
        workflow.evaluate(run)


@pytest.mark.parametrize(
    "changed", ["training", "settings", "assignments", "source"]
)
def test_evaluation_rejects_changed_validation_state(prepared_run, changed):
    data_dir, run, _, _ = prepared_run
    if changed == "training":
        path = data_dir / "train_set_1.csv"
        path.write_text(path.read_text() + "\n")
    elif changed == "settings":
        path = run / "selected_settings.json"
        path.write_text(path.read_text() + "\n")
    elif changed == "assignments":
        path = run / "fold_1_assignments.csv"
        path.write_text(path.read_text() + "\n")
    else:
        path = run / "experiment.json"
        manifest = json.loads(path.read_text())
        manifest["implementation"] = {}
        workflow.write_json(path, manifest)
    with pytest.raises(ValueError, match="changed"):
        workflow.evaluate(run)
    assert not (run / "evaluation").exists()


def test_evaluation_rejects_outer_project_overlap(prepared_run):
    data_dir, run, _, _ = prepared_run
    sample_data("train1").to_csv(data_dir / "test_set_1.csv", index=False)
    with pytest.raises(ValueError, match="overlap"):
        workflow.evaluate(run)
    assert not (run / "evaluation").exists()


def test_preprocessing_is_fitted_on_inner_training_only():
    training = sample_data(projects=3)
    testing = sample_data("test", projects=2)
    testing["full_code"] = "assertTrue(true);" * 40
    analyzer = MLAnalyzer()
    candidate = workflow.parse_candidates([{
        "candidate_id": "lr", "analyzer": "Logistic Reg",
    }])[0]
    # Inspect the pipeline fitted by score without altering its implementation.
    candidate.build = lambda: analyzer
    workflow.score(candidate, training, testing)
    features = analyzer.extract_features(training)
    expected = np.log1p(np.clip(features.to_numpy(), 0, 5)).mean(axis=0)
    scaler = analyzer.model.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, expected)
    assert scaler.n_samples_seen_ == len(training)


def test_outputs_cannot_be_written_inside_dataset_directory(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        workflow.validate(tmp_path, tmp_path / "outputs")


def test_cli_default_workflow_with_real_analyzers(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    training = sample_data()
    testing = sample_data("test", projects=2)
    training.to_csv(data_dir / "train_set_1.csv", index=False)
    testing.to_csv(data_dir / "test_set_1.csv", index=False)
    run = tmp_path / "run"
    workflow.main([
        "validate", "--data-dir", str(data_dir),
        "--output-dir", str(run), "--folds", "1",
    ])
    workflow.main(["evaluate", "--run-dir", str(run)])
    metrics = pd.read_csv(run / "evaluation/outer_metrics.csv")
    assert len(metrics) == 4
    candidates = {c.analyzer: c for c in workflow.parse_candidates()}
    for row in metrics.itertuples(index=False):
        analyzer = candidates[row.analyzer].build()
        if analyzer.requires_training:
            analyzer.train(training)
        expected = analyzer.analyze(testing)
        actual = pd.read_csv(run / "evaluation" / row.predictions_file)
        assert actual.id.tolist() == expected.id.tolist()
        np.testing.assert_allclose(actual.flaky_score, expected.flaky_score)


def test_raw_and_masked_rules_share_splits_and_preserve_source(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    training = sample_data()
    clean = training.category == 5
    training.loc[clean, "full_code"] = "// Thread.sleep(999);"
    source = data_dir / "train_set_1.csv"
    training.to_csv(source, index=False)
    original_hash = workflow.checksum(source)
    config = [
        {"candidate_id": f"rules_{mode}", "analyzer": "Rule-Based",
         "parameters": {"source_mode": mode}}
        for mode in ("raw", "masked")
    ]
    run = workflow.validate(data_dir, tmp_path / "run", config, folds=(1,))
    metrics = pd.read_csv(run / "validation_metrics.csv")
    raw = metrics[metrics.candidate_id == "rules_raw"]
    masked = metrics[metrics.candidate_id == "rules_masked"]
    assert raw.total_tests.tolist() == masked.total_tests.tolist()
    assert raw.total_flaky.tolist() == masked.total_flaky.tolist()
    assert (masked.pr_auc.to_numpy() > raw.pr_auc.to_numpy()).all()
    selected = json.loads((run / "selected_settings.json").read_text())
    assert selected["1"][0]["candidate_id"] == "rules_masked"
    assert workflow.checksum(source) == original_hash
    assert not (run / "evaluation").exists()

"""Unit tests for dataset ingestion utilities and main benchmark runner.

Verifies:
- Safe loading of valid CSV datasets and handling of missing files.
- Composite unique identifier generation and ground truth column mapping.
- Stateless scoring across repeated runs in RuleBasedAnalyzer.
- Silent CSV output in save_results_to_csv without console emissions.
- Summary metrics aggregation and single-line user confirmation in
  compare_folds.
"""

from unittest.mock import patch

import pandas as pd

from src.main import compare_folds
from src.RuleBasedAnalyzer import RuleBasedAnalyzer
from src.utility import get_comparison_data, load_data, save_results_to_csv


def test_load_data_valid_file(tmp_path):
    """Verify load_data successfully reads an existing CSV into a DataFrame."""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("full_code\nThread.sleep(1000);\n")

    data = load_data(csv_file)

    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert list(data.columns) == ["full_code"]
    assert data.loc[0, "full_code"] == "Thread.sleep(1000);"


def test_load_data_missing_file(tmp_path):
    """Verify load_data returns None when the target file does not exist."""
    missing_file = tmp_path / "does_not_exist.csv"

    data = load_data(missing_file)

    assert data is None


def test_get_comparison_data_valid_file(tmp_path):
    """Verify ground truth binary labels and unique identifiers are created."""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(
        "id,project,test_name,category,label\n"
        "1,Project1,testOne,5,non-flaky\n"
        "2,Project2,testTwo,0,async wait\n"
    )

    data = get_comparison_data(csv_file)

    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert list(data.columns) == [
        "unique_identifier", "category", "label", "is_flaky"
    ]
    assert data.loc[0, "unique_identifier"] == "1_Project1_testOne"
    assert data.loc[0, "is_flaky"] == "no"
    assert data.loc[1, "unique_identifier"] == "2_Project2_testTwo"
    assert data.loc[1, "is_flaky"] == "yes"


def test_rule_based_analyzer_scores_rows_independently(tmp_path):
    """Verify rule analyzer resets state and scores rows independently."""
    csv_file = tmp_path / "rules.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "project": ["Project1"] * 3,
            "test_name": ["sleepTest", "cleanTest", "threadTest"],
            "full_code": [
                "Thread.sleep(1000);",
                'System.out.println("clean");',
                "new Thread(() -> {}).start();",
            ],
        }
    ).to_csv(csv_file, index=False)

    analyzer = RuleBasedAnalyzer()
    first_result = analyzer.analyze(csv_file)
    second_result = analyzer.analyze(csv_file)

    assert first_result["flaky_score"].tolist() == [0.35, 0.0, 0.25]
    assert second_result["flaky_score"].tolist() == [0.35, 0.0, 0.25]


def test_save_results_to_csv(tmp_path, capsys):
    """Verify save_results_to_csv writes to disk without stdout output."""
    out = tmp_path / "saved.csv"
    df = pd.DataFrame({"a": [1, 2]})
    save_results_to_csv(df, out)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert out.exists()


def test_compare_folds_saves_csv_and_prints_single_line(tmp_path, capsys):
    """Verify compare_folds produces summary CSV and single message line."""
    dummy_metrics = {
        "roc_auc": 0.8,
        "pr_auc": 0.5,
        "recall@5%": 0.2,
        "recall@10%": 0.4,
        "recall@20%": 0.6,
        "precision@5%": 0.3,
        "precision@10%": 0.25,
    }
    dummy_routing = {
        "fast_clean_pct": 0.8,
        "ml_deep_pct": 0.15,
        "fast_flaky_pct": 0.05,
    }

    out_file = str(tmp_path / "custom_summary.csv")

    with patch("src.main.run_rule_based_fold", return_value=dummy_metrics), \
         patch("src.main.run_ml_fold", return_value=dummy_metrics), \
         patch("src.main.run_rf_fold", return_value=dummy_metrics), \
         patch(
             "src.main.run_adaptive_fold",
             return_value=(dummy_metrics, dummy_routing),
         ):
        result_df = compare_folds(folds=(1,), output_file=out_file)

    captured = capsys.readouterr()
    assert captured.out == f'Process done, saved in the "{out_file}"\n'
    assert (tmp_path / "custom_summary.csv").exists()
    assert list(result_df.columns) == [
        "Metric", "Rule-Based", "Logistic Reg",
        "Random Forest", "Adaptive Hybrid"
    ]
    assert "ROC-AUC" in result_df["Metric"].values
    assert "Total ML Workload Saved" in result_df["Metric"].values


def test_run_rule_based_fold(tmp_path):
    """Verify run_rule_based_fold computes ranking metrics on fold data."""
    from src.main import run_rule_based_fold

    sample_test = pd.DataFrame({
        "full_code": ["Thread.sleep(100);", "assertTrue(true);"],
        "category": [0, 5],
    })

    with patch("src.main.load_data", return_value=sample_test), \
         patch("src.main.save_results_to_csv"):
        metrics = run_rule_based_fold(99)

    assert metrics is not None
    assert "roc_auc" in metrics
    assert "recall@5%" in metrics

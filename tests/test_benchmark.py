"""Tests for the polymorphic benchmark lifecycle."""

from unittest.mock import Mock

import pandas as pd
import pytest

from src.BaseAnalyzer import BaseAnalyzer
from src.benchmark import AnalyzerSpec, BenchmarkRunner


class RecordingAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.training_data = None

    def train(self, data):
        self.training_data = data
        return self

    def analyze(self, data):
        result = data.copy()
        result["flaky_score"] = [0.9, 0.1]
        return result


@pytest.mark.parametrize("trainable", [False, True])
def test_fold_lifecycle(tmp_path, trainable):
    training = pd.DataFrame({"full_code": ["training only"]})
    testing = pd.DataFrame({"full_code": ["sleep", "clean"], "category": [0, 5]})
    instances = []

    def factory():
        analyzer = RecordingAnalyzer()
        analyzer.requires_training = trainable
        instances.append(analyzer)
        return analyzer

    loader = Mock(side_effect=lambda path: (
        training if path.name.startswith("train_") else testing
    ))
    saver = Mock()
    runner = BenchmarkRunner(tmp_path, tmp_path, loader=loader, saver=saver)
    spec = AnalyzerSpec("Custom", factory, "_custom")
    for fold in (1, 2):
        result = runner.run_fold(spec, fold)
        assert result.metrics["roc_auc"] == 1.0
        assert result.routing is None
    assert instances[0] is not instances[1]
    assert loader.call_count == (4 if trainable else 2)
    for instance in instances:
        assert instance.training_data is (training if trainable else None)
    assert saver.call_args.args[1] == tmp_path / "test_set_2_custom_results.csv"
    assert "flaky_score" not in testing.columns


@pytest.mark.parametrize("missing", ["train", "test"])
def test_missing_data_skips_scoring_and_saving(missing):
    analyzer = Mock(spec=BaseAnalyzer)
    analyzer.requires_training = True
    loader = Mock(side_effect=lambda path: (
        None if path.name.startswith(missing) else pd.DataFrame()
    ))
    saver = Mock()
    runner = BenchmarkRunner(loader=loader, saver=saver)
    result = runner.run_fold(AnalyzerSpec("Missing", lambda: analyzer, ""), 1)
    assert result.metrics is None
    analyzer.train.assert_not_called()
    analyzer.analyze.assert_not_called()
    saver.assert_not_called()


def test_unlabeled_data_is_saved_without_metrics(tmp_path):
    data = pd.DataFrame({"full_code": ["sleep", "clean"]})
    saver = Mock()
    runner = BenchmarkRunner(loader=lambda path: data, saver=saver)
    result = runner.run_fold(AnalyzerSpec("Custom", RecordingAnalyzer, ""), 1)
    assert result.metrics is None
    saver.assert_called_once()


def test_empty_benchmark_does_not_claim_success(tmp_path, capsys):
    output = tmp_path / "summary.csv"
    assert BenchmarkRunner().compare_folds((), output) is None
    assert not output.exists()
    assert capsys.readouterr().out == ""


def test_stateless_training_hook():
    analyzer = RecordingAnalyzer()
    assert BaseAnalyzer.train(analyzer, pd.DataFrame()) is analyzer
    assert analyzer.get_routing_summary() == {}
    analyzer.requires_training = True
    with pytest.raises(NotImplementedError):
        BaseAnalyzer.train(analyzer, pd.DataFrame())

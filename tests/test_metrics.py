import numpy as np
import pytest

from src.metrics import (
    compute_ranking_metrics,
    precision_at_k_percent,
    recall_at_k_percent,
)


def test_recall_at_k_percent_perfect_ranking():
    # 10 tests, 2 flaky tests ranked at the very top
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    # Top 10% (1 test): captures 1 of 2 flaky tests -> recall = 0.5
    assert recall_at_k_percent(y_true, y_scores, k_percent=10) == 0.5

    # Top 20% (2 tests): captures both flaky tests -> recall = 1.0
    assert recall_at_k_percent(y_true, y_scores, k_percent=20) == 1.0


def test_recall_at_k_percent_worst_ranking():
    # Flaky tests ranked at the bottom
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    assert recall_at_k_percent(y_true, y_scores, k_percent=20) == 0.0


def test_recall_at_k_percent_zero_positives():
    y_true = np.array([0, 0, 0, 0])
    y_scores = np.array([0.5, 0.4, 0.3, 0.2])
    assert recall_at_k_percent(y_true, y_scores, k_percent=50) == 0.0


def test_precision_at_k_percent():
    # 10 tests, 2 positives
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    # Top 10% (1 test): test 0 is positive -> precision = 1.0
    assert precision_at_k_percent(y_true, y_scores, k_percent=10) == 1.0

    # Top 20% (2 tests): 1 positive, 1 negative -> precision = 0.5
    assert precision_at_k_percent(y_true, y_scores, k_percent=20) == 0.5


def test_compute_ranking_metrics_keys_and_values():
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    metrics = compute_ranking_metrics(y_true, y_scores, k_percents=(10, 20))

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "recall@10%" in metrics
    assert "recall@20%" in metrics
    assert "precision@10%" in metrics
    assert "precision@20%" in metrics
    assert metrics["total_tests"] == 10.0
    assert metrics["total_flaky"] == 2.0
    assert metrics["flaky_rate"] == pytest.approx(0.2)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_compute_ranking_metrics_single_class_guard():
    # All non-flaky
    y_true = np.array([0, 0, 0, 0, 0])
    y_scores = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    metrics = compute_ranking_metrics(y_true, y_scores)
    assert metrics["total_flaky"] == 0.0
    assert metrics["roc_auc"] == 0.5

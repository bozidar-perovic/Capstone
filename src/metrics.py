"""Evaluation metrics for flaky test prioritization.

Implements the quality grading metrics defined in the capstone guide:
- ROC-AUC: Area under ROC curve.
- Recall@K%: Fraction of flaky tests identified in top K% tests.
- Precision@K%: Fraction of top K% highest-risk tests that are flaky.
"""

from typing import Dict, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_k_percent(
    y_true: Union[np.ndarray, pd.Series, Sequence[int]],
    y_scores: Union[np.ndarray, pd.Series, Sequence[float]],
    k_percent: float = 10.0,
) -> float:
    """Calculate the recall of positive cases in the top K% ranked items.

    Args:
        y_true: Binary ground truth indicators (1 for flaky, 0 for non-flaky).
        y_scores: Continuous risk scores / probabilities (higher = more flaky).
        k_percent: Cutoff percentile between 0 and 100 (e.g. 5, 10, 20).

    Returns:
        Fraction of all positive items captured within the top K% ranking.
    """
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    total_positives = int(np.sum(y_true_arr == 1))
    if total_positives == 0:
        return 0.0

    n = len(y_true_arr)
    top_n = max(1, int(np.ceil(n * (k_percent / 100.0))))

    # Rank descending by predicted risk score
    ranked_indices = np.argsort(y_scores_arr)[::-1][:top_n]
    detected_positives = int(np.sum(y_true_arr[ranked_indices] == 1))

    return detected_positives / total_positives


def precision_at_k_percent(
    y_true: Union[np.ndarray, pd.Series, Sequence[int]],
    y_scores: Union[np.ndarray, pd.Series, Sequence[float]],
    k_percent: float = 10.0,
) -> float:
    """Calculate precision among the top K% ranked items.

    Args:
        y_true: Binary ground truth indicators (1 for flaky, 0 for non-flaky).
        y_scores: Continuous risk scores / probabilities (higher = more flaky).
        k_percent: Cutoff percentile between 0 and 100 (e.g. 5, 10, 20).

    Returns:
        Fraction of items in the top K% ranking that are positive.
    """
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    n = len(y_true_arr)
    if n == 0:
        return 0.0

    top_n = max(1, int(np.ceil(n * (k_percent / 100.0))))
    ranked_indices = np.argsort(y_scores_arr)[::-1][:top_n]
    detected_positives = int(np.sum(y_true_arr[ranked_indices] == 1))

    return detected_positives / top_n


def compute_ranking_metrics(
    y_true: Union[np.ndarray, pd.Series, Sequence[int]],
    y_scores: Union[np.ndarray, pd.Series, Sequence[float]],
    k_percents: Sequence[float] = (5.0, 10.0, 20.0),
) -> Dict[str, float]:
    """Compute all evaluation metrics defined in the capstone guide.

    Returns a dictionary of scalar metrics for reporting and comparison.
    """
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    total_tests = len(y_true_arr)
    total_flaky = int(np.sum(y_true_arr == 1))
    flaky_rate = (total_flaky / total_tests) if total_tests > 0 else 0.0

    # Guard against single-class splits in small tests
    if len(np.unique(y_true_arr)) > 1:
        roc_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_scores_arr))
    else:
        roc_auc = 0.5
        pr_auc = flaky_rate

    metrics: Dict[str, float] = {
        "total_tests": float(total_tests),
        "total_flaky": float(total_flaky),
        "flaky_rate": flaky_rate,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    for k in k_percents:
        k_int = int(k) if int(k) == k else k
        metrics[f"recall@{k_int}%"] = recall_at_k_percent(
            y_true_arr, y_scores_arr, k_percent=k
        )
        metrics[f"precision@{k_int}%"] = precision_at_k_percent(
            y_true_arr, y_scores_arr, k_percent=k
        )

    return metrics

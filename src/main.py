"""Execution and cross-validation benchmark runner for flaky test analyzers.

Orchestrates 4-fold project-based cross-validation across all static analyzers:
1. Rule-Based (RuleBasedAnalyzer): Transparent keyword heuristic scorer.
2. Logistic Regression (MLAnalyzer): Regularized, class-weighted linear model.
3. Random Forest (RandomForestAnalyzer): Non-linear ensemble model.
4. Adaptive Hybrid (AdaptiveAnalyzer): 3-zone heuristic triage routing.

Each model is evaluated on held-out projects (Group K-Fold by repository)
to ensure models generalize to unseen codebases without data leakage.

Saves predictions for each fold to
`src/results/test_set_{k}_{model}_results.csv` and writes the
consolidated 4-fold benchmark summary table to `src/benchmark_summary.csv`.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .AdaptiveAnalyzer import AdaptiveAnalyzer
from .metrics import compute_ranking_metrics
from .MLAnalyzer import MLAnalyzer
from .RandomForestAnalyzer import RandomForestAnalyzer
from .RuleBasedAnalyzer import RuleBasedAnalyzer
from .utility import load_data, save_results_to_csv


def run_rule_based_fold(fold_number: int) -> Optional[Dict[str, float]]:
    """Score one project-based test fold with RuleBasedAnalyzer and evaluate.

    Args:
        fold_number: Index of the cross-validation fold (1 through 4).

    Returns:
        Dictionary of ranking metrics on the test fold, or None if data
        loading failed.
    """
    test_path = f"src/data/test_set_{fold_number}.csv"
    output_path = f"src/results/test_set_{fold_number}_rule_based_results.csv"

    test_data = load_data(test_path)
    if test_data is None:
        return None

    # Score held-out test fold using static heuristic indicators
    analyzer = RuleBasedAnalyzer()
    results = analyzer.analyze(test_data)
    if results is None:
        return None
    save_results_to_csv(results, output_path)

    metrics = None
    if "category" in results.columns:
        y_true = (results["category"] != 5).astype(int)
        metrics = compute_ranking_metrics(y_true, results["flaky_score"])

    return metrics


def run_ml_fold(fold_number: int) -> Optional[Dict[str, float]]:
    """Train on one project-based fold, score it, and evaluate.

    Args:
        fold_number: Index of the cross-validation fold (1 through 4).

    Returns:
        Dictionary of ranking metrics on the held-out test fold, or None
        if data loading failed.
    """
    train_path = f"src/data/train_set_{fold_number}.csv"
    test_path = f"src/data/test_set_{fold_number}.csv"
    output_path = f"src/results/test_set_{fold_number}_results.csv"

    train_data = load_data(train_path)
    test_data = load_data(test_path)
    if train_data is None or test_data is None:
        return None

    # Fit Logistic Regression on training fold and score held-out test fold
    analyzer = MLAnalyzer()
    analyzer.train(train_data)
    results = analyzer.analyze(test_data)
    save_results_to_csv(results, output_path)

    metrics = None
    if "category" in results.columns:
        # Convert multiclass labels to binary:
        # category 5 = non-flaky (0), categories 0-4 = flaky (1)
        y_true = (results["category"] != 5).astype(int)
        metrics = compute_ranking_metrics(y_true, results["flaky_score"])

    return metrics


def run_rf_fold(fold_number: int) -> Optional[Dict[str, float]]:
    """Train Random Forest on one fold, score it, and evaluate.

    Args:
        fold_number: Index of the cross-validation fold (1 through 4).

    Returns:
        Dictionary of ranking metrics on the held-out test fold, or None
        if data loading failed.
    """
    train_path = f"src/data/train_set_{fold_number}.csv"
    test_path = f"src/data/test_set_{fold_number}.csv"
    output_path = f"src/results/test_set_{fold_number}_rf_results.csv"

    train_data = load_data(train_path)
    test_data = load_data(test_path)
    if train_data is None or test_data is None:
        return None

    # Fit Random Forest classifier with seed 42 for reproducibility
    analyzer = RandomForestAnalyzer(random_state=42)
    analyzer.train(train_data)
    results = analyzer.analyze(test_data)
    save_results_to_csv(results, output_path)

    metrics = None
    if "category" in results.columns:
        y_true = (results["category"] != 5).astype(int)
        metrics = compute_ranking_metrics(y_true, results["flaky_score"])

    return metrics


def run_adaptive_fold(
    fold_number: int
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
    """Train on one fold, score with AdaptiveAnalyzer, and evaluate.

    Args:
        fold_number: Index of the cross-validation fold (1 through 4).

    Returns:
        Tuple of (metrics_dict, routing_summary_dict), or (None, None)
        if data loading failed.
    """
    train_path = f"src/data/train_set_{fold_number}.csv"
    test_path = f"src/data/test_set_{fold_number}.csv"
    output_path = f"src/results/test_set_{fold_number}_adaptive_results.csv"

    train_data = load_data(train_path)
    test_data = load_data(test_path)
    if train_data is None or test_data is None:
        return None, None

    # Adaptive Stage 1 heuristic scan + Stage 2 ML triage
    analyzer = AdaptiveAnalyzer()
    analyzer.train(train_data)
    results = analyzer.analyze(test_data)
    save_results_to_csv(results, output_path)

    routing = analyzer.get_routing_summary()

    metrics = None
    if "category" in results.columns:
        y_true = (results["category"] != 5).astype(int)
        metrics = compute_ranking_metrics(y_true, results["flaky_score"])

    return metrics, routing


def compare_folds(
    folds: Sequence[int] = (1, 2, 3, 4),
    output_file: Union[str, Path] = "src/benchmark_summary.csv",
) -> Optional[pd.DataFrame]:
    """Run all cross-validation folds and save summary benchmark results.

    Evaluates Rule-Based, Logistic Regression, Random Forest, and Adaptive
    Hybrid analyzers across each fold, aggregates ranking metrics across folds,
    calculates adaptive routing efficiency, and saves the summary CSV.

    Args:
        folds: Sequence of fold indices to evaluate (default: 1, 2, 3, 4).
        output_file: Destination file path for the benchmark summary CSV.

    Returns:
        Summary DataFrame comparing models across metrics, or None if models
        failed to evaluate.
    """
    rb_all = []
    lr_all = []
    rf_all = []
    adapt_all = []
    routings = []

    # Execute all folds sequentially
    for fold in folds:
        m_rb = run_rule_based_fold(fold)
        m_lr = run_ml_fold(fold)
        m_rf = run_rf_fold(fold)
        m_ad, r_ad = run_adaptive_fold(fold)

        if m_rb:
            rb_all.append(m_rb)
        if m_lr:
            lr_all.append(m_lr)
        if m_rf:
            rf_all.append(m_rf)
        if m_ad:
            adapt_all.append(m_ad)
        if r_ad:
            routings.append(r_ad)

    summary_df = None
    if rb_all and lr_all and rf_all and adapt_all:
        df_rb = pd.DataFrame(rb_all)
        df_lr = pd.DataFrame(lr_all)
        df_rf = pd.DataFrame(rf_all)
        df_ad = pd.DataFrame(adapt_all)

        # Aggregate ranking metrics (mean across folds)
        rows = []
        for key, label in [
            ("roc_auc", "ROC-AUC"),
            ("pr_auc", "PR-AUC"),
            ("recall@5%", "Recall@5%"),
            ("recall@10%", "Recall@10%"),
            ("recall@20%", "Recall@20%"),
            ("precision@5%", "Precision@5%"),
            ("precision@10%", "Precision@10%"),
        ]:
            v_rb = df_rb[key].mean()
            v_lr = df_lr[key].mean()
            v_rf = df_rf[key].mean()
            v_ad = df_ad[key].mean()
            fmt = ".1%" if "recall" in key or "precision" in key else ".4f"
            rows.append({
                "Metric": label,
                "Rule-Based": f"{v_rb:{fmt}}",
                "Logistic Reg": f"{v_lr:{fmt}}",
                "Random Forest": f"{v_rf:{fmt}}",
                "Adaptive Hybrid": f"{v_ad:{fmt}}",
            })

        # Append Adaptive Stage 1 triage and efficiency breakdown
        if routings:
            avg_clean = np.mean([r["fast_clean_pct"] for r in routings])
            avg_ml = np.mean([r["ml_deep_pct"] for r in routings])
            avg_flaky = np.mean([r["fast_flaky_pct"] for r in routings])
            rows.extend([
                {
                    "Metric": "Fast Clean (Bypassed ML)",
                    "Rule-Based": "N/A",
                    "Logistic Reg": "N/A",
                    "Random Forest": "N/A",
                    "Adaptive Hybrid": f"{avg_clean:.1%}",
                },
                {
                    "Metric": "ML Routed (Evaluated)",
                    "Rule-Based": "0.0%",
                    "Logistic Reg": "100.0%",
                    "Random Forest": "100.0%",
                    "Adaptive Hybrid": f"{avg_ml:.1%}",
                },
                {
                    "Metric": "Fast Flaky (Bypassed ML)",
                    "Rule-Based": "N/A",
                    "Logistic Reg": "N/A",
                    "Random Forest": "N/A",
                    "Adaptive Hybrid": f"{avg_flaky:.1%}",
                },
                {
                    "Metric": "Total ML Workload Saved",
                    "Rule-Based": "100.0%",
                    "Logistic Reg": "0.0%",
                    "Random Forest": "0.0%",
                    "Adaptive Hybrid": f"{(1.0 - avg_ml):.1%}",
                },
            ])

        # Write summary DataFrame to destination CSV file
        summary_df = pd.DataFrame(rows)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_file, index=False)

    print(f'Process done, saved in the "{output_file}"')
    return summary_df


if __name__ == "__main__":
    out_path = (
        sys.argv[1] if len(sys.argv) > 1 else "src/benchmark_summary.csv"
    )
    compare_folds(folds=(1, 2, 3, 4), output_file=out_path)

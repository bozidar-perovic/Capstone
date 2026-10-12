import numpy as np
import pandas as pd

from .AdaptiveAnalyzer import AdaptiveAnalyzer
from .metrics import compute_ranking_metrics
from .MLAnalyzer import MLAnalyzer
from .RandomForestAnalyzer import RandomForestAnalyzer
from .utility import load_data, save_results_to_csv


def print_metrics(name, metrics):
    """Print formatted evaluation metrics for one model."""
    print(f"--- {name} Evaluation ---")
    print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:         {metrics['pr_auc']:.4f}")
    print(f"  Recall@5%:      {metrics['recall@5%']:.1%}")
    print(f"  Recall@10%:     {metrics['recall@10%']:.1%}")
    print(f"  Recall@20%:     {metrics['recall@20%']:.1%}")
    print(f"  Precision@5%:   {metrics['precision@5%']:.1%}")
    print(f"  Precision@10%:  {metrics['precision@10%']:.1%}")


def run_ml_fold(fold_number):
    """Train on one project-based fold, score it, and evaluate."""
    train_path = f"src/data/train_set_{fold_number}.csv"
    test_path = f"src/data/test_set_{fold_number}.csv"
    output_path = f"src/results/test_set_{fold_number}_results.csv"

    print(f"Running ML analyzer (Logistic Regression) for fold {fold_number}")
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    if train_data is None or test_data is None:
        print("Could not load the fold data; skipping this fold.")
        return None

    print(
        f"Loaded {len(train_data)} training tests and "
        f"{len(test_data)} test tests."
    )
    print("Extracting features and training logistic regression...")
    analyzer = MLAnalyzer()
    analyzer.train(train_data)

    print("Scoring held-out test rows...")
    results = analyzer.analyze(test_data)
    save_results_to_csv(results, output_path)

    metrics = None
    if "category" in results.columns:
        y_true = (results["category"] != 5).astype(int)
        metrics = compute_ranking_metrics(y_true, results["flaky_score"])
        print_metrics("Logistic Regression", metrics)

    if "test_name" in results.columns:
        print("Top 5 tests by predicted flaky risk:")
        top5 = results.nlargest(
            5, "flaky_score"
        )[["test_name", "flaky_score"]]
        print(top5.to_string(index=False))

    return metrics


def run_rf_fold(fold_number):
    """Train Random Forest on one fold, score it, and evaluate."""
    train_path = f"src/data/train_set_{fold_number}.csv"
    test_path = f"src/data/test_set_{fold_number}.csv"
    output_path = f"src/results/test_set_{fold_number}_rf_results.csv"

    print(f"Running Random Forest analyzer for fold {fold_number}")
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    if train_data is None or test_data is None:
        print("Could not load the fold data; skipping this fold.")
        return None

    print(
        f"Loaded {len(train_data)} training tests and "
        f"{len(test_data)} test tests."
    )
    print("Extracting features and training Random Forest...")
    analyzer = RandomForestAnalyzer(random_state=42)
    analyzer.train(train_data)

    print("Scoring held-out test rows...")
    results = analyzer.analyze(test_data)
    save_results_to_csv(results, output_path)

    metrics = None
    if "category" in results.columns:
        y_true = (results["category"] != 5).astype(int)
        metrics = compute_ranking_metrics(y_true, results["flaky_score"])
        print_metrics("Random Forest", metrics)

    if "test_name" in results.columns:
        print("Top 5 tests by predicted flaky risk:")
        top5 = results.nlargest(
            5, "flaky_score"
        )[["test_name", "flaky_score"]]
        print(top5.to_string(index=False))

    return metrics


def run_adaptive_fold(fold_number):
    """Train on one fold, score with AdaptiveAnalyzer, and evaluate."""
    train_path = f"src/data/train_set_{fold_number}.csv"
    test_path = f"src/data/test_set_{fold_number}.csv"
    output_path = f"src/results/test_set_{fold_number}_adaptive_results.csv"

    print(f"Running Adaptive Hybrid analyzer for fold {fold_number}")
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    if train_data is None or test_data is None:
        print("Could not load the fold data; skipping this fold.")
        return None, None

    print(
        f"Loaded {len(train_data)} training tests and "
        f"{len(test_data)} test tests."
    )
    print("Training Adaptive Analyzer (Stage 1 Rules + Stage 2 ML)...")
    analyzer = AdaptiveAnalyzer()
    analyzer.train(train_data)

    print("Scoring held-out test rows...")
    results = analyzer.analyze(test_data)
    save_results_to_csv(results, output_path)

    routing = analyzer.get_routing_summary()
    print("Stage 1 Routing Breakdown:")
    print(
        f"  Fast Clean:  {routing['fast_clean_count']} tests "
        f"({routing['fast_clean_pct']:.1%}) [Skips ML]"
    )
    print(
        f"  ML Routed:   {routing['ml_deep_count']} tests "
        f"({routing['ml_deep_pct']:.1%}) [Requires ML]"
    )
    print(
        f"  Fast Flaky:  {routing['fast_flaky_count']} tests "
        f"({routing['fast_flaky_pct']:.1%}) [Immediate Flag]"
    )

    metrics = None
    if "category" in results.columns:
        y_true = (results["category"] != 5).astype(int)
        metrics = compute_ranking_metrics(y_true, results["flaky_score"])
        print_metrics("Adaptive Hybrid", metrics)

    if "test_name" in results.columns:
        print("Top 5 tests by predicted flaky risk:")
        top5 = results.nlargest(
            5, "flaky_score"
        )[["test_name", "flaky_score", "route"]]
        print(top5.to_string(index=False))

    return metrics, routing


def compare_folds(folds=(1, 2, 3, 4)):
    """Run all folds and print comparative summary table."""
    lr_all = []
    rf_all = []
    adapt_all = []
    routings = []

    for fold in folds:
        print(f"\n{'='*25} EVALUATING FOLD {fold} {'='*25}")
        m_lr = run_ml_fold(fold)
        print()
        m_rf = run_rf_fold(fold)
        print()
        m_ad, r_ad = run_adaptive_fold(fold)

        if m_lr:
            lr_all.append(m_lr)
        if m_rf:
            rf_all.append(m_rf)
        if m_ad:
            adapt_all.append(m_ad)
        if r_ad:
            routings.append(r_ad)

    if lr_all and rf_all and adapt_all:
        df_lr = pd.DataFrame(lr_all)
        df_rf = pd.DataFrame(rf_all)
        df_ad = pd.DataFrame(adapt_all)

        print(f"\n{'='*25} 4-FOLD BENCHMARK SUMMARY {'='*25}")
        header = (
            f"{'Metric':<16} | {'Logistic Reg':<14} | "
            f"{'Random Forest':<14} | {'Adaptive Hybrid':<16}"
        )
        print(header)
        print("-" * len(header))
        for key, label in [
            ("roc_auc", "ROC-AUC"),
            ("pr_auc", "PR-AUC"),
            ("recall@5%", "Recall@5%"),
            ("recall@10%", "Recall@10%"),
            ("recall@20%", "Recall@20%"),
            ("precision@5%", "Precision@5%"),
            ("precision@10%", "Precision@10%"),
        ]:
            v_lr = df_lr[key].mean()
            v_rf = df_rf[key].mean()
            v_ad = df_ad[key].mean()
            fmt = ".1%" if "recall" in key or "precision" in key else ".4f"
            print(
                f"{label:<16} | {v_lr:<14{fmt}} | "
                f"{v_rf:<14{fmt}} | {v_ad:<16{fmt}}"
            )
        print("=" * len(header))

        if routings:
            avg_clean = np.mean([r["fast_clean_pct"] for r in routings])
            avg_ml = np.mean([r["ml_deep_pct"] for r in routings])
            avg_flaky = np.mean([r["fast_flaky_pct"] for r in routings])
            print("\nAdaptive Efficiency (Stage 1 Routing Across 4 Folds):")
            print(f"  Fast Clean (Bypassed ML):  {avg_clean:.1%}")
            print(f"  ML Routed  (Evaluated):    {avg_ml:.1%}")
            print(f"  Fast Flaky (Bypassed ML):  {avg_flaky:.1%}")
            print(f"  Total ML Workload Saved:   {(1.0 - avg_ml):.1%}")


if __name__ == "__main__":
    compare_folds(folds=(1, 2, 3, 4))

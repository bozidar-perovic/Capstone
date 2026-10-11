from .MLAnalyzer import MLAnalyzer
from .utility import load_data, save_results_to_csv


def run_ml_fold(fold_number):
    """Train on one project-based fold, score it, and save results."""
    train_path = f"src/data/train_set_{fold_number}.csv"
    test_path = f"src/data/test_set_{fold_number}.csv"
    output_path = f"src/results/test_set_{fold_number}_results.csv"

    print(f"Running ML analyzer for fold {fold_number}")
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    if train_data is None or test_data is None:
        print("Could not load the fold data; skipping this fold.")
        return

    print(
        f"Loaded {len(train_data)} training tests and " f"{len(test_data)} test tests."
    )
    print("Extracting source-code features and training " "logistic regression...")
    analyzer = MLAnalyzer()
    analyzer.train(train_data)
    print(f"Model trained with features: {list(analyzer.FEATURE_PATTERNS)}")

    print("Scoring held-out test rows...")
    results = analyzer.analyze(test_data)
    save_results_to_csv(results, output_path)

    if "test_name" in results.columns:
        print("Top 5 tests by predicted flaky risk:")
        print(
            results.nlargest(5, "flaky_score")[["test_name", "flaky_score"]].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    for fold in (1, 2):
        run_ml_fold(fold)

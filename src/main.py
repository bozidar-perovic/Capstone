from .RuleBasedAnalyzer import RuleBasedAnalyzer
from .MLAnalyzer import MLAnalyzer
from .utility import load_data, save_results_to_csv

if __name__ == "__main__":
    rule_based_analyzer = RuleBasedAnalyzer()
    ml_analyzer = MLAnalyzer()

    # print(rule_based_analyzer.analyze(file_path))
    # print(get_comparison_data(file_path))

    train_data = load_data("src/data/train_set_1.csv")
    ml_analyzer.train(train_data)
    ml_analyzer.analyze(train_data)
    test_data = load_data("src/data/test_set_1.csv")
    results = ml_analyzer.analyze(test_data)
    save_results_to_csv(results, "src/results/test_set_1_results.csv")

    train_data = load_data("src/data/train_set_2.csv")
    ml_analyzer.train(train_data)
    ml_analyzer.analyze(train_data)
    test_data = load_data("src/data/test_set_2.csv")
    results = ml_analyzer.analyze(test_data)
    save_results_to_csv(results, "src/results/test_set_2_results.csv")

    train_data = load_data("src/data/train_set_3.csv")
    ml_analyzer.train(train_data)
    ml_analyzer.analyze(train_data)
    test_data = load_data("src/data/test_set_3.csv")
    results = ml_analyzer.analyze(test_data)
    save_results_to_csv(results, "src/results/test_set_3_results.csv")

    train_data = load_data("src/data/train_set_4.csv")
    ml_analyzer.train(train_data)
    ml_analyzer.analyze(train_data)
    test_data = load_data("src/data/test_set_4.csv")
    results = ml_analyzer.analyze(test_data)
    save_results_to_csv(results, "src/results/test_set_4_results.csv")

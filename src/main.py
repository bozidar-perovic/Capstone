from RuleBasedAnalyzer import RuleBasedAnalyzer
from utility import get_comparison_data

if __name__ == "__main__":
    rule_based_analyzer = RuleBasedAnalyzer()
    file_path = "src/data/test_set_1.csv"
    print(rule_based_analyzer.analyze(file_path))

    print(get_comparison_data(file_path))

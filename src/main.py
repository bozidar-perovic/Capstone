from RuleBasedAnalyzer import RuleBasedAnalyzer
from utility import load_data, get_comparison_data

if __name__ == "__main__":
    rule_based_analyzer = RuleBasedAnalyzer()
    file_path = "src/data/testing_code.csv"
    print(rule_based_analyzer.analyze(file_path))

    # data_insformation = get_comparison_data("src/data/testing_code.csv")
    # print(data_information)

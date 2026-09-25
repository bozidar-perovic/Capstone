from .RuleBasedAnalyzer import RuleBasedAnalyzer

if __name__ == "__main__":
    rule_based_analyzer = RuleBasedAnalyzer()
    file_path = "src/data/testing_code.csv"
    score = rule_based_analyzer.analyze(file_path)

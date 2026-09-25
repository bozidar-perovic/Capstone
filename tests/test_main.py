import pandas as pd

from src.RuleBasedAnalyzer import RuleBasedAnalyzer
from src.utility import load_data


def test_load_data_valid_file(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("full_code\nThread.sleep(1000);\n")

    data = load_data(csv_file)

    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert list(data.columns) == ["full_code"]
    assert data.loc[0, "full_code"] == "Thread.sleep(1000);"


def test_load_data_missing_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.csv"

    data = load_data(missing_file)

    assert data is None


def test_rule_based_solution_counts_all_matching_rows(tmp_path):
    csv_file = tmp_path / "rules.csv"
    csv_file.write_text(
        "full_code\n"
        "Thread.sleep(1000);\n"
        "new Thread(() -> {}).start();\n"
        'System.out.println("clean");\n'
    )
    analyzer = RuleBasedAnalyzer()
    analyzer.analyze(csv_file)

import pandas as pd

from src.RuleBasedAnalyzer import RuleBasedAnalyzer
from src.utility import load_data, get_comparison_data


# Testing the load_data function with valid data
def test_load_data_valid_file(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("full_code\nThread.sleep(1000);\n")

    data = load_data(csv_file)

    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert list(data.columns) == ["full_code"]
    assert data.loc[0, "full_code"] == "Thread.sleep(1000);"


# Testing the load_data function with a missing file
def test_load_data_missing_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.csv"

    data = load_data(missing_file)

    assert data is None


# Check comparison fields using a CSV with all required columns.
def test_get_comparison_data_valid_file(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(
        "id,project,test_name,category,label\n"
        "1,Project1,testOne,5,non-flaky\n"
        "2,Project2,testTwo,0,async wait\n"
    )

    data = get_comparison_data(csv_file)

    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert list(data.columns) == [
        "unique_identifier", "category", "label", "is_flaky"
    ]
    assert data.loc[0, "unique_identifier"] == "1_Project1_testOne"
    assert data.loc[0, "is_flaky"] == "no"
    assert data.loc[1, "unique_identifier"] == "2_Project2_testTwo"
    assert data.loc[1, "is_flaky"] == "yes"


# The analyzer scores every row and resets scores between rows and runs.
def test_rule_based_analyzer_scores_rows_independently(tmp_path):
    csv_file = tmp_path / "rules.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "project": ["Project1"] * 3,
            "test_name": ["sleepTest", "cleanTest", "threadTest"],
            "full_code": [
                "Thread.sleep(1000);",
                'System.out.println("clean");',
                "new Thread(() -> {}).start();",
            ],
        }
    ).to_csv(csv_file, index=False)

    analyzer = RuleBasedAnalyzer()
    first_result = analyzer.analyze(csv_file)
    second_result = analyzer.analyze(csv_file)

    assert first_result["flaky_score"].tolist() == [0.35, 0.0, 0.25]
    assert second_result["flaky_score"].tolist() == [0.35, 0.0, 0.25]

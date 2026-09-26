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


# Testing the load data function with a valid file that has the required columns
def test_get_comparison_data_valid_file(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(
        "id,project,category\n1,Project1,Category1\n2,Project2,Category2\n"
    )

    data = get_comparison_data(csv_file)

    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert list(data.columns) == ["id", "project", "category"]
    assert data.loc[0, "id"] == 1
    assert data.loc[0, "project"] == "Project1"
    assert data.loc[0, "category"] == "Category1"


# Testing the hardcoded solution with all the matching rows in the CSV file
def test_rule_based_solution_counts_all_matching_rows(tmp_path):
    csv_file = tmp_path / "rules.csv"
    csv_file.write_text(
        "full_code\n"
        "Thread.sleep(1000);\n"
        "TimeUnit.SECONDS.sleep(1);\n"
        "await someAsyncMethod();\n"
        "ExecutorService executor = Executors.newFixedThreadPool(10);\n"
        "AtomicInteger atomicInt = new AtomicInteger(0);\n"
        "Date date = new Date();\n"
        "Random random = new Random();\n"
        "SharedResource shared = new SharedResource();\n"
        "synchronized (lock) { /* critical section */ }\n"
        "new Thread(() -> {}).start();\n"
        "api_call();\n"
        'System.out.println("clean");\n'
    )
    analyzer = RuleBasedAnalyzer()
    analyzer.analyze(csv_file)

import pandas as pd
import pytest

from src.AdaptiveAnalyzer import AdaptiveAnalyzer
from src.RandomForestAnalyzer import RandomForestAnalyzer


def test_adaptive_analyzer_requires_training_before_scoring():
    analyzer = AdaptiveAnalyzer()
    with pytest.raises(ValueError, match="Train the model"):
        analyzer.analyze(pd.DataFrame({"full_code": ["Thread.sleep(1);"]}))


def test_adaptive_analyzer_routes_three_zones_correctly():
    train_data = pd.DataFrame(
        {
            "full_code": [
                "Thread.sleep(100);",
                "new Thread(task);",
                "assertTrue(true);",
                "assertEquals(1, 1);",
            ],
            "category": [0, 1, 5, 5],
        }
    )

    test_data = pd.DataFrame(
        {
            "full_code": [
                'System.out.println("hello");',  # Clean (0.0) -> fast_clean
                "await();",  # Ambiguous (0.15) -> ml_deep
                # High (0.60) -> fast_flaky
                "Thread.sleep(100); new Thread(r);",
            ]
        }
    )

    analyzer = AdaptiveAnalyzer().train(train_data)
    results = analyzer.analyze(test_data)

    assert results["route"].tolist() == ["fast_clean", "ml_deep", "fast_flaky"]
    assert results["flaky_score"].iloc[0] == 0.0
    assert 0.0 <= results["flaky_score"].iloc[1] <= 1.0
    assert results["flaky_score"].iloc[2] == 0.95


def test_adaptive_analyzer_routing_summary():
    train_data = pd.DataFrame(
        {
            "full_code": ["Thread.sleep(1);", "assertTrue(ok);"],
            "category": [0, 5],
        }
    )
    test_data = pd.DataFrame(
        {
            "full_code": [
                "assertTrue(true);",
                "await();",
                "Thread.sleep(1); new Thread(t);",
            ]
        }
    )

    analyzer = AdaptiveAnalyzer().train(train_data)
    analyzer.analyze(test_data)
    summary = analyzer.get_routing_summary()

    assert summary["total_tests"] == 3
    assert summary["fast_clean_count"] == 1
    assert summary["fast_clean_pct"] == pytest.approx(1 / 3)
    assert summary["ml_deep_count"] == 1
    assert summary["fast_flaky_count"] == 1


def test_adaptive_analyzer_with_random_forest_backend():
    train_data = pd.DataFrame(
        {
            "full_code": ["Thread.sleep(1);", "assertTrue(ok);"],
            "category": [0, 5],
        }
    )
    test_data = pd.DataFrame({"full_code": ["await();"]})

    rf = RandomForestAnalyzer(random_state=42)
    analyzer = AdaptiveAnalyzer(ml_analyzer=rf).train(train_data)
    results = analyzer.analyze(test_data)

    assert results["route"].iloc[0] == "ml_deep"
    assert 0.0 <= results["flaky_score"].iloc[0] <= 1.0


def test_adaptive_analyzer_does_not_double_count_thread_start():
    analyzer = AdaptiveAnalyzer()
    # "new Thread(task).start();" has both thread_creation and thread_start
    score = analyzer.compute_rule_score("new Thread(task).start();")
    assert score == pytest.approx(0.25)

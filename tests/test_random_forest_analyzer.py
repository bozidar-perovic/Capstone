"""Unit tests for RandomForestAnalyzer.

Verifies:
- Accurate risk probability estimation on test data.
- Strict training requirement before scoring or querying importances.
- Gini feature importances formatting and monotonic descending order.
- Deterministic score reproducibility with fixed random_state.
- Appropriate weighting ensuring assertion volume alone does not outweigh
  active concurrency indicators.
"""

import pandas as pd
import pytest

from src.RandomForestAnalyzer import RandomForestAnalyzer


def test_rf_analyzer_scores_unlabeled_test_rows():
    """Verify RandomForestAnalyzer predicts continuous probabilities."""
    train_data = pd.DataFrame(
        {
            "full_code": [
                "Thread.sleep(1);",
                "new Thread(task);",
                "Thread.sleep(2);",
                "assertTrue(true);",
                "assertEquals(1, 1);",
                "assertFalse(false);",
            ],
            "category": [0, 1, 0, 5, 5, 5],
        }
    )
    test_data = pd.DataFrame(
        {"full_code": ["Thread.sleep(3);", "assertTrue(true);"]}
    )

    analyzer = RandomForestAnalyzer(random_state=42)
    analyzer.train(train_data)
    result = analyzer.analyze(test_data)

    assert len(result) == len(test_data)
    assert result["flaky_score"].between(0, 1).all()
    assert "flaky_score" not in test_data.columns


def test_rf_analyzer_requires_training_before_scoring():
    """Verify ValueError is raised if analyze is called before training."""
    analyzer = RandomForestAnalyzer()

    with pytest.raises(ValueError, match="Train the model"):
        analyzer.analyze(pd.DataFrame({"full_code": ["Thread.sleep(1);"]}))


def test_rf_analyzer_feature_importances():
    """Verify feature importances sum to 1.0 and are sorted descending."""
    train_data = pd.DataFrame(
        {
            "full_code": [
                "Thread.sleep(1);",
                "new Thread(task);",
                "Thread.sleep(2);",
                "assertTrue(true);",
                "assertEquals(1, 1);",
                "assertFalse(false);",
            ],
            "category": [0, 1, 0, 5, 5, 5],
        }
    )
    analyzer = RandomForestAnalyzer(random_state=42)
    with pytest.raises(ValueError, match="Train the model"):
        analyzer.get_feature_importances()

    analyzer.train(train_data)
    importances = analyzer.get_feature_importances()

    assert len(importances) == 14
    assert pytest.approx(importances.sum(), abs=1e-5) == 1.0
    # Verified sorted descending
    assert importances.is_monotonic_decreasing


def test_rf_analyzer_reproducibility():
    """Verify identical risk predictions across runs with matching seeds."""
    train_data = pd.DataFrame(
        {
            "full_code": [
                "Thread.sleep(1);",
                "new Thread(task);",
                "assertTrue(true);",
                "assertEquals(1, 1);",
            ],
            "category": [0, 1, 5, 5],
        }
    )
    test_data = pd.DataFrame({"full_code": ["Thread.sleep(5);"]})

    rf1 = RandomForestAnalyzer(random_state=42).train(train_data)
    rf2 = RandomForestAnalyzer(random_state=42).train(train_data)

    score1 = rf1.analyze(test_data)["flaky_score"].iloc[0]
    score2 = rf2.analyze(test_data)["flaky_score"].iloc[0]

    assert score1 == score2


def test_rf_analyzer_does_not_overpenalize_assertions_without_concurrency():
    """Verify that assertion volume alone does not outweigh concurrency."""
    train_data = pd.DataFrame(
        {
            "full_code": [
                "Thread.sleep(100);",
                "await();",
                "Thread.sleep(50);",
                "assertTrue(true);",
                "assertEquals(1, 1);",
                "assertNotNull(obj);",
            ],
            "category": [0, 0, 0, 5, 5, 5],
        }
    )
    test_data = pd.DataFrame(
        {
            "full_code": [
                "Thread.sleep(100); assertTrue(ok);",
                "assertTrue(ok);" * 100,  # 100 assertions, zero concurrency
            ]
        }
    )

    analyzer = RandomForestAnalyzer(random_state=42).train(train_data)
    results = analyzer.analyze(test_data)

    concurrent_score = results["flaky_score"].iloc[0]
    pure_assert_score = results["flaky_score"].iloc[1]

    # Concurrent test must have significantly higher flaky risk
    # than pure assertions
    assert concurrent_score > pure_assert_score

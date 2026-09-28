import pandas as pd
import pytest

from src.MLAnalyzer import MLAnalyzer


def test_ml_analyzer_scores_unlabeled_test_rows():
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

    analyzer = MLAnalyzer()
    analyzer.train(train_data)
    result = analyzer.analyze(test_data)

    assert len(result) == len(test_data)
    assert result["flaky_score"].between(0, 1).all()
    assert "flaky_score" not in test_data.columns


def test_ml_analyzer_requires_training_before_scoring():
    analyzer = MLAnalyzer()

    with pytest.raises(ValueError, match="Train the model"):
        analyzer.analyze(pd.DataFrame({"full_code": ["Thread.sleep(1);"]}))

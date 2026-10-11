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
    test_data = pd.DataFrame({"full_code": ["Thread.sleep(3);", "assertTrue(true);"]})

    analyzer = MLAnalyzer()
    analyzer.train(train_data)
    result = analyzer.analyze(test_data)

    assert len(result) == len(test_data)
    assert result["flaky_score"].between(0, 1).all()
    assert "flaky_score" not in test_data.columns


def test_extract_features_counts_each_row_independently():
    data = pd.DataFrame(
        {
            "full_code": [
                "Thread.sleep(1); Thread.sleep (timeout); new Thread(task); await();",
                "new  Thread (task); await(); await();",
                None,
            ]
        },
        index=[10, 20, 30],
    )

    features = MLAnalyzer().extract_features(data)

    assert features[
        [
            "sleep_count",
            "thread_creation_count",
            "await_count",
        ]
    ].values.tolist() == [[2, 1, 1], [0, 1, 2], [0, 0, 0]]
    assert features.shape == (3, 14)
    assert features.index.tolist() == [10, 20, 30]


def test_extract_features_detects_additional_signal_groups():
    data = pd.DataFrame(
        {
            "full_code": [
                "worker.start(); Executors.newFixedThreadPool(2); "
                "Future result; CountDownLatch latch; "
                "System.nanoTime(); new Random(); new HashMap<>(); "
                "Files.readString(path); assertTrue(ok); "
                "for (;;) { if (ready) break; }"
            ]
        }
    )

    features = MLAnalyzer().extract_features(data).iloc[0]

    expected = {
        "thread_start_count": 1,
        "executor_count": 1,
        "future_count": 1,
        "latch_barrier_count": 1,
        "time_api_count": 1,
        "random_api_count": 1,
        "unordered_collection_count": 1,
        "file_network_count": 1,
        "assertion_count": 1,
        "loop_count": 1,
        "branch_count": 1,
    }
    for name, count in expected.items():
        assert features[name] == count


def test_ml_analyzer_requires_training_before_scoring():
    analyzer = MLAnalyzer()

    with pytest.raises(ValueError, match="Train the model"):
        analyzer.analyze(pd.DataFrame({"full_code": ["Thread.sleep(1);"]}))

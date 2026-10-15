"""Unit tests for MLAnalyzer.

Verifies:
- Risk probability prediction on unlabeled test inputs.
- Independent row-by-row feature count extraction.
- Pattern matching across all 14 static signal groups.
- Proper feature clipping (max_count=5), log1p scaling, and StandardScaler fit.
- Clean execution in uncapped (max_count=None) mode.
- Strict enforcement of training before inference.
"""

import numpy as np
import pandas as pd
import pytest

from src.MLAnalyzer import MLAnalyzer


def test_ml_analyzer_scores_unlabeled_test_rows():
    """Verify MLAnalyzer outputs valid probability scores in [0, 1]."""
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


def test_extract_features_counts_each_row_independently():
    """Verify feature counts are computed independently for each row."""
    data = pd.DataFrame(
        {
            "full_code": [
                "Thread.sleep(1); Thread.sleep (timeout); "
                "new Thread(task); await();",
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
    """Verify regex patterns match each individual feature group."""
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


def test_ml_pipeline_log_scales_and_fits_scaler_only_on_training_rows():
    """Verify pipeline clips, log1p-scales, and isolates scaler parameters."""
    train = pd.DataFrame({
        "full_code": ["", "assertTrue(ok);" * 93, "Thread.sleep(1);"],
        "category": [5, 5, 0],
    })
    analyzer = MLAnalyzer(max_count=5).train(train)
    raw = analyzer.extract_features(train)
    clipped = analyzer.model.named_steps["clip"].transform(raw)
    np.testing.assert_allclose(clipped, np.clip(raw, 0, 5))

    logged = analyzer.model.named_steps["log"].transform(clipped)
    np.testing.assert_allclose(logged, np.log1p(np.clip(raw, 0, 5)))

    assertion_column = raw.columns.get_loc("assertion_count")
    assert logged[0, assertion_column] == 0
    # 93 assertions is capped at 5 before log-scaling
    assert logged[1, assertion_column] == pytest.approx(np.log1p(5))

    scaler = analyzer.model.named_steps["scaler"]
    expected_mean = np.log1p(np.clip(raw, 0, 5)).mean(axis=0)
    np.testing.assert_allclose(scaler.mean_, expected_mean)
    original_mean = scaler.mean_.copy()
    original_scale = scaler.scale_.copy()
    analyzer.analyze(pd.DataFrame({"full_code": ["assertTrue(ok);" * 1000]}))
    np.testing.assert_array_equal(scaler.mean_, original_mean)
    np.testing.assert_array_equal(scaler.scale_, original_scale)
    assert scaler.n_samples_seen_ == len(train)


def test_ml_analyzer_supports_uncapped_mode():
    """Verify MLAnalyzer can run without clipping when max_count is None."""
    train = pd.DataFrame({
        "full_code": ["", "assertTrue(ok);" * 93, "Thread.sleep(1);"],
        "category": [5, 5, 0],
    })
    analyzer = MLAnalyzer(max_count=None).train(train)
    assert "clip" not in analyzer.model.named_steps
    raw = analyzer.extract_features(train)
    logged = analyzer.model.named_steps["log"].transform(raw)
    assertion_column = raw.columns.get_loc("assertion_count")
    assert logged[1, assertion_column] == pytest.approx(np.log1p(93))


def test_ml_analyzer_requires_training_before_scoring():
    """Verify ValueError is raised when scoring without prior training."""
    analyzer = MLAnalyzer()

    with pytest.raises(ValueError, match="Train the model"):
        analyzer.analyze(pd.DataFrame({"full_code": ["Thread.sleep(1);"]}))

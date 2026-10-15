"""Unit tests for RuleBasedAnalyzer.

Verifies:
- Presence-based scoring (indicator fires once regardless of count).
- Deduplication of thread creation and start calls.
- Correct weight summation across distinct flakiness indicators.
- Safe handling of empty inputs and state reset across consecutive calls.
"""

import pandas as pd
import pytest

from src.RuleBasedAnalyzer import RuleBasedAnalyzer


def test_rules_use_shared_patterns_and_presence_not_counts():
    """Verify indicators score feature presence rather than frequency."""
    data = pd.DataFrame({"full_code": [
        "Thread . sleep (1);",
        "Thread . sleep (1);" * 100,
        "assertTrue(ok); if (ok) {} for (;;) {}" * 93,
        "runtime(); apiTest(); awaitResult(); syncTest();",
        None,
    ]}, index=[10, 20, 30, 40, 50])
    result = RuleBasedAnalyzer().analyze(data)
    assert result["flaky_score"].tolist() == [0.35, 0.35, 0, 0, 0]
    assert result.index.tolist() == data.index.tolist()
    assert "flaky_score" not in data


def test_rules_do_not_double_count_thread_creation_and_start():
    """Verify that new Thread() and .start() share a single weight."""
    data = pd.DataFrame({"full_code": [
        "new Thread(task).start();", "worker.start();",
    ]})
    scores = RuleBasedAnalyzer().analyze(data)["flaky_score"]
    assert scores.tolist() == [0.25, 0.25]


def test_rules_score_specific_time_and_random_apis():
    """Verify accurate weight summation for time and random API calls."""
    data = pd.DataFrame({"full_code": [
        "System . nanoTime (); Math . random ();",
    ]})
    score = RuleBasedAnalyzer().analyze(data)["flaky_score"].iloc[0]
    assert score == pytest.approx(0.4)


def test_rules_handle_empty_input_and_reset_state():
    """Verify that empty DataFrames are handled cleanly and state is reset."""
    analyzer = RuleBasedAnalyzer()
    analyzer.analyze(pd.DataFrame({"full_code": ["Thread.sleep(1);"]}))
    result = analyzer.analyze(pd.DataFrame({"full_code": []}))
    assert result.empty
    assert "flaky_score" in result
    assert analyzer.scores == []
    assert analyzer.total_score == 0

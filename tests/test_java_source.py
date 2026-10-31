"""Masking preserves executable code and excludes comment/literal text."""

import pandas as pd
import pytest

from src.features import extract_features
from src.java_source import mask_non_code
from src.RuleBasedAnalyzer import RuleBasedAnalyzer


@pytest.mark.parametrize("source,expected", [
    ("// Thread.sleep(500);", 0.0),
    ("/* Thread.sleep(500); await(); */", 0.0),
    ('String s = "Thread.sleep(500); await();";', 0.0),
    (r'String s = "escaped \" Thread.sleep(1); /* await */";', 0.0),
    ('String url = "https://host/*await*/"; Thread.sleep(1);', 0.35),
    ("char quote = '\\''; Thread.sleep(1);", 0.35),
    (r"char slash = '\\'; Thread.sleep(1);", 0.35),
    ("char marker = '/'; Math.random();", 0.3),
    ('String s = """\nThread.sleep(1); await()\n"""; Math.random();', 0.3),
    ('String s = """\n' + r'\""" Thread.sleep(1);' +
     '\n"""; Math.random();', 0.3),
    ("Thread/* await */.sleep(1);", 0.35),
    ("Thr/* comment */ead.sleep(1);", 0.0),
    ("// Thread.sleep(1);\rawait();", 0.15),
    ("// Thread.sleep(1);\r\nawait();", 0.15),
    ("/* unfinished Thread.sleep(1);", 0.0),
    ('String s = "unfinished Thread.sleep(1);', 0.0),
    ('String s = """\nunfinished Thread.sleep(1);', 0.0),
    (None, 0.0),
])
def test_masked_rules_exclude_non_code_and_keep_real_calls(source, expected):
    data = pd.DataFrame({"full_code": [source]}, index=[17])
    original = data.copy(deep=True)
    result = RuleBasedAnalyzer(source_mode="masked").analyze(data)
    assert result.flaky_score.iloc[0] == pytest.approx(expected)
    assert result.index.tolist() == [17]
    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_series_equal(result.full_code, original.full_code)


@pytest.mark.parametrize("source", [
    r'String s = \u0022Thread.sleep(1); await()\u0022; Math.random();',
    r'String s = \uuuu0022Thread.sleep(1)\uu0022; Math.random();',
    r'\u002f\u002f Thread.sleep(1);\u000a Math.random();',
    r'\u002f* Thread.sleep(1); *\u002f Math.random();',
    r'String s = "\\u0022 Thread.sleep(1)"; Math.random();',
    r'String s = "\u005c" Thread.sleep(1)"; Math.random();',
    r'String s = "\u005c\u0022 Thread.sleep(1)"; Math.random();',
])
def test_unicode_escapes_recognize_boundaries_without_fake_quotes(source):
    result = mask_non_code(source)
    assert len(result) == len(source)
    assert "Thread.sleep" not in result
    data = pd.DataFrame({"full_code": [source]})
    scores = RuleBasedAnalyzer(source_mode="masked").analyze(data)
    assert scores.flaky_score.iloc[0] == pytest.approx(0.3)


def test_spacing_line_endings_and_executable_spelling_are_preserved():
    source = (
        'Thread/* await\r\n */.sleep("Random"); // new Thread()\r'
        '\nassertTrue(ok);\n'
    )
    expected = (
        'Thread        \r\n   .sleep("      ");                \r'
        '\nassertTrue(ok);\n'
    )
    masked = mask_non_code(source)
    assert masked == expected
    assert len(masked) == len(source)
    assert [i for i, c in enumerate(masked) if c in "\r\n"] == [
        i for i, c in enumerate(source) if c in "\r\n"
    ]


def test_assertion_call_survives_masking_its_argument():
    source = 'assertEquals("await Thread.sleep(1); for if Random", actual);'
    counts = extract_features(pd.DataFrame({
        "full_code": [mask_non_code(source)],
    })).iloc[0]
    assert counts.assertion_count == 1
    for name in ("sleep_count", "await_count", "loop_count", "branch_count",
                 "random_api_count"):
        assert counts[name] == 0


def test_real_code_is_unchanged_and_comments_do_not_join_identifiers():
    code = "Thread.sleep(1); new Thread(task).start(); Math.random();"
    assert mask_non_code(code) == code
    assert mask_non_code("Thr/**/ead") == "Thr    ead"


def test_masked_thread_start_can_contain_punctuation_inside_argument():
    source = 'new Thread("await; Thread.sleep(1)").start();'
    masked = mask_non_code(source)
    counts = extract_features(pd.DataFrame({"full_code": [masked]})).iloc[0]
    assert counts.thread_creation_count == counts.thread_start_count == 1
    assert counts.sleep_count == counts.await_count == 0


def test_raw_mode_keeps_baseline_and_invalid_mode_is_rejected():
    data = pd.DataFrame({"full_code": ['// Thread.sleep(1);']})
    assert RuleBasedAnalyzer().analyze(data).flaky_score.iloc[0] == 0.35
    assert RuleBasedAnalyzer("raw").analyze(data).flaky_score.iloc[0] == 0.35
    assert RuleBasedAnalyzer("masked").analyze(data).flaky_score.iloc[0] == 0
    with pytest.raises(ValueError, match="source_mode"):
        RuleBasedAnalyzer("maskedd")


def test_masked_mode_handles_empty_frames_and_csv_input(tmp_path):
    analyzer = RuleBasedAnalyzer("masked")
    assert analyzer.analyze(pd.DataFrame({"full_code": []})).empty
    data = pd.DataFrame({
        "id": [1], "project": ["project"], "test_name": ["test"],
        "full_code": ['String s = "Thread.sleep(1);";'],
    })
    source = tmp_path / "tests.csv"
    data.to_csv(source, index=False)
    result = analyzer.analyze(source)
    assert result.full_code.tolist() == data.full_code.tolist()
    assert result.flaky_score.iloc[0] == 0

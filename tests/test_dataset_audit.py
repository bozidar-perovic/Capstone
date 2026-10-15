"""Unit tests for dataset_audit data validation and leakage detection tool.

Verifies:
- Tokenizer precision: comment removal while preserving string literals.
- Structural issue auditing (unbalanced braces, missing bodies, annotations).
- Duplicate source hashing and conflicting label detection.
- Schema validation and category-label alignment.
- Cross-split overlap detection (projects, code hashes, method names, IDs).
- CLI report generation without mutating dataset inputs.
"""

import json

import pandas as pd

from src.dataset_audit import audit_frame, audit_overlap, main, source_tokens


def frame(codes, categories=None):
    """Helper creating a standard test DataFrame for audit tests."""
    count = len(codes)
    return pd.DataFrame({
        "id": list(range(count)), "project": ["example"] * count,
        "test_name": [f"Test.method{i}" for i in range(count)],
        "full_code": codes, "label": ["non-flaky"] * count,
        "category": categories or [5] * count,
    })


def test_source_tokens_preserve_literals_and_ignore_comments():
    """Verify tokenizer strips comments while preserving string literals."""
    tokens = source_tokens(
        'String url = "http://host/*path*/"; // await\n'
        'char brace = \'}\'; /* { */ String quote = "escaped\\\"quote";'
    )
    assert '"http://host/*path*/"' in tokens
    assert "'}'" in tokens
    assert '"escaped\\\"quote"' in tokens
    assert "await" not in tokens
    assert "{" not in tokens


def test_audit_flags_source_problems_without_changing_input():
    """Verify audit flags malformed syntax without modifying input data."""
    data = frame([
        None, "  ", "// @Test void test() {}", "@BeforeEach void setup() {}",
        "@Test void test()", "@Test void test() { new Runnable() {",
        '@Test void test() { String s = "}"; /* { */ }',
        "@TestAnnotation void helper() {}",
    ])
    original = data.copy(deep=True)
    summary, issues, _ = audit_frame(data, "tests.csv")
    by_row = {}
    for issue in issues:
        by_row.setdefault(issue["row_number"], set()).add(issue["issue"])
    assert summary["rows"] == 8
    assert "missing_source" in by_row[2]
    assert "missing_source" in by_row[3]
    assert "comment_only_source" in by_row[4]
    assert "lifecycle_method" in by_row[5]
    assert "missing_method_body" in by_row[6]
    assert "unbalanced_braces" in by_row[7]
    assert 8 not in by_row
    assert "no_recognized_test_annotation" in by_row[9]
    pd.testing.assert_frame_equal(data, original)


def test_duplicate_conflicts_ignore_comments_not_literal_whitespace():
    """Verify duplicate source hashing ignores whitespace differences."""
    data = frame([
        '@Test void test() { String s = "a b"; }',
        '@Test void test(){/* comment */String s="a b";}',
        '@Test void test() { String s = "ab"; }',
        None, None,
    ], [0, 5, 5, 5, 5])
    data.loc[0, "label"] = "async wait"
    summary, _, duplicates = audit_frame(data, "tests.csv")
    assert summary["duplicate_groups"] == 1
    assert summary["conflicting_duplicate_groups"] == 1
    assert len(duplicates) == 2
    assert all(d["binary_label_conflict"] for d in duplicates)


def test_labels_and_schema_validation():
    """Verify schema column validation and invalid category detection."""
    data = frame(["@Test void test() {}"] * 4, ["5", "invalid", 6, 1])
    _, issues, _ = audit_frame(data, "tests.csv")
    assert sum(i["issue"] == "invalid_category" for i in issues) == 2
    assert sum(i["issue"] == "label_category_mismatch" for i in issues) == 1
    summary, issues, duplicates = audit_frame(pd.DataFrame(), "empty.csv")
    assert "full_code" in summary["missing_columns"]
    assert issues[0]["issue"] == "missing_columns"
    assert duplicates == []


def test_overlap_detects_project_identity_code_name_and_id():
    """Verify cross-split leakage detector flags project and code overlap."""
    train = frame(["@Test void test() {}"])
    test = train.copy()
    assert {i["kind"] for i in audit_overlap(train, test, 1)} == {
        "project", "test_identity", "source_hash", "test_name", "id",
    }
    # When project differs, source code, name, and ID overlap are still caught
    test["project"] = "different"
    assert {i["kind"] for i in audit_overlap(train, test, 1)} == {
        "source_hash", "test_name", "id",
    }
    # When all identifiers and code differ, no overlap remains
    test["id"] = 999
    test["test_name"] = "Other.other"
    test["full_code"] = "@Test void other() { int x = 1; }"
    assert audit_overlap(train, test, 1) == []


def test_overlap_detects_conflicting_source_hash():
    """Verify conflicting source hashes across splits are flagged."""
    train = frame(["@Test void test() {}"], categories=[5])
    test = frame(["@Test void test() {}"], categories=[0])
    test["project"] = "different"
    test["id"] = 999
    test["test_name"] = "Other.other"
    overlaps = audit_overlap(train, test, 1)
    assert {i["kind"] for i in overlaps} == {
        "source_hash", "conflicting_source_hash",
    }


def test_cli_writes_reports_and_leaves_files_unchanged(tmp_path):
    """Verify audit CLI generates reporting files without altering dataset."""
    for split in ("train", "test"):
        frame(["@Test void test() {}"]).to_csv(
            tmp_path / f"{split}_set_1.csv", index=False
        )
    path = tmp_path / "test_set_1.csv"
    original = path.read_bytes()
    output = tmp_path / "audit"
    main(["--data-dir", str(tmp_path), "--folds", "1",
          "--output-dir", str(output)])
    assert path.read_bytes() == original
    summary = json.loads((output / "summary.json").read_text())
    assert summary["fold_overlap_count"] == 5
    assert summary["fold_overlap_by_kind"] == {
        "project": 1, "test_identity": 1, "source_hash": 1,
        "test_name": 1, "id": 1,
    }
    assert pd.read_csv(output / "row_issues.csv").empty
    assert pd.read_csv(output / "duplicate_sources.csv").empty
    assert len(pd.read_csv(output / "fold_overlap.csv")) == 5

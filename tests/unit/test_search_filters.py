"""Tests for search filters in MCP server."""

from __future__ import annotations

import pytest

from stropha.server import _apply_filters
from stropha.models import SearchHit


def _make_hit(
    rel_path: str = "src/main.py",
    language: str = "python",
    kind: str = "function",
    rank: int = 1,
) -> SearchHit:
    """Create a minimal SearchHit for testing."""
    return SearchHit(
        chunk_id="test-chunk",
        rel_path=rel_path,
        start_line=1,
        end_line=10,
        language=language,
        kind=kind,
        symbol="test_func",
        snippet="def test_func(): pass",
        score=1.0,
        rank=rank,
    )


class TestApplyFilters:
    """Tests for _apply_filters helper function."""

    def test_no_filters_returns_all(self) -> None:
        hits = [_make_hit(), _make_hit(), _make_hit()]
        result = _apply_filters(hits)
        assert len(result) == 3

    def test_language_filter_single(self) -> None:
        hits = [
            _make_hit(language="python"),
            _make_hit(language="java"),
            _make_hit(language="python"),
        ]
        result = _apply_filters(hits, language=["python"])
        assert len(result) == 2
        assert all(h.language == "python" for h in result)

    def test_language_filter_multiple(self) -> None:
        hits = [
            _make_hit(language="python"),
            _make_hit(language="java"),
            _make_hit(language="typescript"),
        ]
        result = _apply_filters(hits, language=["python", "java"])
        assert len(result) == 2
        assert {h.language for h in result} == {"python", "java"}

    def test_language_filter_case_insensitive(self) -> None:
        hits = [_make_hit(language="Python"), _make_hit(language="JAVA")]
        result = _apply_filters(hits, language=["python", "java"])
        assert len(result) == 2

    def test_path_prefix_filter(self) -> None:
        hits = [
            _make_hit(rel_path="backend/src/main.py"),
            _make_hit(rel_path="frontend/src/app.ts"),
            _make_hit(rel_path="backend/test/test_main.py"),
        ]
        result = _apply_filters(hits, path_prefix="backend/")
        assert len(result) == 2
        assert all(h.rel_path.startswith("backend/") for h in result)

    def test_kind_filter_single(self) -> None:
        hits = [
            _make_hit(kind="function"),
            _make_hit(kind="class"),
            _make_hit(kind="method"),
        ]
        result = _apply_filters(hits, kind=["class"])
        assert len(result) == 1
        assert result[0].kind == "class"

    def test_kind_filter_multiple(self) -> None:
        hits = [
            _make_hit(kind="function"),
            _make_hit(kind="class"),
            _make_hit(kind="method"),
        ]
        result = _apply_filters(hits, kind=["class", "method"])
        assert len(result) == 2
        assert {h.kind for h in result} == {"class", "method"}

    def test_exclude_tests_filter(self) -> None:
        hits = [
            _make_hit(rel_path="src/main.py"),
            _make_hit(rel_path="tests/test_main.py"),
            _make_hit(rel_path="src/test_utils.py"),  # test_ prefix
            _make_hit(rel_path="backend/test/foo.py"),  # /test/ folder
        ]
        result = _apply_filters(hits, exclude_tests=True)
        assert len(result) == 1
        assert result[0].rel_path == "src/main.py"

    def test_combined_filters(self) -> None:
        hits = [
            _make_hit(rel_path="backend/src/main.py", language="python", kind="function"),
            _make_hit(rel_path="backend/src/main.py", language="python", kind="class"),
            _make_hit(rel_path="backend/test/test_main.py", language="python", kind="function"),
            _make_hit(rel_path="frontend/src/app.ts", language="typescript", kind="function"),
        ]
        result = _apply_filters(
            hits,
            language=["python"],
            path_prefix="backend/",
            kind=["function"],
            exclude_tests=True,
        )
        assert len(result) == 1
        assert result[0].rel_path == "backend/src/main.py"
        assert result[0].kind == "function"

    def test_empty_hits_returns_empty(self) -> None:
        result = _apply_filters([], language=["python"])
        assert result == []

    def test_no_matches_returns_empty(self) -> None:
        hits = [_make_hit(language="python")]
        result = _apply_filters(hits, language=["java"])
        assert result == []

"""Tests for query decomposition."""

from __future__ import annotations

import pytest

from stropha.retrieval.query_decomposition import (
    DecompositionResult,
    _is_compound_query,
    _pattern_decompose,
    decompose_query,
    is_decomposition_enabled,
)


class TestIsCompoundQuery:
    """Test compound query detection."""

    @pytest.mark.parametrize("query", [
        "how does X work and what calls Y",
        "where is the scheduler and how does mastery work",
        "how does authentication work or how does authorization work",
        "what is X and also what is Y",
    ])
    def test_detects_compound_queries(self, query: str):
        assert _is_compound_query(query) is True

    @pytest.mark.parametrize("query", [
        "how does X work",
        "what calls FsrsCalculator",
        "find all API endpoints",
        "show me the authentication flow",
    ])
    def test_detects_simple_queries(self, query: str):
        assert _is_compound_query(query) is False


class TestPatternDecompose:
    """Test pattern-based decomposition."""

    def test_splits_on_and_how(self):
        result = _pattern_decompose("how does X work and how does Y work")
        assert result is not None
        assert len(result) == 2
        assert "X" in result[0]
        assert "Y" in result[1]

    def test_splits_on_and_what(self):
        result = _pattern_decompose("what is X and what calls Y")
        assert result is not None
        assert len(result) == 2

    def test_splits_on_or_where(self):
        result = _pattern_decompose("where is X or where is Y defined")
        assert result is not None
        assert len(result) == 2

    def test_no_split_on_simple_query(self):
        result = _pattern_decompose("how does authentication work")
        assert result is None

    def test_no_split_on_and_without_question(self):
        result = _pattern_decompose("authentication and authorization")
        assert result is None


class TestDecomposeQuery:
    """Test the main decompose_query function."""

    def test_returns_original_for_simple_query(self):
        result = decompose_query("how does X work", use_llm=False)
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "how does X work"
        assert result.method == "none"

    def test_decomposes_compound_query(self):
        result = decompose_query(
            "how does X work and what calls Y",
            use_llm=False
        )
        assert len(result.sub_queries) >= 2
        assert result.method == "pattern"

    def test_empty_query(self):
        result = decompose_query("", use_llm=False)
        assert len(result.sub_queries) == 0
        assert result.method == "none"

    def test_whitespace_query(self):
        result = decompose_query("   ", use_llm=False)
        assert len(result.sub_queries) == 0
        assert result.method == "none"

    def test_force_decomposition(self):
        # Even simple queries can be forced through decomposition
        result = decompose_query(
            "how does authentication work",
            use_llm=False,
            force=True
        )
        # Should still return [query] since pattern doesn't match
        assert len(result.sub_queries) == 1

    def test_decomposition_result_dataclass(self):
        result = DecompositionResult(
            sub_queries=["q1", "q2"],
            original_query="q1 and q2",
            method="pattern",
        )
        assert result.sub_queries == ["q1", "q2"]
        assert result.original_query == "q1 and q2"
        assert result.method == "pattern"


class TestIsDecompositionEnabled:
    """Test environment-based enable check."""

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("STROPHA_QUERY_DECOMPOSITION_ENABLED", raising=False)
        assert is_decomposition_enabled() is False

    def test_enabled_when_set(self, monkeypatch):
        monkeypatch.setenv("STROPHA_QUERY_DECOMPOSITION_ENABLED", "1")
        assert is_decomposition_enabled() is True

    def test_disabled_when_zero(self, monkeypatch):
        monkeypatch.setenv("STROPHA_QUERY_DECOMPOSITION_ENABLED", "0")
        assert is_decomposition_enabled() is False


class TestRealWorldQueries:
    """Test with realistic code search queries."""

    @pytest.mark.parametrize("query,expected_parts", [
        (
            "how does the FSRS algorithm calculate intervals and how does it update card state",
            2
        ),
        (
            "where is the scheduler defined and what methods does it call",
            2
        ),
        (
            "how does authentication work and how does authorization work and what checks permissions",
            3
        ),
    ])
    def test_realistic_compound_queries(self, query: str, expected_parts: int):
        result = decompose_query(query, use_llm=False)
        assert len(result.sub_queries) >= expected_parts
        assert result.method == "pattern"

    @pytest.mark.parametrize("query", [
        "explain the FSRS algorithm implementation",
        "show me all database migration files",
        "find the authentication middleware",
        "how does the caching layer work",
    ])
    def test_simple_queries_not_decomposed(self, query: str):
        result = decompose_query(query, use_llm=False)
        assert len(result.sub_queries) == 1
        assert result.method == "none"

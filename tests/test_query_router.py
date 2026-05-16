"""Tests for query routing and smart search."""

from __future__ import annotations

import pytest

from stropha.retrieval.query_router import (
    QueryIntent,
    RoutedQuery,
    is_graph_intent,
    route_query,
)


class TestPatternMatching:
    """Test pattern-based intent classification."""

    # find_callers patterns
    @pytest.mark.parametrize("query,expected_symbol", [
        ("what calls FsrsCalculator", "FsrsCalculator"),
        ("who calls submitAnswer", "submitAnswer"),
        ("what method calls UserService.authenticate", "UserService.authenticate"),
        ("callers of processCard", "processCard"),
        ("callers for StudyService", "StudyService"),
        ("where is calculateMastery called", "calculateMastery"),
        ("FsrsCalculator is called by", "FsrsCalculator"),
    ])
    def test_find_callers_patterns(self, query: str, expected_symbol: str):
        result = route_query(query, use_llm=False)
        assert result.intent == QueryIntent.FIND_CALLERS
        assert result.symbol == expected_symbol
        assert result.confidence >= 0.8

    # find_tests patterns
    @pytest.mark.parametrize("query,expected_symbol", [
        ("tests for FsrsCalculator", "FsrsCalculator"),
        ("tests of UserService", "UserService"),
        ("how is submitAnswer tested", "submitAnswer"),
        ("what tests FsrsCalculator", "FsrsCalculator"),
        ("test coverage for processCard", "processCard"),
    ])
    def test_find_tests_patterns(self, query: str, expected_symbol: str):
        result = route_query(query, use_llm=False)
        assert result.intent == QueryIntent.FIND_TESTS
        assert result.symbol == expected_symbol
        assert result.confidence >= 0.8

    # find_related patterns
    @pytest.mark.parametrize("query,expected_symbol", [
        ("what relates to FsrsCalculator", "FsrsCalculator"),
        ("related code for StudyService", "StudyService"),
        ("UserService dependencies", "UserService"),
    ])
    def test_find_related_patterns(self, query: str, expected_symbol: str):
        result = route_query(query, use_llm=False)
        assert result.intent == QueryIntent.FIND_RELATED
        assert result.symbol == expected_symbol
        assert result.confidence >= 0.8

    # trace_feature patterns
    @pytest.mark.parametrize("query,expected_symbol", [
        ("trace feature card review", "card review"),
        ("trace the login flow", "login"),  # "the" is captured but trimmed
        ("how does mastery calculation work", "mastery calculation"),
        ("how does scheduler flow", "scheduler"),
        ("execution path for submitAnswer", "submitAnswer"),
    ])
    def test_trace_feature_patterns(self, query: str, expected_symbol: str):
        result = route_query(query, use_llm=False)
        assert result.intent == QueryIntent.TRACE_FEATURE
        assert result.symbol == expected_symbol
        assert result.confidence >= 0.8

    # get_symbol patterns (exact symbol lookup)
    @pytest.mark.parametrize("query,expected_symbol", [
        ("FsrsCalculator", "FsrsCalculator"),
        ("UserService.authenticate", "UserService.authenticate"),
        ("show me processCard", "processCard"),
        ("get the definition of FsrsCalculator", "FsrsCalculator"),
        ("where is calculateMastery defined", "calculateMastery"),
    ])
    def test_get_symbol_patterns(self, query: str, expected_symbol: str):
        result = route_query(query, use_llm=False)
        assert result.intent == QueryIntent.GET_SYMBOL
        assert result.symbol == expected_symbol
        assert result.confidence >= 0.8

    # search_code fallback (no pattern match)
    @pytest.mark.parametrize("query", [
        "explain the FSRS algorithm",
        "what is the purpose of the scheduler",
        "find all database queries",
        "show me authentication logic",
        "list all API endpoints",
    ])
    def test_search_code_fallback(self, query: str):
        result = route_query(query, use_llm=False)
        assert result.intent == QueryIntent.SEARCH_CODE
        assert result.confidence <= 0.6


class TestIsGraphIntent:
    """Test graph intent detection."""

    def test_graph_intents(self):
        assert is_graph_intent(QueryIntent.FIND_CALLERS) is True
        assert is_graph_intent(QueryIntent.FIND_TESTS) is True
        assert is_graph_intent(QueryIntent.FIND_RELATED) is True
        assert is_graph_intent(QueryIntent.TRACE_FEATURE) is True

    def test_non_graph_intents(self):
        assert is_graph_intent(QueryIntent.SEARCH_CODE) is False
        assert is_graph_intent(QueryIntent.GET_SYMBOL) is False


class TestEmptyQuery:
    """Test edge cases."""

    def test_empty_query(self):
        result = route_query("", use_llm=False)
        assert result.intent == QueryIntent.SEARCH_CODE
        assert result.confidence == 1.0

    def test_whitespace_query(self):
        result = route_query("   ", use_llm=False)
        assert result.intent == QueryIntent.SEARCH_CODE
        assert result.confidence == 1.0


class TestSymbolExtraction:
    """Test symbol extraction from various formats."""

    @pytest.mark.parametrize("query,expected_symbol", [
        ("what calls `FsrsCalculator`", "FsrsCalculator"),
        ("what calls 'FsrsCalculator'", "FsrsCalculator"),
        ('what calls "FsrsCalculator"', "FsrsCalculator"),
        ("callers of `UserService.authenticate`", "UserService.authenticate"),
    ])
    def test_quoted_symbols(self, query: str, expected_symbol: str):
        result = route_query(query, use_llm=False)
        assert result.symbol == expected_symbol


class TestLLMRouter:
    """Test LLM-based routing (mocked)."""

    def test_llm_disabled_by_default(self):
        """LLM router should be disabled by default."""
        result = route_query("explain the FSRS algorithm")
        # Should fall back to search_code without calling LLM
        assert result.intent == QueryIntent.SEARCH_CODE

    def test_llm_fallback_on_pattern_match(self):
        """When pattern matches, LLM should not be called."""
        result = route_query("what calls FsrsCalculator", use_llm=True)
        assert result.intent == QueryIntent.FIND_CALLERS
        assert result.reason.startswith("matched pattern")


class TestRoutedQueryDataclass:
    """Test RoutedQuery dataclass."""

    def test_routed_query_creation(self):
        rq = RoutedQuery(
            intent=QueryIntent.FIND_CALLERS,
            symbol="FsrsCalculator",
            original_query="what calls FsrsCalculator",
            confidence=0.95,
            reason="test",
        )
        assert rq.intent == QueryIntent.FIND_CALLERS
        assert rq.symbol == "FsrsCalculator"
        assert rq.confidence == 0.95

"""Unit tests for multi-query expansion.

Tests cover:
- Enable/disable toggles (env var + config)
- Paraphrase generation via inference backend mock
- Cache behavior
- Graceful fallback on failures
- Integration with HybridRrfRetrieval
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from stropha.retrieval.multi_query import (
    generate_paraphrases,
    MultiQueryExpander,
    clear_cache,
    _cache_key,
)


# --------------------------------------------------------------------------- Basic


def test_multi_query_disabled_by_default(monkeypatch) -> None:
    """Multi-query is disabled when env var is not set."""
    monkeypatch.delenv("STROPHA_MULTI_QUERY_ENABLED", raising=False)
    clear_cache()
    
    result = generate_paraphrases("how does authentication work")
    
    assert result == ["how does authentication work"]


def test_multi_query_returns_paraphrases_when_enabled(monkeypatch) -> None:
    """Multi-query generates paraphrases when enabled."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "authentication login flow user session\n"
            "security middleware auth check validation\n"
            "user credentials token JWT verification"
        )
        result = generate_paraphrases("how does authentication work")
    
    assert len(result) >= 2  # Original + at least 1 paraphrase
    assert result[0] == "how does authentication work"  # Original first
    assert "authentication login flow user session" in result


def test_multi_query_force_enabled(monkeypatch) -> None:
    """force_enabled bypasses env var check."""
    monkeypatch.delenv("STROPHA_MULTI_QUERY_ENABLED", raising=False)
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "search terms expanded query"
        result = generate_paraphrases(
            "how does auth work",
            force_enabled=True,
        )
    
    assert len(result) >= 2


def test_multi_query_empty_query() -> None:
    """Empty query returns itself."""
    clear_cache()
    
    assert generate_paraphrases("", force_enabled=True) == [""]
    assert generate_paraphrases("  ", force_enabled=True) == ["  "]


def test_multi_query_generation_failure_fallback(monkeypatch) -> None:
    """Fallback to original query when inference fails."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = None  # Simulate failure
        result = generate_paraphrases("test query")
    
    assert result == ["test query"]


def test_multi_query_empty_response_fallback(monkeypatch) -> None:
    """Fallback to original query when response is empty."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = ""
        result = generate_paraphrases("test query")
    
    assert result == ["test query"]


# --------------------------------------------------------------------------- Cache


def test_multi_query_cache_hit(monkeypatch) -> None:
    """Cached paraphrases are returned without LLM call."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    query = "how does caching work"
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "caching layer memory storage"
        # First call - hits LLM
        result1 = generate_paraphrases(query)
        call_count_1 = mock_gen.call_count
        
        # Second call - should use cache
        result2 = generate_paraphrases(query)
        call_count_2 = mock_gen.call_count
    
    assert result1 == result2
    assert call_count_1 == call_count_2  # No new LLM call


def test_multi_query_cache_key_normalization() -> None:
    """Cache keys normalize case and whitespace."""
    key1 = _cache_key("How Does Auth Work")
    key2 = _cache_key("how does auth work")
    key3 = _cache_key("  how does auth work  ")
    
    # All should produce same key (lowercase + strip)
    assert key1 == key2
    assert key1 == key3


def test_cache_clear() -> None:
    """clear_cache empties the cache."""
    clear_cache()
    
    # Simulate a cached entry via the public API
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "cached result"
        generate_paraphrases("cached query", force_enabled=True)
    
    clear_cache()
    
    # Now should call LLM again
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "new result"
        result = generate_paraphrases("cached query", force_enabled=True)
        assert mock_gen.called  # Had to call LLM


# --------------------------------------------------------------------------- Expander Class


def test_expander_class_disabled() -> None:
    """MultiQueryExpander respects enabled flag."""
    clear_cache()
    
    expander = MultiQueryExpander(enabled=False)
    result = expander.expand("test query")
    
    assert result == ["test query"]
    assert not expander.is_enabled


def test_expander_class_enabled(monkeypatch) -> None:
    """MultiQueryExpander generates paraphrases when enabled."""
    monkeypatch.delenv("STROPHA_MULTI_QUERY_ENABLED", raising=False)
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "expanded search terms"
        expander = MultiQueryExpander(enabled=True, num_paraphrases=3)
        result = expander.expand("test query")
    
    assert len(result) >= 2
    assert expander.is_enabled


def test_expander_respects_num_paraphrases(monkeypatch) -> None:
    """MultiQueryExpander limits number of paraphrases."""
    monkeypatch.delenv("STROPHA_MULTI_QUERY_ENABLED", raising=False)
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "paraphrase one\n"
            "paraphrase two\n"
            "paraphrase three\n"
            "paraphrase four\n"
            "paraphrase five"
        )
        expander = MultiQueryExpander(enabled=True, num_paraphrases=2)
        result = expander.expand("test query")
    
    # Original + 2 paraphrases max
    assert len(result) <= 3


# --------------------------------------------------------------------------- Env Vars


def test_env_var_count(monkeypatch) -> None:
    """STROPHA_MULTI_QUERY_COUNT env var is respected."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    monkeypatch.setenv("STROPHA_MULTI_QUERY_COUNT", "5")
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "p1\np2\np3\np4\np5"
        result = generate_paraphrases("test query")
        
        # Should use 5 paraphrases
        mock_gen.assert_called_once()
        # The call should have a prompt with {num_paraphrases}=5
        call_args = mock_gen.call_args
        assert "5" in call_args[0][0]  # Prompt contains "5"


# --------------------------------------------------------------------------- Filtering


def test_paraphrases_deduplicated(monkeypatch) -> None:
    """Duplicate paraphrases are removed."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "same query text\n"
            "same query text\n"
            "different query text"
        )
        result = generate_paraphrases("test query")
    
    # Should deduplicate
    assert len(result) == 3  # original + 2 unique
    assert result.count("same query text") == 1


def test_original_query_always_first(monkeypatch) -> None:
    """Original query is always first in results."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    original = "my original query"
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "paraphrase one\nparaphrase two"
        result = generate_paraphrases(original)
    
    assert result[0] == original


def test_numbered_lists_filtered(monkeypatch) -> None:
    """Numbered list markers are filtered out."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "1. should be filtered\n"
            "valid paraphrase\n"
            "- also filtered\n"
            "* and this"
        )
        result = generate_paraphrases("test query")
    
    # Only "valid paraphrase" should remain (plus original)
    assert "valid paraphrase" in result
    assert not any(p.startswith("1.") for p in result)
    assert not any(p.startswith("-") for p in result)
    assert not any(p.startswith("*") for p in result)


def test_short_paraphrases_filtered(monkeypatch) -> None:
    """Very short paraphrases are filtered."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "ok\nab\nvalid paraphrase here"
        result = generate_paraphrases("test query")
    
    # "ok" and "ab" should be filtered (< 6 chars)
    assert "ok" not in result
    assert "ab" not in result
    assert "valid paraphrase here" in result

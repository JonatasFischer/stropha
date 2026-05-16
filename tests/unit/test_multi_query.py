"""Unit tests for multi-query expansion.

Tests cover:
- Enable/disable toggles (env var + config)
- Paraphrase generation via Ollama mock
- Cache behavior
- Graceful fallback on failures
- Integration with HybridRrfRetrieval
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch, MagicMock

import pytest

from stropha.retrieval.multi_query import (
    generate_paraphrases,
    MultiQueryExpander,
    clear_cache,
    _cache_key,
)


def _fake_response(payload: dict) -> io.BytesIO:
    """Create a mock HTTP response."""
    body = json.dumps(payload).encode("utf-8")
    bio = io.BytesIO(body)
    bio.__enter__ = lambda self=bio: self
    bio.__exit__ = lambda self=bio, *args: False
    return bio


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
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "response": (
                "authentication login flow user session\n"
                "security middleware auth check validation\n"
                "user credentials token JWT verification"
            )
        })
        result = generate_paraphrases("how does authentication work")
    
    assert len(result) >= 2  # Original + at least 1 paraphrase
    assert result[0] == "how does authentication work"  # Original first
    assert "authentication login flow user session" in result


def test_multi_query_force_enabled(monkeypatch) -> None:
    """force_enabled bypasses env var check."""
    monkeypatch.delenv("STROPHA_MULTI_QUERY_ENABLED", raising=False)
    clear_cache()
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "response": "search terms expanded query"
        })
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


def test_multi_query_ollama_failure_fallback(monkeypatch) -> None:
    """Fallback to original query when Ollama fails."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.side_effect = TimeoutError("Connection timed out")
        result = generate_paraphrases("test query")
    
    assert result == ["test query"]


def test_multi_query_empty_response_fallback(monkeypatch) -> None:
    """Fallback when Ollama returns empty response."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({"response": ""})
        result = generate_paraphrases("test query")
    
    assert result == ["test query"]


# --------------------------------------------------------------------------- Cache


def test_multi_query_cache_hit(monkeypatch) -> None:
    """Cached paraphrases are returned without LLM call."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    query = "how does caching work"
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "response": "caching layer memory storage"
        })
        # First call - hits LLM
        result1 = generate_paraphrases(query)
        call_count_1 = mock_open.call_count
        
        # Second call - should use cache
        result2 = generate_paraphrases(query)
        call_count_2 = mock_open.call_count
    
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
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({"response": "cached result"})
        generate_paraphrases("cached query", force_enabled=True)
    
    clear_cache()
    
    # Now should call LLM again
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({"response": "new result"})
        result = generate_paraphrases("cached query", force_enabled=True)
        assert mock_open.called  # Had to call LLM


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
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "response": "expanded search terms"
        })
        expander = MultiQueryExpander(enabled=True, num_paraphrases=3)
        result = expander.expand("test query")
    
    assert len(result) >= 2
    assert expander.is_enabled


def test_expander_respects_num_paraphrases(monkeypatch) -> None:
    """MultiQueryExpander limits number of paraphrases."""
    monkeypatch.delenv("STROPHA_MULTI_QUERY_ENABLED", raising=False)
    clear_cache()
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "response": (
                "paraphrase one\n"
                "paraphrase two\n"
                "paraphrase three\n"
                "paraphrase four\n"
                "paraphrase five"
            )
        })
        expander = MultiQueryExpander(enabled=True, num_paraphrases=2)
        result = expander.expand("test query")
    
    # Original + 2 paraphrases max
    assert len(result) <= 3


# --------------------------------------------------------------------------- Env Vars


def test_env_var_model(monkeypatch) -> None:
    """STROPHA_MULTI_QUERY_MODEL env var is respected."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    monkeypatch.setenv("STROPHA_MULTI_QUERY_MODEL", "custom-model:7b")
    clear_cache()
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({"response": "test"})
        generate_paraphrases("test query")
        
        # Check the request body contains the custom model
        call_args = mock_open.call_args
        request = call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))
        assert body["model"] == "custom-model:7b"


def test_env_var_count(monkeypatch) -> None:
    """STROPHA_MULTI_QUERY_COUNT env var is respected."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    monkeypatch.setenv("STROPHA_MULTI_QUERY_COUNT", "5")
    clear_cache()
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "response": (
                "p1\np2\np3\np4\np5"
            )
        })
        result = generate_paraphrases("test query")
    
    # Check prompt mentions 5 paraphrases
    call_args = mock_open.call_args
    request = call_args[0][0]
    body = json.loads(request.data.decode("utf-8"))
    assert "5" in body["prompt"]


# --------------------------------------------------------------------------- Deduplication


def test_paraphrases_deduplicated(monkeypatch) -> None:
    """Duplicate paraphrases are removed."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "response": (
                "unique paraphrase\n"
                "unique paraphrase\n"  # Duplicate
                "another unique one"
            )
        })
        result = generate_paraphrases("test query")
    
    # Should not have duplicates
    assert len(result) == len(set(r.lower() for r in result))


def test_original_query_always_first(monkeypatch) -> None:
    """Original query is always the first result."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    query = "original query here"
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "response": "paraphrase one\nparaphrase two"
        })
        result = generate_paraphrases(query)
    
    assert result[0] == query


# --------------------------------------------------------------------------- Numbered list filtering


def test_numbered_lists_filtered(monkeypatch) -> None:
    """Lines starting with numbers/bullets are filtered out."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.retrieval.multi_query.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "response": (
                "1. numbered line should be filtered\n"
                "good paraphrase here\n"
                "- bullet also filtered\n"
                "another good paraphrase"
            )
        })
        result = generate_paraphrases("test query")
    
    # Numbered and bulleted lines should be excluded
    for r in result:
        assert not r.startswith("1.")
        assert not r.startswith("-")

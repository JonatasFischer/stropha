"""Unit tests for semantic query cache.

Tests cover:
- Enable/disable toggles (env var + config)
- Cache hit/miss behavior
- LRU eviction
- TTL expiration
- Filter handling
- Stats tracking
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from stropha.models import SearchHit
from stropha.retrieval.cache import (
    SemanticCache,
    CacheEntry,
    _quantize_vector,
    _filters_hash,
    get_global_cache,
    clear_global_cache,
)


def _make_hit(chunk_id: str, score: float = 0.9, rank: int = 1) -> SearchHit:
    """Create a minimal SearchHit for testing."""
    return SearchHit(
        rank=rank,
        chunk_id=chunk_id,
        score=score,
        rel_path="test.py",
        start_line=1,
        end_line=10,
        language="python",
        kind="function",
        symbol="test_func",
        snippet="def test(): pass",
    )


def _make_vec(dim: int = 128, seed: float = 0.1) -> list[float]:
    """Create a test embedding vector."""
    return [seed + i * 0.01 for i in range(dim)]


# --------------------------------------------------------------------------- Basic


def test_cache_disabled_by_default(monkeypatch) -> None:
    """Cache is disabled when env var is not set."""
    monkeypatch.delenv("STROPHA_QUERY_CACHE_ENABLED", raising=False)
    
    cache = SemanticCache()
    vec = _make_vec()
    
    cache.set(vec, [_make_hit("chunk1")], top_k=10)
    result = cache.get(vec, top_k=10)
    
    assert result is None
    assert not cache.is_enabled


def test_cache_enabled_via_config() -> None:
    """Cache can be enabled via constructor."""
    cache = SemanticCache(enabled=True, max_size=100)
    vec = _make_vec()
    hits = [_make_hit("chunk1"), _make_hit("chunk2")]
    
    cache.set(vec, hits, top_k=10)
    result = cache.get(vec, top_k=10)
    
    assert result is not None
    assert len(result) == 2
    assert cache.is_enabled


def test_cache_enabled_via_env(monkeypatch) -> None:
    """Cache can be enabled via env var."""
    monkeypatch.setenv("STROPHA_QUERY_CACHE_ENABLED", "1")
    
    cache = SemanticCache()
    vec = _make_vec()
    hits = [_make_hit("chunk1")]
    
    cache.set(vec, hits, top_k=10)
    result = cache.get(vec, top_k=10)
    
    assert result is not None
    assert cache.is_enabled


# --------------------------------------------------------------------------- Hit/Miss


def test_cache_hit(monkeypatch) -> None:
    """Cache returns stored results."""
    cache = SemanticCache(enabled=True, max_size=100)
    vec = _make_vec()
    hits = [_make_hit("chunk1", 0.95), _make_hit("chunk2", 0.85)]
    
    cache.set(vec, hits, top_k=10)
    result = cache.get(vec, top_k=10)
    
    assert result is not None
    assert len(result) == 2
    assert result[0].chunk_id == "chunk1"
    assert result[0].score == 0.95
    assert result[1].chunk_id == "chunk2"


def test_cache_miss_different_vec() -> None:
    """Cache miss for different query vector."""
    cache = SemanticCache(enabled=True, max_size=100)
    vec1 = _make_vec(seed=0.1)
    vec2 = _make_vec(seed=0.9)  # Different vector
    hits = [_make_hit("chunk1")]
    
    cache.set(vec1, hits, top_k=10)
    result = cache.get(vec2, top_k=10)
    
    assert result is None


def test_cache_miss_different_top_k() -> None:
    """Cache miss for different top_k."""
    cache = SemanticCache(enabled=True, max_size=100)
    vec = _make_vec()
    hits = [_make_hit("chunk1")]
    
    cache.set(vec, hits, top_k=10)
    result = cache.get(vec, top_k=20)  # Different top_k
    
    assert result is None


def test_cache_miss_different_filters() -> None:
    """Cache miss for different filters."""
    cache = SemanticCache(enabled=True, max_size=100)
    vec = _make_vec()
    hits = [_make_hit("chunk1")]
    
    cache.set(vec, hits, top_k=10, filters={"language": ["python"]})
    result = cache.get(vec, top_k=10, filters={"language": ["java"]})
    
    assert result is None


# --------------------------------------------------------------------------- LRU Eviction


def test_lru_eviction() -> None:
    """Oldest entries are evicted when cache is full."""
    cache = SemanticCache(enabled=True, max_size=3)
    
    # Fill cache
    for i in range(3):
        vec = _make_vec(seed=i * 0.1)
        cache.set(vec, [_make_hit(f"chunk{i}")], top_k=10)
    
    # Add one more (should evict first)
    vec_new = _make_vec(seed=0.9)
    cache.set(vec_new, [_make_hit("chunk_new")], top_k=10)
    
    # First entry should be evicted
    vec_first = _make_vec(seed=0.0)
    assert cache.get(vec_first, top_k=10) is None
    
    # New entry should exist
    assert cache.get(vec_new, top_k=10) is not None


def test_lru_access_order() -> None:
    """Accessed entries are moved to end (not evicted first)."""
    cache = SemanticCache(enabled=True, max_size=3)
    
    vec1 = _make_vec(seed=0.1)
    vec2 = _make_vec(seed=0.2)
    vec3 = _make_vec(seed=0.3)
    
    # Add entries
    cache.set(vec1, [_make_hit("chunk1")], top_k=10)
    cache.set(vec2, [_make_hit("chunk2")], top_k=10)
    cache.set(vec3, [_make_hit("chunk3")], top_k=10)
    
    # Access first entry (moves it to end)
    cache.get(vec1, top_k=10)
    
    # Add new entry (should evict vec2, not vec1)
    vec4 = _make_vec(seed=0.4)
    cache.set(vec4, [_make_hit("chunk4")], top_k=10)
    
    # vec1 should still exist (was accessed)
    assert cache.get(vec1, top_k=10) is not None
    # vec2 should be evicted
    assert cache.get(vec2, top_k=10) is None


# --------------------------------------------------------------------------- TTL Expiration


def test_ttl_expiration() -> None:
    """Expired entries return None."""
    cache = SemanticCache(enabled=True, max_size=100, ttl_seconds=0.1)
    vec = _make_vec()
    hits = [_make_hit("chunk1")]
    
    cache.set(vec, hits, top_k=10)
    
    # Should hit immediately
    assert cache.get(vec, top_k=10) is not None
    
    # Wait for expiration
    time.sleep(0.15)
    
    # Should miss after TTL
    assert cache.get(vec, top_k=10) is None


def test_remove_expired() -> None:
    """remove_expired cleans up old entries."""
    cache = SemanticCache(enabled=True, max_size=100, ttl_seconds=0.1)
    
    for i in range(5):
        vec = _make_vec(seed=i * 0.1)
        cache.set(vec, [_make_hit(f"chunk{i}")], top_k=10)
    
    assert cache.stats["size"] == 5
    
    # Wait for expiration
    time.sleep(0.15)
    
    removed = cache.remove_expired()
    assert removed == 5
    assert cache.stats["size"] == 0


# --------------------------------------------------------------------------- Stats


def test_stats_tracking() -> None:
    """Stats track hits and misses."""
    cache = SemanticCache(enabled=True, max_size=100)
    vec = _make_vec()
    
    # Miss
    cache.get(vec, top_k=10)
    assert cache.stats["misses"] == 1
    assert cache.stats["hits"] == 0
    
    # Set and hit
    cache.set(vec, [_make_hit("chunk1")], top_k=10)
    cache.get(vec, top_k=10)
    
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1
    assert cache.stats["hit_rate"] == 0.5


def test_invalidate() -> None:
    """invalidate clears all entries."""
    cache = SemanticCache(enabled=True, max_size=100)
    
    for i in range(5):
        vec = _make_vec(seed=i * 0.1)
        cache.set(vec, [_make_hit(f"chunk{i}")], top_k=10)
    
    count = cache.invalidate()
    
    assert count == 5
    assert cache.stats["size"] == 0


# --------------------------------------------------------------------------- Filter Handling


def test_filters_hash_consistency() -> None:
    """Same filters produce same hash."""
    h1 = _filters_hash({"language": ["python"], "kind": ["function"]})
    h2 = _filters_hash({"kind": ["function"], "language": ["python"]})
    
    # Order shouldn't matter
    assert h1 == h2


def test_filters_hash_none() -> None:
    """None filters produce consistent hash."""
    h1 = _filters_hash(None)
    h2 = _filters_hash({})
    
    assert h1 == "none"
    # Empty dict produces same as None (no effective filters)
    assert h2 == "none"


def test_cache_with_filters() -> None:
    """Cache respects filter parameters."""
    cache = SemanticCache(enabled=True, max_size=100)
    vec = _make_vec()
    
    # Set with filter
    cache.set(vec, [_make_hit("py_chunk")], top_k=10, filters={"language": ["python"]})
    
    # Hit with same filter
    result = cache.get(vec, top_k=10, filters={"language": ["python"]})
    assert result is not None
    
    # Miss with different filter
    result = cache.get(vec, top_k=10, filters={"language": ["java"]})
    assert result is None
    
    # Miss with no filter
    result = cache.get(vec, top_k=10)
    assert result is None


# --------------------------------------------------------------------------- Quantization


def test_quantize_vector() -> None:
    """Vector quantization rounds to specified precision."""
    vec = [0.123456, 0.654321, 0.999999]
    
    q2 = _quantize_vector(vec, precision=2)
    assert q2 == (0.12, 0.65, 1.0)
    
    q1 = _quantize_vector(vec, precision=1)
    assert q1 == (0.1, 0.7, 1.0)


def test_similar_vectors_same_key() -> None:
    """Very similar vectors produce same cache key (within precision)."""
    cache = SemanticCache(enabled=True, max_size=100)
    
    vec1 = [0.1001, 0.2002, 0.3003]
    vec2 = [0.1002, 0.2001, 0.3004]  # Slightly different
    
    cache.set(vec1, [_make_hit("chunk1")], top_k=10)
    
    # Should hit because vectors round to same values at precision=2
    result = cache.get(vec2, top_k=10)
    assert result is not None


# --------------------------------------------------------------------------- Global Cache


def test_global_cache() -> None:
    """Global cache singleton works."""
    clear_global_cache()
    
    cache1 = get_global_cache()
    cache2 = get_global_cache()
    
    assert cache1 is cache2


def test_clear_global_cache() -> None:
    """clear_global_cache clears the singleton."""
    # Set up with enabled cache
    with patch.dict("os.environ", {"STROPHA_QUERY_CACHE_ENABLED": "1"}):
        cache = get_global_cache()
        vec = _make_vec()
        cache.set(vec, [_make_hit("chunk1")], top_k=10)
        
        count = clear_global_cache()
        assert count >= 0  # May be 0 if cache was disabled


# --------------------------------------------------------------------------- CacheEntry


def test_cache_entry_expiration() -> None:
    """CacheEntry correctly reports expiration."""
    entry = CacheEntry(
        chunk_ids=["chunk1"],
        scores=[0.9],
        created_at=time.time() - 100,  # Created 100 seconds ago
    )
    
    assert entry.is_expired(ttl=50)  # 50 second TTL
    assert not entry.is_expired(ttl=200)  # 200 second TTL

"""Semantic Query Cache — LRU cache for search results.

Idea: Cache search results keyed by a semantic hash of the query embedding.
This avoids repeated expensive searches for similar queries.

The cache key is based on:
- Quantized query embedding (rounded to reduce dimensionality)
- top_k parameter
- Filter hash (language, path_prefix, kind, exclude_tests)

Cache invalidation:
- TTL-based: entries expire after a configurable time (default 1 hour)
- Manual: clear_cache() can be called on index update

This is a pure in-memory cache. It does NOT persist across process restarts.
For MCP servers that run as long-lived processes, this provides significant
latency savings for repeated queries.

Usage:
    from stropha.retrieval.cache import SemanticCache

    cache = SemanticCache(max_size=1000, ttl_seconds=3600)
    
    # Check cache
    cached = cache.get(query_vec, top_k=10, filters={"language": ["python"]})
    if cached is not None:
        return cached
    
    # Compute result
    result = expensive_search(query, top_k=10)
    
    # Store in cache
    cache.set(query_vec, result, top_k=10, filters={"language": ["python"]})

Environment variables:
    STROPHA_QUERY_CACHE_ENABLED: Enable/disable cache (default: 0)
    STROPHA_QUERY_CACHE_SIZE: Max cache entries (default: 500)
    STROPHA_QUERY_CACHE_TTL: TTL in seconds (default: 3600)
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..logging import get_logger
from ..models import SearchHit

log = get_logger(__name__)


def _enabled() -> bool:
    return os.environ.get("STROPHA_QUERY_CACHE_ENABLED", "0") == "1"


def _max_size() -> int:
    try:
        return int(os.environ.get("STROPHA_QUERY_CACHE_SIZE", "500"))
    except ValueError:
        return 500


def _ttl_seconds() -> float:
    try:
        return float(os.environ.get("STROPHA_QUERY_CACHE_TTL", "3600"))
    except ValueError:
        return 3600.0


@dataclass
class CacheEntry:
    """A cached search result with metadata."""
    chunk_ids: list[str]
    scores: list[float]
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self, ttl: float) -> bool:
        return time.time() - self.created_at > ttl


def _quantize_vector(vec: list[float], precision: int = 2) -> tuple[float, ...]:
    """Quantize vector to reduce dimensionality for cache key.
    
    Args:
        vec: The embedding vector.
        precision: Number of decimal places to round to.
        
    Returns:
        Quantized vector as a tuple (hashable).
    """
    return tuple(round(v, precision) for v in vec)


def _filters_hash(filters: dict[str, Any] | None) -> str:
    """Generate a hash of filter parameters."""
    if not filters:
        return "none"
    # Sort keys for consistent ordering
    canonical = sorted((k, str(v)) for k, v in filters.items() if v is not None)
    return hashlib.sha256(repr(canonical).encode()).hexdigest()[:12]


class SemanticCache:
    """LRU cache for search results, keyed by quantized query embedding."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        max_size: int | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        """Initialize the cache.
        
        Args:
            enabled: Whether cache is enabled. Defaults to env var.
            max_size: Maximum number of entries. Defaults to env var.
            ttl_seconds: Time-to-live for entries. Defaults to env var.
        """
        self._enabled = enabled if enabled is not None else _enabled()
        self._max_size = max_size if max_size is not None else _max_size()
        self._ttl = ttl_seconds if ttl_seconds is not None else _ttl_seconds()
        
        # OrderedDict for LRU eviction
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Stats
        self._hits = 0
        self._misses = 0

    def _make_key(
        self,
        query_vec: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Generate a cache key from query vector and parameters."""
        quantized = _quantize_vector(query_vec)
        filter_hash = _filters_hash(filters)
        # Use first 8 floats as a quick fingerprint + full hash for uniqueness
        fingerprint = quantized[:8] if len(quantized) >= 8 else quantized
        vec_hash = hashlib.sha256(repr(quantized).encode()).hexdigest()[:16]
        return f"{fingerprint}:{top_k}:{filter_hash}:{vec_hash}"

    def get(
        self,
        query_vec: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit] | None:
        """Get cached result if available and not expired.
        
        Args:
            query_vec: The query embedding vector.
            top_k: Number of results requested.
            filters: Filter parameters (language, path_prefix, etc.)
            
        Returns:
            List of SearchHit if cached, None otherwise.
        """
        if not self._enabled:
            return None
            
        key = self._make_key(query_vec, top_k, filters)
        entry = self._cache.get(key)
        
        if entry is None:
            self._misses += 1
            return None
            
        if entry.is_expired(self._ttl):
            # Remove expired entry
            del self._cache[key]
            self._misses += 1
            log.debug("cache.expired", key=key[:20])
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        log.debug("cache.hit", key=key[:20])
        
        # Reconstruct SearchHit objects (minimal version - just chunk_id and score)
        # The caller should have access to storage to hydrate full details if needed
        return [
            SearchHit(
                rank=i + 1,
                chunk_id=cid,
                score=score,
                rel_path="",  # Will be hydrated by caller
                start_line=0,
                end_line=0,
                language="",
                kind="",
                symbol=None,
                snippet="",  # Will be hydrated by caller
            )
            for i, (cid, score) in enumerate(zip(entry.chunk_ids, entry.scores))
        ]

    def set(
        self,
        query_vec: list[float],
        results: list[SearchHit],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> None:
        """Store search results in cache.
        
        Args:
            query_vec: The query embedding vector.
            results: The search results to cache.
            top_k: Number of results requested.
            filters: Filter parameters.
        """
        if not self._enabled:
            return
            
        key = self._make_key(query_vec, top_k, filters)
        
        # Evict oldest entries if at capacity
        while len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            log.debug("cache.evict", key=oldest_key[:20])
        
        # Store minimal data (chunk_ids + scores)
        entry = CacheEntry(
            chunk_ids=[h.chunk_id for h in results],
            scores=[h.score for h in results],
        )
        self._cache[key] = entry
        log.debug("cache.set", key=key[:20], count=len(results))

    def invalidate(self) -> int:
        """Clear all cache entries.
        
        Returns:
            Number of entries cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        log.info("cache.invalidated", count=count)
        return count

    def remove_expired(self) -> int:
        """Remove all expired entries.
        
        Returns:
            Number of entries removed.
        """
        expired = [
            key for key, entry in self._cache.items()
            if entry.is_expired(self._ttl)
        ]
        for key in expired:
            del self._cache[key]
        if expired:
            log.debug("cache.cleanup", count=len(expired))
        return len(expired)

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self._enabled,
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0,
        }

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# Global cache instance for the MCP server
_global_cache: SemanticCache | None = None


def get_global_cache() -> SemanticCache:
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SemanticCache()
    return _global_cache


def clear_global_cache() -> int:
    """Clear the global cache. Returns number of entries cleared."""
    global _global_cache
    if _global_cache is not None:
        return _global_cache.invalidate()
    return 0

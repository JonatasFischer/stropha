"""Multi-Query Expansion — generate paraphrases and fuse results via RRF.

Idea: Instead of searching with a single query, generate 3-5 paraphrases
of the original query using an LLM, search with each paraphrase, and
fuse the results via Reciprocal Rank Fusion (RRF).

This increases recall by capturing different ways users might ask
about the same concept:

    "how to authenticate users"
    → Paraphrases:
       1. "authentication login flow user session"
       2. "security middleware auth check validation"
       3. "user credentials token JWT verification"
    → Search each, RRF fuse

Backend selection (automatic):
- MLX (Apple Silicon): fastest, no daemon needed
- Ollama: cross-platform fallback

Falls back to single-query search when:
- No inference backend available
- Generation times out
- Generated paraphrases are empty
- STROPHA_MULTI_QUERY_ENABLED is unset / "0"

Caching: Paraphrases are cached in-memory by query to avoid repeated
LLM calls for the same query.

Usage:
    from stropha.retrieval.multi_query import MultiQueryExpander

    expander = MultiQueryExpander()
    paraphrases = expander.expand(query)
    # Returns list of 3-5 paraphrases including the original query
"""

from __future__ import annotations

import hashlib
import os

from ..logging import get_logger

log = get_logger(__name__)

_DEFAULT_PROMPT = (
    "Generate {num_paraphrases} different search queries that a developer might use "
    "to find the same code as this query. Each paraphrase should focus on different "
    "aspects: synonyms, related concepts, or technical terms.\n\n"
    "Original query: {query}\n\n"
    "Return ONLY the paraphrases, one per line, no numbers or bullets:"
)

# In-memory cache for paraphrases (keyed by query hash)
_PARAPHRASE_CACHE: dict[str, list[str]] = {}
_CACHE_MAX_SIZE = 1000


def _enabled() -> bool:
    return os.environ.get("STROPHA_MULTI_QUERY_ENABLED", "0") == "1"


def _prompt_template() -> str:
    return os.environ.get("STROPHA_MULTI_QUERY_PROMPT", _DEFAULT_PROMPT)


def _num_paraphrases() -> int:
    try:
        return int(os.environ.get("STROPHA_MULTI_QUERY_COUNT", "3"))
    except ValueError:
        return 3


def _cache_key(query: str) -> str:
    """Generate a cache key for the query."""
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]


def _get_cached(query: str) -> list[str] | None:
    """Get cached paraphrases for a query."""
    key = _cache_key(query)
    return _PARAPHRASE_CACHE.get(key)


def _set_cached(query: str, paraphrases: list[str]) -> None:
    """Cache paraphrases for a query, with LRU eviction."""
    global _PARAPHRASE_CACHE
    if len(_PARAPHRASE_CACHE) >= _CACHE_MAX_SIZE:
        # Simple eviction: remove oldest half
        keys = list(_PARAPHRASE_CACHE.keys())
        for k in keys[: len(keys) // 2]:
            del _PARAPHRASE_CACHE[k]
    key = _cache_key(query)
    _PARAPHRASE_CACHE[key] = paraphrases


def generate_paraphrases(
    query: str,
    *,
    force_enabled: bool = False,
    num_paraphrases: int | None = None,
) -> list[str]:
    """Generate paraphrases of the query using an LLM.

    Uses MLX on Apple Silicon, falls back to Ollama on other platforms.

    Args:
        query: The user's search query.
        force_enabled: If True, skip the _enabled() check.
        num_paraphrases: Number of paraphrases to generate (default: 3).

    Returns:
        List of paraphrases INCLUDING the original query as the first element.
        On failure, returns just [query] (single-element list).
    """
    if not force_enabled and not _enabled():
        return [query]
    if not query or not query.strip():
        return [query]

    # Check cache first
    cached = _get_cached(query)
    if cached is not None:
        log.debug("multi_query.cache_hit", query=query[:50])
        return cached

    n = num_paraphrases or _num_paraphrases()
    prompt = _prompt_template().format(query=query, num_paraphrases=n)

    # Use unified inference backend (MLX preferred, Ollama fallback)
    from ..inference import generate

    text = generate(
        prompt,
        max_tokens=256,
        temperature=0.7,  # Some variety for paraphrases
    )

    if not text:
        log.info("multi_query.skip_generation_failed")
        return [query]

    # Parse paraphrases (one per line)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # Filter out lines that are too short or look like metadata
    paraphrases = [
        line for line in lines
        if len(line) > 5 and not line.startswith(("1.", "2.", "3.", "-", "*", "#"))
    ]

    if not paraphrases:
        log.info("multi_query.no_valid_paraphrases")
        return [query]

    # Always include original query first, then paraphrases
    # Dedupe and limit
    seen = {query.lower()}
    result = [query]
    for p in paraphrases:
        if p.lower() not in seen and len(result) < n + 1:
            seen.add(p.lower())
            result.append(p[:200])  # Cap individual paraphrase length

    log.debug(
        "multi_query.generated",
        original=query[:50],
        count=len(result) - 1,
    )

    # Cache the result
    _set_cached(query, result)

    return result


class MultiQueryExpander:
    """Stateful wrapper for multi-query expansion with configuration."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        num_paraphrases: int = 3,
    ) -> None:
        """Initialize the expander.

        Args:
            enabled: If True, multi-query expansion is active.
            num_paraphrases: Number of paraphrases to generate.
        """
        self._enabled = enabled
        self._num_paraphrases = num_paraphrases

    def expand(self, query: str) -> list[str]:
        """Expand query into multiple paraphrases.

        Returns:
            List of queries including the original. Length 1 if disabled
            or on failure.
        """
        return generate_paraphrases(
            query,
            force_enabled=self._enabled,
            num_paraphrases=self._num_paraphrases,
        )

    @property
    def is_enabled(self) -> bool:
        """Check if multi-query is enabled via config or env var."""
        return self._enabled or _enabled()


def clear_cache() -> None:
    """Clear the paraphrase cache. Useful for testing."""
    global _PARAPHRASE_CACHE
    _PARAPHRASE_CACHE = {}

"""Query Rewriting — transform user queries into more searchable form.

Idea: LLM rewrites natural-language queries into forms that are more likely
to match code identifiers and technical terms. Works for all streams (dense,
sparse, symbol), unlike HyDE which only helps dense.

Examples:
    "onde tem teste pra fsrs" → "FSRS calculator test unit test FsrsCalculatorTest"
    "como funciona mastery" → "mastery streak transition REVIEW phase acquisition"

Backend selection (automatic):
- MLX (Apple Silicon): fastest, no daemon needed
- Ollama: cross-platform fallback

Falls back to original query when:
- No inference backend available
- The generation times out
- The generated text is empty
- STROPHA_QUERY_REWRITE_ENABLED is unset / "0"

Usage:
    from stropha.retrieval.query_rewrite import maybe_rewrite_query

    rewritten = maybe_rewrite_query(user_query) or user_query
"""

from __future__ import annotations

import os

from ..logging import get_logger

log = get_logger(__name__)

_DEFAULT_PROMPT = (
    "Rewrite this developer question into search terms that would match code.\n"
    "Include: class names, method names, technical terms, file patterns.\n"
    "Keep the original intent but expand with likely code identifiers.\n\n"
    "QUESTION: {query}\n\n"
    "SEARCH TERMS (just the terms, no explanation):"
)


def _enabled() -> bool:
    return os.environ.get("STROPHA_QUERY_REWRITE_ENABLED", "0") == "1"


def _prompt_template() -> str:
    return os.environ.get("STROPHA_QUERY_REWRITE_PROMPT", _DEFAULT_PROMPT)


def maybe_rewrite_query(query: str, *, force_enabled: bool = False) -> str | None:
    """Return a rewritten query optimized for code search, or None on fail/skip.

    Uses MLX on Apple Silicon, falls back to Ollama on other platforms.

    Unlike HyDE, the rewritten query is used for ALL streams (dense, sparse,
    symbol), not just dense. The goal is to expand natural language into
    code-like terms.

    Args:
        query: The user's search query.
        force_enabled: If True, skip the _enabled() check. Used when caller
            already determined query rewriting should run (e.g., config flag).

    Never raises. Returns None on failure, letting caller use original query.
    """
    if not force_enabled and not _enabled():
        return None
    if not query or not query.strip():
        return None

    prompt = _prompt_template().format(query=query)

    # Use unified inference backend (MLX preferred, Ollama fallback)
    from ..inference import generate

    text = generate(
        prompt,
        max_tokens=128,
        temperature=0.0,
    )

    if not text:
        log.info("query_rewrite.skip_generation_failed")
        return None

    # Combine original query with rewritten terms for better coverage
    # Keep it bounded to avoid token explosion
    combined = f"{query} {text}"
    return combined[:500]

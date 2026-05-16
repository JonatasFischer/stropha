"""Query Rewriting — transform user queries into more searchable form.

Idea: LLM rewrites natural-language queries into forms that are more likely
to match code identifiers and technical terms. Works for all streams (dense,
sparse, symbol), unlike HyDE which only helps dense.

Examples:
    "onde tem teste pra fsrs" → "FSRS calculator test unit test FsrsCalculatorTest"
    "como funciona mastery" → "mastery streak transition REVIEW phase acquisition"

Local implementation: route through Ollama (same as HyDE). Falls back to
original query when:
- Ollama is unreachable
- HTTP call times out
- Generated text is empty
- STROPHA_QUERY_REWRITE_ENABLED is unset / "0"

Usage:
    from stropha.retrieval.query_rewrite import maybe_rewrite_query

    rewritten = maybe_rewrite_query(user_query) or user_query
"""

from __future__ import annotations

import json
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

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


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _ollama_model() -> str:
    return os.environ.get("STROPHA_QUERY_REWRITE_MODEL", "qwen2.5-coder:1.5b")


def _prompt_template() -> str:
    return os.environ.get("STROPHA_QUERY_REWRITE_PROMPT", _DEFAULT_PROMPT)


def _timeout_s() -> float:
    try:
        return float(os.environ.get("STROPHA_QUERY_REWRITE_TIMEOUT_S", "5"))
    except ValueError:
        return 5.0


def maybe_rewrite_query(query: str, *, force_enabled: bool = False) -> str | None:
    """Return a rewritten query optimized for code search, or None on fail/skip.

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
    body = json.dumps(
        {
            "model": _ollama_model(),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
    ).encode("utf-8")

    try:
        req = urllib_request.Request(
            f"{_ollama_url()}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=_timeout_s()) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib_error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        log.info("query_rewrite.skip_ollama_failure", error=str(exc))
        return None

    text = (payload.get("response") or "").strip()
    if not text:
        log.info("query_rewrite.skip_empty")
        return None
    
    # Combine original query with rewritten terms for better coverage
    # Keep it bounded to avoid token explosion
    combined = f"{query} {text}"
    return combined[:500]

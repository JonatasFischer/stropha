"""HyDE — Hypothetical Document Embeddings (Phase 3 §6.3).

Idea (Gao et al., 2022): generate a hypothetical answer to the query
using an LLM, then embed *that answer* instead of the query itself. The
hypothetical document lives in the same space as your indexed chunks,
so the dense stream sees a much better neighbour match than it does
from the raw question.

Backend selection (automatic):
- MLX (Apple Silicon): fastest, no daemon needed
- Ollama: cross-platform fallback

Falls back to the raw query when:
- No inference backend available
- The generation times out
- The generated text is empty
- ``STROPHA_HYDE_ENABLED`` is unset / "0"

Drift-safety note: HyDE changes only the *query path*. It does not
mutate stored chunks or their embeddings, so toggling it on and off is
zero-cost — no re-index needed.

Usage:

    from stropha.retrieval.hyde import maybe_hyde_rewrite

    text_to_embed = maybe_hyde_rewrite(user_query) or user_query
    qvec = embedder.embed_query(text_to_embed)
"""

from __future__ import annotations

import os

from ..logging import get_logger

log = get_logger(__name__)

_DEFAULT_PROMPT = (
    "Write a 2-3 sentence imaginary code snippet or doc paragraph that would "
    "directly answer this developer question:\n\n"
    "QUESTION: {query}\n\n"
    "Reply with just the snippet/paragraph — no preamble, no \"sure here is\".")


def _enabled() -> bool:
    return os.environ.get("STROPHA_HYDE_ENABLED", "0") == "1"


def _prompt_template() -> str:
    return os.environ.get("STROPHA_HYDE_PROMPT", _DEFAULT_PROMPT)


def maybe_hyde_rewrite(query: str, *, force_enabled: bool = False) -> str | None:
    """Return a hypothetical document for ``query`` or ``None`` on fail/skip.

    Uses MLX on Apple Silicon, falls back to Ollama on other platforms.

    Caller is expected to use the returned text *instead of* the raw
    query for the dense embedding step; the BM25 / symbol streams keep
    using the raw query (HyDE doesn't help literal-match lanes).

    Args:
        query: The user's search query.
        force_enabled: If True, skip the _enabled() check. Used when caller
            already determined HyDE should run (e.g., config flag).

    Never raises. Worst case it returns None and the caller falls back
    to the raw query — same behaviour as if the feature were off.
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
        max_tokens=512,
        temperature=0.0,
    )

    if not text:
        log.info("hyde.skip_generation_failed")
        return None

    # Cap to keep dense embedding time bounded.
    return text[:2000]

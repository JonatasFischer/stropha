"""HyDE — Hypothetical Document Embeddings (Phase 3 §6.3).

Idea (Gao et al., 2022): generate a hypothetical answer to the query
using an LLM, then embed *that answer* instead of the query itself. The
hypothetical document lives in the same space as your indexed chunks,
so the dense stream sees a much better neighbour match than it does
from the raw question.

Local implementation: route the prompt through the already-installed
Ollama daemon (default model ``qwen2.5-coder:1.5b``). No network, no API
key. Falls back to the raw query when:

- Ollama is unreachable (daemon not running, model not pulled)
- The HTTP call times out
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

import json
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..logging import get_logger

log = get_logger(__name__)

_DEFAULT_PROMPT = (
    "Write a 2-3 sentence imaginary code snippet or doc paragraph that would "
    "directly answer this developer question:\n\n"
    "QUESTION: {query}\n\n"
    "Reply with just the snippet/paragraph — no preamble, no \"sure here is\".")


def _enabled() -> bool:
    return os.environ.get("STROPHA_HYDE_ENABLED", "0") == "1"


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _ollama_model() -> str:
    return os.environ.get("STROPHA_HYDE_MODEL", "qwen2.5-coder:1.5b")


def _prompt_template() -> str:
    return os.environ.get("STROPHA_HYDE_PROMPT", _DEFAULT_PROMPT)


def _timeout_s() -> float:
    try:
        return float(os.environ.get("STROPHA_HYDE_TIMEOUT_S", "8"))
    except ValueError:
        return 8.0


def maybe_hyde_rewrite(query: str, *, force_enabled: bool = False) -> str | None:
    """Return a hypothetical document for ``query`` or ``None`` on fail/skip.

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
        log.info("hyde.skip_ollama_failure", error=str(exc))
        return None

    text = (payload.get("response") or "").strip()
    if not text:
        log.info("hyde.skip_empty")
        return None
    # Cap to keep dense embedding time bounded.
    return text[:2000]

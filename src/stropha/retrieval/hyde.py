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
import re

from ..logging import get_logger

log = get_logger(__name__)

# Code-tuned prompts for different query types
# The hypothetical document should look like actual code/docs from the codebase

_PROMPT_CODE_FUNCTION = """\
Write a hypothetical code snippet that would answer this question about a function or method.
Include: function signature, brief docstring, and key implementation lines.

Question: {query}

```
"""

_PROMPT_CODE_CLASS = """\
Write a hypothetical code snippet that would answer this question about a class or type.
Include: class definition, key attributes, and important method signatures.

Question: {query}

```
"""

_PROMPT_CODE_HOW = """\
Write a hypothetical code comment or docstring that would explain how this works.
Be specific about the implementation approach, algorithms used, or data flow.

Question: {query}

/**
"""

_PROMPT_CODE_WHERE = """\
Write a hypothetical code snippet showing where this functionality is implemented.
Include: the file path as a comment, class/function name, and key lines.

Question: {query}

// File: 
"""

_PROMPT_CODE_TEST = """\
Write a hypothetical test case that would test this functionality.
Include: test method name, setup, assertion, and what it verifies.

Question: {query}

@Test
"""

_PROMPT_CODE_ERROR = """\
Write a hypothetical code snippet showing how this error or bug would be fixed.
Include: the problematic code pattern and the corrected version.

Question: {query}

// Before (bug):
"""

_PROMPT_CODE_DEFAULT = """\
Write a 2-4 line hypothetical code snippet or documentation that answers this developer question.
Match the style of real production code with proper naming and structure.

Question: {query}

"""

# Query pattern detection for selecting the best prompt
_PATTERNS = [
    (re.compile(r'\b(function|method|def|fn)\b', re.I), _PROMPT_CODE_FUNCTION),
    (re.compile(r'\b(class|type|interface|struct|enum)\b', re.I), _PROMPT_CODE_CLASS),
    (re.compile(r'^how\b', re.I), _PROMPT_CODE_HOW),
    (re.compile(r'\b(where|find|locate)\b', re.I), _PROMPT_CODE_WHERE),
    (re.compile(r'\b(tests?|specs?|asserts?|assertions?)\b', re.I), _PROMPT_CODE_TEST),
    (re.compile(r'\b(bug|fix|error|issue|debug)\b', re.I), _PROMPT_CODE_ERROR),
]

# Legacy default prompt (used when STROPHA_HYDE_PROMPT is set)
_DEFAULT_PROMPT = (
    "Write a 2-3 sentence imaginary code snippet or doc paragraph that would "
    "directly answer this developer question:\n\n"
    "QUESTION: {query}\n\n"
    "Reply with just the snippet/paragraph — no preamble, no \"sure here is\"."
)


def _select_prompt(query: str) -> str:
    """Select the best code-tuned prompt based on query patterns."""
    for pattern, prompt in _PATTERNS:
        if pattern.search(query):
            return prompt
    return _PROMPT_CODE_DEFAULT


def _enabled() -> bool:
    return os.environ.get("STROPHA_HYDE_ENABLED", "0") == "1"


def _use_code_tuned() -> bool:
    """Check if code-tuned prompts should be used (default: yes)."""
    return os.environ.get("STROPHA_HYDE_CODE_TUNED", "1") == "1"


def _prompt_template(query: str) -> str:
    """Get the appropriate prompt template for the query."""
    # Check for custom prompt override
    custom = os.environ.get("STROPHA_HYDE_PROMPT")
    if custom:
        return custom
    
    # Use code-tuned prompts by default
    if _use_code_tuned():
        return _select_prompt(query)
    
    return _DEFAULT_PROMPT


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

    prompt = _prompt_template(query).format(query=query)
    
    log.debug("hyde.prompt_selected", query=query[:50], prompt_start=prompt[:30])

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

    # Clean up the response - remove markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    
    # Cap to keep dense embedding time bounded.
    return text[:2000]


def get_prompt_type(query: str) -> str:
    """Return the prompt type that would be used for this query (for debugging)."""
    for pattern, prompt in _PATTERNS:
        if pattern.search(query):
            # Extract the prompt name from the variable name
            for name, val in globals().items():
                if val is prompt and name.startswith("_PROMPT_CODE_"):
                    return name.replace("_PROMPT_CODE_", "").lower()
    return "default"

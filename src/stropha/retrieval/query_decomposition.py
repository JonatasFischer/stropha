"""Query Decomposition — split complex queries into atomic sub-queries.

For natural language questions that ask about multiple things:
    "how does the scheduler calculate mastery and when does it schedule cards"
    → Sub-queries:
       1. "how does the scheduler calculate mastery"
       2. "when does the scheduler schedule cards"
    → Search each, RRF fuse

This differs from multi-query expansion (which generates paraphrases of
a single concept). Query decomposition handles compound questions.

Detection heuristics:
1. Conjunction words: "and", "or", "also", "as well as"
2. Multiple question words: "how...what...", "where...when..."
3. Comma-separated items: "X, Y, and Z"

Backend selection (automatic):
- Pattern-based decomposition for clear conjunctions (zero cost)
- LLM decomposition for ambiguous queries (STROPHA_QUERY_DECOMPOSITION_LLM=1)

Usage:
    from stropha.retrieval.query_decomposition import decompose_query
    
    sub_queries = decompose_query("how does X work and what calls Y")
    # Returns ["how does X work", "what calls Y"]
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from ..logging import get_logger

log = get_logger(__name__)

# Conjunction patterns for splitting
_CONJUNCTIONS = re.compile(
    r"\s+(?:and\s+(?:also\s+)?|or\s+|as\s+well\s+as\s+|,\s*(?:and\s+)?)"
    r"(?:how|what|where|when|who|which|why|does|is|are|can|should)",
    re.IGNORECASE
)

# Question word pattern
_QUESTION_WORDS = re.compile(
    r"\b(how|what|where|when|who|which|why)\b",
    re.IGNORECASE
)

# Pattern for "X and Y" where both are noun phrases (not questions)
_NOUN_CONJUNCTION = re.compile(
    r"\b(\w+(?:\s+\w+)?)\s+and\s+(\w+(?:\s+\w+)?)\b",
    re.IGNORECASE
)


@dataclass
class DecompositionResult:
    """Result of query decomposition."""
    sub_queries: list[str]
    original_query: str
    method: str  # "pattern", "llm", or "none"
    

def _is_compound_query(query: str) -> bool:
    """Check if query appears to be compound (multiple questions)."""
    # Multiple question words
    question_words = _QUESTION_WORDS.findall(query)
    if len(question_words) >= 2:
        return True
    
    # Conjunction before a question word
    if _CONJUNCTIONS.search(query):
        return True
    
    return False


def _pattern_decompose(query: str) -> list[str] | None:
    """Try to decompose query using patterns (zero cost).
    
    Returns list of sub-queries or None if pattern doesn't match.
    """
    # Split on conjunction + question word
    parts = _CONJUNCTIONS.split(query)
    if len(parts) >= 2:
        # The split removes the conjunction, but keeps the question word
        # We need to reconstruct by finding the split points
        result = []
        last_end = 0
        for match in _CONJUNCTIONS.finditer(query):
            if last_end < match.start():
                sub = query[last_end:match.start()].strip()
                if sub:
                    result.append(sub)
            last_end = match.start()
            # Include the question word that starts the next part
            # The regex captures "and how", "or what", etc.
            # We want to keep "how...", "what..." in the next part
        
        # Add the last part
        if last_end < len(query):
            # Find where the question word starts in the match
            remaining = query[last_end:]
            # Remove the conjunction prefix
            conj_match = _CONJUNCTIONS.match(remaining)
            if conj_match:
                # Find the question word position
                for word in ["how", "what", "where", "when", "who", "which", "why", "does", "is", "are", "can", "should"]:
                    idx = remaining.lower().find(word)
                    if idx >= 0:
                        result.append(remaining[idx:].strip())
                        break
            else:
                result.append(remaining.strip())
        
        if len(result) >= 2:
            return [q for q in result if len(q) > 5]
    
    return None


def _llm_decompose(query: str) -> list[str] | None:
    """Use LLM to decompose complex query into sub-queries."""
    try:
        from ..inference import generate
        
        prompt = f"""Break this complex question into simpler, independent sub-questions.
If the question is already simple (asks about one thing), return just the original.

Question: {query}

Return ONLY the sub-questions, one per line. No numbers, bullets, or explanations.
Each sub-question should be complete and searchable on its own.
"""
        
        response = generate(prompt, max_tokens=200, temperature=0.0)
        
        if not response:
            return None
        
        # Parse response
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        # Filter out meta-text
        sub_queries = [
            line for line in lines
            if len(line) > 5 
            and not line.lower().startswith(("here", "the ", "sub-question", "1.", "2.", "-", "*"))
        ]
        
        if len(sub_queries) >= 2:
            return sub_queries[:5]  # Cap at 5 sub-queries
        
        return None
    except Exception as e:
        log.warning("query_decomposition.llm_failed", error=str(e))
        return None


def decompose_query(
    query: str,
    *,
    use_llm: bool | None = None,
    force: bool = False,
) -> DecompositionResult:
    """Decompose a complex query into atomic sub-queries.
    
    Args:
        query: The user's search query.
        use_llm: Whether to use LLM for ambiguous cases. Defaults to
                 STROPHA_QUERY_DECOMPOSITION_LLM env var.
        force: If True, always attempt decomposition even if query
               doesn't appear compound.
    
    Returns:
        DecompositionResult with sub_queries list (may be just [query]
        if no decomposition was needed/possible).
    """
    if use_llm is None:
        use_llm = os.environ.get("STROPHA_QUERY_DECOMPOSITION_LLM", "0") == "1"
    
    query = query.strip()
    if not query:
        return DecompositionResult(
            sub_queries=[query] if query else [],
            original_query=query,
            method="none",
        )
    
    # Quick check: is this even a compound query?
    if not force and not _is_compound_query(query):
        return DecompositionResult(
            sub_queries=[query],
            original_query=query,
            method="none",
        )
    
    # Try pattern-based decomposition first (zero cost)
    pattern_result = _pattern_decompose(query)
    if pattern_result:
        log.info(
            "query_decomposition.pattern_split",
            original=query[:50],
            count=len(pattern_result),
        )
        return DecompositionResult(
            sub_queries=pattern_result,
            original_query=query,
            method="pattern",
        )
    
    # Try LLM decomposition if enabled
    if use_llm:
        llm_result = _llm_decompose(query)
        if llm_result:
            log.info(
                "query_decomposition.llm_split",
                original=query[:50],
                count=len(llm_result),
            )
            return DecompositionResult(
                sub_queries=llm_result,
                original_query=query,
                method="llm",
            )
    
    # No decomposition needed/possible
    return DecompositionResult(
        sub_queries=[query],
        original_query=query,
        method="none",
    )


def is_decomposition_enabled() -> bool:
    """Check if query decomposition is enabled via env var."""
    return os.environ.get("STROPHA_QUERY_DECOMPOSITION_ENABLED", "0") == "1"

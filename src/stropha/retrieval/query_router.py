"""Query routing — classify intent and dispatch to appropriate tool.

Routes queries like:
- "what calls X" → find_callers
- "tests for X" / "how is X tested" → find_tests_for
- "what does X relate to" / "what is related to X" → find_related
- "trace feature X" / "how does X flow" → trace_feature
- Everything else → search_code (hybrid retrieval)

Uses pattern matching first (zero cost), then optional LLM classification
for ambiguous queries when STROPHA_QUERY_ROUTER_LLM=1.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


class QueryIntent(Enum):
    """Classified query intent."""
    FIND_CALLERS = "find_callers"
    FIND_TESTS = "find_tests_for"
    FIND_RELATED = "find_related"
    TRACE_FEATURE = "trace_feature"
    GET_SYMBOL = "get_symbol"
    SEARCH_CODE = "search_code"  # default fallback


@dataclass
class RoutedQuery:
    """Result of query routing."""
    intent: QueryIntent
    symbol: str | None  # extracted symbol for graph tools
    original_query: str
    confidence: float  # 0.0-1.0, higher = more confident
    reason: str  # why this intent was chosen


# Pattern-based intent classification (zero LLM cost)
# Each pattern: (compiled_regex, intent, symbol_group_index)
# Symbol pattern: matches CamelCase or snake_case identifiers with optional dots
_SYMBOL = r"[`'\"]?([A-Z_a-z]\w*(?:\.[A-Z_a-z]\w*)*)[`'\"]?"

_INTENT_PATTERNS: list[tuple[re.Pattern, QueryIntent, int | None]] = [
    # find_callers patterns - be specific about what precedes the symbol
    (re.compile(rf"(?:what|who|which)\s+(?:\w+\s+)?(?:calls?|invokes?|uses?)\s+(?:the\s+)?{_SYMBOL}", re.I), QueryIntent.FIND_CALLERS, 1),
    (re.compile(rf"callers?\s+(?:of|for)\s+{_SYMBOL}", re.I), QueryIntent.FIND_CALLERS, 1),
    (re.compile(rf"{_SYMBOL}\s+(?:is\s+)?called\s+(?:by|from|where)", re.I), QueryIntent.FIND_CALLERS, 1),
    (re.compile(rf"where\s+is\s+{_SYMBOL}\s+(?:called|invoked|used)", re.I), QueryIntent.FIND_CALLERS, 1),
    
    # find_tests patterns
    (re.compile(rf"tests?\s+(?:for|of|covering)\s+{_SYMBOL}", re.I), QueryIntent.FIND_TESTS, 1),
    (re.compile(rf"how\s+is\s+{_SYMBOL}\s+tested", re.I), QueryIntent.FIND_TESTS, 1),
    (re.compile(rf"what\s+tests?\s+{_SYMBOL}", re.I), QueryIntent.FIND_TESTS, 1),
    (re.compile(rf"test\s+coverage\s+(?:for|of)\s+{_SYMBOL}", re.I), QueryIntent.FIND_TESTS, 1),
    
    # find_related patterns
    (re.compile(rf"what\s+(?:is\s+)?relates?\s+to\s+{_SYMBOL}", re.I), QueryIntent.FIND_RELATED, 1),
    (re.compile(rf"related\s+(?:to|code|symbols?)\s+(?:for|of)?\s*{_SYMBOL}", re.I), QueryIntent.FIND_RELATED, 1),
    (re.compile(rf"{_SYMBOL}\s+(?:dependencies|dependents|connections)", re.I), QueryIntent.FIND_RELATED, 1),
    
    # trace_feature patterns - these capture more text, not just symbols
    (re.compile(r"trace\s+(?:the\s+)?(?:feature\s+)?[`'\"]?(.+?)[`'\"]?(?:\s+flow)?$", re.I), QueryIntent.TRACE_FEATURE, 1),
    (re.compile(r"follow\s+(?:the\s+)?[`'\"]?(.+?)[`'\"]?\s+(?:flow|path)", re.I), QueryIntent.TRACE_FEATURE, 1),
    (re.compile(r"how\s+does\s+(?:the\s+)?[`'\"]?(.+?)[`'\"]?\s+(?:flow|work|execute)", re.I), QueryIntent.TRACE_FEATURE, 1),
    (re.compile(r"execution\s+(?:path|flow)\s+(?:for|of)\s+[`'\"]?(.+?)[`'\"]?$", re.I), QueryIntent.TRACE_FEATURE, 1),
    
    # get_symbol patterns (exact symbol lookup) - stricter, at end of query
    (re.compile(rf"show\s+(?:me\s+)?(?:the\s+)?{_SYMBOL}$", re.I), QueryIntent.GET_SYMBOL, 1),
    (re.compile(rf"get\s+(?:the\s+)?(?:definition\s+(?:of|for)\s+)?{_SYMBOL}$", re.I), QueryIntent.GET_SYMBOL, 1),
    (re.compile(rf"where\s+is\s+{_SYMBOL}\s+defined", re.I), QueryIntent.GET_SYMBOL, 1),
    (re.compile(rf"^{_SYMBOL}$", re.I), QueryIntent.GET_SYMBOL, 1),  # bare symbol name
]

# LLM prompt for ambiguous queries
_ROUTER_PROMPT = """You are a query router for a code search system. Classify the user's query intent.

Available intents:
- find_callers: "what calls X", "who uses X" - find code that calls a symbol
- find_tests_for: "tests for X", "how is X tested" - find test code for a symbol
- find_related: "what relates to X", "dependencies of X" - find related symbols
- trace_feature: "trace feature X", "how does X flow" - trace execution paths
- get_symbol: "show me X", "definition of X" - lookup exact symbol definition
- search_code: general code search, conceptual questions, anything else

Query: {query}

Respond with ONLY a JSON object (no markdown):
{{"intent": "<intent_name>", "symbol": "<extracted_symbol_or_null>", "confidence": <0.0-1.0>}}

Examples:
Query: "what calls the FsrsCalculator"
{{"intent": "find_callers", "symbol": "FsrsCalculator", "confidence": 0.95}}

Query: "how does mastery calculation work"
{{"intent": "search_code", "symbol": null, "confidence": 0.85}}

Query: "tests for UserService.authenticate"
{{"intent": "find_tests_for", "symbol": "UserService.authenticate", "confidence": 0.95}}
"""


def _pattern_match(query: str) -> RoutedQuery | None:
    """Try pattern-based classification (fast, no LLM)."""
    query = query.strip()
    
    for pattern, intent, symbol_group in _INTENT_PATTERNS:
        match = pattern.search(query)
        if match:
            symbol = match.group(symbol_group) if symbol_group else None
            # Clean up symbol (remove quotes, backticks)
            if symbol:
                symbol = symbol.strip("`'\"")
            return RoutedQuery(
                intent=intent,
                symbol=symbol,
                original_query=query,
                confidence=0.9,  # patterns are high confidence
                reason=f"matched pattern: {pattern.pattern[:50]}...",
            )
    return None


def _llm_classify(query: str) -> RoutedQuery | None:
    """Use LLM for ambiguous query classification."""
    try:
        from ..inference import generate
        
        prompt = _ROUTER_PROMPT.format(query=query)
        response = generate(prompt, max_tokens=100, temperature=0.0)
        
        if not response:
            return None
        
        # Parse JSON response
        import json
        
        # Clean up response (remove markdown code blocks if present)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        data = json.loads(response)
        intent_name = data.get("intent", "search_code")
        symbol = data.get("symbol")
        confidence = float(data.get("confidence", 0.5))
        
        # Map intent name to enum
        intent_map = {
            "find_callers": QueryIntent.FIND_CALLERS,
            "find_tests_for": QueryIntent.FIND_TESTS,
            "find_related": QueryIntent.FIND_RELATED,
            "trace_feature": QueryIntent.TRACE_FEATURE,
            "get_symbol": QueryIntent.GET_SYMBOL,
            "search_code": QueryIntent.SEARCH_CODE,
        }
        intent = intent_map.get(intent_name, QueryIntent.SEARCH_CODE)
        
        return RoutedQuery(
            intent=intent,
            symbol=symbol,
            original_query=query,
            confidence=confidence,
            reason="LLM classification",
        )
    except Exception as e:
        log.warning("query_router.llm_classify_failed", error=str(e))
        return None


def route_query(
    query: str,
    *,
    use_llm: bool | None = None,
) -> RoutedQuery:
    """Route a query to the appropriate intent.
    
    Args:
        query: The user's search query.
        use_llm: Whether to use LLM for ambiguous queries. Defaults to
                 STROPHA_QUERY_ROUTER_LLM env var (0 = off, 1 = on).
    
    Returns:
        RoutedQuery with intent, extracted symbol, and confidence.
    """
    if use_llm is None:
        use_llm = os.environ.get("STROPHA_QUERY_ROUTER_LLM", "0") == "1"
    
    query = query.strip()
    if not query:
        return RoutedQuery(
            intent=QueryIntent.SEARCH_CODE,
            symbol=None,
            original_query=query,
            confidence=1.0,
            reason="empty query",
        )
    
    # Try pattern matching first (fast)
    result = _pattern_match(query)
    if result:
        log.info(
            "query_router.pattern_match",
            intent=result.intent.value,
            symbol=result.symbol,
            confidence=result.confidence,
        )
        return result
    
    # Try LLM classification if enabled
    if use_llm:
        result = _llm_classify(query)
        if result and result.confidence >= 0.7:
            log.info(
                "query_router.llm_classify",
                intent=result.intent.value,
                symbol=result.symbol,
                confidence=result.confidence,
            )
            return result
    
    # Default to search_code
    return RoutedQuery(
        intent=QueryIntent.SEARCH_CODE,
        symbol=None,
        original_query=query,
        confidence=0.5,
        reason="no pattern match, defaulting to search_code",
    )


def is_graph_intent(intent: QueryIntent) -> bool:
    """Check if intent requires the graph to be loaded."""
    return intent in {
        QueryIntent.FIND_CALLERS,
        QueryIntent.FIND_TESTS,
        QueryIntent.FIND_RELATED,
        QueryIntent.TRACE_FEATURE,
    }

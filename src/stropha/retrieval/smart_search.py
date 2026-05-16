"""Smart search — unified search with automatic query routing.

Combines query intent classification with automatic dispatch to the
appropriate tool (search_code, find_callers, find_tests_for, etc.).

Usage:
    from stropha.retrieval.smart_search import smart_search
    
    results = smart_search(
        query="what calls FsrsCalculator",
        storage=storage,
        embedder=embedder,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..logging import get_logger
from ..models import SearchHit
from .query_router import QueryIntent, RoutedQuery, is_graph_intent, route_query

if TYPE_CHECKING:
    from ..embeddings.base import Embedder
    from ..storage import Storage

log = get_logger(__name__)


@dataclass
class SmartSearchResult:
    """Result of smart search."""
    hits: list[SearchHit]
    routed: RoutedQuery
    tool_used: str
    graph_result: dict[str, Any] | None = None  # raw result from graph tools


def smart_search(
    query: str,
    storage: "Storage",
    embedder: "Embedder",
    *,
    top_k: int = 10,
    use_router: bool = True,
    use_llm_router: bool | None = None,
) -> SmartSearchResult:
    """Unified search with automatic query routing.
    
    Routes queries to the appropriate tool:
    - "what calls X" → find_callers
    - "tests for X" → find_tests_for
    - "what relates to X" → find_related
    - "trace feature X" → trace_feature
    - Everything else → hybrid search
    
    Args:
        query: The search query.
        storage: Storage instance.
        embedder: Embedder instance.
        top_k: Maximum results to return.
        use_router: Whether to use query routing. If False, always uses
                    hybrid search.
        use_llm_router: Whether to use LLM for ambiguous queries.
    
    Returns:
        SmartSearchResult with hits, routing info, and tool used.
    """
    from . import graph
    from .search import SearchEngine
    
    # Route the query
    if use_router:
        routed = route_query(query, use_llm=use_llm_router)
    else:
        routed = RoutedQuery(
            intent=QueryIntent.SEARCH_CODE,
            symbol=None,
            original_query=query,
            confidence=1.0,
            reason="routing disabled",
        )
    
    # Check if graph is required and available
    if is_graph_intent(routed.intent):
        if not graph.graph_loaded(storage):
            log.info(
                "smart_search.graph_not_loaded_fallback",
                intent=routed.intent.value,
            )
            # Fall back to search_code
            routed = RoutedQuery(
                intent=QueryIntent.SEARCH_CODE,
                symbol=routed.symbol,
                original_query=query,
                confidence=routed.confidence * 0.5,
                reason=f"graph not loaded, fallback from {routed.intent.value}",
            )
    
    # Dispatch to appropriate tool
    if routed.intent == QueryIntent.FIND_CALLERS and routed.symbol:
        result = graph.find_callers(storage, routed.symbol, limit=top_k)
        hits = _graph_nodes_to_hits(result.get("callers", []), storage)
        return SmartSearchResult(
            hits=hits,
            routed=routed,
            tool_used="find_callers",
            graph_result=result,
        )
    
    elif routed.intent == QueryIntent.FIND_TESTS and routed.symbol:
        result = graph.find_tests_for(storage, routed.symbol, limit=top_k)
        hits = _graph_nodes_to_hits(result.get("tests", []), storage)
        return SmartSearchResult(
            hits=hits,
            routed=routed,
            tool_used="find_tests_for",
            graph_result=result,
        )
    
    elif routed.intent == QueryIntent.FIND_RELATED and routed.symbol:
        result = graph.find_related(storage, routed.symbol, limit=top_k)
        hits = _graph_nodes_to_hits(result.get("related", []), storage)
        return SmartSearchResult(
            hits=hits,
            routed=routed,
            tool_used="find_related",
            graph_result=result,
        )
    
    elif routed.intent == QueryIntent.TRACE_FEATURE:
        # trace_feature uses the full query, not just symbol
        feature = routed.symbol or routed.original_query
        result = graph.trace_feature(storage, feature, max_paths=top_k)
        # Flatten paths to hits
        all_nodes = []
        for path in result.get("paths", []):
            all_nodes.extend(path.get("nodes", []))
        # Dedupe by node_id
        seen = set()
        unique_nodes = []
        for node in all_nodes:
            node_id = node.get("node_id") or node.get("symbol")
            if node_id and node_id not in seen:
                seen.add(node_id)
                unique_nodes.append(node)
        hits = _graph_nodes_to_hits(unique_nodes[:top_k], storage)
        return SmartSearchResult(
            hits=hits,
            routed=routed,
            tool_used="trace_feature",
            graph_result=result,
        )
    
    elif routed.intent == QueryIntent.GET_SYMBOL and routed.symbol:
        # Direct symbol lookup
        hits = storage.lookup_by_symbol(routed.symbol, limit=top_k)
        return SmartSearchResult(
            hits=hits,
            routed=routed,
            tool_used="get_symbol",
        )
    
    # Default: hybrid search
    engine = SearchEngine(storage, embedder)
    hits = engine.search(query, top_k=top_k)
    return SmartSearchResult(
        hits=hits,
        routed=routed,
        tool_used="search_code",
    )


def _graph_nodes_to_hits(
    nodes: list[dict[str, Any]],
    storage: "Storage",
) -> list[SearchHit]:
    """Convert graph node dicts to SearchHit objects."""
    hits = []
    for i, node in enumerate(nodes):
        # Try to find chunk from node info
        rel_path = node.get("chunk_rel_path") or node.get("source_file")
        start_line = node.get("chunk_start_line") or _parse_line(node.get("source_location"))
        end_line = node.get("chunk_end_line") or start_line
        snippet = node.get("chunk_snippet") or node.get("label", "")
        
        if not rel_path:
            continue
        
        hit = SearchHit(
            chunk_id=node.get("chunk_id", f"graph:{node.get('node_id', '')}"),
            rel_path=rel_path,
            start_line=start_line or 1,
            end_line=end_line or 1,
            language=_guess_language(rel_path),
            kind=node.get("kind", "symbol"),
            symbol=node.get("symbol") or node.get("node_id", "").split(":")[-1],
            content=snippet,
            snippet=snippet[:200] if snippet else "",
            score=1.0 - (i * 0.05),  # rank-based score
            rank=i + 1,
            repo=None,
        )
        hits.append(hit)
    
    return hits


def _parse_line(location: str | None) -> int | None:
    """Parse line number from source_location like 'file.py:42'."""
    if not location:
        return None
    if ":" in location:
        try:
            return int(location.split(":")[-1])
        except ValueError:
            pass
    return None


def _guess_language(path: str) -> str:
    """Guess language from file extension."""
    ext_map = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "jsx",
        ".java": "java",
        ".kt": "kotlin",
        ".go": "go",
        ".rs": "rust",
        ".vue": "vue",
        ".md": "markdown",
        ".feature": "gherkin",
    }
    for ext, lang in ext_map.items():
        if path.endswith(ext):
            return lang
    return "unknown"

"""MCP server entry point.

Wires Config → Embedder → Storage → SearchEngine inside a FastMCP lifespan,
and exposes the Phase 1 tool surface from spec §9.2:

- `search_code`         — hybrid retrieval (dense + BM25 + RRF).
- `get_symbol`          — direct symbol lookup.
- `get_file_outline`    — symbolic outline of a single file.
- `list_repos`          — enumerate repositories present in the index.
- `index_stats` resource (stropha://stats).

Every search result carries a ``repo`` field (URL, default branch, HEAD)
so a remote MCP client can ``git clone`` the source. The server speaks
stdio by default; that is the transport Claude Code, Cursor and friends
launch automatically when configured via `.mcp.json`.

Background file watcher (enabled by default via ``STROPHA_MCP_WATCH=1``):
The server starts a background thread that polls the target repo for file
changes. After a debounce period (default 2s of no further changes), it
triggers a light re-index pass so MCP queries always see recent edits
without requiring a separate ``stropha watch`` command. Disable with
``STROPHA_MCP_WATCH=0`` if you prefer manual indexing or use the hook only.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from . import __version__
from .config import Config, get_config, get_config_info
from .embeddings import build_embedder
from .embeddings.base import Embedder
from .errors import StrophaError
from .logging import configure_logging, get_logger
from .retrieval import SearchEngine
from .retrieval.graph import (
    find_callers as graph_find_callers,
    find_rationale as graph_find_rationale,
    find_related as graph_find_related,
    find_tests_for as graph_find_tests_for,
    get_community as graph_get_community,
    graph_loaded,
    has_rationale_edges,
    trace_feature as graph_trace_feature,
)
from .retrieval.smart_search import SmartSearchResult, smart_search
from .retrieval.graph import (
    find_rationale as graph_find_rationale,
)
from .retrieval.graph import (
    find_related as graph_find_related,
)
from .retrieval.graph import (
    get_community as graph_get_community,
)
from .retrieval.graph import (
    graph_loaded,
    has_rationale_edges,
)
from .storage import Storage
from .watch import WatchController

log = get_logger(__name__)


# ---- composition root ------------------------------------------------------

@dataclass
class AppContext:
    """Long-lived dependencies, injected into every tool via the MCP Context."""

    config: Config
    storage: Storage
    embedder: Embedder
    search_engine: SearchEngine


@asynccontextmanager
async def _lifespan(app: FastMCP) -> AsyncIterator[AppContext]:
    # Use the centralized config singleton. It handles .env loading with proper
    # precedence (env vars from MCP client override .env file values).
    cfg = get_config()
    configure_logging(cfg.log_level)
    log.info(
        "mcp.startup",
        version=__version__,
        index_path=str(cfg.resolve_index_path()),
        target=str(cfg.target_repo),
        mcp_watch=cfg.mcp_watch,
    )
    embedder = build_embedder(cfg)
    storage = Storage(cfg.resolve_index_path(), embedding_dim=embedder.dim)
    engine = SearchEngine(storage, embedder)

    # Start background file watcher if enabled (default: on).
    # This keeps the index fresh as the user edits files, so MCP queries
    # always see recent changes without requiring a separate `stropha watch`.
    watcher: WatchController | None = None
    if cfg.mcp_watch:
        watcher = WatchController(
            repo=cfg.target_repo,
            interval_s=cfg.mcp_watch_interval,
            debounce_s=cfg.mcp_watch_debounce,
            full_refresh=False,  # Light refresh; hook handles graphify
        )
        watcher.start()

    try:
        yield AppContext(cfg, storage, embedder, engine)
    finally:
        if watcher is not None:
            watcher.stop()
        log.info("mcp.shutdown")
        storage.close()


_INSTRUCTIONS = """\
stropha — RAG over the local codebase. Prefer these tools over directory walks
or grep when looking for code by intent ("how does mastery work?"), by symbol
("EnrollmentRepository.findByUsername"), or for a file's symbolic outline.

Tool selection guide:
- Free-text or conceptual query → search_code
- Known symbol / fully-qualified name → get_symbol  (cheaper, exact)
- Need a file's structure before reading → get_file_outline
- "Who calls X?" → find_callers (structural — uses graphify graph)
- "What relates to X?" → find_related (structural — graph traversal)
- "Show me the community of X" → get_community (cluster grouping)
- "Why was X built this way?" → find_rationale (links code to docs/ADRs)

The find_* tools require a graphify graph in `graphify-out/graph.json`.
When the graph is missing they return {"graph_loaded": false, ...} —
run `graphify .` (or the post-commit hook installed by `stropha hook install`)
to bootstrap.
"""


mcp = FastMCP(
    name="stropha_rag",
    instructions=_INSTRUCTIONS,
    lifespan=_lifespan,
)


# ---- response models --------------------------------------------------------

class RepoInfo(BaseModel):
    """Source-repo identity attached to every search result.

    A client can run `git clone {url}` (when `url` is non-null) to obtain
    the source tree, then check out `default_branch` (or `head_commit`).
    """

    normalized_key: str = Field(
        description="Stable cross-user key (e.g. 'github.com/foo/bar')."
    )
    url: str | None = Field(default=None, description="HTTPS clone URL.")
    default_branch: str | None = None
    head_commit: str | None = None


class SearchResult(BaseModel):
    rank: int
    score: float = Field(description="Hybrid (RRF) score; higher = more relevant.")
    path: str = Field(description="File path relative to the repo root.")
    start_line: int
    end_line: int
    language: str
    kind: str = Field(description="file | class | method | function | section | scenario | …")
    symbol: str | None = None
    snippet: str = Field(description="Truncated preview. Use Read tool for full content.")
    chunk_id: str
    repo: RepoInfo | None = Field(
        default=None,
        description="Source repository identity (URL, branch, HEAD).",
    )


class FacetCounts(BaseModel):
    """Counts per facet dimension, computed from search results."""

    language: dict[str, int] = Field(
        default_factory=dict,
        description="Count of results per language (e.g. {'python': 15, 'java': 8}).",
    )
    kind: dict[str, int] = Field(
        default_factory=dict,
        description="Count of results per chunk kind (e.g. {'method': 20, 'class': 5}).",
    )
    repo: dict[str, int] = Field(
        default_factory=dict,
        description="Count of results per repository.",
    )


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_candidates: int = Field(
        description="Total candidates considered before fusion (sum of dense + sparse top-50)."
    )
    query: str
    facets: FacetCounts | None = Field(
        default=None,
        description="Facet counts when include_facets=True. Shows distribution by language, kind, repo.",
    )


class OutlineEntry(BaseModel):
    chunk_id: str
    kind: str
    symbol: str | None
    parent_chunk_id: str | None
    start_line: int
    end_line: int


class RepoSummary(BaseModel):
    """One entry in the `list_repos` response."""

    normalized_key: str
    url: str | None
    default_branch: str | None
    head_commit: str | None
    files: int
    chunks: int
    last_indexed_at: str | None


class StatsPayload(BaseModel):
    db_path: str
    size_bytes: int
    chunks: int
    files: int
    index_dim: int
    models: list[dict]
    repos: list[RepoSummary] = Field(default_factory=list)
    enrichers: list[dict] = Field(default_factory=list)
    enrichment_cache_size: int = 0
    graph: dict | None = Field(
        default=None,
        description="Graphify mirror summary (None when no graph loaded).",
    )


# ---- helpers ---------------------------------------------------------------

def _ctx(mcp_ctx: Context) -> AppContext:
    return mcp_ctx.request_context.lifespan_context


def _to_result(hit) -> SearchResult:  # type: ignore[no-untyped-def]
    repo: RepoInfo | None = None
    if hit.repo is not None:
        repo = RepoInfo(
            normalized_key=hit.repo.normalized_key,
            url=hit.repo.url,
            default_branch=hit.repo.default_branch,
            head_commit=hit.repo.head_commit,
        )
    return SearchResult(
        rank=hit.rank,
        score=hit.score,
        path=hit.rel_path,
        start_line=hit.start_line,
        end_line=hit.end_line,
        language=hit.language,
        kind=hit.kind,
        symbol=hit.symbol,
        snippet=hit.snippet,
        chunk_id=hit.chunk_id,
        repo=repo,
    )


# ---- tools ------------------------------------------------------------------

def _apply_filters(
    hits: list,
    *,
    language: list[str] | None = None,
    path_prefix: str | None = None,
    kind: list[str] | None = None,
    exclude_tests: bool = False,
) -> list:
    """Apply post-retrieval filters to search hits.
    
    Note: This is post-filtering on already-retrieved candidates. For large
    result sets, pre-filtering in SQL would be more efficient (future work).
    """
    result = hits
    
    if language:
        lang_set = {lang.lower() for lang in language}
        result = [h for h in result if h.language.lower() in lang_set]
    
    if path_prefix:
        result = [h for h in result if h.rel_path.startswith(path_prefix)]
    
    if kind:
        kind_set = {k.lower() for k in kind}
        result = [h for h in result if h.kind.lower() in kind_set]
    
    if exclude_tests:
        test_patterns = ("test_", "_test.", ".test.", "/tests/", "/test/", "Test.")
        result = [
            h for h in result
            if not any(p in h.rel_path for p in test_patterns)
        ]
    
    return result


@mcp.tool(
    title="Search code",
    description=(
        "Hybrid semantic + lexical search over the indexed codebase. "
        "Use for queries like 'where is X', 'how does Y work', "
        "'show examples of Z'. For an exact symbol name, prefer get_symbol. "
        "Supports optional filters: language, path_prefix, kind, exclude_tests. "
        "Set include_facets=True to get counts by language/kind/repo. "
        "Set recursive=True to auto-merge sibling and adjacent chunks."
    ),
)
def search_code(
    query: str,
    top_k: int = 10,
    language: list[str] | None = None,
    path_prefix: str | None = None,
    kind: list[str] | None = None,
    exclude_tests: bool = False,
    include_facets: bool = False,
    recursive: bool = False,
    *,
    ctx: Context,
) -> SearchResponse:
    """Hybrid search (dense + BM25, fused via RRF). Returns top_k results
    with path, line range, language, kind, symbol, and a snippet.

    Args:
        query: Natural-language question or technical terms.
        top_k: 1–30. Default 10.
        language: Filter by language(s), e.g. ["java", "python"].
        path_prefix: Filter by path prefix, e.g. "backend/src/".
        kind: Filter by chunk kind(s), e.g. ["method", "class", "function"].
        exclude_tests: If True, exclude test files from results.
        include_facets: If True, include facet counts (language, kind, repo distribution).
        recursive: If True, auto-merge sibling chunks (same parent) and adjacent chunks.
    """
    import os
    
    app = _ctx(ctx)
    top_k = max(1, min(top_k, 30))

    # Enable recursive retrieval for this search if requested
    prev_recursive = os.environ.get("STROPHA_RECURSIVE_RETRIEVAL")
    if recursive:
        os.environ["STROPHA_RECURSIVE_RETRIEVAL"] = "1"

    # Fetch more candidates when filtering to ensure we get enough results
    has_filters = any([language, path_prefix, kind, exclude_tests])
    fetch_k = top_k * 3 if has_filters else top_k

    try:
        hits = app.search_engine.search(query, top_k=fetch_k)
    except StrophaError as exc:
        log.warning("mcp.search_code.error", error=str(exc))
        return SearchResponse(results=[], total_candidates=0, query=query, facets=None)
    finally:
        # Restore previous env var state
        if recursive:
            if prev_recursive is None:
                os.environ.pop("STROPHA_RECURSIVE_RETRIEVAL", None)
            else:
                os.environ["STROPHA_RECURSIVE_RETRIEVAL"] = prev_recursive

    # Apply filters
    if has_filters:
        hits = _apply_filters(
            hits,
            language=language,
            path_prefix=path_prefix,
            kind=kind,
            exclude_tests=exclude_tests,
        )

    # Compute facets before truncating (on filtered results)
    facets: FacetCounts | None = None
    if include_facets and hits:
        chunk_ids = [h.chunk_id for h in hits]
        facet_data = app.storage.compute_facets(chunk_ids)
        facets = FacetCounts(
            language=facet_data["language"],
            kind=facet_data["kind"],
            repo=facet_data["repo"],
        )

    # Truncate to requested top_k
    hits = hits[:top_k]

    return SearchResponse(
        results=[_to_result(h) for h in hits],
        total_candidates=len(hits),
        query=query,
        facets=facets,
    )


# ---- smart search (query routing) ------------------------------------------

class SmartSearchResponse(BaseModel):
    """Response from smart_search with routing metadata."""
    results: list[SearchResult]
    total_candidates: int
    query: str
    intent: str = Field(description="Detected query intent (find_callers, find_tests_for, search_code, etc.)")
    symbol: str | None = Field(default=None, description="Extracted symbol name, if any")
    tool_used: str = Field(description="The actual tool invoked (find_callers, search_code, etc.)")
    confidence: float = Field(description="Routing confidence (0.0-1.0)")
    graph_available: bool = Field(default=True, description="Whether the graph is loaded")


@mcp.tool(
    title="Smart search",
    description=(
        "Intelligent search with automatic query routing. Analyzes the query "
        "intent and dispatches to the best tool:\n"
        "- 'what calls X' → find_callers\n"
        "- 'tests for X' → find_tests_for\n"
        "- 'what relates to X' → find_related\n"
        "- 'trace feature X' → trace_feature\n"
        "- Everything else → hybrid search\n\n"
        "Use this when you're not sure which tool to use. Returns results "
        "with metadata about which tool was used and why."
    ),
)
def smart_search_tool(
    query: str,
    top_k: int = 10,
    use_llm_router: bool = False,
    *,
    ctx: Context,
) -> SmartSearchResponse:
    """Smart search with automatic query routing.
    
    Args:
        query: Natural-language question or technical terms.
        top_k: 1–30. Default 10.
        use_llm_router: If True, use LLM for ambiguous queries (slower but more accurate).
    """
    app = _ctx(ctx)
    top_k = max(1, min(top_k, 30))
    
    try:
        result = smart_search(
            query,
            app.storage,
            app.embedder,
            top_k=top_k,
            use_router=True,
            use_llm_router=use_llm_router,
        )
        return SmartSearchResponse(
            results=[_to_result(h) for h in result.hits],
            total_candidates=len(result.hits),
            query=query,
            intent=result.routed.intent.value,
            symbol=result.routed.symbol,
            tool_used=result.tool_used,
            confidence=result.routed.confidence,
            graph_available=graph_loaded(app.storage),
        )
    except StrophaError as exc:
        log.warning("mcp.smart_search.error", error=str(exc))
        return SmartSearchResponse(
            results=[],
            total_candidates=0,
            query=query,
            intent="search_code",
            symbol=None,
            tool_used="search_code",
            confidence=0.0,
            graph_available=graph_loaded(app.storage),
        )


@mcp.tool(
    title="Get symbol",
    description=(
        "Return chunks whose `symbol` matches the given name. "
        "Cheaper and more precise than search_code when the symbol is known. "
        "Accepts simple names ('FsrsCalculator') or qualified names "
        "('StudyService.submitAnswer')."
    ),
)
def get_symbol(symbol: str, limit: int = 5, *, ctx: Context) -> list[SearchResult]:
    """Find by symbol name. Returns up to `limit` matches, exact-match first."""
    app = _ctx(ctx)
    hits = app.storage.lookup_by_symbol(symbol, limit=max(1, min(limit, 20)))
    return [_to_result(h) for h in hits]


@mcp.tool(
    title="List repositories",
    description=(
        "Enumerate every repository present in this index, with file and "
        "chunk counts. Useful when the index spans multiple repos and the "
        "client needs to know what is available before issuing search_code. "
        "Each entry includes the clone URL when available."
    ),
)
def list_repos(*, ctx: Context) -> list[RepoSummary]:
    """Aggregate per-repo stats. Cheap query — single GROUP BY."""
    app = _ctx(ctx)
    return [
        RepoSummary(
            normalized_key=r.normalized_key,
            url=r.url,
            default_branch=r.default_branch,
            head_commit=r.head_commit,
            files=r.files,
            chunks=r.chunks,
            last_indexed_at=r.last_indexed_at,
        )
        for r in app.storage.list_repos()
    ]


@mcp.tool(
    title="Get file outline",
    description=(
        "Symbolic outline of a single file: every chunk's kind, symbol, and "
        "line range, sorted by position. Use this to plan a Read before "
        "consuming a whole file into context."
    ),
)
def get_file_outline(path: str, *, ctx: Context) -> list[OutlineEntry]:
    """Return the chunk-level outline of a file path (repo-relative)."""
    app = _ctx(ctx)
    rows = app.storage.file_outline(path)
    return [
        OutlineEntry(
            chunk_id=r["chunk_id"],
            kind=r["kind"],
            symbol=r["symbol"],
            parent_chunk_id=r["parent_chunk_id"],
            start_line=int(r["start_line"]),
            end_line=int(r["end_line"]),
        )
        for r in rows
    ]


# ---- graph tools (RFC §5 — Trilha A 1.5b) ----------------------------------
# Always registered for tool-discovery stability. When the graphify mirror
# tables are empty (no `graphify-out/graph.json` ever loaded), the tools
# short-circuit with `{graph_loaded: False, message: ...}` so the calling
# agent gets actionable feedback instead of a silent empty list.


def _graph_unavailable() -> dict:
    return {
        "graph_loaded": False,
        "message": (
            "graphify graph not loaded. Run `graphify .` in the repo root to "
            "bootstrap, then `stropha index` to mirror it into SQLite."
        ),
    }


@mcp.tool(
    title="Find callers of a symbol",
    description=(
        "Lists code locations that call `symbol`. Direct traversal over the "
        "graphify mirror filtered by relation='calls' and confidence='EXTRACTED'. "
        "Returns more precise results than `search_code` for queries like "
        "'who calls X?'. Requires `graphify-out/graph.json` (run `graphify .` "
        "to bootstrap)."
    ),
)
def find_callers(
    symbol: str,
    *,
    ctx: Context,
    depth: int = 1,
    limit: int = 20,
) -> dict:
    """Return callers of ``symbol`` (BFS up incoming `calls` edges, depth ≤ 3)."""
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"symbol": symbol}
    return graph_find_callers(app.storage, symbol, depth=depth, limit=limit)


@mcp.tool(
    title="Find related nodes",
    description=(
        "Symmetric BFS over the graphify graph (in + out edges). Useful when "
        "you don't know the relationship type ahead of time. Pass `relations` "
        "to filter (e.g. ['calls','implements','references']). Depth capped at 3."
    ),
)
def find_related(
    symbol: str,
    *,
    ctx: Context,
    depth: int = 1,
    limit: int = 20,
    relations: list[str] | None = None,
) -> dict:
    """Return any node connected to ``symbol`` via the graph."""
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"symbol": symbol}
    return graph_find_related(
        app.storage, symbol, depth=depth, limit=limit,
        relations=tuple(relations) if relations else None,
    )


@mcp.tool(
    title="Get community of a symbol",
    description=(
        "Return all nodes in the same community as `symbol_or_community_id`. "
        "Communities are pre-computed by graphify clustering and represent "
        "tight cohesive groups (e.g. 'Hybrid Retrieval & RRF'). Use this to "
        "see everything that participates in a coherent feature/module."
    ),
)
def get_community(
    symbol_or_community_id: str,
    *,
    ctx: Context,
    limit: int = 50,
) -> dict:
    """Return the community membership of a symbol or community id."""
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"query": symbol_or_community_id}
    # Accept stringified int as community id
    query: str | int = symbol_or_community_id
    try:
        query = int(symbol_or_community_id)
    except (ValueError, TypeError):
        pass
    return graph_get_community(app.storage, query, limit=limit)


@mcp.tool(
    title="Find rationale for a symbol",
    description=(
        "Return rationale-style nodes (typically docs / ADRs / design "
        "comments) that explain why `symbol` exists or was built this way. "
        "Walks `rationale_for` edges from the graphify graph. Returns empty "
        "when no rationale edges target the symbol."
    ),
)
def find_rationale(
    symbol: str,
    *,
    ctx: Context,
    limit: int = 10,
) -> dict:
    """Return rationale nodes (docs/ADRs) explaining ``symbol``."""
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"symbol": symbol}
    if not has_rationale_edges(app.storage):
        return {
            "symbol": symbol,
            "rationale": [],
            "message": "Graph loaded but no `rationale_for` edges present yet.",
        }
    return graph_find_rationale(app.storage, symbol, limit=limit)


@mcp.tool(
    title="Find tests covering a symbol",
    description=(
        "Return code chunks in test files that call / reference / implement "
        "the given symbol. Walks `calls`, `references`, `implements`, `tests` "
        "edges from the graphify graph and filters callers whose source_file "
        "matches a common test convention (test_*, *_test, *.spec.*, "
        "*.test.*, /tests/, /test/). Pass custom patterns when your project "
        "uses other conventions. Requires the graphify graph."
    ),
)
def find_tests_for(
    symbol: str,
    *,
    ctx: Context,
    limit: int = 20,
    test_path_patterns: list[str] | None = None,
) -> dict:
    """Return tests that exercise ``symbol`` (path-pattern filtered)."""
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"symbol": symbol}
    patterns = (
        tuple(test_path_patterns)
        if test_path_patterns
        else ("test_", "_test", "/tests/", "/test/", ".spec.", ".test.")
    )
    return graph_find_tests_for(
        app.storage, symbol, limit=limit, test_path_patterns=patterns,
    )


@mcp.tool(
    title="Trace a feature through the call graph",
    description=(
        "Trace a feature description (e.g. 'user submits an answer') down "
        "through the call graph to surface every code chunk that "
        "participates in it. Picks entry-point nodes by token overlap, "
        "then DFS along EXTRACTED `calls` edges. Useful for Gherkin "
        "scenario → step-definition → method chains, but works for any "
        "free-text feature. Requires the graphify graph."
    ),
)
def trace_feature(
    feature: str,
    *,
    ctx: Context,
    max_paths: int = 5,
    max_depth: int = 6,
) -> dict:
    """Walk the call graph from feature-matching entry points."""
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"feature": feature}
    return graph_trace_feature(
        app.storage, feature, max_paths=max_paths, max_depth=max_depth,
    )


@mcp.tool(
    name="get_config",
    description=(
        "Show the active stropha configuration: index path, target repo, "
        "embedding model, and environment sources. Use this to debug which "
        "database the MCP server is using and where the config values came from."
    ),
)
def get_config_tool(*, ctx: Context) -> dict:
    """Return the active configuration for debugging."""
    app = _ctx(ctx)
    stats = app.storage.stats()
    
    # Get the centralized config info
    info = get_config_info()
    
    # Add runtime stats from the active storage/embedder
    info["embedder"] = {
        "model": app.embedder.model_name,
        "dim": app.embedder.dim,
    }
    info["index_stats"] = {
        "total_chunks": stats.get("chunks", 0),
        "total_files": stats.get("files", 0),
        "repos": [
            {"name": r.get("normalized_key"), "chunks": r.get("chunks")}
            for r in stats.get("repos", [])
        ],
    }
    
    return info


@mcp.resource("stropha://stats")
def index_stats_resource() -> StatsPayload:
    """Index statistics. Exposed as both a Tool and a Resource for discovery."""
    # FastMCP resources cannot use the lifespan context directly today, so we
    # re-derive what we need from the config singleton + a short-lived Storage
    # handle. This is fine because the resource is read-only and called rarely.
    cfg = get_config()
    embedder = build_embedder(cfg)
    with Storage(cfg.resolve_index_path(), embedding_dim=embedder.dim) as storage:
        info = storage.stats()
    return StatsPayload(**info)


# ---- entry point -----------------------------------------------------------

def main() -> None:
    """Run the MCP server over stdio. Used by `[project.scripts]` console entry."""
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

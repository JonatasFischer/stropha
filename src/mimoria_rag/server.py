"""MCP server entry point.

Wires Config → Embedder → Storage → SearchEngine inside a FastMCP lifespan,
and exposes the Phase 1 tool surface from spec §9.2:

- `search_code`         — hybrid retrieval (dense + BM25 + RRF).
- `get_symbol`          — direct symbol lookup.
- `get_file_outline`    — symbolic outline of a single file.
- `index_stats` resource (rag://stats).

The server speaks stdio by default; that is the transport Claude Code, Cursor
and friends launch automatically when configured via `.mcp.json`.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from . import __version__
from .config import Config
from .embeddings import build_embedder
from .embeddings.base import Embedder
from .errors import RagError
from .logging import configure_logging, get_logger
from .retrieval import SearchEngine
from .storage import Storage

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
    load_dotenv()
    cfg = Config()  # type: ignore[call-arg]
    configure_logging(cfg.log_level)
    log.info(
        "mcp.startup",
        version=__version__,
        index_path=str(cfg.resolve_index_path()),
        target=str(cfg.target_repo),
    )
    embedder = build_embedder(cfg)
    storage = Storage(cfg.resolve_index_path(), embedding_dim=embedder.dim)
    engine = SearchEngine(storage, embedder)
    try:
        yield AppContext(cfg, storage, embedder, engine)
    finally:
        log.info("mcp.shutdown")
        storage.close()


_INSTRUCTIONS = """\
RAG over the Mimoria codebase. Prefer these tools over directory walks or
grep when looking for code by intent ("how does mastery work?"), by symbol
("FsrsCalculator.calculateNewStability"), or for a file's symbolic outline.

Tool selection guide:
- Free-text or conceptual query → search_code
- Known symbol / fully-qualified name → get_symbol  (cheaper, exact)
- Need a file's structure before reading → get_file_outline
"""


mcp = FastMCP(
    name="mimoria-rag",
    instructions=_INSTRUCTIONS,
    lifespan=_lifespan,
)


# ---- response models --------------------------------------------------------

class SearchResult(BaseModel):
    rank: int
    score: float = Field(description="Hybrid (RRF) score; higher = more relevant.")
    path: str
    start_line: int
    end_line: int
    language: str
    kind: str = Field(description="file | class | method | function | section | scenario | …")
    symbol: str | None = None
    snippet: str = Field(description="Truncated preview. Use Read tool for full content.")
    chunk_id: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_candidates: int = Field(
        description="Total candidates considered before fusion (sum of dense + sparse top-50)."
    )
    query: str


class OutlineEntry(BaseModel):
    chunk_id: str
    kind: str
    symbol: str | None
    parent_chunk_id: str | None
    start_line: int
    end_line: int


class StatsPayload(BaseModel):
    db_path: str
    size_bytes: int
    chunks: int
    files: int
    index_dim: int
    models: list[dict]


# ---- helpers ---------------------------------------------------------------

def _ctx(mcp_ctx: Context) -> AppContext:
    return mcp_ctx.request_context.lifespan_context


def _to_result(hit) -> SearchResult:  # type: ignore[no-untyped-def]
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
    )


# ---- tools ------------------------------------------------------------------

@mcp.tool(
    title="Search code",
    description=(
        "Hybrid semantic + lexical search over the Mimoria codebase. "
        "Use for queries like 'where is X', 'how does Y work', "
        "'show examples of Z'. For an exact symbol name, prefer get_symbol."
    ),
)
def search_code(
    query: str,
    top_k: int = 10,
    *,
    ctx: Context,
) -> SearchResponse:
    """Hybrid search (dense + BM25, fused via RRF). Returns top_k results
    with path, line range, language, kind, symbol, and a snippet.

    Args:
        query: Natural-language question or technical terms.
        top_k: 1–30. Default 10.
    """
    app = _ctx(ctx)
    top_k = max(1, min(top_k, 30))
    try:
        hits = app.search_engine.search(query, top_k=top_k)
    except RagError as exc:
        log.warning("mcp.search_code.error", error=str(exc))
        return SearchResponse(results=[], total_candidates=0, query=query)
    return SearchResponse(
        results=[_to_result(h) for h in hits],
        total_candidates=top_k,
        query=query,
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


@mcp.resource("rag://stats")
def index_stats_resource() -> StatsPayload:
    """Index statistics. Exposed as both a Tool and a Resource for discovery."""
    # FastMCP resources cannot use the lifespan context directly today, so we
    # re-derive what we need from Config + a short-lived Storage handle. This
    # is fine because the resource is read-only and called rarely.
    cfg = Config()  # type: ignore[call-arg]
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

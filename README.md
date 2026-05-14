# mimoria-rag

RAG system over the [Mimoria](https://github.com/anomalyco/mimoria) codebase, exposed to LLM clients via the **Model Context Protocol (MCP)**.

> **Status:** Phase 1 — MVP MCP. Tree-sitter AST chunking (Java / TS / JS / Python / Rust / Go / Kotlin) plus custom chunkers for Vue / Markdown / Gherkin. Three-stream hybrid retrieval (dense + BM25 + symbol-lookup) fused via RRF. MCP server (`mimoria-rag-mcp`) exposes `search_code`, `get_symbol`, `get_file_outline` over stdio. Reranking, Contextual Retrieval, symbol graph, and git-diff incremental reindex are planned for Phase 2/3. See `docs/architecture/rag-system.md` §16.

## Quick start

```bash
# 1. Install deps (Python 3.12 via uv)
uv sync

# 2. Configure
cp .env.example .env
# Optional but recommended: set VOYAGE_API_KEY in .env.
# Without it, a local ONNX fallback model is used.

# 3. Point at the target repo (in .env: RAG_TARGET_REPO)
# Defaults to ../mimoria when run from this directory.

# 4. Build the index
uv run mimoria-rag index

# 5. Search
uv run mimoria-rag search "how does FSRS mastery work"
uv run mimoria-rag stats
```

## Connect Claude Code (MCP)

```bash
# Copy the template into the repo whose code you want indexed,
# then rename it. Claude Code reads `.mcp.json` automatically.
cp .mcp.example.json ../mimoria/.mcp.json
# (edit the absolute path in the file if your checkout lives elsewhere)
```

The server (`mimoria-rag-mcp`) exposes three tools:

| Tool | Use for |
|---|---|
| `search_code(query, top_k)` | Free-text or conceptual queries ("how does mastery work?"). |
| `get_symbol(symbol, limit)`  | Known identifier — `FsrsCalculator`, `StudyService.submitAnswer`. |
| `get_file_outline(path)`     | Structure of a single file before reading it in full. |

A read-only resource `rag://stats` reports index size and the embedding models in use.

## What's here

- **Spec:** `docs/architecture/rag-system.md` — full technical design (20 sections, ~57 KB).
- **Agent guide:** `CLAUDE.md` — for LLM coding agents working on this repo.
- **Code:** `src/mimoria_rag/` — Python 3.12, organized per spec §9.4.

## Architecture (one screen)

```
target repo (git ls-files)
        │
        ▼
   Walker ──► Chunker ──► Embedder ──► Storage (SQLite)
                                          │  ├─ chunks (metadata)
                                          │  ├─ vec_chunks (sqlite-vec)
                                          │  └─ fts_chunks (FTS5, Phase 1)
                                          ▼
                                     Retrieval ──► CLI
                                                   └─► MCP server (Phase 1)
```

## License

MIT.

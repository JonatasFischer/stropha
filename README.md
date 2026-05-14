# stropha

A Retrieval-Augmented Generation indexer for codebases, exposed to LLM clients via the **Model Context Protocol (MCP)**.

Originally built to give Claude Code precise, local-first retrieval over the [Mimoria](https://github.com/anomalyco/mimoria) codebase, but repo-agnostic — point `STROPHA_TARGET_REPO` anywhere.

> **Status:** Phase 1 — MVP MCP. Tree-sitter AST chunking (Java / TS / JS / Python / Rust / Go / Kotlin) plus custom chunkers for Vue / Markdown / Gherkin. Three-stream hybrid retrieval (dense + BM25 + symbol-lookup) fused via RRF. MCP server (`stropha-mcp`) exposes `search_code`, `get_symbol`, `get_file_outline` over stdio. Reranking, Contextual Retrieval, symbol-graph integration and post-commit hooks are planned for Phase 2/3. See `docs/architecture/stropha-system.md` §16.

## Quick start

```bash
# 1. Install deps (Python 3.12 via uv)
uv sync

# 2. Configure
cp .env.example .env
# Optional but recommended: set VOYAGE_API_KEY in .env.
# Without it, a local ONNX model is used (mixedbread-ai/mxbai-embed-large-v1 by default).

# 3. Point at the target repo (in .env: STROPHA_TARGET_REPO)
# Defaults to the current working directory.

# 4. Build the index
uv run stropha index

# 5. Search
uv run stropha search "how does FSRS mastery work"
uv run stropha stats
```

## Connect Claude Code (MCP)

```bash
# Copy the template into the repo whose code you want indexed,
# then rename it. Claude Code reads `.mcp.json` automatically.
cp .mcp.example.json ../<your-repo>/.mcp.json
# (edit the absolute path in the file to point at this stropha checkout)
```

The server (`stropha-mcp`) exposes three tools:

| Tool | Use for |
|---|---|
| `search_code(query, top_k)` | Free-text or conceptual queries ("how does mastery work?"). |
| `get_symbol(symbol, limit)`  | Known identifier — `FsrsCalculator`, `StudyService.submitAnswer`. |
| `get_file_outline(path)`     | Structure of a single file before reading it in full. |

A read-only resource `stropha://stats` reports index size and the embedding models in use.

## What's here

- **Spec:** `docs/architecture/stropha-system.md` — full technical design (20 sections, ~57 KB).
- **Integration plan:** `docs/architecture/stropha-graphify-integration.md` — RFC-style design for symbol-graph integration via graphify + post-commit hook automation (proposed, not yet implemented).
- **Agent guide:** `CLAUDE.md` — for LLM coding agents working on this repo.
- **Code:** `src/stropha/` — Python 3.12, organized per spec §9.4.

## Architecture (one screen)

```
target repo (git ls-files)
        │
        ▼
   Walker ──► Chunker ──► Embedder ──► Storage (SQLite)
                                          │  ├─ chunks (metadata)
                                          │  ├─ vec_chunks (sqlite-vec)
                                          │  └─ fts_chunks (FTS5 BM25)
                                          ▼
                                     Retrieval ──► CLI
                                                   └─► MCP server (stropha-mcp)
```

## License

MIT.

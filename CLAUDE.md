# CLAUDE.md — stropha

Operational guide for LLM coding agents (Claude Code, OpenCode, etc.) working on this repository.

## Project purpose

**stropha** is a Retrieval-Augmented Generation system for arbitrary codebases, exposed to LLM clients via the **Model Context Protocol (MCP)**.

Goal: when an agent like Claude Code is editing a target codebase, it should be able to call `search_code("how does mastery work?")` and get back precise file/line/snippet results without burning context window on directory walks.

The original development target is the sibling **Mimoria** repository at `../mimoria`, but stropha is repo-agnostic — point `STROPHA_TARGET_REPO` anywhere and it works.

## Authoritative spec

**`docs/architecture/stropha-system.md`** is the single source of truth for design decisions. Read it before making non-trivial changes. If you change behavior that contradicts the spec, update the spec in the same commit.

The companion document **`docs/architecture/stropha-graphify-integration.md`** specifies the planned symbol-graph integration (consuming `graphify-out/graph.json`) and the post-commit hook automation.

## Current phase

**Phase 1 — MVP MCP** complete (per spec §16):

Phase 0 (spike) ✓ — kept for reference:
- [x] `uv` project scaffold, walker, dummy chunker, embedder abstraction,
      sqlite-vec storage, CLI `index` / `search` / `stats`.

Phase 1 ✓:
- [x] Tree-sitter AST chunking for Java, TypeScript, JavaScript, Python,
      Rust, Go, Kotlin (via `tree-sitter-language-pack.process()`).
- [x] Custom chunkers for Vue SFC, Markdown (heading split), Gherkin.
- [x] Class/interface skeleton chunks: emit a meso-level chunk per
      container with qualified name + member list so BM25 matches the
      type by identifier.
- [x] Hybrid search (spec §6.1): **three streams** fused via RRF —
      dense (sqlite-vec) + sparse (FTS5 BM25, with CamelCase splitting +
      path/symbol tokens augmenting the FTS document) + symbol-token
      lookup (query routing per spec §6.3.5).
- [x] MCP server (`stropha-mcp`) over stdio. Tools: `search_code`,
      `get_symbol`, `get_file_outline`, `list_repos`.
      Resource: `stropha://stats`.
- [x] `.mcp.example.json` template for Claude Code / Cursor integration.
- [x] Chunk-level freshness skip — re-running `index` on a stable repo
      is near-instant (no API calls, no DB writes).
- [x] Local embedder default: `mixedbread-ai/mxbai-embed-large-v1`
      (1024-dim, top open MTEB at this scale, ONNX-stable on macOS aarch64).
- [x] **Per-chunk repository identity** (schema v2): every chunk records
      the git repo it came from (`normalized_key`, clone URL, default
      branch, HEAD commit at index time). Returned in every `SearchHit`
      so MCP clients can `git clone <url>` to fetch the source. Schema
      auto-migrates from v1 with sanity-checked backfill.

Phase 4 (multi-repo) ✓:
- [x] `stropha index --repo A --repo B ...` walks each repo sequentially
      sharing the same Storage and Embedder.
- [x] `chunk_id` is namespaced by repo (`make_chunk_id(..., repo_key=...)`)
      so identical files in distinct repos do not collide on the global
      `chunks.chunk_id` UNIQUE constraint.
- [x] `IndexStats` aggregates per-repo counters (`stats.repos: list[RepoStats]`)
      while keeping single-repo back-compat accessors.
- [x] `--rebuild` clears chunks but preserves the `repos` table — identities
      survive rebuilds so FK references stay stable across runs.

Pipeline-adapters Phase 1 ✓ (per `docs/architecture/stropha-pipeline-adapters.md`):
- [x] `Stage` protocol + `StageContext`/`StageHealth` + `@register_adapter`
      decorator with auto-import discovery (`stropha.pipeline`,
      `stropha.adapters`).
- [x] `EmbedderStage` and `EnricherStage` protocols (`stropha.stages.*`).
      Legacy `stropha.embeddings.Embedder` Protocol kept as the minimal
      back-compat subset.
- [x] Migrated `LocalEmbedder` + `VoyageEmbedder` to
      `stropha.adapters.embedder.*` with `@register_adapter`. Old import
      paths (`stropha.embeddings.local`, `…voyage`) still resolve via
      shims.
- [x] New enricher adapters: `noop` (default — identity) and
      `hierarchical` (prepends parent skeleton).
- [x] Pydantic-validated config: YAML (`./stropha.yaml`,
      `~/.stropha/config.yaml`) + env-var overlay
      (`STROPHA_<STAGE>__<KEY>__<SUBKEY>`) + CLI flags
      (`--enricher`, `--embedder`). Legacy env vars
      (`STROPHA_LOCAL_EMBED_MODEL`, `STROPHA_VOYAGE_EMBED_MODEL`,
      `STROPHA_VOYAGE_EMBED_DIM`) keep working as aliases routed
      strictly by the resolved adapter.
- [x] Schema v3: `chunks.embedding_text` + `chunks.enricher_id` +
      `enrichments(content_hash, enricher_id)` cache table.
      Migration v1→v2→v3 is idempotent; legacy NULL `enricher_id`
      is treated as `'noop'` so upgrades do NOT trigger full re-embed.
- [x] Adapter-drift detection (ADR-004): switching the active enricher
      (e.g. `--enricher hierarchical`) automatically re-enriches +
      re-embeds the drifted chunks — no `--rebuild` needed. Per-chunk
      cache via `enrichments` makes repeat enrichment near-free.
- [x] New CLI: `stropha pipeline show` (resolved composition + health),
      `stropha pipeline validate` (probe), `stropha adapters list`.
- [x] 30 new unit tests (`test_pipeline_framework.py`,
      `test_enricher_adapters.py`, `test_pipeline_drift.py`) on top of
      the 53 pre-existing → 83 total, all green.

Pipeline-adapters Phase 2 ✓:
- [x] `WalkerStage` / `StorageStage` / `RetrievalStage` protocols.
- [x] `adapters/walker/git_ls_files.py`, `adapters/storage/sqlite_vec.py`
      (subclass of `Storage` so the full read/write surface is inherited),
      `adapters/retrieval/hybrid_rrf.py`.
- [x] `build_stages()` extended to all 5 stages with cross-stage
      injection (storage gets embedder.dim; retrieval gets storage +
      embedder).
- [x] `Pipeline` class uses walker + storage adapters; `cli.py` `index`
      / `search` / `stats` route through the builder; `pipeline show`
      drops the "(legacy)" rows for migrated stages.
- [x] Legacy env aliases extended: `STROPHA_INDEX_PATH` →
      `pipeline.storage.config.path`; `STROPHA_MAX_FILE_BYTES` →
      `pipeline.walker.config.max_file_bytes`.
- [x] 14 new unit tests (`test_phase2_adapters.py`).

Pipeline-adapters Phase 3 ✓:
- [x] `ChunkerStage` + `LanguageChunkerStage` protocols.
- [x] `adapters/chunker/tree_sitter_dispatch.py` — dispatcher adapter
      with sub-pipeline (language → sub-adapter) exposed via YAML
      `chunker.config.languages.*`.
- [x] Per-language sub-adapters under `adapters/chunker/languages/`:
      `ast-generic`, `heading-split`, `sfc-split`,
      `regex-feature-scenario`, `file-level`. All wrap the existing
      classes under `ingest/chunkers/` to preserve test coverage.
- [x] `Pipeline` builds + uses the chunker adapter; output is
      byte-identical (same `chunk_id`s) to the legacy `Chunker` for
      every supported language.
- [x] `pipeline show` / `pipeline validate` cover all 6 stages — no
      more "(legacy)" rows.
- [x] 11 new unit tests (`test_phase3_chunker.py`).

Pipeline-adapters Phase 4 ✓:
- [x] `RetrievalStreamStage` protocol; sub-adapters under
      `adapters/retrieval/streams/`: `vec-cosine`, `fts5-bm25`,
      `like-tokens`.
- [x] `hybrid-rrf` refactored to read `config.streams.{name: {adapter,
      config}}` and instantiate registered streams. Setting a stream to
      `null` disables it; omitted streams inherit the legacy default.
- [x] Performance: `hybrid-rrf.search` skips `embed_query` when no dense
      stream is enabled.
- [x] `adapter_id` of `hybrid-rrf` digests the stream composition so
      changing any sub-adapter forces a fresh `adapter_id` (cache /
      drift hooks pick it up).
- [x] 12 new unit tests (`test_phase4_retrieval_streams.py`). Total
      suite: **120 tests, all green**.

Pipeline-adapters Phase 5 (pending — sob demanda):
- [ ] LLM enricher adapters (`ollama`, `anthropic`, `mlx`, `openai`).
- [ ] Walker variants (`filesystem`, `nested-git`).
- [ ] Storage variants (`qdrant`, `pgvector`, `lancedb`).
- [ ] Retrieval `hybrid-rrf-rerank` (Voyage rerank-2.5 stage hook).

Exit criterion for Phase 0: `stropha search "where is the FSRS calculator"` returns the right file in the top 3 — ✓.

Phase 1 exit criterion is qualitative ("Claude Code uses stropha without being instructed"). The objective floor is: hybrid search returns the right code chunk in the top 3 for symbol+conceptual queries — ✓.

**Not yet implemented (deferred to Phase 2/3 — see specs):**
- Reranking (Voyage `rerank-2.5`).
- Contextual retrieval (Anthropic technique — LLM-generated chunk prefixes).
- Symbol graph integration via graphify (`docs/architecture/stropha-graphify-integration.md`).
- Post-commit hook (`stropha hook install`) — designed in the integration spec.
- Golden dataset + RAGAS evaluation harness.
- OpenTelemetry tracing → Langfuse.

## How to run

```bash
# One-time setup
uv sync

# Copy and edit env
cp .env.example .env
# (optional) set VOYAGE_API_KEY for best quality; otherwise local fallback is used.

# Index the target repo (defaults to current working dir; override via STROPHA_TARGET_REPO)
uv run stropha index

# Search
uv run stropha search "how does mastery work"

# Stats
uv run stropha stats
```

## Code conventions

- **Python 3.12+**, type hints mandatory on public functions, `pydantic` for structured data.
- **Async by default** in IO-bound paths (embedding API, file reads). CPU-bound stays sync.
- **No global state.** Pass `Config` / `Storage` explicitly. Single composition root in `cli.py`.
- **Errors are structured.** Raise `stropha.errors.StrophaError` subclasses; never bare `Exception`.
- **Logging via `structlog`.** Always include `request_id` / `chunk_id` when relevant; never log raw secrets.
- **File references in messages.** Use `path:line` format so editors can jump (e.g. `walker.py:42`).
- Line length 100. `ruff` + `mypy` clean in `dev` extra.

## Architecture map (current)

| Module | Responsibility |
|---|---|
| `cli.py` | Typer entry points (`index`, `search`, `stats`); wires Config → components. |
| `server.py` | MCP server entry (`stropha-mcp`); FastMCP with lifespan composition root. |
| `config.py` | Pydantic settings loaded from `.env`. |
| `ingest/walker.py` | Discover indexable files via `git ls-files` + `.strophaignore` + binary/size filters. |
| `ingest/git_meta.py` | Detect repo identity (normalized key, clone URL, default branch, HEAD) with auth-token stripping. |
| `ingest/chunker.py` | Dispatcher: picks per-language chunker, falls back to file-level on errors. |
| `ingest/chunkers/ast_generic.py` | Tree-sitter via `tree-sitter-language-pack.process()` (Java, TS, JS, Python, Rust, Go, Kotlin). |
| `ingest/chunkers/markdown.py` | Heading-based section split. |
| `ingest/chunkers/vue.py` | SFC block split (`<script>` / `<template>` / `<style>`). |
| `ingest/chunkers/gherkin.py` | Feature / Scenario split (regex; no tree-sitter grammar in the pack). |
| `ingest/chunkers/fallback.py` | File-level chunker used for unsupported languages. |
| `ingest/pipeline.py` | Walker → chunker → embedder → storage; freshness skip. |
| `embeddings/base.py` | `Embedder` protocol; dimension + batch size contract. |
| `embeddings/voyage.py` | Voyage AI client (active when `VOYAGE_API_KEY` set). |
| `embeddings/local.py` | fastembed (ONNX) local default. |
| `storage/sqlite.py` | sqlite-vec + FTS5 (BM25 with CamelCase / path / symbol expansion) + metadata. |
| `retrieval/rrf.py` | Reciprocal Rank Fusion (k=60). |
| `retrieval/search.py` | Three-stream hybrid: dense + BM25 + symbol-token lookup. |

Planned (not yet present): `ingest/enricher.py` (Contextual Retrieval), `ingest/graphify_loader.py` (symbol graph mirror), `retrieval/rerank.py` (Voyage rerank-2.5), `retrieval/graph.py` (graph traversal).

## Embedding strategy

Two providers behind one `Embedder` protocol:

1. **Voyage `voyage-code-3` @ 512 dims (Matryoshka)** — preferred. Activates when `VOYAGE_API_KEY` is set. SOTA for code per spec §4.1.
2. **fastembed `mixedbread-ai/mxbai-embed-large-v1` @ 1024 dims** — local ONNX default. Top open-source English MTEB at this size class. Zero cost, zero network. Configurable via `STROPHA_LOCAL_EMBED_MODEL`.

> **Avoid `jinaai/jina-embeddings-v2-base-code`** as a local model on macOS aarch64. It exhibits an ONNX-runtime instability that hangs or crashes the process on the second consecutive embed call. This is documented in stropha-graphify-integration.md ADR-008.

Both write the model name + dimension into the chunk row (`embedding_model`, `embedding_dim`). Switching providers does **not** invalidate the index; rows from a different model are simply ignored at query time until reindex. The CLI prints a warning if it finds mixed models.

## Testing

```bash
uv run pytest                          # unit tests (~25 fast tests)
uv run pytest -m e2e                   # end-to-end (requires indexed fixture repo)
```

Phase 2 will add a golden dataset under `tests/eval/golden/` and a RAGAS harness. Until then, smoke tests live in `tests/unit/`.

## Hygiene rules for agents

1. **Never invent symbols or APIs.** If unsure whether `voyageai` has `embed_async`, read the installed source under `.venv/`.
2. **Never bypass the embedder abstraction.** No direct `voyageai.Client()` calls outside `embeddings/voyage.py`.
3. **Never write to the index outside `storage/sqlite.py`.** All schema migrations go through `storage/sqlite.py:Storage.migrate`.
4. **Never index secrets.** The walker excludes `.env*`; if you add a new file type, confirm gitleaks-style scanning is in place before merge.
5. **Spec drift = update the spec.** If you change behavior, edit `docs/architecture/stropha-system.md` (or `stropha-graphify-integration.md`) in the same PR.

## Useful pointers

- Spec: `docs/architecture/stropha-system.md`
- Integration spec: `docs/architecture/stropha-graphify-integration.md`
- Voyage docs: https://docs.voyageai.com
- sqlite-vec docs: https://github.com/asg017/sqlite-vec
- MCP spec: https://modelcontextprotocol.io/specification

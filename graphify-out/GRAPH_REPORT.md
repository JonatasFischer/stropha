# Graph Report - .  (2026-05-15)

## Corpus Check
- 90 files · ~50,608 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1140 nodes · 2096 edges · 94 communities (52 shown, 42 thin omitted)
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 530 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Architecture Docs & Rationale|Architecture Docs & Rationale]]
- [[_COMMUNITY_Ingest Pipeline Core|Ingest Pipeline Core]]
- [[_COMMUNITY_Local Embedder Adapter|Local Embedder Adapter]]
- [[_COMMUNITY_AST Generic Chunker|AST Generic Chunker]]
- [[_COMMUNITY_CLI & Embedder Factory|CLI & Embedder Factory]]
- [[_COMMUNITY_Per-Language Chunkers|Per-Language Chunkers]]
- [[_COMMUNITY_Chunker Family Helpers|Chunker Family Helpers]]
- [[_COMMUNITY_Hierarchical Enricher|Hierarchical Enricher]]
- [[_COMMUNITY_SqliteVec Storage Adapter|SqliteVec Storage Adapter]]
- [[_COMMUNITY_Stage Protocols|Stage Protocols]]
- [[_COMMUNITY_File Walker & Discovery|File Walker & Discovery]]
- [[_COMMUNITY_Storage Layer Tests|Storage Layer Tests]]
- [[_COMMUNITY_Adapter Auto-Loading|Adapter Auto-Loading]]
- [[_COMMUNITY_Hybrid Retrieval & RRF|Hybrid Retrieval & RRF]]
- [[_COMMUNITY_SQLite Storage Internals|SQLite Storage Internals]]
- [[_COMMUNITY_Pipeline Orchestrator|Pipeline Orchestrator]]
- [[_COMMUNITY_Git Metadata Detection|Git Metadata Detection]]
- [[_COMMUNITY_TreeSitter Dispatch Chunker|TreeSitter Dispatch Chunker]]
- [[_COMMUNITY_Storage Stage Surface|Storage Stage Surface]]
- [[_COMMUNITY_Hybrid RRF Adapter|Hybrid RRF Adapter]]
- [[_COMMUNITY_Hierarchical Enricher Adapter|Hierarchical Enricher Adapter]]
- [[_COMMUNITY_Adapter Drift Tests|Adapter Drift Tests]]
- [[_COMMUNITY_Pipeline Config Loader|Pipeline Config Loader]]
- [[_COMMUNITY_MCP Server Tools|MCP Server Tools]]
- [[_COMMUNITY_Retrieval Streams Tests|Retrieval Streams Tests]]
- [[_COMMUNITY_Adapter Builder|Adapter Builder]]
- [[_COMMUNITY_Vec Cosine Stream|Vec Cosine Stream]]
- [[_COMMUNITY_Error Hierarchy|Error Hierarchy]]
- [[_COMMUNITY_Storage Stage Protocol|Storage Stage Protocol]]
- [[_COMMUNITY_Hybrid Retrieval Stage|Hybrid Retrieval Stage]]
- [[_COMMUNITY_CLI Composition Root|CLI Composition Root]]
- [[_COMMUNITY_SQLite Storage Module|SQLite Storage Module]]
- [[_COMMUNITY_Walker Stage Adapter|Walker Stage Adapter]]
- [[_COMMUNITY_CLI Pipeline Subcommands|CLI Pipeline Subcommands]]
- [[_COMMUNITY_Pipeline Config Merging|Pipeline Config Merging]]
- [[_COMMUNITY_Storage Search Methods|Storage Search Methods]]
- [[_COMMUNITY_Stage Protocol Base|Stage Protocol Base]]
- [[_COMMUNITY_Pipeline Stats Counters|Pipeline Stats Counters]]
- [[_COMMUNITY_File-Level Chunker Fallback|File-Level Chunker Fallback]]
- [[_COMMUNITY_Vector Serialization|Vector Serialization]]
- [[_COMMUNITY_Embedder Stage|Embedder Stage]]
- [[_COMMUNITY_Legacy Embedder Factory|Legacy Embedder Factory]]
- [[_COMMUNITY_Structured Logging|Structured Logging]]
- [[_COMMUNITY_Adapter Registry Lookup|Adapter Registry Lookup]]
- [[_COMMUNITY_Adapter Registry Tests|Adapter Registry Tests]]
- [[_COMMUNITY_Like Tokens Stream|Like Tokens Stream]]
- [[_COMMUNITY_Embedder Base Protocol|Embedder Base Protocol]]
- [[_COMMUNITY_Chunker Dispatcher Stage|Chunker Dispatcher Stage]]
- [[_COMMUNITY_Runtime Config|Runtime Config]]
- [[_COMMUNITY_Like Tokens Module|Like Tokens Module]]
- [[_COMMUNITY_FTS5 BM25 Module|FTS5 BM25 Module]]
- [[_COMMUNITY_AST Generic Sub-Adapter|AST Generic Sub-Adapter]]
- [[_COMMUNITY_Markdown Sub-Adapter|Markdown Sub-Adapter]]
- [[_COMMUNITY_Gherkin Sub-Adapter|Gherkin Sub-Adapter]]
- [[_COMMUNITY_Vue SFC Sub-Adapter|Vue SFC Sub-Adapter]]
- [[_COMMUNITY_Storage Migrations|Storage Migrations]]
- [[_COMMUNITY_FTS5 BM25 Stream|FTS5 BM25 Stream]]
- [[_COMMUNITY_Config Defaults|Config Defaults]]
- [[_COMMUNITY_Retrieval Stream Protocol|Retrieval Stream Protocol]]
- [[_COMMUNITY_Stats Surface|Stats Surface]]
- [[_COMMUNITY_Chunker Adapter Package|Chunker Adapter Package]]
- [[_COMMUNITY_Pipeline Stats Models|Pipeline Stats Models]]
- [[_COMMUNITY_File Outline Surface|File Outline Surface]]
- [[_COMMUNITY_Language Sub-Adapter Package|Language Sub-Adapter Package]]
- [[_COMMUNITY_Enricher Adapter Package|Enricher Adapter Package]]
- [[_COMMUNITY_Retrieval Streams Package|Retrieval Streams Package]]
- [[_COMMUNITY_Ingest Package Init|Ingest Package Init]]
- [[_COMMUNITY_Retrieval Stage Protocols|Retrieval Stage Protocols]]
- [[_COMMUNITY_Enrichment Cache Put|Enrichment Cache Put]]
- [[_COMMUNITY_Repo Stats Counters|Repo Stats Counters]]
- [[_COMMUNITY_MCP Tools Package|MCP Tools Package]]
- [[_COMMUNITY_Stage Name Constants|Stage Name Constants]]
- [[_COMMUNITY_Adapter Short ID|Adapter Short ID]]
- [[_COMMUNITY_Adapter Fully-Qualified ID|Adapter Fully-Qualified ID]]
- [[_COMMUNITY_Adapter Config Schema|Adapter Config Schema]]
- [[_COMMUNITY_Embedder Model Identifier|Embedder Model Identifier]]
- [[_COMMUNITY_Embedder Vector Dim|Embedder Vector Dim]]
- [[_COMMUNITY_Embedder Batch Size|Embedder Batch Size]]
- [[_COMMUNITY_Stream Top-K Budget|Stream Top-K Budget]]
- [[_COMMUNITY_Storage Vector Dim|Storage Vector Dim]]
- [[_COMMUNITY_Storage Model Identifier|Storage Model Identifier]]
- [[_COMMUNITY_SqliteVec Vector Dim|SqliteVec Vector Dim]]
- [[_COMMUNITY_SqliteVec Batch Size|SqliteVec Batch Size]]
- [[_COMMUNITY_Chunk ID Re-derivation|Chunk ID Re-derivation]]
- [[_COMMUNITY_Stropha Package Root|Stropha Package Root]]
- [[_COMMUNITY_SourceFile Model|SourceFile Model]]
- [[_COMMUNITY_StageHealth Type|StageHealth Type]]
- [[_COMMUNITY_Stages Package Init|Stages Package Init]]

## God Nodes (most connected - your core abstractions)
1. `StageHealth` - 67 edges
2. `Chunk` - 41 edges
3. `StorageStage` - 40 edges
4. `Storage` - 37 edges
5. `SourceFile` - 36 edges
6. `_RecordingStorage` - 27 edges
7. `HybridRrfRetrieval` - 27 edges
8. `SearchHit` - 24 edges
9. `HybridRrfConfig` - 23 edges
10. `TreeSitterDispatchChunker` - 23 edges

## Surprising Connections (you probably didn't know these)
- `test_chunk_id_is_deterministic_under_namespace()` --calls--> `make_chunk_id()`  [INFERRED]
  tests/unit/test_pipeline_multirepo.py → src/stropha/ingest/chunkers/base.py
- `test_stage_context_defaults()` --calls--> `StageContext`  [INFERRED]
  tests/unit/test_pipeline_framework.py → src/stropha/pipeline/base.py
- `test_stage_health_ready()` --calls--> `StageHealth`  [INFERRED]
  tests/unit/test_pipeline_framework.py → src/stropha/pipeline/base.py
- `md_file()` --calls--> `SourceFile`  [INFERRED]
  tests/unit/test_phase3_chunker.py → src/stropha/models.py
- `txt_file()` --calls--> `SourceFile`  [INFERRED]
  tests/unit/test_phase3_chunker.py → src/stropha/models.py

## Hyperedges (group relationships)
- **Phase 4 retrieval sub-pipeline: 3 streams fused via RRF** — stropha_retrieval_hybrid_rrf, stropha_stream_vec_cosine, stropha_stream_fts5_bm25, stropha_stream_like_tokens, stropha_rrf_fuse [EXTRACTED 1.00]
- **Index pipeline flow: walker → chunker → enricher → embedder → storage** — stropha_walker_class, stropha_chunker_class, stropha_enricher_noop, stropha_local_embedder_class, stropha_storage_class, stropha_index_pipeline_class [INFERRED 0.85]
- **Adapter framework: protocol + registry + loader + drift detection** — stropha_pipeline_registry, stropha_pipeline_loader, concept_single_active_adapter, concept_adapter_drift_adr004, concept_subpipeline_pattern [EXTRACTED 0.95]
- **Hybrid retrieval (dense + BM25 + symbol tokens fused via RRF)** — retrieval_search_engine, storage_search_dense, storage_search_bm25, storage_search_symbol_tokens, retrieval_rrf_fuse [EXTRACTED 1.00]
- **Indexing pipeline: walker → chunker → enricher → embedder → storage** — pipeline_orchestrator, storage_chunk_is_fresh, storage_get_enrichment, storage_upsert_chunk, embeddings_embedder_protocol [EXTRACTED 0.95]
- **Adapter framework (registry + builder + Stage protocol)** — pipeline_stage_protocol, pipeline_register_adapter, pipeline_lookup_adapter, pipeline_builder_build_stages, pipeline_load_pipeline_config [EXTRACTED 0.95]
- **Indexing pipeline: Walker → Chunker → Embedder → Storage** — ingest_Walker, ingest_Chunker, stages_EmbedderStage, stages_StorageStage, pipeline_IndexPipeline [EXTRACTED 0.95]
- **Language-specific chunkers dispatched by Chunker registry** — ingest_Chunker, chunkers_AstGenericChunker, chunkers_VueChunker, chunkers_MarkdownChunker, chunkers_GherkinChunker, chunkers_FallbackChunker [EXTRACTED 0.95]
- **Stage protocol family (pluggable adapter contracts)** — stages_ChunkerStage, stages_EmbedderStage, stages_EnricherStage, stages_RetrievalStage, stages_StorageStage, stages_WalkerStage [EXTRACTED 0.90]
- **Per-language chunker sub-adapters dispatched by tree-sitter-dispatch** — ast_generic_language_chunker, file_level_language_chunker, heading_split_language_chunker, regex_feature_scenario_language_chunker, sfc_split_language_chunker, tree_sitter_dispatch_chunker [EXTRACTED 0.95]
- **Hybrid RRF retrieval streams (dense + sparse + symbol)** — hybrid_rrf_retrieval, vec_cosine_stream, fts5_bm25_stream, like_tokens_stream [EXTRACTED 0.95]
- **Full stropha pipeline stage adapters (walk → chunk → enrich → embed → store → retrieve)** — git_ls_files_walker, tree_sitter_dispatch_chunker, hierarchical_enricher, noop_enricher, local_embedder, voyage_embedder, sqlite_vec_storage, hybrid_rrf_retrieval [INFERRED 0.90]

## Communities (94 total, 42 thin omitted)

### Community 0 - "Architecture Docs & Rationale"
Cohesion: 0.05
Nodes (60): Adapter-drift detection (ADR-004), Per-repo chunk_id namespacing, Contextual Retrieval (Anthropic), Chunk-level freshness skip, graphify (external symbol-graph tool), Three-stream hybrid retrieval fused via RRF, Avoid jinaai/jina-embeddings-v2-base-code on macOS aarch64 (ADR-008), Local fastembed mxbai-embed-large-v1 (+52 more)

### Community 1 - "Ingest Pipeline Core"
Cohesion: 0.05
Nodes (40): FallbackChunker, One chunk per file; line-based split when file exceeds MAX_CHARS_PER_CHUNK., Embedder, Chunker, Public entry point. Replaces Phase 0's FileChunker., IndexPipeline, IndexStats, Indexing pipeline orchestration.  Composition: Walker → Chunker → Embedder → Sto (+32 more)

### Community 2 - "Local Embedder Adapter"
Cohesion: 0.05
Nodes (22): LocalEmbedder, LocalEmbedderConfig, Local fastembed embedder (ONNX, no torch, no network).  Default model per ADR-00, Config schema for the local fastembed adapter., ONNX-based embedder, runs on CPU., Voyage AI embedder adapter.  Active when ``VOYAGE_API_KEY`` is set. ``voyage-cod, Config schema for the Voyage adapter., Wraps the official ``voyageai`` SDK. (+14 more)

### Community 3 - "AST Generic Chunker"
Cohesion: 0.08
Nodes (29): AstGenericChunker, _kind_str(), _qualified_name(), Generic AST chunker built on tree-sitter-language-pack's `process()`.  Strategy, Emit a single file-level chunk (or split if oversized)., Split an oversized span on line boundaries., Normalize StructureKind enum to a comparable string., Uses tslp.process() for languages with structure extraction support. (+21 more)

### Community 4 - "CLI & Embedder Factory"
Cohesion: 0.06
Nodes (45): CLI: adapters list, Typer CLI App, CLI: index command, CLI: search command, Config (pydantic settings), build_embedder factory, Embedder Protocol (legacy), embeddings.local shim → adapters.embedder.local (+37 more)

### Community 5 - "Per-Language Chunkers"
Cohesion: 0.1
Nodes (26): BaseModel, GherkinChunker, MarkdownChunker, Section-per-chunk. Parent chain encoded via parent_chunk_id., VueChunker, _build_registry(), AstGenericConfig, AstGenericLanguageChunker (+18 more)

### Community 6 - "Chunker Family Helpers"
Cohesion: 0.09
Nodes (36): AstGenericChunker (tree-sitter), FallbackChunker (file-level), GherkinChunker (.feature), LanguageChunker Protocol (concrete), MarkdownChunker (heading-split), VueChunker (SFC split), count_lines, make_chunk_id (+28 more)

### Community 7 - "Hierarchical Enricher"
Cohesion: 0.1
Nodes (25): HierarchicalEnricher, HierarchicalEnricherConfig, NoopEnricher, NoopEnricherConfig, Empty config. The noop enricher has no tunables., Returns ``chunk.content`` unchanged., EnricherStage, Cross-cutting context every stage may consult.      Stages NEVER mutate the cont (+17 more)

### Community 8 - "SqliteVec Storage Adapter"
Cohesion: 0.08
Nodes (15): Storage, Storage adapter for sqlite-vec.  Subclasses the existing ``stropha.storage.sqlit, Storage stage backed by SQLite + sqlite-vec + FTS5., SqliteVecStorage, SqliteVecStorageConfig, Tests for Phase 2 adapters (walker, storage, retrieval) + builder wiring., STROPHA_INDEX_PATH must route into pipeline.storage.config.path., Init a tiny git repo with a Python and a Markdown file. (+7 more)

### Community 9 - "Stage Protocols"
Cohesion: 0.08
Nodes (18): LanguageChunker, Splits a single file into one or more Chunks., BuiltStages, Set of constructed adapters returned by :func:`build_stages`.      All 6 stages, Protocol, ChunkerStage, LanguageChunkerStage, Split an iterable of ``SourceFile`` into ``Chunk`` records. (+10 more)

### Community 10 - "File Walker & Discovery"
Cohesion: 0.11
Nodes (20): detect_language(), _filesystem_walk(), _git_ls_files(), _is_binary(), _load_ragignore(), Source file discovery.  Per spec §3.1: - Use `git ls-files` when target is a git, Fallback walker for non-git directories., Load .strophaignore patterns from the target repo and from this project.      Bo (+12 more)

### Community 11 - "Storage Layer Tests"
Cohesion: 0.2
Nodes (23): _identity(), _make_chunk(), Tests for SQLite + sqlite-vec + FTS5 layer., Auto-backfill assigns orphan chunks when target_repo paths match., Auto-backfill is a no-op when the target_repo changed since indexing., Return a unit vector aligned to one axis., storage(), test_backfill_refuses_when_paths_dont_match() (+15 more)

### Community 12 - "Adapter Auto-Loading"
Cohesion: 0.2
Nodes (23): _autoload(), Concrete adapters for each pipeline stage.  Per ADR-003 (``docs/architecture/str, ADR-003: stropha pipeline adapters, ADR-008: graphify integration (default model choice), AstGenericLanguageChunker (tree-sitter sub-adapter), FileLevelLanguageChunker (single-chunk fallback), Fts5Bm25Stream (sparse BM25), GitLsFilesWalker (git ls-files + fs fallback) (+15 more)

### Community 13 - "Hybrid Retrieval & RRF"
Cohesion: 0.12
Nodes (15): Retrieval pipeline. Phase 1: hybrid dense+BM25 + RRF. Phase 2 will add rerank., Reciprocal Rank Fusion (Cormack et al., 2009 — spec §6.1).  Formula: score(d) =, Merge any number of ranked lists into a single ranked list.      Identity is by, rrf_fuse(), Hybrid retrieval (spec §6.1): dense (vec) + sparse (FTS5) fused via RRF.  Phase, Hybrid dense+BM25 + RRF. Falls back gracefully if one side is empty., Three-stream hybrid retrieval, fused with RRF.          Streams:         - dense, Escape hatch for debugging or A/B comparison. (+7 more)

### Community 14 - "SQLite Storage Internals"
Cohesion: 0.09
Nodes (9): Owns the SQLite connection. All schema mutations go through here., Insert or update a repo row keyed by ``normalized_key``. Returns its id., Best-effort assignment of orphan chunks to ``repo_id``.          Used on upgrade, Per-repo aggregate counters., True if the stored chunk matches every (hash, model, enricher).          Per ADR, Return cached embedding_text for ``(content_hash, enricher_id)`` or None., Remove every chunk whose rel_path is in the list. Returns count., Symbolic outline of a file (chunks sorted by start_line). (+1 more)

### Community 15 - "Pipeline Orchestrator"
Cohesion: 0.13
Nodes (15): Pipeline, PipelineStats, Per-repo counters returned by :meth:`Pipeline.run`., Aggregate result of a :meth:`Pipeline.run`., Adapter-aware walker → chunker → enricher → embedder → storage., RepoStats, Discover ``SourceFile`` records under a repo root., Yield every file the indexer should consider for chunking. (+7 more)

### Community 16 - "Git Metadata Detection"
Cohesion: 0.15
Nodes (21): detect(), _detect_default_branch(), normalize_url(), Git repository identity extraction.  For every file we index we want to remember, Normalize a git remote URL.      Returns a tuple ``(normalized_key, sanitized_ht, Run a git subcommand. Returns stdout trimmed, or ``None`` on failure.      Never, Try three sources, in order of fidelity., Stable identification of a git repository (or a fallback). (+13 more)

### Community 17 - "TreeSitter Dispatch Chunker"
Cohesion: 0.13
Nodes (18): Sub-pipeline config: each language → ``{adapter, config}``., Stage adapter — dispatches per file to a language sub-adapter., TreeSitterDispatchChunker, TreeSitterDispatchConfig, md_file(), py_file(), Tests for Phase 3 chunker dispatcher and language sub-adapters., Phase 3 must NOT change chunk_id of unchanged files (no spurious re-embed). (+10 more)

### Community 19 - "Hybrid RRF Adapter"
Cohesion: 0.16
Nodes (14): HybridRrfConfig, HybridRrfRetrieval, RRF over N retrieval streams, each itself a registered adapter., RetrievalStage, ConfigError, Invalid or missing configuration., _DummyStorage, Storage stub: returns empty hit lists for all read methods. (+6 more)

### Community 20 - "Hierarchical Enricher Adapter"
Cohesion: 0.11
Nodes (4): Hierarchical enricher — prepend a small skeleton of the parent chunk.  Rationale, Noop enricher — identity. Default; preserves Phase 0/1 behavior exactly., EnricherStage protocol — transform a chunk before embedding.  Phase 1 introduces, Shared pydantic models used across modules.

### Community 21 - "Adapter Drift Tests"
Cohesion: 0.1
Nodes (6): Tests for adapter-drift detection (ADR-004) and enrichment cache.  Drift = store, Upgrading from v0.1.0 (NULL enricher_id) MUST NOT trigger re-embed for noop., Init a tiny git repo with one Python file., repo_with_one_file(), test_chunk_is_fresh_treats_legacy_null_as_noop(), _vec()

### Community 22 - "Pipeline Config Loader"
Cohesion: 0.15
Nodes (18): load_pipeline_config(), Resolve the full pipeline config dict.      Args:         project_root: dir to l, Class decorator. Registers ``cls`` under ``(stage, name)``.      The adapter cla, register_adapter(), Tests for the pipeline framework (base + registry + config + builder)., Regression: STROPHA_VOYAGE_EMBED_MODEL must NOT contaminate a local adapter., test_config_cli_override_wins_over_env(), test_config_defaults_pick_local_when_no_voyage_key() (+10 more)

### Community 23 - "MCP Server Tools"
Cohesion: 0.16
Nodes (18): _ctx(), get_file_outline(), get_symbol(), list_repos(), main(), OutlineEntry, MCP server entry point.  Wires Config → Embedder → Storage → SearchEngine inside, One entry in the `list_repos` response. (+10 more)

### Community 24 - "Retrieval Streams Tests"
Cohesion: 0.13
Nodes (6): Tests for Phase 4 retrieval sub-pipeline (streams as adapters)., Performance: no dense stream → don't pay embed_query cost., _StubEmbedder, test_hybrid_rrf_adapter_id_changes_with_stream_swap(), test_hybrid_rrf_skips_embedder_when_no_dense_stream(), test_hybrid_rrf_unknown_sub_adapter_raises()

### Community 25 - "Adapter Builder"
Cohesion: 0.13
Nodes (4): Builder: resolved config dict → instantiated adapter objects.  Importing this mo, ChunkerStage protocol — split files into indexable chunks.  The default adapter, Stage-specific protocols.  Each module defines the contract a specific stage's a, RetrievalStage protocol — query string → ranked SearchHits.  Phase 2 ships ``hyb

### Community 26 - "Vec Cosine Stream"
Cohesion: 0.2
Nodes (8): VecCosineConfig, VecCosineStream, _hit(), Records calls + returns deterministic synthetic hits., _RecordingStorage, test_hybrid_rrf_calls_all_default_streams(), test_vec_cosine_stream_returns_empty_when_no_query_vec(), test_vec_cosine_stream_routes_to_storage_search_dense()

### Community 27 - "Error Hierarchy"
Cohesion: 0.18
Nodes (14): Exception, AdapterError, ChunkerError, PipelineError, Structured exceptions. Per CLAUDE.md, never raise bare Exception., Failure while discovering source files., Failure while splitting a file into chunks., Failure during query execution. (+6 more)

### Community 30 - "CLI Composition Root"
Cohesion: 0.18
Nodes (11): index(), _load_config(), pipeline_show(), Typer CLI — composition root.  Subcommands: - ``index``    : walk → chunk → enri, Hybrid retrieval (dense + BM25 + symbol-token fused via RRF)., Print the fully-resolved pipeline composition (YAML + env + CLI)., Initialize logging before any subcommand runs., Walk one or more repos, chunk every file, enrich, embed, and store. (+3 more)

### Community 31 - "SQLite Storage Module"
Cohesion: 0.19
Nodes (10): SQLite + sqlite-vec storage layer. Single source of truth for the index., _fts_text(), _identifier_tokens(), sqlite-vec + FTS5 + metadata, all in one SQLite database.  Per spec §5: keep eve, Extract identifier-like tokens from a free-text query.      Drops stopwords and, Expand CamelCase + dotted identifiers into separate tokens.      Used on BOTH in, Turn free text into a safe FTS5 MATCH expression (OR-joined tokens)., Pre-process chunk content before indexing in FTS5.      We assemble a single FTS (+2 more)

### Community 33 - "CLI Pipeline Subcommands"
Cohesion: 0.17
Nodes (12): CLI: pipeline show, CLI: pipeline validate, build_stages(), Instantiate every adapter named in ``resolved_config``.      ``open_storage=Fals, pipeline_validate(), Print index metadata., Run a lightweight health probe on every adapter. Exit non-zero on error., stats() (+4 more)

### Community 34 - "Pipeline Config Merging"
Cohesion: 0.22
Nodes (10): _coerce(), _env_to_dict(), _legacy_env_to_dict(), _merge(), Pipeline configuration loader.  Cascading precedence (per ``docs/architecture/st, Translate ``STROPHA_<STAGE>__<KEY>__...`` into nested dict overlay.      ``STROP, Map deprecated env vars onto the new shape. Logs a warning.      Caller MUST res, Cheap coercion for env-var strings to int/float/bool/str. (+2 more)

### Community 35 - "Storage Search Methods"
Cohesion: 0.22
Nodes (8): Top-k via FTS5 BM25 ranking. Returns [] for empty/sanitized queries., Match chunks whose `symbol` column contains identifier tokens from query., Exact + suffix match on symbol column. Used by `get_symbol`.          Matches `F, Build a ``RepoRef`` from a row that LEFT JOINed ``repos``.      Returns ``None``, _repo_from_row(), _snippet(), Stable identifier of the git repository a chunk belongs to.      Returned alongs, RepoRef

### Community 36 - "Stage Protocol Base"
Cohesion: 0.2
Nodes (4): Stage protocol + shared types.  Per ``docs/architecture/stropha-pipeline-adapter, Lightweight readiness probe, ≤2 s, non-blocking.          Called by ``stropha pi, One responsibility in the pipeline. Exactly one adapter active per run.      Ada, Stage

### Community 38 - "File-Level Chunker Fallback"
Cohesion: 0.2
Nodes (4): File-level fallback sub-adapter — one chunk per file.  Used as the dispatcher's, Adapter registry — populated at import-time via the ``@register_adapter`` decora, Test helper: clear the registry. Do NOT call from production code., _reset_for_tests()

### Community 39 - "Vector Serialization"
Cohesion: 0.24
Nodes (6): Float32 little-endian, the format sqlite-vec expects for BLOB inputs., Insert (or update) a chunk and its vector atomically. Returns rowid.          ``, Top-k dense nearest neighbors., _serialize_vector(), Failure in the SQLite/vec layer (schema, IO, corruption)., StorageError

### Community 41 - "Legacy Embedder Factory"
Cohesion: 0.25
Nodes (3): Legacy embedder protocol.  Kept for back-compat with code (and test stubs) writt, Builds the right Embedder based on Config., Embedding providers. Single abstraction (`Embedder`), swappable backends.

### Community 42 - "Structured Logging"
Cohesion: 0.22
Nodes (8): configure_logging(), get_logger(), Structured logging via structlog. Single configuration entry point., Configure structlog + stdlib logging.      Uses a console renderer when stderr i, Return a structlog logger. Use module __name__ as `name`., AppContext, _lifespan(), Long-lived dependencies, injected into every tool via the MCP Context.

### Community 43 - "Adapter Registry Lookup"
Cohesion: 0.25
Nodes (8): _build_simple(), Build a stage that needs no cross-stage dependency injection., available_for_stage(), lookup_adapter(), Return the adapter class registered for ``(stage, name)``.      Raises ``ConfigE, Return sorted adapter names registered for ``stage``., test_available_for_stage_returns_sorted(), test_lookup_unknown_adapter_raises_config_error_with_alternatives()

### Community 44 - "Adapter Registry Tests"
Cohesion: 0.25
Nodes (8): all_adapters(), Snapshot of the registry, grouped by stage. Used by ``adapters list``., adapters_list(), List every adapter the registry knows about., test_phase2_stages_all_registered(), test_chunker_and_language_chunker_stages_registered(), test_streams_registered(), test_default_adapters_registered()

### Community 45 - "Like Tokens Stream"
Cohesion: 0.32
Nodes (5): LikeTokensConfig, LikeTokensStream, A single result from retrieval., SearchHit, test_like_tokens_stream_routes_to_storage_search_symbol()

### Community 46 - "Embedder Base Protocol"
Cohesion: 0.25
Nodes (6): Embedder, Synchronous embedder. Async wrappers (when justified) live in subclasses.      P, Embed a batch of documents (chunks during indexing)., Embed a single user query. Providers may use a different input_type., Source-repo identity attached to every search result.      A client can run `git, RepoInfo

### Community 48 - "Runtime Config"
Cohesion: 0.29
Nodes (7): BaseSettings, Config, Runtime configuration. All product env vars are prefixed `STROPHA_`., Expand ~ and env vars in the configured index path., index_stats_resource(), Index statistics. Exposed as both a Tool and a Resource for discovery., StatsPayload

### Community 56 - "FTS5 BM25 Stream"
Cohesion: 0.4
Nodes (3): Fts5Bm25Config, Fts5Bm25Stream, test_fts5_bm25_stream_routes_to_storage_search_bm25()

### Community 57 - "Config Defaults"
Cohesion: 0.33
Nodes (3): _default_target_repo(), Configuration loaded from environment / .env via pydantic-settings., Default target = current working directory.

### Community 58 - "Retrieval Stream Protocol"
Cohesion: 0.4
Nodes (3): Returns a ranked list of ``SearchHit`` for a query., Return ranked hits. ``query_vec`` may be None for non-dense streams., RetrievalStreamStage

### Community 59 - "Stats Surface"
Cohesion: 1.0
Nodes (3): CLI: stats command, MCP resource: index_stats, Storage.stats

## Knowledge Gaps
- **302 isolated node(s):** `Tests for the chunker dispatcher and per-language chunkers.`, `Smoke test: launch the MCP server as a subprocess and exchange JSON-RPC.  This t`, `Spawn the server, initialize, list tools, call search_code.`, `Tests for the git identity helper.`, `Tests for Phase 2 adapters (walker, storage, retrieval) + builder wiring.` (+297 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **42 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Chunk` connect `Per-Language Chunkers` to `Architecture Docs & Rationale`, `Ingest Pipeline Core`, `AST Generic Chunker`, `Hierarchical Enricher`, `Stage Protocols`, `SQLite Storage Internals`, `Pipeline Orchestrator`, `TreeSitter Dispatch Chunker`, `Storage Stage Surface`, `Hybrid RRF Adapter`, `Hierarchical Enricher Adapter`?**
  _High betweenness centrality (0.139) - this node is a cross-community bridge._
- **Why does `StageHealth` connect `Per-Language Chunkers` to `Local Embedder Adapter`, `FTS5 BM25 Stream`, `Stage Protocol Base`, `Retrieval Stream Protocol`, `Hierarchical Enricher`, `SqliteVec Storage Adapter`, `Stage Protocols`, `Like Tokens Stream`, `Pipeline Orchestrator`, `TreeSitter Dispatch Chunker`, `Storage Stage Surface`, `Hybrid RRF Adapter`, `Pipeline Config Loader`, `Retrieval Streams Tests`, `Vec Cosine Stream`?**
  _High betweenness centrality (0.128) - this node is a cross-community bridge._
- **Why does `Storage` connect `SQLite Storage Internals` to `Storage Search Methods`, `Enrichment Cache Put`, `Per-Language Chunkers`, `Repo Stats Counters`, `Vector Serialization`, `SqliteVec Storage Adapter`, `Like Tokens Stream`, `Git Metadata Detection`, `Storage Migrations`, `SQLite Storage Module`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Are the 49 inferred relationships involving `StageHealth` (e.g. with `_StubEmbedder` and `_DummyStorage`) actually correct?**
  _`StageHealth` has 49 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `Chunk` (e.g. with `_StubEmbedder` and `_DummyStorage`) actually correct?**
  _`Chunk` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `StorageStage` (e.g. with `BuiltStages` and `RepoStats`) actually correct?**
  _`StorageStage` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Storage` (e.g. with `StorageError` and `RepoIdentity`) actually correct?**
  _`Storage` has 8 INFERRED edges - model-reasoned connections that need verification._
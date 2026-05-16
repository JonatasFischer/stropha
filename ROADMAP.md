# Stropha RAG Roadmap — Best-in-Class RAG Backend for MCP

> **Strategy:** Be the **best RAG backend** that MCP clients (OpenCode, Continue, Cursor, Zed, etc.) can use. Double down on graph intelligence as the unique differentiator.
>
> **Non-goals:** Stropha is NOT an IDE. We don't compete with Cursor's editor, Tab completions, or cloud agents. We provide context retrieval that makes ANY MCP client smarter.
>
> **Date:** 2026-05-16
> **Status:** In progress

---

## Strategic Focus

### Option A: Best RAG Backend for MCP Clients
- Focus on retrieval quality (already strong)
- Ship integrations for OpenCode, Continue, Zed, etc.
- Make stropha the "best RAG backend for MCP clients"

### Option C: Graph Intelligence Moat
- Cursor has basic codebase indexing
- Stropha has graphify with community detection, god nodes, cross-file relationships
- Double down on "understand architecture, not just search code"

---

## Current Status

| Capability | Status | Notes |
|------------|--------|-------|
| Hybrid retrieval (4-stream RRF) | ✅ | dense + BM25 + symbol + graph-vec |
| HyDE | ✅ | Hypothetical document via Ollama |
| Query rewriting | ✅ | LLM-powered query optimization |
| Reranking | ✅ | Local cross-encoder (fastembed) |
| MCP filters | ✅ | language, path_prefix, kind, exclude_tests |
| Graph traversal | ✅ | find_callers, find_related, trace_feature |
| Community detection | ✅ | Unique in market |
| Local-first | ✅ | Zero cloud dependency |

---

## Remaining Gaps

| Gap | Priority | Effort | Impact | Status |
|-----|----------|--------|--------|--------|
| Faceted search | P1 | Small | Medium | **Done** |
| Multi-query expansion | P2 | Small | Medium | **Done** |
| Semantic query cache | P2 | Small | Medium | **Done** |
| Contextual enricher | P1 | Medium | High | **Done** |
| Auto-merging retrieval | P2 | Medium | Medium | **Done** |
| Recursive --flag | P2 | Small | Medium | **Done** |
| SPLADE stream | P3 | Medium | Low | Deferred |
| Glossary | P3 | Small | Low | Deferred |

---

## Sprint 1: Reranking + Filtros (Alta Prioridade)

**Duração:** 1-2 semanas
**Impacto:** +15-25% precision@10

### 1.1 Reranker Local (Cross-Encoder ONNX)

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/adapters/retrieval/reranker/`
**Prioridade:** P0

Implementar reranker local usando cross-encoder ONNX:

```
adapters/retrieval/
├── reranker/
│   ├── __init__.py
│   ├── base.py              # RerankerStage protocol
│   ├── mxbai_rerank.py      # mixedbread mxbai-rerank-large-v1 (ONNX)
│   ├── bge_rerank.py        # BAAI/bge-reranker-v2-m3 (ONNX)
│   └── noop.py              # Pass-through (default)
```

**Modelo recomendado:** `mixedbread-ai/mxbai-rerank-large-v1`
- ~500MB ONNX
- Latência: ~200ms para 50 docs em CPU
- Top open-source em MTEB reranking

**Integração com hybrid-rrf:**
```yaml
retrieval:
  adapter: hybrid-rrf
  config:
    streams: { ... }
    reranker:
      adapter: mxbai-rerank
      config:
        model: mixedbread-ai/mxbai-rerank-large-v1
        top_k: 10        # final output
        candidates: 50   # input from RRF
```

**Critério de saída:**
- [ ] `stropha search "como funciona X"` usa reranker quando configurado
- [ ] Benchmark mostra ganho de precision vs sem reranker
- [ ] Fallback graceful quando modelo não disponível

### 1.2 Filtros Expostos no MCP

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/server.py` (tool `search_code`)
**Prioridade:** P0

O backend já suporta filtros, mas não estão expostos no MCP. Adicionar parâmetros:

```python
@mcp.tool(
    name="search_code",
    description="...",
)
async def search_code(
    query: str,
    top_k: int = 10,
    # Novos filtros:
    language: list[str] | None = None,      # ["java", "python"]
    path_glob: str | None = None,           # "backend/src/**"
    kind: list[str] | None = None,          # ["method", "class"]
    exclude_tests: bool = False,
    repo: str | None = None,                # "github.com/org/repo"
    modified_since: str | None = None,      # "2026-01-01"
) -> list[SearchResult]:
```

**Critério de saída:**
- [ ] `search_code` aceita todos os filtros acima
- [ ] Filtros aplicados como pre-filter (antes do ANN) quando possível
- [ ] Schema MCP documenta os filtros

### 1.3 Faceted Search Response

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/server.py`, `src/stropha/retrieval/`
**Prioridade:** P1

Retornar contagens por faceta junto com resultados:

```json
{
  "hits": [...],
  "facets": {
    "language": {"java": 45, "kotlin": 12, "typescript": 8},
    "kind": {"method": 40, "class": 15, "function": 10},
    "repo": {"backend": 50, "frontend": 15}
  },
  "total": 65
}
```

**Critério de saída:**
- [ ] `search_code` retorna campo `facets` opcional
- [ ] Facetas computadas em uma query SQL (não N+1)

---

## Sprint 2: Query Intelligence (Alta Prioridade)

**Duração:** 1-2 semanas
**Impacto:** +10-20% recall em queries vagas

### 2.1 HyDE (Hypothetical Document Embeddings)

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/adapters/retrieval/streams/hyde.py`
**Prioridade:** P0

Em vez de embeddar a query diretamente, gerar um "documento hipotético" que responderia a query:

```
Query: "como salvar enrollment"
→ LLM gera: "public void saveEnrollment(Enrollment e) { repository.persist(e); }"
→ Embedda o código hipotético
→ Busca por similaridade com esse embedding
```

**Implementação como stream:**
```yaml
retrieval:
  adapter: hybrid-rrf
  config:
    streams:
      dense:
        adapter: vec-cosine
      hyde:
        adapter: hyde-vec
        config:
          llm: ollama          # ou mlx
          model: qwen2.5-coder:1.5b
          fallback_to_dense: true
```

**Critério de saída:**
- [ ] Stream `hyde-vec` gera código hipotético via Ollama/MLX
- [ ] Skip automático se query já contém código/símbolos
- [ ] Latência < 500ms (LLM local)

### 2.2 Query Rewriting

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/retrieval/query_rewrite.py`
**Prioridade:** P1

LLM reescreve query para forma mais "pesquisável":

```
"onde tem teste pra fsrs" → "FSRS calculator test unit test FsrsCalculatorTest"
"como funciona mastery" → "mastery streak transition REVIEW phase acquisition"
```

**Integração:**
```python
class QueryRewriter:
    def rewrite(self, query: str) -> str:
        # LLM call com prompt específico
        # Retorna query expandida com termos técnicos
```

**Critério de saída:**
- [ ] Rewriter opcional via config `STROPHA_QUERY_REWRITE=true`
- [ ] Cache por query (evita LLM call repetido)
- [ ] Latência < 100ms com prompt cacheado

### 2.3 Multi-Query Expansion

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/retrieval/multi_query.py`
**Prioridade:** P2

Gerar 3-5 paráfrases da query, buscar cada uma, fundir via RRF:

```
Query: "autenticação de usuário"
→ Paráfrases:
   1. "authentication login flow"
   2. "user session token validation"
   3. "security middleware auth check"
→ Buscar cada uma
→ RRF fusion
```

**Critério de saída:**
- [ ] Geração de paráfrases via LLM
- [ ] RRF fusion das N buscas
- [ ] Config para número de paráfrases (default: 3)

---

## Sprint 3: Retrieval Avançado (Média Prioridade)

**Duração:** 1-2 semanas
**Impacto:** Melhor UX, economia de context window

### 3.1 Auto-Merging Retrieval

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/retrieval/auto_merge.py`
**Prioridade:** P1

Se múltiplos chunks do mesmo parent aparecem nos resultados, promover o parent:

```
Resultados:
  - StudyService.submitAnswer (rank 1)
  - StudyService.validateAnswer (rank 3)  
  - StudyService.recordStreak (rank 5)

Auto-merge:
  - StudyService (classe inteira, rank 1)
  - OtherClass.method (rank 2)
```

**Config:**
```yaml
retrieval:
  config:
    auto_merge:
      enabled: true
      threshold: 3    # mínimo de siblings para promover parent
```

**Critério de saída:**
- [ ] Promoção automática de parent quando threshold atingido
- [ ] Mantém children no resultado se parent muito grande
- [ ] Metadata indica que houve merge

### 3.2 Recursive Retrieval (completar)

**Status:** [ ] Parcialmente implementado (`STROPHA_RECURSIVE_RETRIEVAL`)
**Arquivo:** `src/stropha/retrieval/recursive.py`
**Prioridade:** P2

Busca em dois níveis:
1. Buscar nível macro (classes/arquivos)
2. Para top N, expandir para nível micro (métodos)
3. Rerank no nível micro

**Critério de saída:**
- [ ] Flag `--recursive` no CLI
- [ ] Parâmetro `recursive: true` no MCP
- [ ] Expansão limitada (max 3 parents expandidos)

### 3.3 SPLADE Stream (Sparse Neural)

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/adapters/retrieval/streams/splade.py`
**Prioridade:** P3

Alternativa neural ao BM25. Expande termos automaticamente:

```
"authentication" → também matcha "login", "session", "token", "credentials"
```

**Modelo:** `naver/splade-cocondenser-ensembledistil` (~500MB)

**Critério de saída:**
- [ ] Stream `splade` como alternativa ao `fts5-bm25`
- [ ] Índice sparse persistido no SQLite
- [ ] Benchmark vs BM25 puro

---

## Sprint 4: Contextual Enrichment (Média Prioridade)

**Duração:** 1-2 semanas
**Impacto:** +35% recall (benchmark Anthropic)

### 4.1 Contextual Prefix Enricher

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/adapters/enricher/contextual.py`
**Prioridade:** P1

Antes de embeddar, prepend contexto gerado por LLM:

```
Chunk original:
  "public void submitAnswer(ExerciseId id, Answer answer) { ... }"

Após enrichment:
  "Este método pertence à classe StudyService (camada de serviço de aplicação) 
   e implementa a lógica de submissão de exercício no fluxo Phase 1 (acquisition).
   
   public void submitAnswer(ExerciseId id, Answer answer) { ... }"
```

**Diferença do `hierarchical`:**
- `hierarchical`: skeleton estático (class name + member list)
- `contextual`: descrição semântica gerada por LLM

**Critério de saída:**
- [ ] Enricher `contextual` usando Ollama/MLX
- [ ] Cache por `content_hash` (não regenera se chunk não mudou)
- [ ] Fallback para `hierarchical` se LLM indisponível

### 4.2 Glossário de Domínio

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/ingest/glossary.py`
**Prioridade:** P2

Extrair termos de domínio e indexar como chunks especiais:

```yaml
glossary:
  - term: "Subject"
    definition: "Atomic knowledge unit (DDD entity); has Content; has RequiredSubjects"
  - term: "Streak of 3"
    definition: "Protocol exiting Phase 1 acquisition into Phase 2 review"
  - term: "NextStudyStep"
    definition: "Server-driven UI navigation contract between backend and mobile"
```

**Critério de saída:**
- [ ] Extração automática via LLM ou manual via YAML
- [ ] Chunks com `kind=glossary`
- [ ] Boost em queries conceituais

### 4.3 Indexação de Comentários Separada

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/ingest/chunkers/` (modificação)
**Prioridade:** P3

Javadoc/comentários como chunks separados linkados ao código:

```
Chunk 1 (kind=comment):
  "Submits an answer for the given exercise. Updates streak and triggers FSRS
   calculation when mastery threshold is reached."

Chunk 2 (kind=method):
  "public void submitAnswer(...) { ... }"

Link: chunk1.related_chunk_id = chunk2.chunk_id
```

**Critério de saída:**
- [ ] Comentários extraídos como chunks `kind=comment`
- [ ] Link bidirecional com chunk de código
- [ ] Boost para queries em linguagem natural

---

## Sprint 5: Soft Index + Polish (Baixa Prioridade)

**Duração:** 1 semana
**Impacto:** UX para desenvolvimento ativo

### 5.1 Soft Index (Working Tree Overlay)

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/storage/overlay.py`
**Prioridade:** P2

Edições não-commitadas refletidas no RAG em tempo real:

```
1. File watcher detecta mudança em arquivo
2. Re-chunk apenas o arquivo modificado
3. Armazena em overlay RAM
4. Queries consultam overlay + base
5. Commit → flush overlay para índice base
```

**Critério de saída:**
- [ ] `stropha watch` mantém overlay atualizado
- [ ] Queries transparentemente consultam overlay
- [ ] Debounce de 2s para evitar thrashing

### 5.2 Detecção de Anchors Arquiteturais

**Status:** [ ] Não iniciado
**Arquivo:** `src/stropha/ingest/anchors.py`
**Prioridade:** P3

Identificar automaticamente pontos críticos:
- Aggregate roots (DDD)
- GraphQL resolvers
- REST controllers
- Security policies
- Config entry points

**Critério de saída:**
- [ ] Flag `architectural_anchor: true` em chunks relevantes
- [ ] Boost no ranking para queries arquiteturais
- [ ] Heurísticas por linguagem (annotations, naming)

---

## Backlog (Futuro)

### B.1 Late Chunking (Jina)
Embeddar documento inteiro, depois fatiar o vetor. Útil para docs longos.

### B.2 ColBERT Multi-Vector
Representação por token em vez de single-vector. +precision, +50x storage.

### B.3 Cross-Repo Federated Search
Query fan-out para múltiplos índices distribuídos.

### B.4 Step-Back Prompting
Gerar versão mais geral da query para queries muito específicas.

### B.5 Self-RAG / CRAG
Modelo decide se documento é relevante; dispara busca alternativa se confiança baixa.

---

## Métricas de Sucesso

| Métrica | Baseline (atual) | Meta Sprint 1-2 | Meta Final |
|---------|------------------|-----------------|------------|
| Recall@10 | ~75% | 85% | 90% |
| Precision@5 | ~60% | 75% | 85% |
| MRR | ~0.65 | 0.75 | 0.85 |
| Latência p50 | 150ms | 200ms (com rerank) | 250ms |
| Latência p95 | 400ms | 500ms | 600ms |

**Benchmark:** Golden dataset em `tests/eval/` + `stropha eval --top-k 10`

---

## Dependências Técnicas

### Novos pacotes Python
```toml
[project.optional-dependencies]
rerank = [
    "sentence-transformers>=2.2.0",  # para cross-encoder
    "onnxruntime>=1.16.0",           # já existe
]
splade = [
    "transformers>=4.35.0",
    "torch>=2.0.0",                  # ou usar ONNX export
]
```

### Modelos ONNX a baixar
| Modelo | Tamanho | Uso |
|--------|---------|-----|
| `mxbai-rerank-large-v1` | ~500MB | Reranking |
| `splade-cocondenser` | ~500MB | Sparse neural (opcional) |

---

## Checklist de Implementação

### Sprint 1
- [x] 1.1 Reranker local (cross-encoder via fastembed) - CONCLUIDO 2026-05-16
- [x] 1.2 Filtros no MCP (language, path_prefix, kind, exclude_tests) - CONCLUIDO 2026-05-16
- [x] 1.3 Faceted response - CONCLUIDO 2026-05-16
  - [x] 1.3.1 Add FacetCounts model to server.py (language, kind, repo counts)
  - [x] 1.3.2 Update SearchResponse model to include optional facets field
  - [x] 1.3.3 Add compute_facets() method to Storage class (single SQL query)
  - [x] 1.3.4 Update search_code tool to compute and return facets
  - [x] 1.3.5 Add include_facets parameter to search_code (default False)
  - [x] 1.3.6 Write unit test for faceted search response

### Sprint 2
- [x] 2.1 HyDE integrado no hybrid-rrf - CONCLUIDO 2026-05-16
- [x] 2.2 Query rewriting - CONCLUIDO 2026-05-16
- [x] 2.3 Multi-query expansion - CONCLUIDO 2026-05-16
  - [x] 2.3.1 Create src/stropha/retrieval/multi_query.py module
  - [x] 2.3.2 Implement MultiQueryExpander class with generate_paraphrases()
  - [x] 2.3.3 Use Ollama/MLX LLM to generate 3-5 paraphrases
  - [x] 2.3.4 Add multi_query_enabled config to HybridRrfConfig
  - [x] 2.3.5 Integrate into HybridRrfRetrieval.search() - run N searches, RRF fuse
  - [x] 2.3.6 Add STROPHA_MULTI_QUERY_ENABLED env var support
  - [x] 2.3.7 Add cache for paraphrases (avoid repeated LLM calls)
  - [x] 2.3.8 Write unit tests for multi-query expansion (17 tests)

### Sprint 3 - Retrieval Avançado
- [x] 3.1 Auto-merging retrieval - ALREADY IMPLEMENTED (see recursive.py)
  - [x] 3.1.1 src/stropha/retrieval/recursive.py already implements parent promotion + adjacency merging
  - [x] 3.1.2 `_maybe_merge_parent()` promotes siblings to parent chunk
  - [x] 3.1.3 Integrated into HybridRrfRetrieval via `_maybe_recursive_merge()`
  - [x] 3.1.4 Config: `recursive_enabled`, `recursive_adjacency` in HybridRrfConfig
  - [x] 3.1.5 Env vars: STROPHA_RECURSIVE_RETRIEVAL, STROPHA_RECURSIVE_ADJACENCY
  - [x] 3.1.6 16 tests in test_hyde_and_recursive.py
- [x] 3.2 Recursive retrieval (completar) - CONCLUIDO 2026-05-16
  - [x] 3.2.1 Add --recursive flag to CLI search command
  - [x] 3.2.2 Add recursive parameter to MCP search_code tool
  - [x] 3.2.3 Parent + adjacency merging implemented in recursive.py
  - [x] 3.2.4 Integration via HybridRrfRetrieval._maybe_recursive_merge()
  - [x] 3.2.5 Adjacency configurable via STROPHA_RECURSIVE_ADJACENCY
  - [x] 3.2.6 Existing tests in test_hyde_and_recursive.py
- [ ] 3.3 SPLADE stream (deferred - BM25 sufficient for now)

### Sprint 4 - Contextual Enrichment (Highest ROI)
- [x] 4.1 Contextual prefix enricher (+35% recall per Anthropic benchmarks) - CONCLUIDO 2026-05-16
  - [x] 4.1.1 Create src/stropha/adapters/enricher/contextual.py
  - [x] 4.1.2 Define ContextualEnricherConfig (model, prompt_template, max_content_chars, max_description_chars)
  - [x] 4.1.3 Implement _generate_context() using Ollama HTTP API
  - [x] 4.1.4 Prompt template with placeholders: {content}, {language}, {rel_path}, {symbol}
  - [x] 4.1.5 Prepend [Context: description] to embedding_text
  - [x] 4.1.6 Cache by content_hash via enrichments table (existing infrastructure)
  - [x] 4.1.7 Fallback to raw content if LLM unavailable (graceful degradation)
  - [x] 4.1.8 Register adapter with @register_adapter(stage="enricher", name="contextual")
  - [x] 4.1.9 Write unit tests (19 tests with mock Ollama responses)
  - [ ] 4.1.10 Benchmark: compare recall with/without contextual enricher (deferred)
- [ ] 4.2 Domain glossary
  - [ ] 4.2.1 Schema v8: Add glossary table (term, definition, embedding, repo_id)
  - [ ] 4.2.2 Create src/stropha/ingest/glossary.py module
  - [ ] 4.2.3 Support manual glossary via YAML file (stropha-glossary.yaml)
  - [ ] 4.2.4 Support auto-extraction via LLM (scan code for domain terms)
  - [ ] 4.2.5 Index glossary terms as special chunks (kind=glossary)
  - [ ] 4.2.6 Boost glossary matches in conceptual queries
  - [ ] 4.2.7 CLI: stropha glossary {add,list,import,export}
  - [ ] 4.2.8 Write unit tests for glossary
- [ ] 4.3 Separate comment indexing
  - [ ] 4.3.1 Extend tree-sitter chunkers to extract docstrings/Javadoc separately
  - [ ] 4.3.2 Create chunks with kind=comment, linked to code chunk
  - [ ] 4.3.3 Add related_chunk_id column to chunks table
  - [ ] 4.3.4 Boost comment chunks for natural language queries
  - [ ] 4.3.5 Write unit tests for comment extraction

### Sprint 5 - Polish & UX
- [ ] 5.1 Soft index overlay (working tree)
  - [ ] 5.1.1 Create src/stropha/storage/overlay.py module
  - [ ] 5.1.2 Implement OverlayStorage wrapping base Storage
  - [ ] 5.1.3 RAM-based overlay for uncommitted changes
  - [ ] 5.1.4 File watcher detects changes, re-chunks modified files only
  - [ ] 5.1.5 Queries transparently merge overlay + base results
  - [ ] 5.1.6 On commit, flush overlay to base index
  - [ ] 5.1.7 Integrate with existing stropha watch command
  - [ ] 5.1.8 Write unit tests for overlay storage
- [ ] 5.2 Architectural anchors detection
  - [ ] 5.2.1 Create src/stropha/ingest/anchors.py module
  - [ ] 5.2.2 Define anchor patterns per language (annotations, naming conventions)
  - [ ] 5.2.3 Detect: aggregate roots, controllers, resolvers, config entry points
  - [ ] 5.2.4 Add is_architectural_anchor boolean to chunks table
  - [ ] 5.2.5 Boost anchors in architectural queries ("entry point", "main flow")
  - [ ] 5.2.6 Expose in MCP: filter by anchor=true
  - [ ] 5.2.7 Write unit tests for anchor detection

### Sprint 6 - Semantic Cache & Performance
- [x] 6.1 Semantic query cache - CONCLUIDO 2026-05-16
  - [x] 6.1.1 Create src/stropha/retrieval/cache.py module
  - [x] 6.1.2 Implement SemanticCache with LRU eviction
  - [x] 6.1.3 Key: (query_embedding_centroid_rounded, top_k, filters_hash)
  - [x] 6.1.4 Store: list of chunk_ids + scores (not full results)
  - [x] 6.1.5 TTL-based invalidation (configurable, default 1 hour)
  - [x] 6.1.6 Global cache singleton for MCP server
  - [x] 6.1.7 Add STROPHA_QUERY_CACHE_ENABLED, STROPHA_QUERY_CACHE_SIZE, STROPHA_QUERY_CACHE_TTL env vars
  - [x] 6.1.8 Write unit tests for semantic cache (21 tests)
- [ ] 6.2 Connection pooling for future remote backends
- [ ] 6.3 Binary quantization for large indexes (deferred)

### Sprint 7 - Documentation & Adoption
- [x] 7.1 MCP integration guide - CONCLUIDO 2026-05-16
  - [x] 7.1.1 Create docs/guides/mcp-integration.md
  - [x] 7.1.2 OpenCode integration
  - [x] 7.1.3 Continue integration
  - [x] 7.1.4 Zed integration
  - [x] 7.1.5 Cursor integration (via MCP)
  - [x] 7.1.6 Claude Desktop integration
  - [x] 7.1.7 Generic MCP client instructions
- [ ] 7.2 Performance tuning guide
  - [ ] 7.2.1 Create docs/guides/performance.md
  - [ ] 7.2.2 Large repo recommendations (>100k files)
  - [ ] 7.2.3 Embedder selection guide
  - [ ] 7.2.4 Enricher tradeoffs
  - [ ] 7.2.5 Hook optimization for monorepos
- [ ] 7.3 Troubleshooting guide
  - [ ] 7.3.1 Create docs/guides/troubleshooting.md
  - [ ] 7.3.2 Common errors and solutions
  - [ ] 7.3.3 Hook debugging
  - [ ] 7.3.4 Index corruption recovery

---

---

## Competitive Position: Stropha as RAG Backend

| Feature | Cursor | Cody | Continue | **Stropha** |
|---------|--------|------|----------|-------------|
| **Retrieval Quality** |
| Reranking | Cloud | Cloud | ✗ | **✓ Local** |
| HyDE | ✓ | ✗ | ✗ | **✓** |
| Query rewrite | ✓ | ✓ | ✗ | **✓** |
| Multi-query | ✓ | ✗ | ✗ | **✓** |
| Rich filters | ✓ | ✓ | Partial | **✓** |
| Faceted search | ✓ | ✓ | ✗ | **✓** |
| **Graph Intelligence (Moat)** |
| Graph traversal | Basic | ✓ SCIP | ✗ | **✓✓ Advanced** |
| Community detection | ✗ | ✗ | ✗ | **✓ Unique** |
| God nodes | ✗ | ✗ | ✗ | **✓ Unique** |
| trace_feature | ✗ | ✗ | ✗ | **✓ Unique** |
| find_rationale | ✗ | ✗ | ✗ | **✓ Unique** |
| **Philosophy** |
| Local-first | ✗ | ✗ | ✓ | **✓✓** |
| MCP native | ✗ | ✗ | ✓ | **✓** |
| Zero cloud cost | ✗ | ✗ | Partial | **✓** |

### Key Insight

Stropha is NOT competing with Cursor as an IDE. It's the **retrieval layer** that can power:
- OpenCode (primary target)
- Continue
- Zed's AI features
- Any MCP-compatible client
- Even Cursor itself (via MCP)

The graph intelligence (community detection, god nodes, trace_feature) is unique in the market and provides architectural understanding that no competitor offers.

---

## Implementation Order (Optimized for Impact)

Based on ROI analysis, here's the recommended implementation order:

### Phase 1: Quick Wins (1-2 days each)
1. **Multi-query expansion** (Sprint 2.3) — leverages existing Ollama/MLX, small effort
2. **Semantic query cache** (Sprint 6.1) — ~0.5 day, immediate latency improvement

### Phase 2: Highest ROI (3-5 days each)
3. **Contextual prefix enricher** (Sprint 4.1) — +35% recall per Anthropic benchmarks
4. **Auto-merging retrieval** (Sprint 3.1) — better UX, saves context window

### Phase 3: Complete the Retrieval Stack (2-3 days each)
5. **Recursive retrieval** (Sprint 3.2) — two-pass search for better coverage
6. **Domain glossary** (Sprint 4.2) — helps conceptual queries

### Phase 4: Polish (2-3 days each)
7. **Soft index overlay** (Sprint 5.1) — real-time uncommitted changes
8. **Architectural anchors** (Sprint 5.2) — auto-detect entry points

### Phase 5: Documentation (1-2 days)
9. **MCP integration guide** (Sprint 7.1) — critical for adoption
10. **Performance & troubleshooting guides** (Sprint 7.2, 7.3)

---

## Next Steps (Priority Order)

1. ~~**Faceted search** — return counts per facet (Sprint 1.3)~~ **Done**
2. ~~**Multi-query expansion** — generate paraphrases and fuse results (Sprint 2.3)~~ **Done**
3. ~~**Semantic query cache** — LRU cache for repeated queries (Sprint 6.1)~~ **Done**
4. ~~**Contextual prefix enricher** — LLM-generated semantic descriptions (Sprint 4.1)~~ **Done**
5. **MCP integration guide** — document usage with OpenCode, Continue, Zed (Sprint 7.1)

---

## History

| Date | Change |
|------|--------|
| 2026-05-16 | Document created with full roadmap |
| 2026-05-16 | Sprint 1.1, 1.2, 2.1, 2.2 completed - reranker, MCP filters, HyDE, query rewrite |
| 2026-05-16 | Detailed subtasks for 1.3 (faceted) and 2.3 (multi-query) |
| 2026-05-16 | Strategic pivot: Option A (best RAG backend) + Option C (graph moat) |
| 2026-05-16 | Sprint 1.3 completed - faceted search with include_facets parameter |
| 2026-05-16 | Sprint 2.3 completed - multi-query expansion with cache + 17 tests |
| 2026-05-16 | Sprint 6.1 completed - semantic query cache with LRU + TTL + 21 tests |
| 2026-05-16 | Sprint 4.1 completed - contextual prefix enricher + 19 tests |
| 2026-05-16 | Sprint 3.1+3.2 completed - --recursive flag for CLI + MCP search_code |
| 2026-05-16 | Sprint 7.1 completed - MCP integration guide (docs/guides/mcp-integration.md) |
| 2026-05-16 | Comprehensive implementation plan added - Sprints 3-7 with detailed subtasks |

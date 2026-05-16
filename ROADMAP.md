# Stropha RAG Roadmap — Feature Parity com Estado da Arte

> **Objetivo:** Implementar funcionalidades RAG que faltam no stropha para atingir paridade com Cursor, Sourcegraph Cody e outras soluções enterprise, mantendo a filosofia local-first.
>
> **Data:** 2026-05-16
> **Status:** Em andamento

---

## Resumo Executivo

| Categoria | Atual | Meta | Gap |
|-----------|-------|------|-----|
| Retrieval | 80% | 100% | Reranking, Auto-merge, SPLADE |
| Query Processing | 40% | 90% | HyDE, Rewrite, Multi-query |
| Filtering | 30% | 90% | Exposição MCP, Facetas |
| Indexing | 90% | 100% | Soft index, Glossário |
| Graph | 95% | 100% | Anchors arquiteturais |

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
- [ ] 1.3 Faceted response
  - [ ] 1.3.1 Add FacetCounts model to server.py (language, kind, repo counts)
  - [ ] 1.3.2 Update SearchResponse model to include optional facets field
  - [ ] 1.3.3 Add compute_facets() method to Storage class (single SQL query)
  - [ ] 1.3.4 Update search_code tool to compute and return facets
  - [ ] 1.3.5 Add include_facets parameter to search_code (default False)
  - [ ] 1.3.6 Write unit test for faceted search response

### Sprint 2
- [x] 2.1 HyDE integrado no hybrid-rrf - CONCLUIDO 2026-05-16
- [x] 2.2 Query rewriting - CONCLUIDO 2026-05-16
- [ ] 2.3 Multi-query expansion
  - [ ] 2.3.1 Create src/stropha/retrieval/multi_query.py module
  - [ ] 2.3.2 Implement MultiQueryExpander class with generate_paraphrases()
  - [ ] 2.3.3 Use Ollama/MLX LLM to generate 3-5 paraphrases
  - [ ] 2.3.4 Add multi_query_enabled config to HybridRrfConfig
  - [ ] 2.3.5 Integrate into HybridRrfRetrieval.search() - run N searches, RRF fuse
  - [ ] 2.3.6 Add STROPHA_MULTI_QUERY_ENABLED env var support
  - [ ] 2.3.7 Add cache for paraphrases (avoid repeated LLM calls)
  - [ ] 2.3.8 Write unit test for multi-query expansion

### Sprint 3
- [ ] 3.1 Auto-merging retrieval
- [ ] 3.2 Recursive retrieval (completar)
- [ ] 3.3 SPLADE stream

### Sprint 4
- [ ] 4.1 Contextual prefix enricher
- [ ] 4.2 Glossário de domínio
- [ ] 4.3 Comentários como chunks separados

### Sprint 5
- [ ] 5.1 Soft index overlay
- [ ] 5.2 Anchors arquiteturais

---

---

## Comparação: Stropha vs. Concorrentes

| Feature | Cursor | Cody | Continue | Stropha |
|---------|--------|------|----------|---------|
| Reranking | ✓ Cloud | ✓ Cloud | ✗ | **✓ Local** |
| HyDE | ✓ | ✗ | ✗ | **✓** |
| Query rewrite | ✓ | ✓ | ✗ | **✓** |
| Multi-query | ✓ | ✗ | ✗ | ✗ |
| Filtros ricos | ✓ | ✓ | Parcial | **✓** |
| Faceted search | ✓ | ✓ | ✗ | ✗ |
| Auto-merge | ✓ | ✗ | ✗ | ✗ |
| SPLADE | ✗ | ✗ | ✗ | ✗ |
| Graph traversal | Básico | ✓ SCIP | ✗ | **✓✓** |
| Community detection | ✗ | ✗ | ✗ | **✓** |
| Local-first | ✗ | ✗ | ✓ | **✓✓** |

### Conclusão

O stropha agora tem:

1. **Reranker local** — ✅ cross-encoder via fastembed (2026-05-16)
2. **HyDE + Query rewriting** — ✅ inteligência de query (2026-05-16)
3. **Filtros expostos no MCP** — ✅ language, path_prefix, kind, exclude_tests (2026-05-16)

Próximos gaps a fechar:
1. **Faceted search** — retornar contagens por faceta
2. **Multi-query expansion** — gerar paráfrases e fusionar resultados

O diferencial do graph já está implementado e é único no mercado.

---

## Histórico

| Data | Mudança |
|------|---------|
| 2026-05-16 | Documento criado com roadmap completo |
| 2026-05-16 | Sprint 1.1, 1.2, 2.1, 2.2 concluídos - reranker, filtros MCP, HyDE, query rewrite |
| 2026-05-16 | Detalhamento das subtarefas de 1.3 (faceted) e 2.3 (multi-query) |

# RAG System para o Codebase Mimoria

Documento de design técnico para um sistema de Retrieval-Augmented Generation (RAG) cuja função é servir como fonte de verdade sobre o codebase Mimoria, exposto ao Claude Code (e outros clientes LLM) via Model Context Protocol (MCP).

> **Status**: Phase 0 (spike) e Phase 1 (MVP MCP) implementados — ver §16. Esquema v2 adiciona identidade de repositório por chunk (foundation para multi-repo da Phase 4). Para o estado vivo da implementação consulte `CLAUDE.md` na raiz do projeto. O documento mapeia o espaço de soluções inteiro — cada decisão tem alternativas listadas com tradeoffs, e o final traz uma recomendação concreta para este projeto.

---

## 1. Objetivos e não-objetivos

### 1.1 Objetivos

1. **Recuperação semântica de código**: dado uma query em linguagem natural ("onde está a lógica que valida resposta de exercício de Speech?"), retornar trechos relevantes do código com path, linhas e contexto.
2. **Recuperação semântica de artefatos não-código**: feature files Gherkin, YAML de seed, documentação Markdown, comentários de domínio.
3. **Navegação simbólica**: dado um nome de classe/método, retornar definição, callers, callees, tipos relacionados.
4. **Atualização incremental** integrada ao fluxo Git, sem reindexação completa a cada commit.
5. **Exposição via MCP** com contrato estável, latência baixa (<500 ms p95) e respostas que economizam o context window do cliente.
6. **Avaliação contínua** com golden dataset versionado e métricas reproduzíveis.

### 1.2 Não-objetivos (escopo explicitamente fora)

- **Geração de código**: o RAG fornece contexto, não escreve código. O Claude Code (ou outro cliente) é o gerador.
- **Substituir LSP/IDE**: análise estática profunda (refactor cross-file, type-checking) continua sendo do IntelliJ/IDE.
- **Indexar dependências externas** (Quarkus, Vue, etc.) numa primeira fase. Isso é explosão de escopo; entra como Phase 4 opcional.
- **Multi-repositório**: foco no monorepo Mimoria. Multi-repo é generalização posterior.

### 1.3 Critérios de sucesso

| Métrica | Alvo |
|---|---|
| Recall@10 em queries semânticas (golden set) | ≥ 0,85 |
| MRR (Mean Reciprocal Rank) em queries simbólicas | ≥ 0,90 |
| Latência p95 do tool `search_code` | ≤ 500 ms |
| Latência p95 da reindexação incremental por commit | ≤ 5 s |
| Custo mensal de embeddings (após bootstrap) | ≤ US$ 5 |
| Cobertura do índice (arquivos relevantes) | ≥ 95 % |

---

## 2. Arquitetura de alto nível

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CAMADA DE INGESTÃO                           │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────────┐  │
│  │ Source   │→ │ Tree-sitter  │→ │ Chunker   │→ │ Enricher (LLM) │  │
│  │ Walker   │  │ Parser (AST) │  │ semântico │  │  + metadata    │  │
│  └──────────┘  └──────────────┘  └───────────┘  └────────────────┘  │
│       │              │                  │                │           │
│       ▼              ▼                  ▼                ▼           │
│  .strophaignore Symbol Graph       Chunk Store      Summary Cache    │
│  + git diff    (call/import)        (SQLite)         (SQLite)        │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      CAMADA DE EMBEDDING                             │
│  ┌────────────────────┐  ┌────────────────────┐                      │
│  │ Dense embedder     │  │ Sparse embedder    │                      │
│  │ (Voyage code-3)    │  │ (BM25 / SPLADE)    │                      │
│  └────────────────────┘  └────────────────────┘                      │
│            │                       │                                 │
│            ▼                       ▼                                 │
│   Vector Index (HNSW)      Inverted Index                            │
│   sqlite-vec / LanceDB     Tantivy / SQLite FTS5                     │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       CAMADA DE RECUPERAÇÃO                          │
│  Query → [Rewriter / HyDE] → [Hybrid Search] → [RRF Fusion]          │
│        → [Reranker (Cohere/Voyage)] → [Context Packer]               │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          MCP SERVER                                  │
│  Tools: search_code · get_symbol · find_callers · get_file_outline   │
│         find_tests_for · explain_module · trace_feature              │
│  Resources: index_stats · indexed_files                              │
│  Transport: stdio (default) · streamable HTTP (opcional)             │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Claude Code / IDE / Agent
```

### 2.1 Princípios de design

1. **Local-first**: tudo roda na máquina do dev por padrão. Sem dependência de serviços hospedados para uso individual.
2. **Single binary preferível**: minimizar processos longos. Servidor MCP em Python com `uv` ou em Rust embedando tudo.
3. **Append-only com compactação**: writes idempotentes; reorganização periódica em background.
4. **Observabilidade desde o dia 0**: tracing OpenTelemetry, métricas Prometheus-compatible, logs estruturados (JSON).
5. **Custo previsível**: embeddings só rodam em chunks novos/modificados; reranking apenas quando hybrid search devolve >K candidatos.
6. **Falha graciosa**: se o reranker estiver fora, devolva resultados de hybrid puro com aviso. Se o índice vetorial falhar, caia para BM25.

---

## 3. Pipeline de indexação

A qualidade do RAG é dominada pela **qualidade da indexação**. Embedding bom não compensa chunking ruim.

### 3.1 Source walker e exclusões

**Descoberta de arquivos**:

- Caminhar a partir da raiz do repo (`git ls-files` é mais confiável que `find` — respeita `.gitignore` automaticamente).
- Aplicar `.strophaignore` adicional (formato igual a `.gitignore`) para excluir lockfiles, builds, fixtures binários:
  ```
  **/target/**
  **/node_modules/**
  **/dist/**
  **/.next/**
  **/*.lock
  **/*.min.js
  **/*.snap
  ```
- Detecção de binários: `file --mime-encoding` ou heurística (presença de bytes nulos nos primeiros 8 KB).
- Limite de tamanho por arquivo: 500 KB. Acima disso, indexar apenas o "outline" (símbolos de topo).

### 3.2 Parsing AST com Tree-sitter

**Tree-sitter** (https://tree-sitter.github.io) é o padrão de facto para parsing incremental de múltiplas linguagens, usado por GitHub, Neovim, Zed, Helix e ferramentas como `ast-grep` e `tree-sitter-graph`.

Linguagens necessárias para Mimoria:

| Linguagem | Grammar |
|---|---|
| Java | `tree-sitter-java` |
| Vue (SFC) | `tree-sitter-vue` (com sub-parsing para `<script>` em TS/JS, `<template>` HTML, `<style>` CSS) |
| TypeScript | `tree-sitter-typescript` |
| JavaScript | `tree-sitter-javascript` |
| YAML | `tree-sitter-yaml` |
| Markdown | `tree-sitter-markdown` |
| Gherkin | `tree-sitter-gherkin` |
| SQL | `tree-sitter-sql` |
| Properties / TOML / JSON | grammars correspondentes |

Para cada arquivo:

1. Detectar linguagem por extensão + shebang.
2. Carregar grammar correspondente.
3. Gerar AST.
4. Aplicar **query S-expression** específica por linguagem para extrair nós-alvo (classes, métodos, componentes Vue, scenarios Gherkin).

Exemplo de query Tree-sitter para Java (extrair métodos com docstring):

```scheme
(method_declaration
  name: (identifier) @method.name
  body: (block) @method.body
  (#set! priority "high")) @method.full
```

### 3.3 Estratégia de chunking

**Princípio fundamental**: um chunk = uma unidade semântica completa que faz sentido isoladamente. Não corte no meio de um método.

#### 3.3.1 Chunking semântico hierárquico (recomendado)

Modelo de **três níveis**:

| Nível | Granularidade Java | Granularidade Vue | Granularidade Markdown |
|---|---|---|---|
| **Macro** | Pacote (sumário das classes) | Diretório de feature (sumário) | Documento inteiro (sumário) |
| **Meso** | Classe (assinatura + javadoc + lista de métodos) | Componente SFC completo | Seção H2/H3 |
| **Micro** | Método individual | `<script setup>` + `<template>` separados, ou função composable | Parágrafo / bloco de código |

Cada chunk no nível N referencia seu pai no nível N+1 via `parent_chunk_id`. Isso permite **recursive retrieval** (recuperar pelo método e expandir para a classe se faltar contexto).

**Regras de quebra**:

- **Java**: nunca quebrar dentro de um método. Se um método > 200 linhas (raro, mas acontece), quebrar em "header" (assinatura + primeiras 30 linhas) e "tail" (resto), ambos referenciando a classe.
- **Vue SFC**: três chunks por componente — `script setup`, `template`, `style`. Em componentes pequenos (<150 linhas totais), um único chunk para o SFC inteiro economiza embeddings.
- **Gherkin**: um chunk por `Scenario`/`Scenario Outline`. Um chunk-pai por `Feature` com Background incluído.
- **Markdown**: split por heading. Se uma seção tem code block grande, ele vira sub-chunk próprio (linkado).
- **YAML de seed**: um chunk por entidade de topo (curso, capítulo, lição) — preserva estrutura aninhada.

#### 3.3.2 Janelas de contexto

Cada chunk tem dois "raios" de contexto adicional, **não embeddados** mas retornáveis:

- `neighbor_above`: 5 linhas anteriores (imports relevantes, decoradores).
- `neighbor_below`: 5 linhas posteriores.

Útil para o consumidor reconstruir o trecho com contexto sem segundo round-trip.

#### 3.3.3 Late chunking (Jina, 2024)

Técnica em que se embedda o documento inteiro com um modelo de contexto longo, depois se "fatia" o vetor em sub-vetores correspondentes a cada chunk, preservando contexto cross-chunk. Recomendado para documentação longa onde referências cruzadas importam (ex.: `CLAUDE.md`, `study-flow.md`). Não compensa para código (chunks já são autocontidos).

#### 3.3.4 Anti-padrões a evitar

- **Fixed-size chunking** (cortar a cada 512 tokens): destrói estruturas, péssimo recall em código.
- **Overlap exagerado** (>20 %): infla custo de armazenamento e degrada precisão (mesma informação em múltiplos chunks confunde rerank).
- **Chunks de 1 linha**: ruído, sem contexto suficiente para embedding.
- **Chunks > 8K tokens**: excede contexto da maioria dos modelos de reranking; trunca silenciosamente.

### 3.4 Extração de metadados

Cada chunk persistido tem schema rico:

```jsonc
{
  "chunk_id": "sha256:abc123...",                     // determinístico (path + start + end + content_hash)
  "path": "backend/src/main/java/com/mimoria/service/StudyService.java",
  "language": "java",
  "kind": "method",                                   // file | class | method | component | scenario | section
  "symbol": "StudyService.submitAnswer",
  "parent_symbol": "StudyService",
  "parent_chunk_id": "sha256:def456...",
  "start_line": 145,
  "end_line": 187,
  "byte_range": [4321, 6543],
  "imports": ["com.mimoria.domain.enrollment.Enrollment", ...],
  "annotations": ["@Transactional", "@Mutation"],
  "calls_out": ["EnrollmentRepository.findByUser", "FsrsCalculator.update"],
  "called_by": [...],                                 // preenchido na 2ª passada
  "tests_referencing": ["StudyServiceTest.shouldRecordCorrectAnswer"],
  "git": {
    "last_commit": "2d22f0a",
    "last_modified": "2026-05-13T10:24:11Z",
    "main_authors": ["jonatas"]                       // top-3 por linhas
  },
  "summary": "Submete a resposta de um exercício, atualiza streak e dispara FSRS quando atinge mastery.",
  "summary_model": "claude-haiku-4-5-20251001",
  "summary_generated_at": "2026-05-14T08:00:00Z",
  "content_hash": "sha256:...",                       // p/ detectar mudança real vs git noise
  "embedding_model": "voyage-code-3",
  "embedding_version": 1
}
```

### 3.5 Symbol graph (grafo de código)

Em paralelo aos chunks, construir um **grafo dirigido**:

- **Nós**: símbolos (classes, métodos, funções, componentes).
- **Arestas**:
  - `defined_in` → arquivo.
  - `calls` → método chamado.
  - `imports` → módulo importado.
  - `extends` / `implements` → herança/interface.
  - `tests` → step definition para feature, teste para método.
  - `references` → uso de tipo/constante.

**Por quê?** Habilita queries que embedding sozinho não resolve bem:

- "Quem chama `submitAnswer`?" → travessia direta do grafo, custo O(1) de lookup.
- "Quais testes cobrem `FsrsCalculator`?" → seguir aresta `tests` reversa.
- "Cadeia de prerequisitos de um Subject" → query sobre dados de domínio (não código), mas mesma estrutura.

**Implementação**:

- Extração: combinar tree-sitter (resolve nomes locais) + análise heurística de imports (resolve cross-file). Para precisão máxima, usar **JavaParser** (Java) e **typescript-eslint** (TS/Vue).
- Persistência: tabelas `nodes(id, kind, name, ...)` e `edges(from_id, to_id, kind, weight)` em SQLite. Para grafos > 1M arestas, considerar Kuzu ou DuckDB com extensão grafo.
- **GraphRAG** (Microsoft, 2024) e **LightRAG** (2024) extraem grafos textualmente via LLM — desnecessário para código onde grafos são derivados deterministicamente do AST. Reservar GraphRAG para documentação não-estruturada se necessário.

### 3.6 Enriquecimento via LLM (summary generation)

Para cada chunk de granularidade média/macro, gerar uma `summary` curta (1-2 sentenças) via LLM barato:

- **Modelo**: `claude-haiku-4-5` (custo ~US$ 1/M input tokens). Prompt-cache as instruções de sistema.
- **Conteúdo embeddado**: `summary + "\n\n" + chunk_text`. O summary aproxima o chunk de queries em linguagem natural (que raramente usam vocabulário do código).
- **Regeneração**: apenas quando `content_hash` muda, não a cada reindex.
- **Custo estimado** (Mimoria, ~5K chunks): ~US$ 0,50 inicial, ~US$ 0,02/dia em incremental.

**Cuidado**: se a summary for muito genérica ("este método faz X"), prejudica mais que ajuda. Prompt deve forçar especificidade com exemplos few-shot e proibir frases vazias.

### 3.7 Pipeline de execução (orquestração)

Frameworks state-of-the-art em 2026:

| Framework | Pontos fortes | Quando escolher |
|---|---|---|
| **LlamaIndex** | Maior ecossistema RAG; primitives prontas (NodeParser, IngestionPipeline, transformations). | Prototipagem rápida, equipe Python. |
| **Haystack 2.x** | Pipelines declarativos, ótima observabilidade, deploy serverless. | Time já familiar com Haystack. |
| **txtai** | Single-binary, embedded, baixa cerimônia. | MVP local ultraleve. |
| **Custom (DIY)** | Sem dependência transitiva, controle total. | Quando os frameworks acima atrapalham mais que ajudam. |

Recomendação: começar com **LlamaIndex `IngestionPipeline`** (com cache de transformações), migrar para custom se gargalo aparecer.

---

## 4. Embeddings

### 4.1 Modelos densos — comparação 2026

| Modelo | Dim | Contexto | MTEB Code | Custo (USD/M tokens) | Notas |
|---|---|---|---|---|---|
| **Voyage `voyage-code-3`** | 1024 (Matryoshka 256/512/1024/2048) | 32K | **SOTA em código (CodeSearchNet, CoIR)** | 0,18 | Melhor qualidade absoluta para código. |
| **Voyage `voyage-3-large`** | 1024 (Matryoshka) | 32K | Excelente geral | 0,18 | Bom se misturar muito texto e código. |
| OpenAI `text-embedding-3-large` | 3072 (truncável) | 8K | Bom | 0,13 | Conhecido, ecossistema enorme. |
| Cohere `embed-v4.0` | 1536 (Matryoshka 256/512/1024/1536) | 128K | Muito bom; multimodal (imagens) | 0,12 | Suporta imagens (diagramas). |
| Jina `jina-embeddings-v3` | 1024 | 8K | Bom; suporta late chunking nativo | 0,02 | Excelente custo-benefício; open weights disponíveis. |
| **NV-Embed-v2** (NVIDIA) | 4096 | 32K | Top MTEB geral | self-host | Open weights; precisa GPU. |
| **Stella-en-1.5B-v5** | 1024 (Matryoshka) | 8K | Top open-source | self-host | 1.5B params; roda em CPU lenta, GPU recomendada. |
| **bge-m3** (BAAI) | 1024 | 8K | Bom; multi-funcional (dense + sparse + multi-vec) | self-host | Único modelo que produz dense + sparse + ColBERT no mesmo forward pass. |

**Recomendação**: `voyage-code-3` para code chunks; `voyage-3-large` para markdown/docs. Configurar dimensão Matryoshka em **512** para economizar storage/latência sem perda significativa.

**Por que não OpenAI por padrão?** Em benchmarks específicos de código (CoIR, CodeSearchNet 2024), Voyage code-3 supera `text-embedding-3-large` por 5-12 pontos de NDCG@10. A diferença é maior em queries que misturam intenção com nome de símbolo.

### 4.2 Embeddings esparsos

Embeddings densos têm um ponto cego: queries com **nomes exatos** ("EnrollmentRepository", "submitExerciseAnswer"). BM25 resolve trivialmente. Hybrid search é mandatório.

Opções:

- **BM25 clássico** (Okapi BM25): implementação trivial via Tantivy ou SQLite FTS5 (BM25 ranking). Zero custo, zero dependência externa.
- **SPLADE** (Sparse Lexical and Expansion): aprende expansões neurais. Melhor que BM25 puro, mas mais complexo (requer modelo).
- **BM42** (Qdrant, 2024): híbrido entre BM25 e atenção de transformer. Ainda controverso (artigo recolhido por inflar resultados); evitar até comunidade decidir.

**Recomendação**: BM25 via SQLite FTS5 (single-file, zero infra). SPLADE só se evaluations mostrarem ganho > 5 pontos NDCG.

### 4.3 Embeddings multi-vetoriais

**ColBERT v2** representa cada token como vetor independente; matching é via "MaxSim" — para cada token da query, achar o token mais similar do doc, somar. Muito mais preciso que dense single-vector, mas storage 50-100x maior.

**ColPali** (2024): variante para documentos visuais (PDFs com diagramas). Não relevante para código puro; relevante se indexar diagramas de arquitetura.

Para Mimoria: **não recomendado na fase inicial**. Single-vector + rerank cobre a necessidade.

### 4.4 Quantização

- **Float16 → Int8**: redução 4x de storage, perda <1% NDCG. Suportado nativo em FAISS, sqlite-vec, Qdrant.
- **Binary quantization** (1-bit): redução 32x, perda 5-10% (recuperável com re-rank). Útil para >10M vetores; overkill para Mimoria.

---

## 5. Camada de armazenamento

### 5.1 Vector stores — comparação

| Store | Tipo | Pros | Contras | Indicação |
|---|---|---|---|---|
| **sqlite-vec** | Embedded SQLite extension | Single-file, zero infra, ACID | Sem replicação, single-writer | **Default para Mimoria.** |
| **LanceDB** | Embedded (Lance format) | Versionamento de dados, multi-modal, columnar | Maturidade média | Quando precisar de versioning/time-travel. |
| **DuckDB + VSS** | Embedded analytical | SQL completo, joins poderosos | Vector search ainda jovem | Análises ad-hoc + RAG na mesma base. |
| **Qdrant** | Server (Rust) | Performático, filtros ricos, payload index | Requer processo separado | Multi-tenant ou >1M vetores. |
| **Weaviate** | Server (Go) | Schema rico, modules (rerank embutido), GraphQL | Mais pesado | Quando precisa de schema strong-typed. |
| **Milvus / Zilliz** | Server (Go/C++) | Escala bilhões | Operação complexa | Escala enterprise. |
| **pgvector** | Postgres extension | Reuso da infra Postgres existente | HNSW só recente; menos features que dedicados | Quando Postgres já é stack obrigatória. |
| **Chroma** | Embedded ou server | API simples | Performance medíocre vs alternativas | Prototipagem inicial. |
| **Vespa** | Server (Java) | Tensor expressions, ranking complexo | Curva aprendizado íngreme | Casos avançados de ranking. |
| **Turbopuffer** | Cloud (serverless) | S3-backed, pay-per-query | Vendor lock-in | Cargas esporádicas. |
| **Pinecone** | Cloud managed | Zero ops | Vendor lock-in, custo alto | Quando ops é proibitivo. |

**Recomendação para Mimoria**: `sqlite-vec` na máquina do dev. Tudo em `~/.stropha/index.db`. Migrar para Qdrant local se passar de ~100k chunks ou exigir filtros complexos.

### 5.2 Algoritmos de indexação aproximada (ANN)

- **HNSW** (Hierarchical Navigable Small World): default da indústria. Recall alto, latência baixa, build moderado.
- **IVF-PQ** (FAISS): compressão extrema, bom para >10M. Build pesado.
- **DiskANN** (Microsoft): vetores em disco SSD. Útil quando RAM limitada.
- **ScaNN** (Google): muito rápido, mas Python-only e pouco mantido.

Para <100k vetores (caso Mimoria), **flat (brute-force)** com SIMD é suficiente e tem recall 100%. HNSW só vira necessário acima de ~500k.

### 5.3 Persistência paralela

| Dado | Store | Por quê |
|---|---|---|
| Vetores densos | sqlite-vec | Hybrid query SQL + ANN |
| Inverted index BM25 | SQLite FTS5 | Mesmo arquivo, transações atômicas |
| Symbol graph | SQLite (tabelas `nodes`/`edges`) | Joins SQL, transacional |
| Chunks raw + metadata | SQLite (tabela `chunks`) | Single source of truth |
| Cache de summaries | SQLite (tabela `summaries`) | Reutilização cross-version do embedding model |
| Cache de embeddings | SQLite (tabela `embeddings_cache` keyed by `content_hash + model`) | Evita reembedding em rebuild |

**Tudo em um único arquivo `.db`** simplifica backup, sync via Dropbox/iCloud, e portabilidade. Se ficar > 5 GB, particionar por sub-projeto.

### 5.4 Versionamento e migração

- Cada chunk tem `embedding_model` + `embedding_version`. Mudança de modelo NÃO invalida o índice — coexistência permite A/B teste.
- Migração de schema via Alembic (Python) ou refinery (Rust).
- **Rebuild completo**: comando `rag rebuild --model voyage-code-3 --version 2`. Preserva o índice antigo até migração validada.

---

## 6. Recuperação (retrieval)

### 6.1 Hybrid search (dense + sparse)

Pipeline padrão:

```
query
  ├──► dense embedding (voyage-code-3)
  │       └──► sqlite-vec ANN search (top 50)
  └──► BM25 tokenization
          └──► SQLite FTS5 (top 50)
                ↓
        Reciprocal Rank Fusion (RRF)
                ↓
            top 30 candidatos
                ↓
        Reranker (Voyage rerank-2.5)
                ↓
            top 10 final
```

#### Reciprocal Rank Fusion (RRF)

Fórmula:
```
score(d) = Σ (1 / (k + rank_i(d)))   para cada lista i, com k=60 (default)
```

Vantagens sobre weighted sum:

- Não requer normalização de scores (BM25 e cosine vivem em escalas diferentes).
- Robusto: outliers em uma lista não dominam.
- Sem hyperparameters críticos (k=60 é resiliente).

### 6.2 Reranking (cross-encoder)

Modelos top em 2026:

| Reranker | Latência (100 docs) | Qualidade | Custo |
|---|---|---|---|
| **Voyage `rerank-2.5`** | ~120 ms | SOTA em código | US$ 0,05 / M tokens |
| **Cohere `rerank-3.5`** | ~150 ms | Excelente geral | US$ 2,00 / 1k searches |
| **Jina `jina-reranker-v2`** | ~80 ms | Muito bom | US$ 0,02 / M tokens |
| **bge-reranker-v2-m3** | self-host (200 ms CPU, 30 ms GPU) | Bom | infra |
| **Mixedbread `mxbai-rerank-large-v2`** | self-host | Top open-source | infra |

**Recomendação**: Voyage rerank-2.5 (mesma família dos embeddings, menos atrito de billing).

**Cuidado com latência**: rerank é cross-encoder (encode query+doc juntos por par). Para 50 candidatos, são 50 forward passes. Sempre rerankar em batch via API.

### 6.3 Query understanding

Querys cruas raramente são ótimas. Pré-processar:

#### 6.3.1 Query rewriting

LLM reescreve query em forma "pesquisável":
- "como funciona a logica de mastery" → "transição para REVIEW após streak de 3 acertos consecutivos"
- "onde tem teste pra fsrs" → "FSRS calculator test scenarios"

Custo: ~50 tokens out via Haiku, ~10 ms.

#### 6.3.2 HyDE (Hypothetical Document Embeddings)

Em vez de embeddar a query, **gerar um documento hipotético** que responderia a query, e embeddar isso. Funciona porque o espaço de embedding está alinhado com docs, não com queries.

- "como salvar enrollment?" → LLM gera: `public void saveEnrollment(Enrollment e) { repository.persist(e); ... }` → embedda este código → busca.

Útil para queries vagas. Adiciona latência (LLM call). Skip se a query já contém termos técnicos.

#### 6.3.3 Multi-query expansion

Gerar 3-5 paráfrases da query, buscar cada uma, fundir resultados via RRF. Aumenta recall ~10% mas multiplica custo.

#### 6.3.4 Step-back prompting

Para queries específicas, gerar uma versão "step back" mais geral. Buscar ambas. Útil quando a query específica não tem chunk exato.
- "como o Hibernate mapeia @TenantId?" → step-back: "como funciona multi-tenancy no projeto?"

#### 6.3.5 Query routing

Classificar a query (regra ou LLM) em rotas:

| Tipo de query | Pipeline |
|---|---|
| **Símbolo exato** ("EnrollmentRepository") | Lookup direto no symbol graph; skip embedding. |
| **Conceitual** ("como funciona spaced repetition") | Hybrid + rerank. |
| **Estrutural** ("quem chama X") | Symbol graph traversal. |
| **Cross-cutting** ("onde tem `@Transactional` em mutations") | FTS5 + filtro de metadados. |
| **Tracing de feature** ("trace do scenario X") | Gherkin → step defs → métodos chamados. |

### 6.4 Técnicas avançadas de recuperação

#### 6.4.1 Contextual Retrieval (Anthropic, set/2024)

Antes de embeddar, prepend ao chunk uma frase de contexto gerada por LLM, do tipo:
> "Este trecho pertence à classe `StudyService` (camada de serviço de aplicação) e implementa a lógica de submissão de exercício no fluxo de Phase 1 (acquisition)."

**Resultado em benchmarks Anthropic**: redução de 35% em failed retrievals; com rerank, 67%.

Custo: 1 chamada LLM por chunk na indexação. Cache by content_hash (regenera só em mudança).

#### 6.4.2 Recursive retrieval / hierarchical retrieval

1. Buscar nível "macro" (sumários de classe).
2. Para os top 3, expandir para nível "micro" (métodos da classe).
3. Rerank no nível micro.

Reduz ruído quando codebase tem muitos métodos curtos similares.

#### 6.4.3 Auto-merging retrieval (LlamaIndex)

Se mais de N chunks-filhos do mesmo pai aparecem nos top-K, retornar o pai inteiro em vez dos filhos. Economiza context window, dá visão coerente.

#### 6.4.4 Self-RAG / Corrective RAG (CRAG)

Modelo decide se cada documento recuperado é relevante; se confiança baixa, dispara busca alternativa (web, outro índice). Útil quando o índice tem cobertura desigual. Para codebase fechado, retorno marginal.

#### 6.4.5 Agentic RAG (ReAct sobre tools)

O cliente (Claude Code) NÃO chama um tool monolítico `rag_query`. Em vez disso, chama tools atômicos (`search_code`, `get_symbol`, `find_callers`) e compõe. Esta é a abordagem MCP — favorece composição sobre encapsulamento.

### 6.5 Filtros e facetas

Toda query deve aceitar filtros estruturados:

```python
search_code(
    query="exercise scoring",
    filters={
        "language": ["java"],
        "path_glob": "backend/src/main/**",
        "kind": ["method", "class"],
        "modified_since": "2026-04-01",
        "annotations_include": ["@Transactional"],
        "exclude_tests": True,
    },
    top_k=10
)
```

Filtros são aplicados **antes** do ANN search (pre-filtering) quando o store suporta (sqlite-vec, Qdrant suportam). Caso contrário, post-filtering com `top_k * 5` candidatos para compensar.

---

## 7. Técnicas específicas para código

### 7.1 Test ↔ implementation linking

Mapear automaticamente:

- Método `StudyServiceTest.shouldRecordCorrectAnswer` → método `StudyService.submitAnswer`.

Estratégias:

1. **Heurística por nome**: `FooTest` → `Foo`. `should<Verb><Object>` → métodos com `<verb>` similar.
2. **Análise de imports**: o teste importa `StudyService` → liga.
3. **Análise de chamadas**: o teste chama `studyService.submitAnswer(...)` → liga ao método específico.

Persistir como aresta `tests` no symbol graph. Tool MCP `find_tests_for("StudyService.submitAnswer")` vira O(1).

### 7.2 Feature ↔ step linking

Para Cucumber:

- Parsear `.feature` files via tree-sitter-gherkin.
- Parsear step definitions Java/TS, extrair regex/cucumber-expression de cada step.
- Casar steps de feature com regex de step def via match exato + similaridade.
- Persistir `feature_step → step_def_method → métodos chamados`.

Resultado: tool `trace_feature("StudentStudySession", "Mastery after 3 correct answers")` devolve a cadeia completa.

### 7.3 Glossário de domínio

Extrair via LLM (Haiku) um glossário de termos do domínio:

```
Subject: atomic knowledge unit (DDD entity); has Content; has RequiredSubjects
Streak of 3: protocol exiting Phase 1 acquisition
NextStudyStep: server-driven UI navigation contract
...
```

Embeddar o glossário como chunks especiais (`kind=glossary`). Queries em linguagem natural batem aqui antes de batem em código, melhorando recall conceitual.

### 7.4 Detecção de "anchors" arquiteturais

Identificar (via análise estática + LLM) os "pontos críticos" do codebase:

- Aggregate roots (DDD).
- GraphQL resources.
- Configuration entry points.
- Security policies.

Marcar com flag `architectural_anchor: true`. Boost no ranking quando query tem termos arquiteturais.

### 7.5 Indexação de comentários e Javadoc separadamente

Comentários explicam intenção; código explica mecanismo. Indexar como chunks separados (kind=`comment`) que linkam ao chunk de código adjacente. Boost para queries em linguagem natural quando matcham comentário.

---

## 8. Atualização incremental

### 8.1 Detecção de mudança

Fonte de verdade: **Git**.

```
git diff --name-status <last_indexed_commit>..HEAD
```

Para cada arquivo:

- `A` (added): chunk + embed + index.
- `M` (modified): rechunk; comparar `content_hash` por chunk; reembed só os mudados.
- `D` (deleted): remover chunks.
- `R` (renamed): atualizar path; preservar embeddings se conteúdo idêntico.

### 8.2 Triggers de reindex

| Trigger | Mecanismo | Latência alvo |
|---|---|---|
| Commit local | Git hook `post-commit` (silencioso, em background) | <5 s |
| Pull / merge | Git hook `post-merge` | <30 s |
| Edição não-commitada | File watcher (chokidar/notify-rs), debounce 2 s | <2 s para "soft index" em RAM |
| Deploy CI | GitHub Actions step | OK ser lento |

### 8.3 Soft index (working tree)

Edições não-commitadas vivem em "overlay" em RAM. Queries consultam overlay + base. Quando commit acontece, overlay é compactado para o índice base. Garante que o RAG sempre reflete o que o dev está vendo, não o último commit.

### 8.4 Compactação e rebuild

- **Compactação semanal**: VACUUM no SQLite, rebuild HNSW.
- **Rebuild completo**: trigger manual ao trocar embedding model. Run em background, swap atômico ao final.

### 8.5 Backfill de novos campos

Schema migrations + lazy backfill: adicionar coluna nullable, preencher em background sem bloquear queries.

---

## 9. MCP Server

### 9.1 Model Context Protocol — visão geral

MCP é o protocolo aberto da Anthropic (2024) para conectar LLMs a ferramentas/dados externos. Especificação: https://modelcontextprotocol.io.

Conceitos:

- **Tools**: funções invocáveis (ex.: `search_code`).
- **Resources**: dados expostos (ex.: estatísticas do índice via URI `stropha://stats`).
- **Prompts**: templates parametrizados (ex.: prompt "explique o módulo X" pré-formatado).
- **Sampling**: o servidor pode pedir ao cliente (LLM) para gerar texto. Útil para HyDE.

Transports:

- **stdio**: processo filho do cliente. Default; menor latência; sem auth.
- **HTTP streamable**: servidor remoto; SSE para responses streaming; suporta auth (OAuth 2.1 com PKCE).

### 9.2 Design dos tools

Princípios:

1. **Atômicos**: cada tool faz uma coisa. Cliente compõe.
2. **Idempotentes**: mesma input → mesma output (até a próxima reindexação).
3. **Output enxuto**: nunca retornar conteúdo grande. Path + lines + snippet curto. Cliente lê arquivo se quiser.
4. **Schemas estritos**: JSON Schema com `additionalProperties: false`, descrições ricas (LLM lê).

#### Catálogo proposto

```jsonc
{
  "tools": [
    {
      "name": "search_code",
      "description": "Busca semântica + lexical (hybrid) no codebase. Retorna trechos com path, linhas e score. Use para 'onde está X', 'como funciona Y', 'mostre exemplos de Z'. Para nome exato de símbolo, prefira get_symbol.",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": { "type": "string", "description": "Pergunta ou descrição em linguagem natural ou termos técnicos." },
          "top_k": { "type": "integer", "default": 10, "minimum": 1, "maximum": 30 },
          "filters": {
            "type": "object",
            "properties": {
              "language": { "type": "array", "items": { "enum": ["java", "vue", "typescript", "yaml", "markdown", "gherkin"] } },
              "path_glob": { "type": "string", "description": "Glob restritivo, ex.: 'backend/src/main/**'." },
              "kind": { "type": "array", "items": { "enum": ["file", "class", "method", "component", "scenario", "section"] } },
              "exclude_tests": { "type": "boolean", "default": false },
              "modified_since": { "type": "string", "format": "date" }
            }
          },
          "rerank": { "type": "boolean", "default": true }
        },
        "required": ["query"]
      }
    },
    {
      "name": "get_symbol",
      "description": "Retorna a definição de um símbolo pelo nome qualificado. Mais barato e preciso que search_code para nomes conhecidos.",
      "input_schema": {
        "type": "object",
        "properties": {
          "symbol": { "type": "string", "description": "Nome qualificado, ex.: 'StudyService.submitAnswer' ou 'StudyView'." },
          "include_callers": { "type": "boolean", "default": false },
          "include_callees": { "type": "boolean", "default": false }
        },
        "required": ["symbol"]
      }
    },
    {
      "name": "find_callers",
      "description": "Lista todos os locais que chamam o símbolo dado. Travessia direta no symbol graph.",
      "input_schema": { "type": "object", "properties": { "symbol": { "type": "string" }, "depth": { "type": "integer", "default": 1 } }, "required": ["symbol"] }
    },
    {
      "name": "get_file_outline",
      "description": "Retorna a estrutura simbólica de um arquivo (classes, métodos, imports) sem ler o conteúdo completo.",
      "input_schema": { "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }
    },
    {
      "name": "find_tests_for",
      "description": "Encontra testes (unit, BDD, E2E) que cobrem o símbolo dado.",
      "input_schema": { "type": "object", "properties": { "symbol": { "type": "string" } }, "required": ["symbol"] }
    },
    {
      "name": "trace_feature",
      "description": "Para um scenario Gherkin, traça step defs → métodos chamados → entidades tocadas.",
      "input_schema": { "type": "object", "properties": { "feature": { "type": "string" }, "scenario": { "type": "string" } }, "required": ["feature", "scenario"] }
    },
    {
      "name": "explain_module",
      "description": "Resumo gerado on-demand de um pacote/módulo (classes principais, responsabilidade, dependências).",
      "input_schema": { "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }
    }
  ],
  "resources": [
    { "uri": "stropha://stats", "description": "Estatísticas do índice (chunks, última atualização, modelo)." },
    { "uri": "stropha://files", "description": "Lista de arquivos indexados com hash." }
  ],
  "prompts": [
    { "name": "investigate-bug", "description": "Template para investigar um bug usando os tools RAG.", "arguments": [{ "name": "symptom", "required": true }] }
  ]
}
```

### 9.3 Formato de resposta

Padrão para `search_code`:

```jsonc
{
  "results": [
    {
      "rank": 1,
      "score": 0.87,
      "path": "backend/src/main/java/com/mimoria/service/StudyService.java",
      "lines": [145, 187],
      "symbol": "StudyService.submitAnswer",
      "kind": "method",
      "snippet": "public StudySubmissionResult submitAnswer(...) {\n    var enrollment = ...\n    // ~30 linhas\n}",
      "summary": "Submete resposta, atualiza streak, dispara FSRS em mastery.",
      "context": {
        "parent": "class StudyService",
        "neighbors": { "imports": ["..."], "annotations": ["@Transactional"] }
      }
    }
  ],
  "query_metadata": {
    "rewritten_query": "...",
    "retrieval_time_ms": 142,
    "rerank_time_ms": 118,
    "total_candidates_before_rerank": 50
  }
}
```

### 9.4 Implementação

**Stack recomendado**:

- **Linguagem**: Python 3.12 (ecossistema RAG mais maduro). Alternativa: Rust se latência crítica.
- **MCP SDK**: `mcp` (Anthropic oficial, Python). Versão ≥ 1.0.
- **Pipeline**: LlamaIndex `IngestionPipeline` + `RetrieverQueryEngine`.
- **Embeddings**: SDK Voyage (`voyageai`).
- **Storage**: `sqlite-vec` (extensão), `apsw` (bindings SQLite Python performáticos).
- **Tree-sitter**: `tree-sitter-language-pack` (todas as grammars empacotadas).
- **Async**: `asyncio` nativo; uvloop em produção.
- **CLI**: `typer` (Click moderno).
- **Empacotamento**: `uv` para gerenciamento, `hatch` para build.

**Estrutura de projeto** (sugestão):

```
stropha/
├── pyproject.toml
├── src/stropha/
│   ├── __init__.py
│   ├── server.py              # MCP server entry
│   ├── tools/
│   │   ├── search.py
│   │   ├── symbol.py
│   │   └── trace.py
│   ├── ingest/
│   │   ├── walker.py
│   │   ├── chunker.py
│   │   ├── enricher.py
│   │   └── pipeline.py
│   ├── retrieval/
│   │   ├── hybrid.py
│   │   ├── rerank.py
│   │   └── rrf.py
│   ├── storage/
│   │   ├── sqlite.py
│   │   ├── vector.py
│   │   └── graph.py
│   ├── embeddings/
│   │   ├── voyage.py
│   │   └── cache.py
│   └── cli.py
└── tests/
    ├── eval/                  # golden dataset, harness
    └── unit/
```

### 9.5 Configuração no Claude Code

`.mcp.json` na raiz do repo:

```json
{
  "mcpServers": {
    "stropha": {
      "command": "uv",
      "args": ["--directory", "/Users/jonatas/sources/stropha", "run", "mcp-server"],
      "env": {
        "VOYAGE_API_KEY": "${VOYAGE_API_KEY}",
        "STROPHA_INDEX_PATH": "${HOME}/.stropha/index.db"
      }
    }
  }
}
```

Tradeoff: `.mcp.json` versionado significa que o time inteiro herda. Para configs pessoais, usar `.mcp.local.json` (gitignored).

### 9.6 Segurança do MCP

- **Path traversal**: validar todos os paths recebidos contra raiz do repo. Rejeitar `..`.
- **Secret leakage**: scanner (gitleaks/trufflehog) na ingestão. Chunks contendo secrets são marcados e nunca retornados a clientes não-autorizados.
- **Rate limiting**: por cliente (token bucket). Default: 100 req/min.
- **Audit log**: cada tool call logado com timestamp, cliente, args, result_hash.
- **Sandboxing**: para deploy remoto, rodar em container sem network egress (exceto Voyage API).

---

## 10. Avaliação

Sem evaluation, qualquer mudança é palpite. Evaluation é o que separa RAG sério de demo.

### 10.1 Golden dataset

Construir manualmente (50-200 exemplos para começar):

```jsonc
{
  "id": "Q-001",
  "query": "como o sistema decide quando agendar review FSRS?",
  "category": "conceptual",
  "relevant_chunks": [
    "backend/.../FsrsCalculator.java:nextReview",
    "backend/.../StudyFlowCoordinator.java:scheduleNext",
    "docs/architecture/study-flow.md#fsrs-cycle"
  ],
  "graded_relevance": { "...:nextReview": 3, "...:scheduleNext": 2, "...#fsrs-cycle": 3 }
}
```

Categorias balanceadas: conceitual, simbólica, estrutural, multi-hop, ambígua, negativa (deve retornar nada).

### 10.2 Métricas

| Métrica | O que mede | Alvo |
|---|---|---|
| **Recall@k** | % de queries onde ≥1 chunk relevante aparece nos top-k | @10 ≥ 0,85 |
| **Precision@k** | % dos top-k que são relevantes | @5 ≥ 0,60 |
| **MRR** | 1/rank do primeiro relevante | ≥ 0,75 |
| **NDCG@k** | Recall ponderado por posição e relevância graduada | @10 ≥ 0,70 |
| **Latência p50/p95/p99** | Tempo de resposta | p95 ≤ 500 ms |
| **Cobertura** | % de chunks que aparecem em ≥1 query top-50 (detecta dead zones) | > 30% |

### 10.3 Frameworks

- **RAGAS** (https://docs.ragas.io): métricas LLM-as-judge — `faithfulness`, `answer_relevance`, `context_precision`, `context_recall`. Útil quando relevância não é binária.
- **DeepEval**: alternativa com mais integrações pytest.
- **TruLens**: bom para tracing + evaluation acoplados.
- **Phoenix (Arize)**: open-source, ótima UI para análise.
- **ARES**: evaluation com synthetic data generation.

### 10.4 LLM-as-judge

Para queries onde relevance gold é caro de produzir:

- LLM (Sonnet) recebe query + chunk + critério, retorna score 0-3 + justificativa.
- Calibração: rodar em 50 exemplos rotulados manualmente, medir kappa (>0,7 aceitável).
- Custo: ~US$ 0,01 por query × top_k.

### 10.5 A/B testing online

Quando rodar como serviço com múltiplos usuários: split traffic entre configurações (ex.: rerank on vs off), medir engagement (cliques nos resultados, tempo até resolver dúvida).

### 10.6 Regression testing em CI

Toda PR ao código do RAG roda golden set. PR bloqueada se NDCG@10 cair >2 pontos vs baseline.

---

## 11. Observabilidade

### 11.1 Tracing

**OpenTelemetry** como padrão. Spans por etapa:

```
mcp.tool.search_code
  ├─ query.rewrite
  ├─ embedding.encode
  ├─ retrieval.dense
  ├─ retrieval.sparse
  ├─ retrieval.rrf
  ├─ rerank.voyage
  └─ response.pack
```

Backends:

- **Langfuse** (open-source, self-hostable): purpose-built para LLM ops.
- **Phoenix** (Arize, OSS): foco em evaluation + tracing.
- **LangSmith**: SaaS, ótimo se já usa LangChain.
- **Honeycomb / Datadog APM**: APM tradicional, OTel-compatible.

### 11.2 Métricas Prometheus

```
rag_query_total{tool="search_code", status="ok"}
rag_query_duration_seconds{tool, quantile}
rag_chunks_indexed_total
rag_embedding_tokens_total{model}
rag_embedding_cost_usd_total{model}
rag_cache_hit_ratio{cache="embedding"}
rag_rerank_skipped_total{reason="latency_budget"}
```

### 11.3 Logging estruturado

JSON logs via `structlog` (Python) ou `tracing` (Rust). Campos obrigatórios: `request_id`, `tool`, `latency_ms`, `result_count`. Nunca logar query crua se contém path sensível.

### 11.4 Cost dashboard

Diário: tokens embeddados, tokens reranked, USD gasto por modelo. Alertas em > 2x média móvel 7d.

---

## 12. Performance e otimização

### 12.1 Caching multi-nível

| Cache | Key | TTL | Backend |
|---|---|---|---|
| **Embedding cache** | `(content_hash, model)` | infinito (até model change) | SQLite tabela |
| **Query embedding cache** | `(query_text, model)` | 1h | SQLite tabela ou Redis |
| **Rerank cache** | `(query_hash, candidates_hash, model)` | 1h | LRU em RAM |
| **Semantic cache** | embedding da query → resposta | sim. cosine > 0.95 → hit | sqlite-vec auxiliar |

Semantic cache de queries é poderoso para usuários humanos (perguntas similares retornam mesmo set). Para Claude Code, menos útil (cada invocação tem contexto novo).

### 12.2 Batching

- **Embeddings**: agrupar até 128 chunks por chamada à API Voyage (limite do provedor). Reduz overhead de RTT.
- **Reranks**: a API aceita 1 query × N docs por chamada. Sempre enviar top 50 de uma vez.

### 12.3 Connection pooling

Para vector stores remotos (Qdrant), pool de conexões HTTP/2. Para SQLite, abrir conexão por query é OK até ~100 QPS; acima, usar pool com `apsw`.

### 12.4 Prompt caching (Anthropic)

Quando o servidor MCP fizer chamadas LLM (HyDE, summary generation), usar prompt caching:

- System prompt + few-shot examples vão em block cacheado.
- Cache hit reduz custo input em 90% e latência em ~50%.
- TTL: 5 min default; 1h se contratado.

### 12.5 Quantização e precisão

- Vetores em `float32` por padrão.
- Quantizar para `int8` quando índice > 1 GB (perda <1% NDCG).
- Binary quantization apenas se índice > 10 GB.

### 12.6 Profiling

- `py-spy` para hot-paths Python.
- `tracy` para Rust se reescrever.
- SQL `EXPLAIN QUERY PLAN` em queries lentas.

---

## 13. Segurança

### 13.1 Detecção de secrets

Antes de embeddar, rodar:

- `gitleaks` ou `trufflehog` no chunk.
- Regex próprias para padrões internos (chaves Google OAuth, JWT tokens).

Chunks com secrets:

- Marcar `has_secret: true`.
- Mascarar (`****`) o secret no texto persistido.
- Excluir de qualquer response a clientes sem flag `--allow-secrets`.

### 13.2 RBAC (multi-usuário)

Se servir vários usuários:

- Token JWT por usuário com claims (`repo_access`, `paths_allowed`).
- Server filtra results por `paths_allowed`.

### 13.3 Auditoria

Append-only log: `(timestamp, user, tool, args_hash, result_hash, latency)`. Retenção 90d. Útil para investigar incidentes.

### 13.4 Supply chain

- Lock files committed (`uv.lock`).
- Verificar checksums de modelos baixados.
- Renovate/Dependabot para CVEs.

---

## 14. Deployment

### 14.1 Topologias

#### Local (desenvolvedor individual)

```
┌────────────────────┐
│ Claude Code        │
│  └─ stdio ──► mcp-server (Python local)
│                  └─ index.db (~/.stropha/)
│                  └─ Voyage API (cloud)
└────────────────────┘
```

Custo: API embeddings/rerank apenas. Setup: `uv tool install stropha`.

#### Self-hosted (time)

```
                       ┌─ Qdrant (Docker)
GitHub Actions ─indexa─┤
                       └─ Postgres (chunks + graph)
                              ▲
                              │ HTTP+SSE
                              │
              dev ─► Claude Code ─► mcp-server (deployed)
```

Single VPS suficiente para times <50 devs. Helm chart se K8s mandatório.

#### Cloud (alta disponibilidade)

- Embedding: Voyage API.
- Vector store: Qdrant Cloud ou Turbopuffer.
- MCP server: AWS Lambda (cold start aceitável para uso esporádico) ou Fargate.
- Storage: S3 para snapshots.

### 14.2 CI/CD para o índice

Pipeline GitHub Actions:

```yaml
on:
  push:
    branches: [develop, main]

jobs:
  reindex:
    runs-on: ubuntu-latest
    steps:
      - checkout
      - uv-install
      - run: uv run stropha reindex --incremental --since=$LAST_INDEXED_SHA
      - upload index.db artifact
      - run evaluation gate (NDCG must not drop > 2 pts)
```

### 14.3 Release artifact

Distribuir como:

- PyPI package (`pip install stropha`).
- Single binary via `pyinstaller` ou `shiv` (~30 MB) para devs não-Python.
- Docker image multi-arch (`linux/amd64`, `linux/arm64`, `darwin/arm64`).

---

## 15. Stack proposto (recomendação concreta)

Decisões para a fase 1 deste projeto:

| Componente | Escolha | Justificativa |
|---|---|---|
| Linguagem servidor | Python 3.12 + `uv` | Ecossistema RAG dominante; iteração rápida. |
| MCP SDK | `mcp` (Anthropic) | Único oficial. |
| Parsing | `tree-sitter-language-pack` | Cobre Java, Vue, TS, Gherkin, YAML, MD num único pacote. |
| Chunking | Custom (semântico hierárquico) | Frameworks genéricos não conhecem Vue SFC nem Gherkin. |
| Embeddings densos | Voyage `voyage-code-3` (dim 512 Matryoshka) | SOTA em código; custo baixo. |
| Embeddings esparsos | SQLite FTS5 (BM25) | Zero infra, suficiente. |
| Vector store | sqlite-vec | Single-file; portabilidade total. |
| Symbol graph | SQLite (mesmo arquivo) | Joins SQL, transacional. |
| Reranker | Voyage `rerank-2.5` | Mesma família, billing único. |
| Query understanding | Heurística + HyDE opcional | Não complicar antes de evaluation justificar. |
| Contextual retrieval | Sim (Anthropic technique) | ROI comprovado, custo aceitável. |
| Pipeline | LlamaIndex `IngestionPipeline` (cacheada) | Reaproveita transformations, fácil swap. |
| Observability | OpenTelemetry → Langfuse local | Open-source, self-hosted, leve. |
| Evaluation | RAGAS + golden set custom | Métricas tradicionais + LLM-as-judge. |
| Deployment | `uv tool install` local; opcional Docker | Atende uso individual primário. |

Custo mensal estimado (uso individual ativo):

| Item | Estimativa |
|---|---|
| Voyage embeddings (incremental) | US$ 0,50 |
| Voyage rerank (~500 queries/mês) | US$ 1,50 |
| Anthropic Haiku (summaries + HyDE) | US$ 1,00 |
| **Total** | **~US$ 3/mês** |

Bootstrap inicial (full reindex Mimoria, ~5K chunks): ~US$ 2 one-shot.

---

## 16. Roadmap em fases

### Phase 0 — Spike (1 dia) ✓
- [x] Setup repo `stropha` com `uv`.
- [x] Walker + chunker dummy (split por arquivo).
- [x] Embedding via Voyage API (com fallback local fastembed quando sem chave).
- [x] sqlite-vec storage.
- [x] CLI `index` / `search` / `stats`.
- [x] Teste manual: "onde está o FSRS calculator?".

**Critério de saída**: query semântica retorna chunk certo no top-3 — ✓.

### Phase 1 — MVP MCP (3-5 dias) ✓
- [x] Tree-sitter parsing para Java, TypeScript, JavaScript, Python, Rust, Go, Kotlin via `tree-sitter-language-pack.process()`.
- [x] Chunkers custom para Vue SFC (script/template/style), Markdown (heading split), Gherkin (feature/scenario).
- [x] Chunking semântico hierárquico (skeleton de classe com nome qualificado + lista de membros; método como chunk filho com `parent_chunk_id`).
- [x] Hybrid search com **três streams** fundidas via RRF (k=60): dense (sqlite-vec) + sparse (FTS5 BM25 com expansão CamelCase + tokens de path + symbol) + symbol-token lookup (query routing §6.3.5).
- [x] Servidor MCP (`stropha-mcp`) sobre stdio. Tools: `search_code`, `get_symbol`, `get_file_outline`. Resource: `stropha://stats`.
- [x] Template `.mcp.example.json` para Claude Code / Cursor.
- [x] Freshness skip por chunk (re-rodar `index` num repo estável é quase instantâneo).
- [ ] **Diferido** Reindexação incremental via `git diff` — `index --rebuild` cobre o caso de uso para Phase 1; hooks `post-commit` e soft-index em RAM vão para Phase 3.

**Critério de saída**: Claude Code usa o RAG sem ser instruído. Critério objetivo: queries simbólicas/conceituais retornam o chunk certo no top-3. Validado com `where is the FSRS calculator` (rank 3), `calculateNewStability` (rank 1), `EnrollmentRepository findByUsername` (rank 1) usando o embedder local de fallback.

### Phase 2 — Qualidade (1 semana)
- [ ] Reranking Voyage rerank-2.5.
- [ ] Contextual retrieval (Anthropic).
- [ ] Summary generation via Haiku (cached).
- [ ] Symbol graph com `find_callers`, `find_tests_for`.
- [ ] Golden dataset inicial (50 queries).
- [ ] Evaluation harness com RAGAS.
- [ ] Tracing OpenTelemetry → Langfuse local.

**Critério de saída**: Recall@10 ≥ 0,85 no golden set.

### Phase 3 — Sofisticação (2 semanas)
- [ ] HyDE opcional, query routing.
- [ ] Recursive retrieval / auto-merging.
- [ ] Tool `trace_feature` (Gherkin → step → método).
- [ ] Glossário de domínio embeddado.
- [ ] Soft index para working tree (file watcher).
- [ ] Cache semântico de queries.
- [ ] Cost dashboard.

### Phase 4 — Escala / extensão
- [x] **Foundation multi-repo**: schema v2 adiciona tabela `repos` + coluna `chunks.repo_id` + tool MCP `list_repos`. Cada `SearchHit` carrega `repo` com URL, branch e HEAD para o cliente fazer `git clone`. Normalização de URL deduplica SSH/HTTPS do mesmo repo; auth tokens são removidos antes da persistência. Detalhes em `src/stropha/ingest/git_meta.py`.
- [x] **Multi-repo indexer UX**: `stropha index --repo A --repo B` (flag `-r` repetível) percorre cada repo sequencialmente compartilhando a mesma Storage e Embedder. `chunk_id` é namespaced por `normalized_key` (via `make_chunk_id(..., repo_key=...)`) — repos diferentes com arquivos idênticos não colidem. `IndexPipeline` aceita `repos: list[Path]`; `IndexStats` agrega contadores por repo (`stats.repos: list[RepoStats]`). `--rebuild` limpa chunks mas preserva tabela `repos`, mantendo FKs estáveis entre rebuilds.
- [ ] Manifest YAML para listas declarativas de repos.
- [ ] Auto-discovery de nested `.git` durante o walk (monorepos com submódulos / vendored deps).
- [ ] Indexação de dependências externas (Quarkus, Vue) on-demand.
- [ ] Modelo de embedding self-hosted (bge-m3) como fallback offline.
- [ ] Deploy remoto com OAuth 2.1.
- [ ] Multi-tenant com RBAC.
- [ ] Web UI para evaluation/debug.

---

## 17. Riscos e mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Custo de embedding explode com refactors massivos | Médio | Médio | Cache por content_hash; alerta de budget. |
| Voyage API down | Baixo | Alto | Fallback para cache + BM25 puro; opção self-host bge-m3. |
| Chunks vazam secrets | Baixo | Crítico | Scanner gitleaks na pipeline; flag `has_secret`. |
| Latência > 1s em queries complexas | Médio | Médio | Latency budget; rerank skip quando exceder. |
| Drift entre código e índice | Alto | Médio | Hooks Git; soft index; reconciliation noturna. |
| Claude Code não usa o tool quando deveria | Alto | Baixo | Documentar no CLAUDE.md quando preferir RAG vs grep. |
| Vendor lock-in Voyage | Médio | Baixo | Embedding model é hot-swappable; abstração `Embedder` interface. |
| Avaliação enviesada (golden set fraco) | Alto | Alto | Crowdsource queries reais via logs; revisão peer trimestral. |

---

## 18. Anti-padrões observados em outras implementações

Lições de RAGs públicos que falham:

1. **Chunking por tamanho fixo** sem respeitar AST → recall ruim em código.
2. **Só dense, sem BM25** → falha em busca por símbolo exato.
3. **Sem rerank** → top-3 frequentemente irrelevante apesar de top-50 relevante.
4. **Reindex full a cada commit** → custo proibitivo, latência inaceitável.
5. **Tool monolítico `query(text)`** → o LLM não consegue compor; melhor expor primitives.
6. **Retornar chunks gigantes** → estoura context window do cliente.
7. **Sem evaluation** → não dá pra saber se mudança ajudou ou piorou.
8. **Sem versionamento de embedding model** → migração quebra produção.
9. **HyDE em tudo** → latência dobra sem ganho de recall na maioria das queries.
10. **GraphRAG aplicado a código** → desperdiça LLM extraindo grafos que AST entrega de graça.

---

## 19. Referências

### Papers fundacionais
- Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (paper original RAG).
- Karpukhin et al., 2020 — *Dense Passage Retrieval for Open-Domain QA* (DPR).
- Khattab & Zaharia, 2020 — *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction*.
- Cormack et al., 2009 — *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*.

### Técnicas modernas (2023-2025)
- Anthropic, 2024 — *Introducing Contextual Retrieval* (https://www.anthropic.com/news/contextual-retrieval).
- Edge et al. (Microsoft), 2024 — *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*.
- Gao et al., 2023 — *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE).
- Asai et al., 2023 — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*.
- Yan et al., 2024 — *Corrective Retrieval Augmented Generation* (CRAG).
- Jina AI, 2024 — *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*.
- Kusupati et al., 2022 — *Matryoshka Representation Learning*.
- Guo et al., 2024 — *LightRAG: Simple and Fast Retrieval-Augmented Generation*.

### Evaluation
- Es et al., 2023 — *RAGAS: Automated Evaluation of Retrieval Augmented Generation*.
- Saad-Falcon et al., 2023 — *ARES: An Automated Evaluation Framework for RAG Systems*.

### Standards e specs
- Model Context Protocol — https://modelcontextprotocol.io/specification
- OpenTelemetry semantic conventions for LLM — https://opentelemetry.io/docs/specs/semconv/gen-ai/

### Ferramentas referenciadas
- Tree-sitter — https://tree-sitter.github.io
- LlamaIndex — https://docs.llamaindex.ai
- Haystack — https://haystack.deepset.ai
- sqlite-vec — https://github.com/asg017/sqlite-vec
- LanceDB — https://lancedb.github.io
- Qdrant — https://qdrant.tech
- Voyage AI — https://docs.voyageai.com
- Langfuse — https://langfuse.com
- Phoenix (Arize) — https://docs.arize.com/phoenix
- gitleaks — https://github.com/gitleaks/gitleaks

---

## 20. Glossário rápido

| Termo | Definição |
|---|---|
| **AST** | Abstract Syntax Tree |
| **ANN** | Approximate Nearest Neighbor |
| **BM25** | Best Match 25 — função de ranking lexical clássica |
| **Chunk** | Unidade indexável (trecho com metadados) |
| **Cross-encoder** | Modelo que processa query+doc juntos (rerank) |
| **HNSW** | Hierarchical Navigable Small World — índice ANN |
| **HyDE** | Hypothetical Document Embeddings |
| **MCP** | Model Context Protocol |
| **MRR** | Mean Reciprocal Rank |
| **NDCG** | Normalized Discounted Cumulative Gain |
| **RAG** | Retrieval-Augmented Generation |
| **Recall@k** | % de queries com ≥1 relevante nos top-k |
| **RRF** | Reciprocal Rank Fusion |
| **SOTA** | State Of The Art |
| **SPLADE** | Sparse Lexical and Expansion model |

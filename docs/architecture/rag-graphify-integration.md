# RAG ↔ Graphify Integration & Post-Commit Automation

> **Status:** Proposed
> **Versão:** 1.0
> **Data:** 2026-05-14
> **Audiência:** maintainers do projeto `mimoria-rag` e agentes LLM operando-o.
> **Documentos relacionados:**
> - `docs/architecture/rag-system.md` (spec mestre — §3.5 symbol graph, §6.3.5 query routing, §8 atualização incremental, §9.2 MCP tools)
> - `CLAUDE.md` (estado vivo da implementação)

---

## 0. TL;DR

Integrar o output do [graphify](https://github.com/safishamsi/graphify) (`graphify-out/graph.json`) ao `mimoria-rag` para entregar a Phase 2 §3.5 do spec mestre (symbol graph + `find_callers` / `find_tests_for` / `trace_feature`) sem reimplementar 5k linhas de extração AST + LLM. Acoplamento opcional: se o arquivo não existir, os tools simplesmente não aparecem no `tools/list` do MCP server.

A integração tem **três níveis sobrepostos** (taxonomia em §4.7):
1. **Mirror relacional (L1)** — tabelas SQLite com índices B-tree para traversal estrutural via tools dedicados (`find_callers`, etc.).
2. **Augmentation lexical (L2, default)** — labels de nó / community entram no documento FTS5 dos chunks correspondentes, melhorando recall do `search_code` existente sem custo recorrente.
3. *(opt-in)* **Indexação semântica (L3)** — embeddings dos node labels e community summaries entram numa virtual table sqlite-vec separada, fundida via 4ª stream RRF. Disponível para usuários sem restrição de custo de embedding.

Como segundo eixo: um git hook **post-commit em background detached** que mantém ambos (graphify graph + RAG SQLite index) atualizados após cada commit, sem bloquear o terminal. Instalável via `mimoria-rag hook install --target <repo>`. Skippa rebases e respeita kill-switch via env var.

Custo estimado de implementação: **4 a 6 dias de trabalho**, distribuído em quatro entregas independentes (L2 incluído por default; L3 opt-in adiciona ~0.5 dia se ativado). Zero custo recorrente de API quando rodando com embedder local — o que move o cálculo de tradeoff em favor de níveis mais profundos de integração (ver ADR-008).

---

## 1. Contexto e motivação

### 1.1 Capacidades hoje (Phase 1 — implementado)

| Capacidade | Estado |
|---|---|
| Tree-sitter AST chunking (Java/TS/JS/Py/Rust/Go/Kotlin) | ✓ |
| Custom chunkers (Vue/Markdown/Gherkin) | ✓ |
| Hybrid retrieval (dense + BM25 + symbol-token) com RRF | ✓ |
| Class skeleton chunks com nome qualificado + lista de membros | ✓ |
| MCP server stdio (`search_code`, `get_symbol`, `get_file_outline`) | ✓ |
| Chunk-level freshness skip (no-op re-run) | ✓ |

### 1.2 Lacuna que justifica este trabalho

Queries estruturais que o spec mestre §6.3.5 categoriza como "Estrutural" ainda não têm resposta direta:

- **"Quem chama `submitAnswer`?"** — hoje cai em `search_code` e depende de BM25 achar referências textuais. Sem garantia.
- **"Quais testes cobrem `FsrsCalculator`?"** — heurística por sufixo de nome funciona para 70% dos casos; o restante (testes parametrizados, helpers, BDD steps) escapa.
- **"Trace do scenario X até o método tocado"** — não tem como sem grafo de chamadas.

A spec mestre §3.5 prevê construir um symbol graph (`nodes(id, kind, name, ...)` + `edges(from_id, to_id, kind, weight)`) na Phase 2. O esforço estimado é alto porque envolve resolução de nomes cross-file, análise de imports, construção do grafo de chamadas — território de ferramentas como JavaParser, typescript-eslint, ou tree-sitter-graph.

### 1.3 Por que graphify

O graphify (`pip install graphifyy`, MIT, ativo) já produz exatamente o grafo necessário:

- **7819 nós × 14477 arestas** extraídos do Mimoria em ~3 minutos
- **68% das arestas são `EXTRACTED`** (derivadas deterministicamente do AST via tree-sitter)
- Tipos de aresta cobrem: `calls`, `method`, `contains`, `imports`, `extends`, `implements`, `references`, `rationale_for` (esta inferida via LLM — liga código a doc/spec que explica)
- Output em JSON estável (formato NetworkX node-link) + cache incremental sob `graphify-out/cache/ast/`
- Roda incrementalmente sem LLM via `graphify update` (custo zero por execução)
- Já tem MCP server próprio (`graphify ... --mcp`), mas focado em graph traversal, não em retrieval

A integração é **complementar, não competitiva**: o graphify resolve a camada estrutural; o mimoria-rag continua resolvendo a camada semântica + lexical.

### 1.4 Posicionamento na roadmap original

Esta proposta **antecipa Phase 2 §3.5** (symbol graph) integrando ferramenta externa em vez de construir do zero. A spec mestre §3.5 inclusive já antecipava esta possibilidade:

> "GraphRAG (Microsoft, 2024) e LightRAG (2024) extraem grafos textualmente via LLM — desnecessário para código onde grafos são derivados deterministicamente do AST."

O graphify é exatamente esse caso: AST-deterministic, sem LLM no caminho crítico.

---

## 2. Objetivos e não-objetivos

### 2.1 Objetivos (MUST)

1. **OBJ-1**: Carregar `graphify-out/graph.json` em tabelas SQLite do índice, joinable com `chunks` por `(rel_path, start_line)`.
2. **OBJ-2**: Expor 3 novos tools MCP — `find_callers`, `find_related`, `get_community` — condicionais à presença do grafo.
3. **OBJ-3**: Instalar git post-commit hook que atualiza graphify + RAG em background, via `mimoria-rag hook install --target <repo>`.
4. **OBJ-4**: O hook NUNCA bloqueia o commit; falhas vão para log e métricas, não para o terminal do usuário.
5. **OBJ-5**: A integração é estritamente opcional. `mimoria-rag` continua funcionando sem o graphify.

### 2.2 Objetivos secundários (SHOULD)

1. **OBJ-6**: Tool opcional `find_rationale` exposto quando edges `rationale_for` existem (>0 no grafo).
2. **OBJ-7**: CLI `mimoria-rag hook status` para inspeção e `mimoria-rag hook uninstall` para remoção limpa.
3. **OBJ-8**: Detectar `core.hooksPath` para coexistir com husky/lefthook caso adotados no futuro.

### 2.3 Não-objetivos (explicitamente fora)

- **NÃO**: Reimplementar extração de grafo. Se graphify quebrar, removemos a integração; não viramos mantenedores do graphify.
- **NÃO**: Suporte a `pre-commit` síncrono. Justificativa em §17 (ADR-003).
- **NÃO**: Auto-bootstrap (rodar `graphify .` automaticamente quando `graphify-out/` não existe). Usuário precisa fazer o bootstrap explicitamente — graphify sem flag custa tokens LLM.
- **NÃO**: Re-indexação automática de mudanças em `graphify-out/` (file watcher). Hook post-commit cobre o cenário pretendido.
- **NÃO**: Suporte multi-repo no `hook install` v1. Um `--target` por instalação. Multi-repo via instalações repetidas.
- **NÃO**: Empacotar graphify como dependência transitiva. Usuário instala via `pipx install graphifyy` separadamente; nosso CLI apenas detecta e usa.

### 2.4 Critérios de sucesso

| Métrica | Alvo |
|---|---|
| `find_callers("StudyService.submitAnswer")` retorna ≥1 chamador correto | 100% (caso golden) |
| `find_related("FsrsCalculator", depth=1)` cobre ≥80% da Community 18 do graphify | manual eval |
| Latência p95 de `find_callers` | ≤ 30 ms (SQL local) |
| Latência adicional do hook no commit | ≤ 100 ms (apenas fork/disown) |
| Tempo de execução do hook em background (commit típico de 5 arquivos) | ≤ 60 s |
| Falsos positivos em "graphify-out missing" (deveria estar presente) | 0 após bootstrap |
| Falsos negativos em `find_callers` (chamada existe mas não retornada) | < 5% — herdado da precisão do graphify |

---

## 3. Arquitetura de alto nível

### 3.1 Visão C4 — Contexto

```
┌──────────────────────────────────────────────────────────────────────┐
│                            DEV / AGENT                               │
│                                                                      │
│  $ git commit -m "…"           (no Mimoria)                          │
│  $ mimoria-rag hook install    (uma vez)                             │
│  Claude Code → MCP tools       (durante sessão)                      │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│      MIMORIA REPO    │ │   GRAPHIFY CLI   │ │   MIMORIA-RAG CLI    │
│  source-of-truth     │ │   /usr/local/.../│ │  /Users/jonatas/sour │
│                      │ │   graphify       │ │  ces/rag/.venv/...   │
│ .git/hooks/          │ │                  │ │                      │
│   post-commit ◄──────┼─┤ update <path>    │ │ index                │
│                      │ │ extract <path>   │ │ hook install/status  │
│ graphify-out/        │ │                  │ │ search / stats       │
│   graph.json    ◄────┘ └─────┬────────────┘ │                      │
│   manifest.json              │              │ MCP stdio server     │
│   GRAPH_REPORT.md            │              │   search_code        │
│                              │              │   get_symbol         │
└──────────────────────────────┼──────────────│   get_file_outline   │
                               │              │   find_callers   ◄───┼─ NEW
                               └──────────────│   find_related   ◄───┼─ NEW
                                              │   get_community  ◄───┼─ NEW
                                              │   find_rationale ◄───┼─ NEW (opt)
                                              │                      │
                                              │  ~/.mimoria-rag/     │
                                              │    index.db          │
                                              └──────────────────────┘
```

### 3.2 Visão C4 — Container (mimoria-rag interno)

```
┌────────────────────────────── mimoria-rag (Python) ─────────────────────────────┐
│                                                                                 │
│  cli.py ──┬──► commands: index, search, stats, hook (install/status/uninstall)  │
│           │                                                                     │
│           └──► server.py ──► FastMCP                                            │
│                                                                                 │
│  ingest/                                                                        │
│    walker.py                                                                    │
│    chunker.py + chunkers/*                                                      │
│    pipeline.py     ──► IndexPipeline                                            │
│    graphify_loader.py  ◄── NEW: lê graphify-out/graph.json, popula SQLite      │
│                                                                                 │
│  storage/                                                                       │
│    sqlite.py ──► tabelas existentes (chunks, vec_chunks, fts_chunks, meta)      │
│              ──► tabelas novas (graph_nodes, graph_edges)  ◄── NEW             │
│                                                                                 │
│  retrieval/                                                                     │
│    search.py    (3 streams existentes)                                          │
│    rrf.py                                                                       │
│    graph.py    ◄── NEW: traversal helpers (callers, neighbors, community)      │
│                                                                                 │
│  tools/                                                                         │
│    (vazio até agora) ──► hook_install.py  ◄── NEW                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Fluxos chave

**Fluxo A — refresh via hook (background)**

```
git commit ─►  .git/hooks/post-commit
                  │
                  ├─ skip if rebase/merge/cherry-pick/empty
                  ├─ skip if MIMORIA_RAG_HOOK_SKIP=1
                  ├─ skip if toplevel ≠ RAG_TARGET_REPO
                  │
                  └─ nohup detached (commit returns) ──►
                         flock /tmp/mimoria-rag-hook.lock
                            │
                            ├─ graphify update <toplevel> --no-cluster
                            │   └─ refreshes graphify-out/graph.json (AST only)
                            │
                            ├─ mimoria-rag index
                            │   └─ walks files, skip-fresh handles unchanged chunks
                            │   └─ if graphify-out/graph.json modified since last load,
                            │      reload into SQLite (graph_nodes, graph_edges)
                            │
                            └─ log → ~/.cache/mimoria-rag-hook.log
```

**Fluxo B — agent invoca `find_callers`**

```
Claude Code ──tool call──► server.py:find_callers(symbol)
                              │
                              ├─ resolve symbol → node_id via graph_nodes table
                              ├─ SELECT * FROM graph_edges
                              │  WHERE target=? AND relation='calls' AND confidence='EXTRACTED'
                              ├─ LEFT JOIN chunks ON (source_file, source_location)
                              │  to fetch snippets when available
                              └─► list[CallerResult]  (path, line, symbol, snippet)
```

---

## 4. Design — Parte A: Ingestão do grafo

### 4.1 Schema SQLite (extensão do `storage/sqlite.py:Storage.migrate`)

```sql
-- Novo: graph_nodes
CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id          TEXT PRIMARY KEY,         -- graphify's "id" field
    label            TEXT NOT NULL,            -- human-readable name
    norm_label       TEXT NOT NULL,            -- lowercased, used for fuzzy match
    file_type        TEXT,                     -- code | document | paper | image | video
    source_file      TEXT,                     -- repo-relative path
    source_location  TEXT,                     -- "L<line>"
    community        INTEGER,                  -- Leiden community id
    loaded_at        TEXT NOT NULL             -- ISO timestamp
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_label      ON graph_nodes(label);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_norm_label ON graph_nodes(norm_label);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_file       ON graph_nodes(source_file);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_community  ON graph_nodes(community);

-- Novo: graph_edges
CREATE TABLE IF NOT EXISTS graph_edges (
    source        TEXT NOT NULL,
    target        TEXT NOT NULL,
    relation      TEXT NOT NULL,                -- calls | method | contains | imports | …
    confidence    TEXT NOT NULL,                -- EXTRACTED | INFERRED | AMBIGUOUS
    weight        REAL DEFAULT 1.0,
    source_file   TEXT,                          -- for join-back to chunks
    source_loc    TEXT,
    FOREIGN KEY (source) REFERENCES graph_nodes(node_id),
    FOREIGN KEY (target) REFERENCES graph_nodes(node_id)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_source     ON graph_edges(source);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target     ON graph_edges(target);
CREATE INDEX IF NOT EXISTS idx_graph_edges_relation   ON graph_edges(relation);
CREATE INDEX IF NOT EXISTS idx_graph_edges_confidence ON graph_edges(confidence);
```

**Por que tabelas separadas** (em vez de embutir em `chunks`):
- O grafo do graphify tem nós que NÃO viraram chunks nossos (nós de documentos, de URLs em `raw/`, de imagens).
- Edges são N:M; embutir como blob de JSON em chunks performaria mal em traversal.
- Tabelas separadas permitem TRUNCATE + reload em < 1s sem tocar o índice vetorial.

**Garantias transacionais**: o reload é um BEGIN ... COMMIT atômico. Falhas no meio do load mantêm a versão anterior do grafo.

### 4.2 Loader (`ingest/graphify_loader.py`)

```python
class GraphifyLoader:
    """Lê graphify-out/graph.json e popula graph_nodes + graph_edges."""

    def __init__(self, storage: Storage, target_repo: Path) -> None: ...

    def find_graph_file(self) -> Path | None:
        """Localiza graph.json respeitando $GRAPHIFY_OUT (default: graphify-out/)."""

    def is_stale(self) -> bool:
        """True se graph.json mais novo que o último carregamento (compare via meta key)."""

    def load(self, *, confidence_filter: set[str] | None = frozenset({"EXTRACTED"})) -> LoadStats:
        """
        Carrega o grafo. Por default só edges EXTRACTED entram (alta precisão).
        INFERRED/AMBIGUOUS são deixadas no JSON original mas não viram queryáveis
        — disponíveis via leitura raw quando o agent pedir explicitamente.

        Returns:
            LoadStats(nodes_loaded, edges_loaded, edges_filtered, source_commit)
        """

    def metadata(self) -> dict[str, Any]:
        """Devolve built_at_commit, node_count, edge_count, community_count, etc."""
```

A função `load()` é idempotente: replay seguro. Implementação:

```python
def load(self, confidence_filter=...):
    payload = json.loads(graph_path.read_text())  # ~8 MB no Mimoria

    with self._storage.transaction():
        # 1. Wipe + reload (faster than diff for graphs < 1M edges)
        self._storage._conn.execute("DELETE FROM graph_edges")
        self._storage._conn.execute("DELETE FROM graph_nodes")

        # 2. Bulk insert
        self._storage._conn.executemany(
            "INSERT INTO graph_nodes(...) VALUES(...)",
            [(n["id"], n["label"], ...) for n in payload["nodes"]],
        )

        # 3. Edges, filtered
        edges = [
            (e["source"], e["target"], e["relation"], e["confidence"], ...)
            for e in payload["links"]
            if confidence_filter is None or e["confidence"] in confidence_filter
        ]
        self._storage._conn.executemany(
            "INSERT INTO graph_edges(...) VALUES(...)",
            edges,
        )

        # 4. Bookkeeping
        self._storage.set_meta(
            "graphify_loaded_commit",
            payload.get("built_at_commit", ""),
        )
        self._storage.set_meta(
            "graphify_loaded_at",
            datetime.now(UTC).isoformat(),
        )

    return LoadStats(...)
```

### 4.3 Política de confiança (default)

| Confidence | Default | Justificativa |
|---|---|---|
| `EXTRACTED` | ✓ carrega | Derivado AST; baixo falso-positivo |
| `INFERRED` | ✗ não carrega | LLM-generated; 32% das edges; ruído alto (ex.: `StudyService references Vue Unit Test Plan`) |
| `AMBIGUOUS` | ✗ não carrega | Reservada pelo graphify; raríssima |

Configurável via env var:

```
RAG_GRAPH_CONFIDENCE=EXTRACTED         # default
RAG_GRAPH_CONFIDENCE=EXTRACTED,INFERRED  # opt-in noise
RAG_GRAPH_CONFIDENCE=*                 # tudo, inclusive AMBIGUOUS
```

### 4.4 Detecção de staleness

Três fontes de truth:

1. **`meta.graphify_loaded_at`** — quando o loader rodou pela última vez
2. **`graph.json` mtime** — quando graphify atualizou o arquivo
3. **`payload.built_at_commit`** — qual commit gerou o grafo

O loader é stale quando `graph.json` mtime > `meta.graphify_loaded_at`. O `IndexPipeline.run()` chama `loader.load()` automaticamente se stale.

O agent vê via `rag://stats`:

```jsonc
{
  "graph": {
    "loaded_commit": "2d22f0aa",
    "head_commit":   "abc12345",  // git rev-parse HEAD do target
    "is_stale":      true,
    "nodes":         7819,
    "edges":         9879,        // só EXTRACTED após filtro
    "edges_filtered": 4598,
    "loaded_at":     "2026-05-14T11:50:45Z"
  }
}
```

### 4.5 Augmentation FTS5 (L2 — default)

Após o `load()` mirrorar nodes + edges em SQLite, um passo subsequente reescreve o documento FTS5 dos chunks correspondentes com tokens extras vindos do grafo:

- Para cada `node` com `source_file` que casa com um `chunk.rel_path` E `source_location` no intervalo `[chunk.start_line, chunk.end_line]`:
  - Adiciona `node.label` e `node.norm_label` ao documento FTS desse chunk
  - Adiciona o **community label** (top-K terms da comunidade Leiden) se computável
- Edges não entram no FTS5 (cardinalidade alta, baixo signal lexical individual)

**Implementação** (`storage/sqlite.py:Storage.augment_fts_with_graph`):

```python
def augment_fts_with_graph(self) -> int:
    """For every chunk overlapping a graph node, append node + community
    labels to the FTS5 document. Returns count of chunks augmented."""
    sql = """
        WITH chunk_nodes AS (
          SELECT c.id AS chunk_rowid,
                 GROUP_CONCAT(n.label || ' ' || COALESCE(n.norm_label,''), ' ') AS extras
          FROM chunks c
          JOIN graph_nodes n
            ON c.rel_path = n.source_file
           AND CAST(SUBSTR(n.source_location, 2) AS INTEGER)
               BETWEEN c.start_line AND c.end_line
          GROUP BY c.id
        )
        UPDATE OR IGNORE fts_chunks
        SET content = (SELECT content FROM chunks WHERE id = fts_chunks.rowid)
                      || char(10) || (SELECT extras FROM chunk_nodes WHERE chunk_rowid = fts_chunks.rowid)
        WHERE rowid IN (SELECT chunk_rowid FROM chunk_nodes);
    """
    cur = self._conn.execute(sql)
    return cur.rowcount
```

**Custo**: O(chunks_with_node_match) UPDATE no FTS5; tipicamente ~3-5 minutos no Mimoria (dominado pela rewrite do FTS5 inverted index, não pela query). Roda apenas quando o grafo é (re)carregado, não a cada index.

**Por que vale**: queries que mencionam "FsrsCalculator" ou termos da comunidade (ex.: "review", "stability", "rating" de Community 18) ganham match em chunks que conteriam só a definição mas não esses tokens explicitamente.

**Toggle**: `RAG_GRAPH_FTS_AUGMENT=1` (default `1`). Setando `0`, pula esta etapa — útil quando o grafo está corrompido ou o usuário quer comparar A/B.

### 4.6 Indexação semântica (L3 — opt-in)

Quando `RAG_GRAPH_VEC_AUGMENT=1`, o loader também:

1. **Embebe cada `node.label`** (ou `label + community summary` quando o node é hub de comunidade) usando o `Embedder` ativo
2. **Persiste em virtual table separada** `vec_graph_nodes` dimensionada pelo embedder atual
3. **Adiciona uma 4ª stream** ao `SearchEngine.search()`: dense (chunks) + sparse (BM25) + symbol-token + **graph-vec**
4. RRF funde as quatro streams igualmente (k=60)

**Schema adicional**:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS vec_graph_nodes USING vec0(
    embedding float[<DIM>]
);
-- vec_graph_nodes.rowid → graph_nodes.rowid

CREATE TABLE IF NOT EXISTS graph_node_embeddings (
    node_rowid       INTEGER PRIMARY KEY,
    embedding_model  TEXT NOT NULL,
    embedding_dim    INTEGER NOT NULL,
    label_hash       TEXT NOT NULL,           -- detect stale embeddings
    embedded_at      TEXT NOT NULL
);
```

Skip-fresh check análogo ao dos chunks: se `(node_rowid, embedding_model, label_hash)` já existe, não re-embebe.

**Custo (local-only)**:
- Mimoria: ~7800 nodes × ~50 chars/label × bge-small @ 384-dim ≈ **~30 segundos one-shot** em CPU aarch64
- Storage: ~30 MB extras na DB
- Latência adicional por search: +~5 ms (uma chamada vec_query extra antes do RRF)

**Justificativa pra opt-in (não default)**:
- Complica composição de fusion (4 streams = mais chance de uma stream ruidosa contaminar top-K)
- Quality real depende fortemente de quão semanticamente significativos são os labels de comunidade do graphify
- Recomendado validar com golden dataset antes de ativar

### 4.7 Níveis de indexação — taxonomia explícita

Para evitar ambiguidade, esta spec adota a seguinte nomenclatura interna:

| Nível | Descrição | Default? | Toggle |
|---|---|---|---|
| **L0** | Consumo puro (parse JSON on-demand) | ✗ rejeitado | n/a — inviável performance |
| **L1** | Mirror relacional + B-tree indexes | ✓ implícito | sempre quando grafo presente |
| **L2** | L1 + augmentation FTS5 com node/community labels | ✓ default | `RAG_GRAPH_FTS_AUGMENT=0` desliga |
| **L3** | L2 + embeddings dos nodes (4ª stream RRF) | ✗ opt-in | `RAG_GRAPH_VEC_AUGMENT=1` ativa |
| **L4** | L3 + ranking signals adicionais (centralidade, community boost) | ✗ futuro | n/a — Phase 3+ |

Os níveis são **acumulativos**: L3 implica L2 implica L1. Desativar um nível nunca quebra o anterior.

### 4.8 Join com `chunks`

A coluna `source_file` (repo-relative path) e `source_location` ("L<line>") permitem associar nó do grafo ao chunk correspondente. Heurística:

```sql
-- Para um nó "FsrsCalculator" em "backend/.../FsrsCalculator.java:L20",
-- encontrar o chunk que cobre essa linha:
SELECT c.*
FROM chunks c, graph_nodes n
WHERE n.node_id = ?
  AND c.rel_path = n.source_file
  AND c.start_line <= CAST(SUBSTR(n.source_location, 2) AS INTEGER)
  AND c.end_line   >= CAST(SUBSTR(n.source_location, 2) AS INTEGER)
ORDER BY (c.end_line - c.start_line) ASC   -- prefer narrowest match
LIMIT 1;
```

Quando não há chunk correspondente (nó documental, URL externa, imagem), o tool retorna apenas os metadados do grafo (path + line), sem snippet.

---

## 5. Design — Parte B: Novos tools MCP

### 5.1 `find_callers`

**Descrição (lida pelo LLM):**
> Lista locais que chamam um símbolo. Travessia direta sobre `graph_edges` filtrada por `relation='calls'` e `confidence='EXTRACTED'`. Mais preciso que `search_code` para queries do tipo "quem chama X". Só disponível se o grafo do graphify estiver carregado.

**Schema:**

```jsonc
{
  "name": "find_callers",
  "input_schema": {
    "type": "object",
    "properties": {
      "symbol": { "type": "string", "description": "Nome qualificado, ex.: 'StudyService.submitAnswer'." },
      "depth":  { "type": "integer", "default": 1, "minimum": 1, "maximum": 3 },
      "limit":  { "type": "integer", "default": 20, "maximum": 100 }
    },
    "required": ["symbol"]
  }
}
```

**Implementação** (`retrieval/graph.py`):

```python
def find_callers(storage: Storage, symbol: str, depth: int = 1, limit: int = 20) -> list[CallerHit]:
    node_id = _resolve_symbol_to_node(storage, symbol)
    if node_id is None:
        return []
    visited: set[str] = {node_id}
    frontier: list[str] = [node_id]
    results: list[CallerHit] = []
    for _ in range(depth):
        if not frontier:
            break
        placeholders = ",".join("?" * len(frontier))
        rows = storage._conn.execute(
            f"""SELECT source, source_file, source_loc
                FROM graph_edges
                WHERE target IN ({placeholders})
                  AND relation = 'calls'
                  AND confidence = 'EXTRACTED'
                LIMIT ?""",
            (*frontier, limit * 2),
        ).fetchall()
        new_frontier: list[str] = []
        for r in rows:
            if r["source"] in visited:
                continue
            visited.add(r["source"])
            new_frontier.append(r["source"])
            results.append(_to_caller_hit(storage, r))
            if len(results) >= limit:
                return results
        frontier = new_frontier
    return results
```

### 5.2 `find_related`

Travessia genérica. Útil quando o agent ainda não sabe a relação exata.

```jsonc
{
  "name": "find_related",
  "input_schema": {
    "type": "object",
    "properties": {
      "symbol":    { "type": "string" },
      "relations": {
        "type": "array",
        "items": { "enum": ["calls", "method", "contains", "imports", "extends",
                            "implements", "references", "tests", "rationale_for"] },
        "description": "Filtros de relação. Vazio = todas."
      },
      "direction": { "enum": ["outgoing", "incoming", "both"], "default": "both" },
      "depth":     { "type": "integer", "default": 1, "maximum": 3 },
      "limit":     { "type": "integer", "default": 20 }
    },
    "required": ["symbol"]
  }
}
```

### 5.3 `get_community`

Retorna tudo na mesma comunidade Leiden do símbolo dado. Cobre queries do tipo "me mostra tudo conexo a FSRS" onde retrieval semântico se espalha demais.

```jsonc
{
  "name": "get_community",
  "input_schema": {
    "type": "object",
    "properties": {
      "symbol": { "type": "string" },
      "limit":  { "type": "integer", "default": 30 }
    },
    "required": ["symbol"]
  }
}
```

Limita o output a comunidades de tamanho razoável: se a comunidade tem >100 nós (god-cluster que captura ruído), retorna apenas os top-N por `degree centrality` calculável on-the-fly via `COUNT(*)` em `graph_edges`.

### 5.4 `find_rationale` (opcional)

Só registrado se o grafo tem `>0` edges com `relation='rationale_for'`. No Mimoria atual: 45 edges. Útil porque é uma relação que tree-sitter puro não consegue derivar — sai de LLM extraction. Apesar de INFERRED, esta relação específica é estável o suficiente para valer expor.

```jsonc
{
  "name": "find_rationale",
  "input_schema": {
    "type": "object",
    "properties": {
      "symbol": { "type": "string", "description": "Símbolo do código (classe/método)." }
    },
    "required": ["symbol"]
  }
}
```

Retorna documentos/seções que o LLM identificou como "razão de existir" do símbolo (typicamente design docs).

### 5.5 Decoração das responses

Todos os 4 tools retornam objetos com campo extra `confidence` (`EXTRACTED` | `INFERRED`) e `provenance` (`"graphify@<commit_sha>"`) para que o agent saiba qual peso dar:

```jsonc
{
  "results": [
    {
      "rank": 1,
      "path": "backend/.../StudyService.java",
      "start_line": 245,
      "end_line": 287,
      "symbol": "StudyService.handleAnswer",
      "snippet": "...",
      "relation": "calls",                           // <-- estes campos extras
      "confidence": "EXTRACTED",                     //
      "provenance": "graphify@2d22f0aa"              //
    }
  ]
}
```

### 5.6 Disponibilidade condicional

Tools são registrados em `tools/list` apenas se:

1. `graphify-out/graph.json` existe E
2. O loader rodou pelo menos uma vez E
3. `graph_nodes` tem `count > 0`

A checagem acontece no lifespan do FastMCP, antes de `tools/list` responder. Se o grafo for adicionado depois, o cliente precisa reconectar (limitação aceita do MCP atual — não há `tools/list_changed` notification ainda em produção).

---

## 6. Design — Parte C: Hook post-commit

### 6.1 Comportamento operacional

```bash
#!/bin/sh
# mimoria-rag-hook-start v=1
# Generated by `mimoria-rag hook install`. Do not edit between markers.

# --- 1. Skip conditions -------------------------------------------------------
[ "${MIMORIA_RAG_HOOK_SKIP:-0}" = "1" ] && exit 0

GIT_DIR=$(git rev-parse --git-dir 2>/dev/null) || exit 0
[ -d "$GIT_DIR/rebase-merge" ]      && exit 0
[ -d "$GIT_DIR/rebase-apply" ]      && exit 0
[ -f "$GIT_DIR/MERGE_HEAD" ]        && exit 0
[ -f "$GIT_DIR/CHERRY_PICK_HEAD" ]  && exit 0

TOPLEVEL=$(git rev-parse --show-toplevel)
EXPECTED="__INSTALL_TARGET__"   # injected at install time
[ "$TOPLEVEL" = "$EXPECTED" ]   || exit 0

CHANGED=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || git diff --name-only HEAD)
[ -z "$CHANGED" ] && exit 0

# --- 2. Detached background run ----------------------------------------------
LOG="${HOME}/.cache/mimoria-rag-hook.log"
LOCK="/tmp/mimoria-rag-hook-$(echo "$TOPLEVEL" | md5).lock"
mkdir -p "$(dirname "$LOG")"

echo "[$(date -u +%FT%TZ)] commit $(git rev-parse --short HEAD) — launching refresh" >> "$LOG"

nohup sh -c "
    # Prevent overlapping runs (commit storm during rebase squash, etc.)
    exec 9>'$LOCK'
    if ! flock -n 9; then
        echo '[mimoria-rag-hook] another refresh running, skipping' >> '$LOG'
        exit 0
    fi

    timeout '${MIMORIA_RAG_HOOK_TIMEOUT:-600}' sh -c '
        # graphify update (AST-only, no API key needed)
        if command -v graphify >/dev/null; then
            if [ -f \"$TOPLEVEL/graphify-out/graph.json\" ]; then
                graphify update \"$TOPLEVEL\" --no-cluster
            else
                echo \"[mimoria-rag-hook] graphify-out missing; skipping graphify update\" >&2
                echo \"[mimoria-rag-hook] run \\\"graphify .\\\" inside $TOPLEVEL once to bootstrap\" >&2
            fi
        fi

        # mimoria-rag index (incremental via chunk_is_fresh skip)
        __RAG_CMD__
    ' || echo '[mimoria-rag-hook] failed (exit $?)' >> '$LOG'
" > "$LOG" 2>&1 < /dev/null &
disown 2>/dev/null || true
# mimoria-rag-hook-end
```

`__INSTALL_TARGET__` e `__RAG_CMD__` são substituídos pelo `hook install` no momento da escrita (atomic write via tmp file + rename).

### 6.2 Composição com hooks existentes

O hook NÃO sobrescreve `.git/hooks/post-commit` se já existir. Em vez disso:

1. **Arquivo não existe** → cria com shebang + bloco entre markers
2. **Arquivo existe sem nossos markers** → faz append do bloco (preserva conteúdo existente)
3. **Arquivo existe com nossos markers** → reescreve apenas o bloco entre markers
4. **Arquivo existe com `graphify-hook-start` markers** → emite warning visível e pergunta confirmação:
   ```
   ⚠ graphify post-commit hook detectado.
   Este script já roda `graphify update`; manter ambos causa duplicação.
   Recomendado: `graphify hook uninstall && mimoria-rag hook install --target …`.
   Continuar mesmo assim? [y/N]
   ```

Markers usados: `# mimoria-rag-hook-start` e `# mimoria-rag-hook-end`. O `v=1` na linha de start permite migrações futuras detectáveis.

### 6.3 Detecção de `core.hooksPath`

Para suportar husky/lefthook quando o usuário adotar futuramente:

```python
def resolve_hooks_dir(repo: Path) -> Path:
    """Replica a lógica de graphify/hooks.py:171-196."""
    try:
        result = subprocess.run(
            ["git", "config", "--get", "core.hooksPath"],
            cwd=repo, capture_output=True, text=True, check=False, timeout=5,
        )
        configured = result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        configured = ""

    if configured:
        # Validate: must be relative to repo OR absolute path inside HOME.
        # Reject anything else to prevent supply-chain via malicious config.
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = repo / candidate
        candidate = candidate.resolve()
        if str(candidate).startswith((str(repo.resolve()), str(Path.home().resolve()))):
            return candidate

    return repo / ".git" / "hooks"
```

### 6.4 Bootstrap (graphify-out ausente)

Conforme decisão registrada ADR-005 (§17): warning ao stderr, prossegue com o RAG indexing.

Visível no terminal apenas na primeira vez (post-commit imprime no log, mas o stderr também é visível porque `nohup` herda stderr até detach). Mensagem one-shot:

```
[mimoria-rag-hook] graphify-out missing — skipping graphify update
[mimoria-rag-hook] run `graphify .` inside /path/to/mimoria once to bootstrap (one-time, may cost LLM tokens)
```

### 6.5 Kill-switch & overrides

| Env var | Efeito |
|---|---|
| `MIMORIA_RAG_HOOK_SKIP=1` | Hook não roda. Útil para `git rebase -i` longos. |
| `MIMORIA_RAG_HOOK_TIMEOUT=600` | Timeout total em segundos (default 600 = 10 min). |
| `MIMORIA_RAG_HOOK_LOG=/tmp/foo.log` | Override do log path. Default: `~/.cache/mimoria-rag-hook.log`. |
| `RAG_GRAPH_CONFIDENCE=...` | Override de filtro de confiança no loader. |

---

## 7. CLI surface

### 7.1 Novo subcomando: `mimoria-rag hook`

```
mimoria-rag hook install   [--target <repo>] [--force]
mimoria-rag hook uninstall [--target <repo>]
mimoria-rag hook status    [--target <repo>]
```

**`install`**:
- Default `target` = `$RAG_TARGET_REPO` ou cwd
- Resolve `hooks_dir` respeitando `core.hooksPath`
- Escreve / atualiza bloco entre markers (atomic via tmp file + rename)
- Garante `chmod +x` no script final
- Imprime instruções de teste: `git commit --allow-empty -m "test hook" && tail -f ~/.cache/mimoria-rag-hook.log`
- `--force`: silencia o prompt de confirmação quando graphify-hook existir

**`uninstall`**:
- Remove APENAS o bloco entre nossos markers; deixa o resto intacto
- Se o arquivo ficar vazio (só shebang), remove o arquivo inteiro

**`status`**:
- Inspeção idempotente, exit 0 se instalado, 1 se não:
  ```
  Hook target:       /Users/jonatas/sources/mimoria
  Hook file:         /Users/jonatas/sources/mimoria/.git/hooks/post-commit
  Installed:         yes (v=1, installed at 2026-05-14T13:00:00Z)
  Graphify cohabit:  no
  Log file:          ~/.cache/mimoria-rag-hook.log (3 entries, last 2026-05-14T13:42:11Z)
  Last refresh:      success (took 7.2s)
  ```

### 7.2 Novo flag em `mimoria-rag index`

```
mimoria-rag index [--rebuild] [--refresh-graph]
```

`--refresh-graph`: força reload de `graphify-out/graph.json` mesmo se o loader não detectar staleness. Útil para corrigir estado inconsistente sem precisar rebuild completo.

### 7.3 Env vars adicionadas ao `.env.example`

```bash
# --- Graphify integration (optional) ---
# Path to graphify-out/. Defaults to <RAG_TARGET_REPO>/graphify-out.
# RAG_GRAPHIFY_OUT=

# Confidence filter for graph edges loaded into SQLite.
# Default: EXTRACTED only (highest precision). Comma-separated.
# RAG_GRAPH_CONFIDENCE=EXTRACTED

# --- Hook automation ---
# Skip the post-commit hook for this run.
# export MIMORIA_RAG_HOOK_SKIP=1
# Timeout (seconds) for the background refresh.
# export MIMORIA_RAG_HOOK_TIMEOUT=600
```

---

## 8. Conformance keywords (RFC 2119)

Conforme a [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119), as palavras MUST, MUST NOT, SHOULD, SHOULD NOT, MAY têm significado normativo neste documento:

- O loader **MUST** ser idempotente: `load()` chamado N vezes produz o mesmo estado de `graph_nodes`/`graph_edges`.
- O loader **MUST** ser transacional: falha no meio mantém a versão anterior.
- O hook **MUST NOT** bloquear o commit por mais de 100 ms.
- O hook **MUST** skippar durante rebase / merge / cherry-pick.
- O hook **SHOULD** detectar `core.hooksPath` e respeitar.
- Os tools `find_*` **MUST NOT** aparecer em `tools/list` quando o grafo não está carregado.
- Toda response dos tools **SHOULD** carregar `provenance` indicando a fonte do dado.
- O loader **MAY** filtrar edges INFERRED por default (decisão de runtime via `RAG_GRAPH_CONFIDENCE`).

---

## 9. Implementação em fases

### Fase 1.5a — Graph loader & SQLite schema (1.5 dia)

| Task | Arquivo | Status |
|---|---|---|
| Adicionar tabelas `graph_nodes`, `graph_edges` ao migrate | `storage/sqlite.py` | TODO |
| Criar `ingest/graphify_loader.py` com `GraphifyLoader` | novo | TODO |
| Hook `IndexPipeline.run()` para chamar loader se grafo stale | `ingest/pipeline.py` | TODO |
| Adicionar campo `graph` ao `Storage.stats()` | `storage/sqlite.py` | TODO |
| Unit tests: loader idempotência, filtro de confiança, staleness | `tests/unit/test_graphify_loader.py` | TODO |
| E2E test: índice + loader contra `graphify-out/` fixture | `tests/eval/test_graph_load.py` | TODO |

**Critério de saída**: `stats` mostra contagem de nós/edges; rodar `index` duas vezes não duplica linhas.

### Fase 1.5b — MCP tools sobre o grafo (1.5 dia)

| Task | Arquivo | Status |
|---|---|---|
| `retrieval/graph.py` com `find_callers`, `find_related`, `get_community` | novo | TODO |
| Wire tools no `server.py` com gating condicional | `server.py` | TODO |
| Symbol resolution helper (`_resolve_symbol_to_node`) | `retrieval/graph.py` | TODO |
| `find_rationale` registrado se edges > 0 | `server.py` | TODO |
| MCP smoke test: handshake + 3 tools | `tests/unit/test_mcp_graph_tools.py` | TODO |
| Golden eval: 5 queries simbólicas com chamadas conhecidas | `tests/eval/golden/symbolic.jsonl` | TODO |

**Critério de saída**: `find_callers("StudyService.submitAnswer")` retorna ≥1 chamador conhecido (validado contra o Mimoria real).

### Fase 1.5c — Post-commit hook (1 dia)

| Task | Arquivo | Status |
|---|---|---|
| `tools/hook_install.py` com install/uninstall/status | novo | TODO |
| Template do script (post-commit.sh.tmpl) | `tools/hook_templates/` | TODO |
| Wire subcomando `hook` no `cli.py` | `cli.py` | TODO |
| Detecção de `core.hooksPath` | `tools/hook_install.py` | TODO |
| Detecção de graphify-hook coexistente | `tools/hook_install.py` | TODO |
| Tests: install em fixture git repo, idempotência, uninstall limpo | `tests/unit/test_hook_install.py` | TODO |
| Smoke: instalar no Mimoria, fazer commit-allow-empty, verificar log | manual | TODO |

**Critério de saída**: commit no Mimoria dispara refresh em background, log mostra "success", `find_callers` reflete o commit em ≤60s.

### Fase 1.5d — Docs & polish (0.5 dia)

| Task | Arquivo |
|---|---|
| Atualizar `CLAUDE.md` com estado do graph integration + hook |
| Atualizar `README.md` com seção "Connecting graphify" |
| Atualizar `.env.example` com novas env vars (`RAG_GRAPHIFY_OUT`, `RAG_GRAPH_CONFIDENCE`, `RAG_GRAPH_FTS_AUGMENT`, `RAG_GRAPH_VEC_AUGMENT`) |
| Atualizar `.mcp.example.json` se mudar args (não deve) |
| Atualizar `docs/architecture/rag-system.md` §3.5, §8 com referências a este doc |

### Fase 1.5e — L2: FTS5 augmentation com graph data (0.5 dia)

Adiciona §4.5 — labels de nó/community entram no documento FTS5 dos chunks correspondentes, melhorando recall lexical do `search_code` existente.

| Task | Arquivo | Status |
|---|---|---|
| `Storage.augment_fts_with_graph()` — UPDATE em FTS5 com JOIN graph_nodes ↔ chunks | `storage/sqlite.py` | TODO |
| Hook em `IndexPipeline.run()`: chamar augment_fts após loader.load() | `ingest/pipeline.py` | TODO |
| Toggle via `RAG_GRAPH_FTS_AUGMENT` (default `1`) | `config.py` | TODO |
| Idempotência: re-augment não duplica tokens (UPDATE em vez de APPEND com guard) | `storage/sqlite.py` | TODO |
| Test: chunk com node match recebe extras; sem match não muda | `tests/unit/test_fts_augment.py` | TODO |
| Quality test: BM25 query por community-term match chunk-related | `tests/eval/golden/lexical.jsonl` | TODO |

**Critério de saída**: query "review stability rating" matcha mais chunks Fsrs-related com L2 que sem (ganho de recall mensurável no golden lexical).

### Fase 1.5f — L3 (opt-in): embeddings de nodes/communities (0.5 dia, opcional)

Adiciona §4.6 — vetores semânticos dos nodes do grafo entram numa 4ª stream RRF.

**Pré-requisito**: golden dataset rodando (Fase 2 do roadmap mestre) para validar que L3 não regrediu top-K em queries simbólicas.

| Task | Arquivo | Status |
|---|---|---|
| Schema: `vec_graph_nodes` virtual table + `graph_node_embeddings` metadata | `storage/sqlite.py` | TODO |
| `embed_graph_nodes()` no loader, com skip-fresh por `(node, model, label_hash)` | `ingest/graphify_loader.py` | TODO |
| `Storage.search_graph_dense()` — análogo a `search_dense` mas sobre `vec_graph_nodes` | `storage/sqlite.py` | TODO |
| `SearchEngine.search()` — adicionar 4ª stream condicional ao toggle | `retrieval/search.py` | TODO |
| Toggle via `RAG_GRAPH_VEC_AUGMENT` (default `0`) | `config.py` | TODO |
| Test: search com toggle on/off compara resultados; idempotência do embed | `tests/unit/test_graph_vec.py` | TODO |
| A/B no golden dataset: NDCG@10 com/sem L3 | `tests/eval/test_graph_vec_quality.py` | TODO |

**Critério de saída**: A/B golden mostra NDCG@10 ≥ baseline (não regride). Se regredir, ficha kept opt-in com warning explícito no help.

**Total estimado: 5 dias úteis** (4.5d sem L3, 5d com L3 ativado). Cada fase é independente e mergeable separadamente. L3 (Fase 1.5f) pode ser pulada e adicionada posteriormente sem retrabalho nas outras fases.

---

## 10. Estratégia de testes

### 10.1 Unit (rápido, deterministic, sem rede)

- `test_graphify_loader.py`: carrega fixture `graph.json` (3 nós, 5 edges), valida queries, testa filtros de confidence, testa idempotência (load 2x → mesmo estado).
- `test_graph_traversal.py`: BFS de callers em grafo sintético; depth limit; cycle detection.
- `test_hook_install.py`: instala em git repo temporário; valida idempotência (install 2x = mesmo arquivo); uninstall limpa; markers presentes/ausentes.

### 10.2 Integração

- `test_storage_graph_schema.py`: migrate aplicado em DB vazio cria tabelas + índices; reaplicar migrate é no-op.
- `test_mcp_graph_tools.py`: spawna MCP server com fixture pré-carregada, chama `find_callers`, `find_related`, `get_community` via JSON-RPC stdio, valida responses.

### 10.3 E2E (com Mimoria real, marker `@pytest.mark.e2e`)

- `test_phase2_canonical.py`:
  - `find_callers("FsrsCalculator.calculateNewStability")` → top hit é `FsrsAlgorithm.recalc` ou similar.
  - `get_community("FsrsCalculator")` → contém ≥3 outros nós Fsrs-relacionados.
  - `find_rationale("FsrsCalculator")` → contém path para `docs/study-flow.md` ou `rag-system.md`.

### 10.4 Golden dataset

`tests/eval/golden/symbolic.jsonl` (5–10 entradas iniciais):

```jsonc
{
  "id": "SYM-001",
  "tool": "find_callers",
  "args": { "symbol": "StudyService.submitAnswer", "depth": 1 },
  "expected_includes_any": [
    "StudyController.submit",
    "StudyServiceTest.shouldRecordCorrectAnswer"
  ]
}
```

Roda em CI; PR é bloqueada se ≥2 entradas regredirem.

### 10.5 Smoke do hook

Script `scripts/smoke-hook.sh`:

```bash
mimoria-rag hook install --target /Users/jonatas/sources/mimoria
( cd /Users/jonatas/sources/mimoria && git commit --allow-empty -m "smoke" )
sleep 30   # tempo para refresh background
tail -50 ~/.cache/mimoria-rag-hook.log
mimoria-rag hook status
```

---

## 11. Observabilidade

### 11.1 Logging estruturado (`structlog`)

Eventos novos:

```
graphify_loader.start          target=/path commit=2d22f0aa
graphify_loader.done           nodes=7819 edges_loaded=9879 edges_filtered=4598 duration_ms=842
graphify_loader.stale          last_load=2026-05-14T11:50 file_mtime=2026-05-14T13:00
graphify_loader.missing        path=/.../graphify-out/graph.json
graph_tool.invocation          tool=find_callers symbol=… results=… duration_ms=…
hook.install                   target=… mode=post-commit hooks_dir=…
hook.install.coexist_warning   other_hook=graphify
hook.refresh.launch            commit=abc123 changed=5
hook.refresh.success           duration_ms=7200
hook.refresh.failure           reason=… exit_code=…
```

### 11.2 Métricas (alvo Phase 2 §11.2)

```
rag_graph_nodes_total
rag_graph_edges_total{confidence}
rag_graph_load_duration_seconds
rag_graph_tool_calls_total{tool, status}
rag_graph_tool_duration_seconds{tool, quantile}
rag_hook_refresh_duration_seconds{quantile}
rag_hook_refresh_failures_total{stage}
```

### 11.3 Tracing (futuro)

Spans propostos sob `mcp.tool.find_callers`:

```
mcp.tool.find_callers
  ├─ graph.resolve_symbol
  ├─ graph.bfs_callers
  └─ graph.hydrate_chunks
```

---

## 12. Segurança e privacidade

### 12.1 Threat model resumido

| Vetor | Risco | Mitigação |
|---|---|---|
| `graph.json` adulterado | Insertion de paths fora do repo | Loader rejeita `source_file` que sai do `target_repo` via `Path.resolve()` |
| Hook executando comando arbitrário | Injection via filename / commit msg | Hook template não interpola dados de commit; só env vars com allowlist |
| `core.hooksPath` malicioso | Escrever script em path inesperado | Validação por allowlist: resolvido deve estar dentro do repo ou de $HOME |
| Hook escapa de chroot | `nohup` vaza para sessão paralela | Aceito; trade-off do detached design. Documentar no install. |
| Secret em chunk indexado entra no `graph.json` | Vazamento via tool response | Heredamos a varredura `gitleaks` da Phase 2 (não escopo deste doc) |
| `flock` race em commits paralelos | Corrupção do SQLite WAL | `flock` exclusivo + SQLite WAL mode já configurado |

### 12.2 Path traversal no loader

```python
def _validate_source_file(self, raw: str, repo: Path) -> str | None:
    candidate = (repo / raw).resolve()
    try:
        candidate.relative_to(repo.resolve())
    except ValueError:
        log.warning("graphify_loader.path_escape", raw=raw)
        return None
    return raw
```

### 12.3 Logs

`~/.cache/mimoria-rag-hook.log` contém commit SHAs e paths, NUNCA conteúdo de arquivo. Rotação manual por enquanto; rotação automática via `logrotate` é tarefa Phase 3.

---

## 13. Performance

### 13.1 Load do grafo

Mimoria atual (`graph.json` = 8.4 MB, 7819 nós, 14477 edges):

- `json.loads()`: ~150 ms
- Bulk insert em SQLite WAL: ~600 ms
- Total alvo: **<1s no caminho quente**

`graph.json` >100 MB (~100k nós): considerar streaming JSON via `ijson` e batched inserts. Não otimizar prematuramente.

### 13.2 Tool latency

| Tool | Operação | Alvo p95 |
|---|---|---|
| `find_callers` (depth=1) | Single SELECT com 1 LEFT JOIN | ≤ 20 ms |
| `find_callers` (depth=3) | BFS em Python + SELECTs | ≤ 80 ms |
| `find_related` (depth=1) | Single SELECT | ≤ 20 ms |
| `get_community` | Single SELECT WHERE community = ? | ≤ 15 ms |
| `find_rationale` | Single SELECT | ≤ 15 ms |

Todos puramente SQLite local; alvos conservadores.

### 13.3 Hook impacto

| Operação | Tempo síncrono |
|---|---|
| Hook script lê env, checa skip conditions | <10 ms |
| `nohup ... &` + `disown` | <50 ms |
| **Total bloqueando o git commit** | **<100 ms** |

O background pode levar 5–600s; usuário não nota.

---

## 14. Failure modes & graceful degradation

| Situação | Comportamento |
|---|---|
| `graphify-out/graph.json` ausente | Loader é no-op; tools `find_*` não aparecem em `tools/list`; MCP server funciona normalmente |
| `graph.json` corrompido (JSON inválido) | Loader log warning + raise; index falha; usuário precisa fixar manualmente |
| `graph.json` com schema desconhecido (graphify nova versão) | Loader ignora campos extras; emite warning; segue |
| Hook: graphify ausente do PATH | Pula etapa graphify; segue para `mimoria-rag index` |
| Hook: `flock` ocupado (commit storm) | Skippa imediatamente; log "skipping due to lock"; commit seguinte triggera refresh |
| Hook: timeout estourado | Kill via `timeout` builtin; log "timeout"; estado fica consistente porque tudo é idempotente |
| Hook: mimoria-rag não está instalado | Falha gracefully; warning no log; commit já foi |
| `core.hooksPath` aponta para caminho inválido | Install falha com mensagem clara; não escreve |

---

## 15. Rollout & migração

### 15.1 Migração de schema

O `Storage.migrate()` já é idempotente. As novas tabelas (`graph_nodes`, `graph_edges`) são adicionadas com `CREATE TABLE IF NOT EXISTS`. Bumping `SCHEMA_VERSION` de 1 para 2 deixa o caminho aberto para migrações futuras.

Usuários com índice da Phase 1 (sem essas tabelas) apenas ganham tabelas vazias. Nada quebra. Próximo `index` (com graphify disponível) popula.

### 15.2 Bootstrap recomendado para um novo target repo

```bash
# Dentro do repo target (ex.: ~/sources/mimoria):

# 1. Bootstrap graphify (uma vez, custa LLM tokens se quiser semantic edges)
graphify .                                           # full pipeline (com LLM)
# ou:
graphify extract . --backend gemini                  # mesmo, controlado
# ou (sem LLM, sem rationale_for / conceptually_related_to):
graphify update . --no-cluster                       # AST-only, gratuito

# 2. Build do índice RAG (já carrega o grafo recém-criado, L1 + L2 default)
uv run --directory ~/sources/rag mimoria-rag index --rebuild

# 3. Instalar o hook para manter ambos frescos
uv run --directory ~/sources/rag mimoria-rag hook install --target $(pwd)

# 4. Sanity check
uv run --directory ~/sources/rag mimoria-rag hook status
uv run --directory ~/sources/rag mimoria-rag stats   # mostra contagem do grafo
```

### 15.3 Upgrade de embedder local (lever de qualidade — recomendado para usuários local-only)

Per ADR-008, o gargalo de qualidade hoje é o embedder, não falta de mais vetores. Para usuários local-only sem restrição de RAM:

```bash
# Em .env, trocar o modelo local. Modelos sugeridos por escala:

# Médio (440 MB, bom equilíbrio)
RAG_LOCAL_EMBED_MODEL=BAAI/bge-base-en-v1.5

# Grande (1.3 GB, top open-source MTEB)
RAG_LOCAL_EMBED_MODEL=mixedbread-ai/mxbai-embed-large-v1

# Code-specialized (550 MB, otimizado para código fonte)
RAG_LOCAL_EMBED_MODEL=nomic-ai/nomic-embed-code

# Multi-lingual + multi-functional (2.3 GB, overkill mas SOTA)
RAG_LOCAL_EMBED_MODEL=BAAI/bge-m3

# Após mudar:
uv run mimoria-rag index --rebuild   # reindex completo (dim mudou; storage detecta)
```

**Trade-offs práticos**:
- Modelo maior = mais RAM no MCP server lifespan (pode dobrar/triplicar o footprint)
- Modelo maior = embedding por chunk fica de ~2ms para ~10–20 ms (CPU aarch64)
- `mimoria-rag index` num repo grande passa de ~3 min para ~10–15 min com modelo grande
- A queries do MCP ganham ~5–15 ms de latência (cabe dentro do budget de p95 ≤ 500 ms da spec mestre §1.3)

**Quando vale L3 vs upgrade de embedder**: o upgrade é o single-hop com maior ROI. L3 é incremental sobre isso. Se você tem orçamento computacional para os dois, faça os dois (upgrade primeiro, L3 depois com A/B). Se só pode escolher um, escolha o upgrade.

### 15.4 Rollback

Se a integração causar problemas, três níveis de desligamento:

```bash
# 1. Desligar o hook (preserva schema, preserva tools)
mimoria-rag hook uninstall --target ~/sources/mimoria

# 2. Desligar os tools (preserva schema, preserva grafo carregado)
#    via env var no .mcp.json:
#    "RAG_GRAPH_TOOLS_DISABLED": "1"

# 3. Limpar tabelas do grafo (mantém chunks/vectors)
sqlite3 ~/.mimoria-rag/index.db "DELETE FROM graph_edges; DELETE FROM graph_nodes;"
```

---

## 16. Alternativas consideradas

### 16.1 Construir symbol graph do zero

**Rejeitado**. Spec §3.5 estima esforço alto (JavaParser para Java, ts-morph para TS, mais resolução de imports cross-file). Graphify cobre o caso, é AST-deterministic, e é MIT. Reescrever seria duplicar 5k+ linhas pelo pouco ganho de "não depender de tool externa". Se o graphify for descontinuado, reescrever então.

### 16.2 Pre-commit síncrono

**Rejeitado**. Custo de UX (15–60s por commit) supera o benefício de "HEAD ≡ índice no instante exato do commit". Agentes consomem o índice via MCP, não no commit; alguns segundos de lag são invisíveis. Graphify chegou à mesma conclusão (vide comentário em `graphify/hooks.py:67`).

### 16.3 Hybrid pre-commit (só graphify) + post-commit (RAG)

**Rejeitado** após análise. Complica o install (dois hooks, dois templates, dois lifecycles). Ganho marginal: gráfico "fresco" alguns segundos antes do RAG. Não justifica.

### 16.4 File watcher (inotify/fsevents) sempre rodando

**Rejeitado para v1**. Custo cognitivo: usuário precisa lembrar que tem um daemon rodando. Custo de CPU: tree-sitter parsing em loop. Custo de ergonomia: precisa de daemonização (launchd/systemd). Hook é mais explícito e barato. Spec §8.3 prevê soft index em RAM via watcher para Phase 3 — mantemos como trabalho futuro.

### 16.5 Auto-bootstrap (rodar `graphify .` se ausente)

**Rejeitado**. `graphify .` aciona LLM extraction, que custa tokens. Acionar sem consentimento explícito do usuário viola o princípio de previsibilidade de custo (spec §1.3: "Custo previsível"). Warning + instrução manual é o padrão.

### 16.6 Persistir o grafo em memória (sem SQLite mirror)

**Rejeitado**. Tempo de cold-start aceitável só para servidor longevo; cada CLI invocation pagaria 1s de load. Mirroring em SQLite paga uma vez por refresh do graphify e amortiza em todas as queries subsequentes.

---

## 17. Decisões registradas (ADRs)

### ADR-001 — Fonte de verdade do grafo

**Contexto**: precisamos de symbol graph; graphify produz um adequado.
**Decisão**: graphify é fonte de verdade; mimoria-rag espelha (read-only) em SQLite.
**Consequência**: dependência opcional, mas única; se graphify mudar o schema do `graph.json`, ajustamos o loader.

### ADR-002 — Política de confidence default

**Contexto**: graphify produz EXTRACTED (68%) e INFERRED (32%).
**Decisão**: default carrega apenas EXTRACTED.
**Justificativa**: 32% de INFERRED gera ruído visível (visto em "Surprising Connections" do GRAPH_REPORT.md do Mimoria — ex. "StudyService references Vue Unit Test Plan"). Usuário avançado pode opt-in via `RAG_GRAPH_CONFIDENCE`.
**Trade-off**: perdemos 45 edges `rationale_for` por default — mas tratamos esta relação como caso especial via tool dedicado `find_rationale` (que retorna sem filtro de confidence).

### ADR-003 — Hook type: post-commit (não pre-commit)

**Contexto**: usuário pediu "antes de commit"; benchmark dos custos mostrou pre-commit > 15s típico.
**Decisão**: post-commit detached background. Confirmado em discussão prévia.
**Trade-off**: índice fica até ~60s atrás do HEAD após cada commit. Aceito porque agentes consomem index lazy, não imediato.

### ADR-004 — Hook composition: combined block

**Contexto**: graphify também instala hook; usuário pode ter ambos.
**Decisão**: nosso hook chama `graphify update` internamente. Avisamos no install se detectarmos `# graphify-hook-start` no arquivo existente. Recomendamos `graphify hook uninstall` antes de prosseguir.
**Razão**: ordem importa quando o RAG passar a depender de `graph.json` fresco; controlar a ordem dentro de um único bloco é mais robusto.

### ADR-005 — Bootstrap policy: stderr warning, sem auto-extract

**Contexto**: graphify-out pode não existir na primeira run.
**Decisão**: warning ao stderr, RAG segue sem grafo, instruções claras de como bootstrap.
**Razão**: auto-run de `graphify .` custaria tokens LLM sem consentimento. Princípio "previsibilidade de custo" da spec mestre §1.3.

### ADR-006 — Install delivery: subcommand do CLI

**Contexto**: alternativa era script `.sh` standalone documentado.
**Decisão**: subcomando `mimoria-rag hook install`.
**Razão**: simetria com `graphify hook install`; cobre status/uninstall/upgrade idempotente; descobrível via `--help`; testável.

### ADR-007 — Single target per install (v1)

**Contexto**: alguém pode querer indexar múltiplos repos.
**Decisão**: `hook install --target <repo>` aceita um repo só. Multi-repo via instalações repetidas.
**Razão**: KISS para v1; multi-target adiciona complexidade de validação e composição com hooks pré-existentes em cada repo. Pode entrar na v2 sem breaking change.

### ADR-008 — Recalibração de níveis de indexação (cost-aware)

**Contexto**: a v0 desta spec parou em L1 (mirror relacional) com a justificativa de que L2/L3 adicionavam custo de embedding/storage não justificado. Essa premissa assume Voyage API. **Quando o usuário roda 100% local** (fastembed/ONNX), o custo de embedar ~7800 nodes adicionais é ~30 s de CPU one-shot — não US$ 2 por reindex.

**Decisão**:
- **L2 vira default** (era opt-in implícito por não-existência). FTS5 augmentation custa ~ms por chunk; benefício de recall lexical é alto e bem documentado em sistemas hybrid retrieval.
- **L3 vira opt-in real** via `RAG_GRAPH_VEC_AUGMENT=1`. Documentado mas desligado por default porque adiciona dimensão a fusion (mais ruído potencial) e requer validação via golden dataset.
- **L4 fica fora desta spec** — entra na Phase 3 quando golden dataset existir.

**Justificativa**:
1. Custo computacional local é trivial (CPU embedding em modelos pequenos é ms-scale por documento)
2. Quality bottleneck atual é o EMBEDDER (`bge-small-en-v1.5` 384-dim), não a falta de mais vetores. L3 adiciona vetores mas não eleva o teto. Recomenda-se primeiro upgrade de modelo (ex.: `mxbai-embed-large-v1`, `bge-large-en-v1.5`, `nomic-embed-code`) — esse é o lever de maior ROI single-hop
3. Princípio de previsibilidade da spec mestre §1.3 ("custo previsível"): default tem que ter custo conhecido (L2 = ms; L3 = segundos one-shot; nada recorrente)

**Consequência**:
- Adiciona §4.5 (FTS5 augmentation) e §4.6 (semantic indexing) ao design
- Adiciona Fase 1.5e (FTS augmentation) à roadmap; total de implementação sobe de 4.5d para 5d
- `Storage.augment_fts_with_graph()` e `vec_graph_nodes` virtual table viram parte do schema
- ADR-002 (default EXTRACTED-only) **não muda** — política de confidence é ortogonal a L1/L2/L3

**Trade-off aceito**: maior superfície de teste; quatro streams no RRF quando L3 ativo (vs três hoje). Risco mitigado por toggles ortogonais que permitem A/B comparison local.

---

## 18. Riscos

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Graphify muda schema do `graph.json` em release futura | Média | Médio | Pin de versão min/max em README; loader tolera campos desconhecidos com warning |
| Graphify deprecado pelo autor | Baixa | Alto | A camada do mimoria-rag fica intacta (tools só ficam off); spec mestre §3.5 ainda pode ser implementada nativamente |
| INFERRED edges enganam usuário que ativa `RAG_GRAPH_CONFIDENCE=*` | Média | Médio | Default seguro; documentar trade-off no help |
| Hook causa loop infinito (commit dentro do refresh trigger novo commit) | Baixa | Alto | Hook NÃO faz commit; só grava em DB externa fora do repo target |
| `flock` indisponível no sistema (Windows nativo) | Média (não-macOS/Linux) | Médio | Hook template usa `flock` quando presente; fallback para PID file em outras plataformas |
| Latência do `find_callers` cresce com depth=3 em grafos densos | Média | Baixo | Limite duro de `depth=3` no schema; documentar |
| Race entre loader (durante index) e tool (durante MCP request) | Baixa | Médio | SQLite WAL + transação atomic no loader; readers veem versão anterior consistente |

---

## 19. Referências

### Documentos internos

- `docs/architecture/rag-system.md` — spec mestre, especialmente §3.5 (symbol graph), §6.3.5 (query routing), §8 (atualização incremental), §9.2 (catálogo de tools), §16 (roadmap)
- `CLAUDE.md` — estado vivo da implementação Phase 0/1

### Documentos externos

- Model Context Protocol — https://modelcontextprotocol.io/specification
- Graphify (origem) — https://github.com/safishamsi/graphify
- Tree-sitter — https://tree-sitter.github.io
- Leiden community detection — Traag, Waltman, Van Eck (2019) "From Louvain to Leiden: guaranteeing well-connected communities"
- RFC 2119 — https://www.rfc-editor.org/rfc/rfc2119
- Architecture Decision Records — https://adr.github.io/
- C4 Model — https://c4model.com/

### Convenções

- OpenTelemetry semantic conventions for LLM — https://opentelemetry.io/docs/specs/semconv/gen-ai/
- SQLite WAL — https://www.sqlite.org/wal.html

---

## 20. Glossário

| Termo | Definição |
|---|---|
| **ADR** | Architecture Decision Record |
| **AST** | Abstract Syntax Tree |
| **BFS / DFS** | Breadth/Depth-First Search |
| **EXTRACTED** | Aresta derivada deterministicamente do AST (alta confiança) |
| **INFERRED** | Aresta inferida por LLM ou heurística (confiança média) |
| **Leiden** | Algoritmo de community detection sucessor do Louvain |
| **NetworkX node-link format** | Schema JSON canônico (`{nodes:[…], links:[…]}`) |
| **post-commit hook** | Script Git executado APÓS um commit bem-sucedido |
| **flock** | Lock advisory POSIX para serialização entre processos |
| **provenance** | Origem rastreável de um dado (qual ferramenta, qual commit) |
| **WAL** | Write-Ahead Logging do SQLite — readers não bloqueiam writers |

"""sqlite-vec + FTS5 + metadata, all in one SQLite database.

Per spec §5: keep everything in a single `.db` file so it is trivially
backed up, synced, and shipped. Phase 0 uses dense search only; the FTS5
table is provisioned now so Phase 1 hybrid retrieval is a query change,
not a schema migration.
"""

from __future__ import annotations

import re
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlite_vec  # type: ignore[import-untyped]

from ..errors import StorageError
from ..ingest.git_meta import RepoIdentity
from ..logging import get_logger
from ..models import Chunk, RepoRef, RepoStats, SearchHit

log = get_logger(__name__)

# v1: initial schema (chunks, vec_chunks, fts_chunks, meta).
# v2: adds `repos` table and `chunks.repo_id` column for multi-repo support.
# v3: pipeline-adapters Phase 1 — `chunks.embedding_text` (what was actually
#     embedded), `chunks.enricher_id` (which adapter produced it), and the
#     `enrichments` cache table. Enables drift detection: changing enricher
#     config triggers re-enrich + re-embed without `--rebuild`.
# v4: graphify integration (RFC §9 Fase 1.5a) — `graph_nodes`, `graph_edges`
#     and `graph_meta` tables backing `find_callers` / `find_related` /
#     `get_community` / `find_rationale` MCP tools. Conditional on
#     `graphify-out/graph.json` being present at index time.
SCHEMA_VERSION = 4


def _serialize_vector(vec: list[float]) -> bytes:
    """Float32 little-endian, the format sqlite-vec expects for BLOB inputs."""
    return struct.pack(f"<{len(vec)}f", *vec)


# Characters that must be stripped from user queries before handing to FTS5.
_FTS_BAD = re.compile(r'[^\w\s\.]+', re.UNICODE)
_CAMEL_BOUNDARY_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


def _split_identifiers(text: str) -> str:
    """Expand CamelCase + dotted identifiers into separate tokens.

    Used on BOTH indexed content and queries so `FsrsCalculator` finds
    `fsrs calculator` after FTS5's unicode61 tokenizer lowercases and splits
    on non-alphanumerics.
    """
    text = _CAMEL_BOUNDARY_1.sub(" ", text)
    text = _CAMEL_BOUNDARY_2.sub(" ", text)
    return text


def _sanitize_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression (OR-joined tokens)."""
    cleaned = _FTS_BAD.sub(" ", query)
    cleaned = _split_identifiers(cleaned)
    tokens = [t for t in cleaned.split() if len(t) > 1]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def _fts_text(content: str, rel_path: str, symbol: str | None) -> str:
    """Pre-process chunk content before indexing in FTS5.

    We assemble a single FTS document from four sources, separated by
    newlines so FTS5 treats them as a single token stream:

    1. Original content — preserves exact identifier matches.
    2. CamelCase / dotted split of content — recovers tokens like `Fsrs`,
       `Calculator` from `FsrsCalculator`.
    3. File path tokens — boosts results whose path mentions the query
       (e.g. `FsrsCalculator.java` is a strong signal for "fsrs calculator").
    4. Symbol (qualified name) — gives chunks a small boost when the user
       asks for them by name.
    """
    path_tokens = _split_identifiers(
        rel_path.replace("/", " ").replace("\\", " ").replace(".", " ")
    )
    parts: list[str] = [content, _split_identifiers(content), path_tokens]
    if symbol:
        parts.append(_split_identifiers(symbol.replace(".", " ")))
    return "\n".join(parts)


def _snippet(content: str, limit: int = 480) -> str:
    return content[:limit].rstrip() + ("…" if len(content) > limit else "")


def _repo_from_row(row: Any) -> RepoRef | None:
    """Build a ``RepoRef`` from a row that LEFT JOINed ``repos``.

    Returns ``None`` for chunks without a repo (legacy data pre-schema-v2).
    """
    key = row["repo_normalized_key"] if "repo_normalized_key" in row.keys() else None
    if not key:
        return None
    return RepoRef(
        normalized_key=key,
        url=row["repo_url"] if "repo_url" in row.keys() else None,
        default_branch=(
            row["repo_default_branch"] if "repo_default_branch" in row.keys() else None
        ),
        head_commit=(
            row["repo_head_commit"] if "repo_head_commit" in row.keys() else None
        ),
    )


# Common SELECT clause emitting both chunk and repo columns. Reused by all
# search_* paths. Aliases are stable so _repo_from_row can rely on them.
_REPO_JOIN_COLUMNS = """
    c.chunk_id, c.rel_path, c.language, c.kind, c.symbol,
    c.start_line, c.end_line, c.content,
    r.normalized_key  AS repo_normalized_key,
    r.remote_url      AS repo_url,
    r.default_branch  AS repo_default_branch,
    r.head_commit     AS repo_head_commit
"""


# Common English stopwords + question words. Filtered out from identifier-token
# extraction so a free-text query like "where is the FSRS calculator" routes
# only the meaningful tokens [FSRS, calculator] to the symbol lookup stream.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done",
    "have", "has", "had",
    "this", "that", "these", "those",
    "what", "where", "when", "why", "how", "who", "which",
    "show", "find", "look", "get",
    "with", "without", "by", "from", "as",
    "it", "its", "i", "you", "we",
    "code", "function", "class", "method", "file",
})


def _identifier_tokens(query: str) -> list[str]:
    """Extract identifier-like tokens from a free-text query.

    Drops stopwords and tokens of length < 3. CamelCase splits before
    filtering so `FsrsCalculator` produces both `Fsrs` and `Calculator`.
    """
    cleaned = _FTS_BAD.sub(" ", query)
    cleaned = _split_identifiers(cleaned)
    out: list[str] = []
    seen: set[str] = set()
    for tok in cleaned.split():
        if len(tok) < 3:
            continue
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(tok)
    return out


class Storage:
    """Owns the SQLite connection. All schema mutations go through here."""

    def __init__(self, db_path: Path, embedding_dim: int) -> None:
        self._db_path = db_path
        self._dim = embedding_dim
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_path))
        try:
            self._conn.enable_load_extension(True)
        except (AttributeError, sqlite3.NotSupportedError) as exc:
            raise StorageError(
                "Your Python sqlite3 was built without extension loading. "
                "Reinstall Python via `uv python install 3.12` and retry."
            ) from exc
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.row_factory = sqlite3.Row

        self._migrate()
        self._check_dimension_consistency()

    @property
    def dim(self) -> int:
        return self._dim

    # ----- schema -----

    def _migrate(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id         TEXT NOT NULL UNIQUE,
                rel_path         TEXT NOT NULL,
                language         TEXT NOT NULL,
                kind             TEXT NOT NULL,
                symbol           TEXT,
                parent_chunk_id  TEXT,
                start_line       INTEGER NOT NULL,
                end_line         INTEGER NOT NULL,
                content          TEXT NOT NULL,
                content_hash     TEXT NOT NULL,
                embedding_model  TEXT NOT NULL,
                embedding_dim    INTEGER NOT NULL,
                indexed_at       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_rel_path     ON chunks(rel_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_language     ON chunks(language);

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding float[{self._dim}]
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                content,
                rel_path UNINDEXED,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61'
            );

            -- v2: repository identity (multi-repo support).
            CREATE TABLE IF NOT EXISTS repos (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                normalized_key     TEXT NOT NULL UNIQUE,
                remote_url         TEXT,
                root_path          TEXT NOT NULL,
                default_branch     TEXT,
                head_commit        TEXT,
                first_indexed_at   TEXT NOT NULL,
                last_indexed_at    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_repos_normalized_key
                ON repos(normalized_key);
            """
        )
        # v2: add chunks.repo_id (ALTER TABLE is idempotent only when guarded).
        self._add_column_if_missing(
            "chunks", "repo_id", "INTEGER REFERENCES repos(id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);"
        )

        # v3: pipeline-adapters Phase 1.
        # - chunks.embedding_text  → exact text fed to the embedder
        # - chunks.enricher_id     → adapter id of the enricher that produced it
        # - enrichments cache     → reusable (content_hash, adapter_id) → text
        self._add_column_if_missing("chunks", "embedding_text", "TEXT")
        self._add_column_if_missing("chunks", "enricher_id", "TEXT")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_enricher ON chunks(enricher_id);"
        )
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS enrichments (
                content_hash   TEXT NOT NULL,
                enricher_id    TEXT NOT NULL,
                embedding_text TEXT NOT NULL,
                cached_at      TEXT NOT NULL,
                PRIMARY KEY (content_hash, enricher_id)
            );
            CREATE INDEX IF NOT EXISTS idx_enrichments_enricher
                ON enrichments(enricher_id);
            """
        )

        # v4: graphify integration tables. Optional — populated by
        # GraphifyLoader iff `graphify-out/graph.json` is present.
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id          TEXT PRIMARY KEY,
                label            TEXT NOT NULL,
                file_type        TEXT,
                source_file      TEXT,
                source_location  TEXT,
                community_id     INTEGER,
                community_label  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_label
                ON graph_nodes(label);
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_source_file
                ON graph_nodes(source_file);
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_community
                ON graph_nodes(community_id);

            CREATE TABLE IF NOT EXISTS graph_edges (
                source           TEXT NOT NULL,
                target           TEXT NOT NULL,
                relation         TEXT NOT NULL,
                confidence       TEXT NOT NULL,
                confidence_score REAL,
                context          TEXT,
                source_file      TEXT,
                source_location  TEXT,
                weight           REAL,
                PRIMARY KEY (source, target, relation, source_file, source_location)
            );
            CREATE INDEX IF NOT EXISTS idx_graph_edges_target_relation
                ON graph_edges(target, relation, confidence);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_source_relation
                ON graph_edges(source, relation, confidence);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_relation
                ON graph_edges(relation, confidence);

            CREATE TABLE IF NOT EXISTS graph_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

        cur.execute(
            "INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        cur.execute(
            "INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)",
            ("embedding_dim", str(self._dim)),
        )
        # Bump schema_version if this is an in-place upgrade.
        cur.execute(
            "UPDATE meta SET value=? WHERE key='schema_version' AND CAST(value AS INTEGER) < ?",
            (str(SCHEMA_VERSION), SCHEMA_VERSION),
        )
        self._conn.commit()

    def _add_column_if_missing(self, table: str, column: str, decl: str) -> None:
        """Idempotent ALTER TABLE — SQLite has no IF NOT EXISTS for ADD COLUMN."""
        cols = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        if any(row["name"] == column for row in cols):
            return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _check_dimension_consistency(self) -> None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='embedding_dim'"
        ).fetchone()
        stored_dim = int(row["value"]) if row else self._dim
        if stored_dim != self._dim:
            raise StorageError(
                f"Index was built with dim={stored_dim} but current embedder uses dim={self._dim}. "
                f"Run `stropha index --rebuild` to rebuild."
            )

    # ----- repos -----

    def register_repo(self, identity: RepoIdentity) -> int:
        """Insert or update a repo row keyed by ``normalized_key``. Returns its id.

        Idempotent: re-running ``index`` against the same repo updates
        ``last_indexed_at`` + ``head_commit`` + ``default_branch`` without
        creating duplicates.
        """
        now = datetime.now(UTC).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id FROM repos WHERE normalized_key = ?",
            (identity.normalized_key,),
        )
        row = cur.fetchone()
        if row is not None:
            repo_id = int(row["id"])
            cur.execute(
                """
                UPDATE repos
                   SET remote_url       = ?,
                       root_path        = ?,
                       default_branch   = ?,
                       head_commit      = ?,
                       last_indexed_at  = ?
                 WHERE id = ?
                """,
                (
                    identity.remote_url,
                    str(identity.root_path),
                    identity.default_branch,
                    identity.head_commit,
                    now,
                    repo_id,
                ),
            )
            return repo_id

        cur.execute(
            """
            INSERT INTO repos (
                normalized_key, remote_url, root_path,
                default_branch, head_commit,
                first_indexed_at, last_indexed_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                identity.normalized_key,
                identity.remote_url,
                str(identity.root_path),
                identity.default_branch,
                identity.head_commit,
                now,
                now,
            ),
        )
        return int(cur.lastrowid or 0)

    def count_chunks_without_repo(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE repo_id IS NULL"
        ).fetchone()
        return int(row["n"])

    def backfill_chunks_to_repo(self, repo_id: int, sample_root: Path) -> int:
        """Best-effort assignment of orphan chunks to ``repo_id``.

        Used on upgrade from schema v1 (no ``repo_id`` column). Performs a
        sanity check first: at least one of the orphan chunks' ``rel_path``
        must exist under ``sample_root``. If the check fails we refuse to
        backfill — the user likely changed ``STROPHA_TARGET_REPO`` between
        indexer runs.

        Returns the number of chunks updated (0 if sanity check failed).
        """
        rows = self._conn.execute(
            "SELECT rel_path FROM chunks WHERE repo_id IS NULL LIMIT 5"
        ).fetchall()
        if not rows:
            return 0
        if not any((sample_root / r["rel_path"]).is_file() for r in rows):
            log.warning(
                "storage.backfill_skipped_sanity",
                sample_root=str(sample_root),
                samples=[r["rel_path"] for r in rows],
            )
            return 0
        cur = self._conn.execute(
            "UPDATE chunks SET repo_id = ? WHERE repo_id IS NULL",
            (repo_id,),
        )
        return int(cur.rowcount or 0)

    def list_repos(self) -> list[RepoStats]:
        """Per-repo aggregate counters."""
        rows = self._conn.execute(
            """
            SELECT
                r.normalized_key,
                r.remote_url      AS url,
                r.default_branch,
                r.head_commit,
                r.last_indexed_at,
                COUNT(DISTINCT c.rel_path) AS files,
                COUNT(c.id)                AS chunks
            FROM repos r
            LEFT JOIN chunks c ON c.repo_id = r.id
            GROUP BY r.id
            ORDER BY chunks DESC
            """
        ).fetchall()
        return [
            RepoStats(
                normalized_key=row["normalized_key"],
                url=row["url"],
                default_branch=row["default_branch"],
                head_commit=row["head_commit"],
                files=int(row["files"]),
                chunks=int(row["chunks"]),
                last_indexed_at=row["last_indexed_at"],
            )
            for row in rows
        ]

    # ----- writes -----

    def chunk_is_fresh(
        self,
        chunk_id: str,
        content_hash: str,
        embedding_model: str,
        enricher_id: str | None = None,
    ) -> bool:
        """True if the stored chunk matches every (hash, model, enricher).

        Per ADR-004 (pipeline-adapters): adapter drift triggers re-processing
        automatically. Pass ``enricher_id=None`` to preserve the v0.1.0
        behavior (only hash + model checked) — used by legacy code paths.

        Back-compat: a stored NULL ``enricher_id`` (chunks indexed before
        schema v3) is treated as equivalent to ``'noop'`` so upgrading from
        v0.1.0 with the default enricher does NOT cause a full re-embed.
        """
        row = self._conn.execute(
            """SELECT content_hash, embedding_model, enricher_id
               FROM chunks WHERE chunk_id = ?""",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return False
        if row["content_hash"] != content_hash:
            return False
        if row["embedding_model"] != embedding_model:
            return False
        if enricher_id is not None:
            stored = row["enricher_id"] or "noop"
            if stored != enricher_id:
                return False
        return True

    # ----- enrichment cache (pipeline-adapters Phase 1) -------------------

    def get_enrichment(self, content_hash: str, enricher_id: str) -> str | None:
        """Return cached embedding_text for ``(content_hash, enricher_id)`` or None."""
        row = self._conn.execute(
            """SELECT embedding_text FROM enrichments
               WHERE content_hash = ? AND enricher_id = ?""",
            (content_hash, enricher_id),
        ).fetchone()
        return row["embedding_text"] if row else None

    def put_enrichment(
        self, content_hash: str, enricher_id: str, embedding_text: str
    ) -> None:
        """Cache an enrichment result. Safe to call concurrently (UPSERT)."""
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO enrichments
               (content_hash, enricher_id, embedding_text, cached_at)
               VALUES (?, ?, ?, ?)""",
            (content_hash, enricher_id, embedding_text, now),
        )

    def delete_by_paths(self, rel_paths: list[str]) -> int:
        """Remove every chunk whose rel_path is in the list. Returns count."""
        if not rel_paths:
            return 0
        placeholders = ",".join("?" * len(rel_paths))
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT id FROM chunks WHERE rel_path IN ({placeholders})", rel_paths
        )
        ids = [int(r["id"]) for r in cur.fetchall()]
        if not ids:
            return 0
        id_placeholders = ",".join("?" * len(ids))
        cur.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({id_placeholders})", ids)
        cur.execute(f"DELETE FROM fts_chunks WHERE rowid IN ({id_placeholders})", ids)
        cur.execute(f"DELETE FROM chunks WHERE id IN ({id_placeholders})", ids)
        return len(ids)

    def upsert_chunk(
        self,
        chunk: Chunk,
        embedding: list[float],
        embedding_model: str,
        embedding_dim: int,
        repo_id: int | None = None,
        embedding_text: str | None = None,
        enricher_id: str | None = None,
    ) -> int:
        """Insert (or update) a chunk and its vector atomically. Returns rowid.

        ``repo_id`` carries the source-repository identity (see ``repos`` table).
        ``embedding_text`` / ``enricher_id`` (schema v3) record what was
        actually embedded and by which enricher adapter, enabling drift
        detection. Both default to None so legacy callers stay compatible —
        in that case ``embedding_text`` falls back to ``chunk.content``.
        """
        if len(embedding) != self._dim:
            raise StorageError(
                f"Embedding dim mismatch: got {len(embedding)}, expected {self._dim}"
            )
        now = datetime.now(UTC).isoformat()
        # Default to source content when caller (e.g. legacy IndexPipeline)
        # has not gone through an enricher.
        effective_text = embedding_text if embedding_text is not None else chunk.content
        cur = self._conn.cursor()
        # Try update first to preserve rowid (vec_chunks linkage).
        cur.execute("SELECT id FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,))
        existing = cur.fetchone()
        if existing is not None:
            rowid = int(existing["id"])
            cur.execute(
                """
                UPDATE chunks SET
                    rel_path=?, language=?, kind=?, symbol=?, parent_chunk_id=?,
                    start_line=?, end_line=?, content=?, content_hash=?,
                    embedding_model=?, embedding_dim=?, indexed_at=?, repo_id=?,
                    embedding_text=?, enricher_id=?
                WHERE id = ?
                """,
                (
                    chunk.rel_path, chunk.language, chunk.kind, chunk.symbol,
                    chunk.parent_chunk_id, chunk.start_line, chunk.end_line,
                    chunk.content, chunk.content_hash, embedding_model,
                    embedding_dim, now, repo_id,
                    effective_text, enricher_id, rowid,
                ),
            )
            cur.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
            cur.execute("DELETE FROM fts_chunks WHERE rowid = ?", (rowid,))
        else:
            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id, rel_path, language, kind, symbol, parent_chunk_id,
                    start_line, end_line, content, content_hash,
                    embedding_model, embedding_dim, indexed_at, repo_id,
                    embedding_text, enricher_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    chunk.chunk_id, chunk.rel_path, chunk.language, chunk.kind,
                    chunk.symbol, chunk.parent_chunk_id, chunk.start_line,
                    chunk.end_line, chunk.content, chunk.content_hash,
                    embedding_model, embedding_dim, now, repo_id,
                    effective_text, enricher_id,
                ),
            )
            rowid = int(cur.lastrowid or 0)

        cur.execute(
            "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize_vector(embedding)),
        )
        cur.execute(
            "INSERT INTO fts_chunks(rowid, content, rel_path) VALUES (?, ?, ?)",
            (rowid, _fts_text(chunk.content, chunk.rel_path, chunk.symbol), chunk.rel_path),
        )
        return rowid

    def commit(self) -> None:
        self._conn.commit()

    def clear(self) -> None:
        """Wipe the index. Used by `index --rebuild`.

        Preserves the ``repos`` table — repository identity survives rebuilds
        so the same id is reused, keeping FK semantics meaningful across runs.
        """
        cur = self._conn.cursor()
        cur.executescript(
            """
            DELETE FROM vec_chunks;
            DELETE FROM fts_chunks;
            DELETE FROM chunks;
            """
        )
        self._conn.commit()

    # ----- reads -----

    def search_dense(self, query_vec: list[float], k: int = 10) -> list[SearchHit]:
        """Top-k dense nearest neighbors."""
        if len(query_vec) != self._dim:
            raise StorageError(
                f"Query embedding dim {len(query_vec)} != index dim {self._dim}"
            )
        rows = self._conn.execute(
            f"""
            SELECT
                {_REPO_JOIN_COLUMNS},
                v.distance AS distance
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.rowid
            LEFT JOIN repos r ON r.id = c.repo_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_vector(query_vec), k),
        ).fetchall()
        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            distance = float(row["distance"])
            # vec_distance is L2 on normalized vectors → score = 1 - d²/2
            score = max(0.0, 1.0 - (distance * distance) / 2.0)
            hits.append(
                SearchHit(
                    rank=rank,
                    score=score,
                    rel_path=row["rel_path"],
                    language=row["language"],
                    kind=row["kind"],
                    symbol=row["symbol"],
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    snippet=_snippet(row["content"]),
                    chunk_id=row["chunk_id"],
                    repo=_repo_from_row(row),
                )
            )
        return hits

    def search_bm25(self, query: str, k: int = 50) -> list[SearchHit]:
        """Top-k via FTS5 BM25 ranking. Returns [] for empty/sanitized queries."""
        match_expr = _sanitize_fts_query(query)
        if not match_expr:
            return []
        try:
            rows = self._conn.execute(
                f"""
                SELECT
                    {_REPO_JOIN_COLUMNS},
                    bm25(fts_chunks) AS score
                FROM fts_chunks
                JOIN chunks c ON c.id = fts_chunks.rowid
                LEFT JOIN repos r ON r.id = c.repo_id
                WHERE fts_chunks MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match_expr, k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            log.warning("storage.fts_query_failed", query=query, error=str(exc))
            return []
        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            # FTS5 bm25() returns a non-positive number; closer to 0 is better.
            # Normalize to [0, 1) where 1 is most relevant.
            raw = float(row["score"])
            score = 1.0 / (1.0 + max(0.0, -raw))
            hits.append(
                SearchHit(
                    rank=rank,
                    score=score,
                    rel_path=row["rel_path"],
                    language=row["language"],
                    kind=row["kind"],
                    symbol=row["symbol"],
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    snippet=_snippet(row["content"]),
                    chunk_id=row["chunk_id"],
                    repo=_repo_from_row(row),
                )
            )
        return hits

    def search_symbol_tokens(self, query: str, k: int = 20) -> list[SearchHit]:
        """Match chunks whose `symbol` column contains identifier tokens from query.

        Used as a third stream in hybrid search (spec §6.3.5 — query routing).
        Cheap because the symbol column is small. Falls back to empty if no
        useful tokens survive the stopword filter.
        """
        tokens = _identifier_tokens(query)
        if not tokens:
            return []
        # OR-join LIKE patterns; rank by token-match count + length (shorter
        # symbols win on ties).
        like_clauses = " OR ".join("symbol LIKE ? COLLATE NOCASE" for _ in tokens)
        params: list[Any] = [f"%{t}%" for t in tokens]
        # Score = number of tokens matched.
        score_terms = " + ".join(
            "(CASE WHEN symbol LIKE ? COLLATE NOCASE THEN 1 ELSE 0 END)"
            for _ in tokens
        )
        params = params + [f"%{t}%" for t in tokens] + [k]
        rows = self._conn.execute(
            f"""
            SELECT {_REPO_JOIN_COLUMNS},
                   ({score_terms}) AS match_count
            FROM chunks c
            LEFT JOIN repos r ON r.id = c.repo_id
            WHERE ({like_clauses.replace("symbol", "c.symbol")})
              AND c.symbol IS NOT NULL
            ORDER BY match_count DESC, LENGTH(c.symbol) ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            SearchHit(
                rank=rank,
                score=float(row["match_count"]),
                rel_path=row["rel_path"],
                language=row["language"],
                kind=row["kind"],
                symbol=row["symbol"],
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                snippet=_snippet(row["content"]),
                chunk_id=row["chunk_id"],
                repo=_repo_from_row(row),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    # ----- structured lookups (used by MCP tools) -----

    def lookup_by_symbol(self, symbol: str, limit: int = 10) -> list[SearchHit]:
        """Exact + suffix match on symbol column. Used by `get_symbol`.

        Matches `Foo.bar`, `bar` (suffix), and case-insensitively as fallback.
        """
        rows = self._conn.execute(
            f"""
            SELECT {_REPO_JOIN_COLUMNS}
            FROM chunks c
            LEFT JOIN repos r ON r.id = c.repo_id
            WHERE c.symbol = ?
               OR c.symbol LIKE ? COLLATE NOCASE
               OR c.symbol LIKE ? COLLATE NOCASE
            ORDER BY
                CASE WHEN c.symbol = ? THEN 0
                     WHEN c.symbol LIKE ? COLLATE NOCASE THEN 1
                     ELSE 2 END,
                LENGTH(c.symbol)
            LIMIT ?
            """,
            (symbol, f"%.{symbol}", f"%{symbol}%", symbol, f"%.{symbol}", limit),
        ).fetchall()
        return [
            SearchHit(
                rank=rank,
                score=1.0,
                rel_path=row["rel_path"],
                language=row["language"],
                kind=row["kind"],
                symbol=row["symbol"],
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                snippet=_snippet(row["content"]),
                chunk_id=row["chunk_id"],
                repo=_repo_from_row(row),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def file_outline(self, rel_path: str) -> list[dict[str, Any]]:
        """Symbolic outline of a file (chunks sorted by start_line)."""
        rows = self._conn.execute(
            """
            SELECT chunk_id, kind, symbol, parent_chunk_id, start_line, end_line
            FROM chunks
            WHERE rel_path = ?
            ORDER BY start_line
            """,
            (rel_path,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """SELECT chunk_id, rel_path, language, kind, symbol, parent_chunk_id,
                      start_line, end_line, content, embedding_model
               FROM chunks WHERE chunk_id = ?""",
            (chunk_id,),
        ).fetchone()
        return dict(row) if row else None

    # ----- meta key/value (used by incremental reindex) -----

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value)
        )

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def stats(self) -> dict[str, Any]:
        cur = self._conn.cursor()
        chunks = cur.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        files = cur.execute(
            "SELECT COUNT(DISTINCT rel_path) AS n FROM chunks"
        ).fetchone()["n"]
        models = [
            dict(row)
            for row in cur.execute(
                """SELECT embedding_model, embedding_dim, COUNT(*) AS n
                   FROM chunks GROUP BY embedding_model, embedding_dim"""
            ).fetchall()
        ]
        # Per-enricher chunk distribution (schema v3). NULL = legacy / noop.
        enrichers = [
            dict(row)
            for row in cur.execute(
                """SELECT COALESCE(enricher_id, '(none)') AS enricher_id,
                          COUNT(*) AS n
                   FROM chunks GROUP BY enricher_id"""
            ).fetchall()
        ]
        enrichment_cache = cur.execute(
            "SELECT COUNT(*) AS n FROM enrichments"
        ).fetchone()["n"]
        size_bytes = self._db_path.stat().st_size if self._db_path.exists() else 0
        repos = [r.model_dump() for r in self.list_repos()]
        # Schema v4: graphify mirror summary (None when never loaded).
        graph_stats = self._graph_stats()
        return {
            "db_path": str(self._db_path),
            "size_bytes": size_bytes,
            "chunks": int(chunks),
            "files": int(files),
            "index_dim": self._dim,
            "models": models,
            "enrichers": enrichers,
            "enrichment_cache_size": int(enrichment_cache),
            "repos": repos,
            "graph": graph_stats,
        }

    def _graph_stats(self) -> dict[str, Any] | None:
        """Aggregate graphify mirror state. Returns None when never loaded."""
        cur = self._conn.cursor()
        # graph_meta is created at v4 but the row only exists after a load.
        try:
            row = cur.execute(
                "SELECT value FROM graph_meta WHERE key = 'last_loaded_at'"
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        n_nodes = cur.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()["n"]
        n_edges = cur.execute("SELECT COUNT(*) AS n FROM graph_edges").fetchone()["n"]
        by_conf = {
            r["confidence"]: int(r["n"])
            for r in cur.execute(
                "SELECT confidence, COUNT(*) AS n FROM graph_edges GROUP BY confidence"
            ).fetchall()
        }
        n_communities = cur.execute(
            "SELECT COUNT(DISTINCT community_id) AS n FROM graph_nodes WHERE community_id IS NOT NULL"
        ).fetchone()["n"]
        path_row = cur.execute(
            "SELECT value FROM graph_meta WHERE key = 'graph_path'"
        ).fetchone()
        conf_row = cur.execute(
            "SELECT value FROM graph_meta WHERE key = 'confidence_filter'"
        ).fetchone()
        return {
            "nodes": int(n_nodes),
            "edges_total": int(n_edges),
            "edges_by_confidence": by_conf,
            "communities": int(n_communities),
            "last_loaded_at": row["value"],
            "graph_path": path_row["value"] if path_row else None,
            "confidence_filter": (
                (conf_row["value"] or "").split(",") if conf_row else None
            ),
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

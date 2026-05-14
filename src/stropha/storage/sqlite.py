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
from ..logging import get_logger
from ..models import Chunk, SearchHit

log = get_logger(__name__)

SCHEMA_VERSION = 1


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
        self._conn.commit()

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

    # ----- writes -----

    def chunk_is_fresh(
        self,
        chunk_id: str,
        content_hash: str,
        embedding_model: str,
    ) -> bool:
        """True if a chunk with this id + hash + model already exists."""
        row = self._conn.execute(
            "SELECT content_hash, embedding_model FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return False
        return (
            row["content_hash"] == content_hash
            and row["embedding_model"] == embedding_model
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
    ) -> int:
        """Insert (or update) a chunk and its vector atomically. Returns rowid."""
        if len(embedding) != self._dim:
            raise StorageError(
                f"Embedding dim mismatch: got {len(embedding)}, expected {self._dim}"
            )
        now = datetime.now(UTC).isoformat()
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
                    embedding_model=?, embedding_dim=?, indexed_at=?
                WHERE id = ?
                """,
                (
                    chunk.rel_path, chunk.language, chunk.kind, chunk.symbol,
                    chunk.parent_chunk_id, chunk.start_line, chunk.end_line,
                    chunk.content, chunk.content_hash, embedding_model,
                    embedding_dim, now, rowid,
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
                    embedding_model, embedding_dim, indexed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    chunk.chunk_id, chunk.rel_path, chunk.language, chunk.kind,
                    chunk.symbol, chunk.parent_chunk_id, chunk.start_line,
                    chunk.end_line, chunk.content, chunk.content_hash,
                    embedding_model, embedding_dim, now,
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
        """Wipe the index. Used by `index --rebuild`."""
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
            """
            SELECT
                c.chunk_id, c.rel_path, c.language, c.kind, c.symbol,
                c.start_line, c.end_line, c.content,
                v.distance AS distance
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.rowid
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
                """
                SELECT
                    c.chunk_id, c.rel_path, c.language, c.kind, c.symbol,
                    c.start_line, c.end_line, c.content,
                    bm25(fts_chunks) AS score
                FROM fts_chunks
                JOIN chunks c ON c.id = fts_chunks.rowid
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
            SELECT chunk_id, rel_path, language, kind, symbol,
                   start_line, end_line, content,
                   ({score_terms}) AS match_count
            FROM chunks
            WHERE ({like_clauses}) AND symbol IS NOT NULL
            ORDER BY match_count DESC, LENGTH(symbol) ASC
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
            )
            for rank, row in enumerate(rows, start=1)
        ]

    # ----- structured lookups (used by MCP tools) -----

    def lookup_by_symbol(self, symbol: str, limit: int = 10) -> list[SearchHit]:
        """Exact + suffix match on symbol column. Used by `get_symbol`.

        Matches `Foo.bar`, `bar` (suffix), and case-insensitively as fallback.
        """
        rows = self._conn.execute(
            """
            SELECT chunk_id, rel_path, language, kind, symbol,
                   start_line, end_line, content
            FROM chunks
            WHERE symbol = ?
               OR symbol LIKE ? COLLATE NOCASE
               OR symbol LIKE ? COLLATE NOCASE
            ORDER BY
                CASE WHEN symbol = ? THEN 0
                     WHEN symbol LIKE ? COLLATE NOCASE THEN 1
                     ELSE 2 END,
                LENGTH(symbol)
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
        size_bytes = self._db_path.stat().st_size if self._db_path.exists() else 0
        return {
            "db_path": str(self._db_path),
            "size_bytes": size_bytes,
            "chunks": int(chunks),
            "files": int(files),
            "index_dim": self._dim,
            "models": models,
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

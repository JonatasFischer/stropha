"""Embed graph node labels into ``graph_nodes.embedding`` for the
``graph-vec`` retrieval stream (Trilha A L3).

Strategy: for every node in ``graph_nodes`` whose ``embedding_model`` does
not match the active embedder, generate a fresh embedding from the node's
``label`` (and optionally its ``community_label``) and store it as a BLOB.

Drift safety: switching to a different embedder writes a different
``embedding_model`` value → the loader re-embeds on the next run. The
loader never re-embeds nodes whose model already matches.

Storage: float32 little-endian packed (same convention as
``stropha.storage.sqlite._serialize_vector``).

Cost: 1 embedder call per uncached node. With local fastembed and a
typical 1500-node graph, embedding finishes in seconds; with Voyage it is
a single batched API call.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from ..embeddings.base import Embedder
from ..logging import get_logger
from ..storage.sqlite import Storage

log = get_logger(__name__)

_BATCH = 64


def _pack_vec(vec: Iterable[float]) -> bytes:
    arr = list(vec)
    return struct.pack(f"<{len(arr)}f", *arr)


def _unpack_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


@dataclass(frozen=True)
class VecLoadResult:
    embedded: int
    skipped_already_current: int
    failed: int
    duration_ms: int
    embedding_model: str

    def as_log_kwargs(self) -> dict[str, object]:
        return {
            "embedded": self.embedded,
            "skipped_already_current": self.skipped_already_current,
            "failed": self.failed,
            "duration_ms": self.duration_ms,
            "embedding_model": self.embedding_model,
        }


class GraphVecLoader:
    """Embed graph node labels using the active embedder.

    Idempotent: nodes whose stored ``embedding_model`` matches the active
    embedder are skipped (zero re-embedding cost on a no-op pass).
    """

    def __init__(
        self,
        storage: Storage,
        embedder: Embedder,
        *,
        include_community_in_label: bool = True,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._include_community = include_community_in_label

    @property
    def embedding_model(self) -> str:
        return self._embedder.model_name

    def needs_run(self) -> bool:
        """True iff at least one node lacks an embedding for the active model."""
        cur = self._storage._conn.cursor()  # noqa: SLF001
        try:
            row = cur.execute(
                """SELECT 1 FROM graph_nodes
                   WHERE embedding IS NULL OR embedding_model IS NOT ?
                   LIMIT 1""",
                (self.embedding_model,),
            ).fetchone()
        except Exception:  # pragma: no cover — table absent
            return False
        return row is not None

    def load(self, *, batch_size: int = _BATCH) -> VecLoadResult:
        """Embed every node whose stored embedding model is not current."""
        start = datetime.now(UTC)
        cur = self._storage._conn.cursor()  # noqa: SLF001
        rows = cur.execute(
            """SELECT node_id, label, community_label
               FROM graph_nodes
               WHERE embedding IS NULL OR embedding_model IS NOT ?""",
            (self.embedding_model,),
        ).fetchall()

        embedded = 0
        failed = 0
        if not rows:
            return VecLoadResult(
                embedded=0, skipped_already_current=self._count_current(),
                failed=0, duration_ms=0,
                embedding_model=self.embedding_model,
            )

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [self._render_text(r) for r in batch]
            try:
                vectors = self._embedder.embed_documents(texts)
            except Exception as exc:
                log.warning(
                    "graph_vec_loader.embed_failed",
                    batch_start=i, batch_size=len(batch), error=str(exc),
                )
                failed += len(batch)
                continue
            if len(vectors) != len(batch):
                log.warning(
                    "graph_vec_loader.length_mismatch",
                    expected=len(batch), got=len(vectors),
                )
                failed += len(batch)
                continue
            cur.executemany(
                """UPDATE graph_nodes
                   SET embedding = ?, embedding_model = ?, embedding_dim = ?
                   WHERE node_id = ?""",
                [
                    (_pack_vec(vec), self.embedding_model, len(vec), r["node_id"])
                    for r, vec in zip(batch, vectors, strict=True)
                ],
            )
            embedded += len(batch)

        self._storage.commit()
        duration_ms = int((datetime.now(UTC) - start).total_seconds() * 1000)
        result = VecLoadResult(
            embedded=embedded,
            skipped_already_current=self._count_current() - embedded,
            failed=failed,
            duration_ms=duration_ms,
            embedding_model=self.embedding_model,
        )
        log.info("graph_vec_loader.done", **result.as_log_kwargs())
        return result

    def _render_text(self, row) -> str:
        label = row["label"] or ""
        if self._include_community and row["community_label"]:
            return f"{row['community_label']}: {label}"
        return label

    def _count_current(self) -> int:
        return int(
            self._storage._conn.execute(  # noqa: SLF001
                """SELECT COUNT(*) AS n FROM graph_nodes
                   WHERE embedding IS NOT NULL AND embedding_model = ?""",
                (self.embedding_model,),
            ).fetchone()["n"]
        )

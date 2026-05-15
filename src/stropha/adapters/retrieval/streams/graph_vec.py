"""Graph node embedding retrieval stream — Trilha A L3.

Maps a query vector against every embedded ``graph_nodes.label`` (cosine
similarity, brute-force in Python). For each top-K matching node we
materialise a :class:`SearchHit` from the corresponding stropha chunk
(matched on ``source_file`` + line containment).

Why brute force: the typical code graph has 10³–10⁴ nodes. A single SIMD
pass over a contiguous numpy array (or Python list when numpy isn't
available) is sub-millisecond at this scale, simpler than maintaining
another sqlite-vec virtual table.

Failure mode: if no embeddings exist (graph never embedded, embedder
mismatch) the stream returns ``[]`` — the RRF coordinator simply ignores
the empty stream and the other streams carry the query.
"""

from __future__ import annotations

import math
import struct
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from ....logging import get_logger
from ....models import RepoRef, SearchHit
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter
from ....stages.storage import StorageStage

log = get_logger(__name__)


def _unpack_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain Python cosine. Returns 0.0 on zero vectors."""
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class GraphVecConfig(BaseModel):
    k: int = Field(default=20, ge=1, le=200)
    min_similarity: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Drop matches below this cosine similarity.",
    )


@register_adapter(stage="retrieval-stream", name="graph-vec")
class GraphVecStream:
    """Top-K graph nodes by cosine similarity to the query vector."""

    Config = GraphVecConfig

    def __init__(
        self,
        config: GraphVecConfig | None = None,
        *,
        storage: StorageStage | None = None,
    ) -> None:
        if storage is None:
            raise ValueError("GraphVecStream requires storage=")
        self._config = config or GraphVecConfig()
        self._storage = storage

    @property
    def stage_name(self) -> str:
        return "retrieval-stream"

    @property
    def adapter_name(self) -> str:
        return "graph-vec"

    @property
    def adapter_id(self) -> str:
        return f"graph-vec:k={self._config.k}:min={self._config.min_similarity}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return GraphVecConfig

    @property
    def k(self) -> int:
        return self._config.k

    def health(self) -> StageHealth:
        try:
            n = int(
                self._storage._conn.execute(  # noqa: SLF001
                    "SELECT COUNT(*) AS n FROM graph_nodes WHERE embedding IS NOT NULL"
                ).fetchone()["n"]
            )
        except Exception:
            return StageHealth(
                status="warning",
                message="graph_nodes table missing — run `stropha index` after `graphify .`",
            )
        if n == 0:
            return StageHealth(
                status="warning",
                message=(
                    "no graph node embeddings — run `stropha index` so the "
                    "GraphVecLoader populates them, or disable this stream."
                ),
            )
        return StageHealth(
            status="ready",
            message=f"graph-vec ready ({n} embedded nodes, min_sim={self._config.min_similarity})",
        )

    def search(self, query: str, query_vec: list[float] | None) -> list[SearchHit]:
        if query_vec is None:
            return []
        rows = self._storage._conn.execute(  # noqa: SLF001
            """SELECT node_id, label, source_file, source_location,
                      community_label, embedding
               FROM graph_nodes WHERE embedding IS NOT NULL"""
        ).fetchall()
        if not rows:
            return []

        # Brute-force cosine — sub-ms for typical N (≤50K nodes, dim 1024).
        scored: list[tuple[float, dict]] = []
        for r in rows:
            sim = _cosine(query_vec, _unpack_vec(r["embedding"]))
            if sim >= self._config.min_similarity:
                scored.append((sim, dict(r)))
        scored.sort(key=lambda t: -t[0])
        scored = scored[: self._config.k]

        # Hydrate to SearchHit by joining with chunks on (rel_path, line range).
        hits: list[SearchHit] = []
        for sim, node in scored:
            hit = self._node_to_hit(node, sim)
            if hit is not None:
                hits.append(hit)
        return hits

    # ------------------------------------------------------------------ helpers

    def _node_to_hit(self, node: dict, sim: float) -> SearchHit | None:
        rel_path = node.get("source_file")
        if not rel_path:
            return None
        line = self._location_to_line(node.get("source_location"))
        cur = self._storage._conn.cursor()  # noqa: SLF001
        chunk = None
        if line is not None:
            chunk = cur.execute(
                """SELECT c.*, r.normalized_key, r.remote_url, r.default_branch,
                          r.head_commit
                   FROM chunks c LEFT JOIN repos r ON c.repo_id = r.id
                   WHERE c.rel_path = ?
                     AND c.start_line <= ? AND c.end_line >= ?
                   ORDER BY (c.end_line - c.start_line) ASC LIMIT 1""",
                (rel_path, line, line),
            ).fetchone()
        if chunk is None:
            chunk = cur.execute(
                """SELECT c.*, r.normalized_key, r.remote_url, r.default_branch,
                          r.head_commit
                   FROM chunks c LEFT JOIN repos r ON c.repo_id = r.id
                   WHERE c.rel_path = ? ORDER BY c.start_line ASC LIMIT 1""",
                (rel_path,),
            ).fetchone()
        if chunk is None:
            return None
        # Build SearchHit. We intentionally use the cosine sim as score so
        # downstream RRF treats higher values as better-ranked.
        repo: RepoRef | None = None
        if chunk["normalized_key"]:
            repo = RepoRef(
                normalized_key=chunk["normalized_key"],
                url=chunk["remote_url"],
                default_branch=chunk["default_branch"],
                head_commit=chunk["head_commit"],
            )
        snippet = (chunk["content"] or "")[:400]
        return SearchHit(
            rank=0,  # populated by the RRF fuser
            score=sim,
            rel_path=chunk["rel_path"],
            language=chunk["language"],
            kind=chunk["kind"],
            symbol=chunk["symbol"],
            start_line=int(chunk["start_line"]),
            end_line=int(chunk["end_line"]),
            snippet=snippet,
            chunk_id=chunk["chunk_id"],
            repo=repo,
        )

    @staticmethod
    def _location_to_line(loc: str | None) -> int | None:
        if not loc:
            return None
        s = loc.strip().lstrip("L")
        try:
            return int(s.split(":")[0])
        except (ValueError, TypeError):
            return None

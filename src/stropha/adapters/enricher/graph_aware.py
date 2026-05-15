"""Graph-aware enricher — L2 augmentation per RFC §0.

Prepends the matching graphify community label and (optionally) the node
label to each chunk's ``embedding_text``. Improves BM25 / FTS5 recall for
queries that mention a community-level concept (e.g. "where is the hybrid
retrieval pipeline?") without requiring any LLM round-trip.

How it works:
1. At indexing time, every chunk has a ``source_file`` (rel_path) and a
   ``start_line`` / ``end_line`` window.
2. We query the storage's ``graph_nodes`` table for the node whose
   ``source_file`` matches and whose ``source_location`` line number falls
   inside the chunk window.
3. If found, we prepend ``# in <community_label>: <node_label>`` to the
   chunk's content and use that as ``embedding_text``.
4. If no node matches (chunk pre-dates graph load, file untracked by
   graphify, etc.) the enricher returns the raw content — graceful fallback.

Drift safety: ``adapter_id`` digests every config flag, so toggling
``include_community`` or ``include_node_label`` triggers re-enrichment on
the next index run (ADR-004 contract).

Cost: zero LLM, zero network. Two indexed SQL lookups per chunk
(``rel_path`` + line range). Negligible vs. embedding cost.
"""

from __future__ import annotations

import re
import sqlite3

from pydantic import BaseModel, Field

from ...models import Chunk
from ...pipeline.base import StageContext, StageHealth
from ...pipeline.registry import register_adapter
from ...stages.enricher import EnricherStage

_LOC_LINE = re.compile(r"L?(\d+)")


def _location_to_line(loc: str | None) -> int | None:
    if not loc:
        return None
    m = _LOC_LINE.search(loc)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (ValueError, TypeError):
        return None


class GraphAwareEnricherConfig(BaseModel):
    include_community: bool = Field(
        default=True,
        description="Prepend the community label (e.g. 'Hybrid Retrieval & RRF').",
    )
    include_node_label: bool = Field(
        default=True,
        description="Prepend the matching graph node label (e.g. 'StudyService.submitAnswer').",
    )
    include_parent_skeleton: bool = Field(
        default=True,
        description="Also prepend parent chunk skeleton (combines hierarchical + L2).",
    )
    separator: str = Field(default="\n", description="Joiner between prefix lines.")


@register_adapter(stage="enricher", name="graph-aware")
class GraphAwareEnricher(EnricherStage):
    """Prepend graphify community + node label, falling back to raw content."""

    Config = GraphAwareEnricherConfig

    def __init__(
        self,
        config: GraphAwareEnricherConfig | None = None,
        *,
        storage: object | None = None,
    ) -> None:
        self._config = config or GraphAwareEnricherConfig()
        # Storage is injected by the pipeline builder. When called from the
        # registry without a builder (CLI `pipeline show --no-open`) the
        # adapter still constructs but health() reports a warning.
        self._storage = storage

    @property
    def stage_name(self) -> str:
        return "enricher"

    @property
    def adapter_name(self) -> str:
        return "graph-aware"

    @property
    def adapter_id(self) -> str:
        # Every flag that affects output is part of the id (drift detection).
        parts = ["graph-aware"]
        if self._config.include_community:
            parts.append("c")
        if self._config.include_node_label:
            parts.append("n")
        if self._config.include_parent_skeleton:
            parts.append("p")
        return ":".join(parts)

    @property
    def config_schema(self) -> type[BaseModel]:
        return GraphAwareEnricherConfig

    def health(self) -> StageHealth:
        if self._storage is None:
            return StageHealth(
                status="warning",
                message=(
                    "graph-aware enricher constructed without storage handle — "
                    "L2 augmentation will be inactive (falls back to raw content)."
                ),
            )
        # Probe graph_nodes existence
        try:
            row = self._storage._conn.execute(  # type: ignore[attr-defined]
                "SELECT COUNT(*) AS n FROM graph_nodes"
            ).fetchone()
            n = int(row["n"])
        except (sqlite3.OperationalError, AttributeError):
            return StageHealth(
                status="warning",
                message=(
                    "graph_nodes table missing or unreadable — first index "
                    "run will populate it via GraphifyLoader."
                ),
            )
        if n == 0:
            return StageHealth(
                status="warning",
                message=(
                    "graph empty — run `graphify .` once to bootstrap. "
                    "L2 augmentation kicks in on the next index."
                ),
            )
        return StageHealth(
            status="ready",
            message=f"graph mirror has {n} nodes; augmentation active",
        )

    def enrich(self, chunk: Chunk, ctx: StageContext) -> str:
        prefix_lines: list[str] = []

        # 1. Optional parent skeleton (hierarchical compatibility)
        if self._config.include_parent_skeleton and ctx.parent_chunk is not None:
            parent: Chunk = ctx.parent_chunk
            symbol = parent.symbol or "<anon>"
            prefix_lines.append(
                f"# in {parent.kind} {symbol} ({parent.rel_path}:{parent.start_line})"
            )

        # 2. Graph augmentation
        if self._storage is not None and (
            self._config.include_community or self._config.include_node_label
        ):
            node = self._lookup_graph_node(chunk)
            if node is not None:
                if self._config.include_community and node["community_label"]:
                    prefix_lines.append(f"# community: {node['community_label']}")
                if self._config.include_node_label and node["label"]:
                    prefix_lines.append(f"# node: {node['label']}")

        if not prefix_lines:
            return chunk.content
        return self._config.separator.join(prefix_lines) + self._config.separator + chunk.content

    # ------------------------------------------------------------------ helpers

    def _lookup_graph_node(self, chunk: Chunk) -> sqlite3.Row | None:
        """Find the graph node whose source_file matches and whose line falls
        inside the chunk window. Tightest match wins.
        """
        try:
            # 1. Exact line containment
            rows = self._storage._conn.execute(  # type: ignore[attr-defined]
                """SELECT label, community_label, source_location
                   FROM graph_nodes
                   WHERE source_file = ?""",
                (chunk.rel_path,),
            ).fetchall()
        except (sqlite3.OperationalError, AttributeError):
            return None
        if not rows:
            return None
        best: sqlite3.Row | None = None
        for r in rows:
            line = _location_to_line(r["source_location"])
            if line is None or not (chunk.start_line <= line <= chunk.end_line):
                continue
            # Prefer the line closest to chunk start (most specific node)
            if best is None or abs(_location_to_line(best["source_location"]) - chunk.start_line) > abs(line - chunk.start_line):  # type: ignore[arg-type]
                best = r
        # Fallback: any node from this file (when no line match)
        if best is None:
            best = rows[0]
        return best

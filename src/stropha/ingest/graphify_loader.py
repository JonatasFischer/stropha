"""Graphify graph loader — populates ``graph_nodes`` / ``graph_edges`` / ``graph_meta``.

Reads ``graphify-out/graph.json`` (NetworkX node-link format produced by the
graphify CLI) and mirrors it into SQLite so the MCP tools (``find_callers``,
``find_related``, ``get_community``, ``find_rationale``) can answer structural
queries without re-loading the JSON on every call.

Per RFC §6.1 (``docs/architecture/stropha-graphify-integration.md``):
- **Idempotent**: ``load()`` called N times produces the same database state.
- **Transactional**: a failure mid-load leaves the previous version intact
  (everything wrapped in a single SQLite transaction).
- **Confidence-filtered**: edges below ``STROPHA_GRAPH_CONFIDENCE`` are
  dropped at load time. Default ``EXTRACTED`` only — the high-precision
  subset derived deterministically from the AST.

The loader does NOT call graphify itself. The graph is produced out-of-band
(``graphify .`` for the LLM-augmented bootstrap, or ``graphify update . --no-cluster``
for code-only incremental refreshes — installed as a post-commit hook step).

Usage:

    from stropha.ingest.graphify_loader import GraphifyLoader

    loader = GraphifyLoader(storage, repo_root)
    if loader.is_stale():
        result = loader.load()
        log.info("graphify_loader.done", **result.as_log_kwargs())
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..logging import get_logger
from ..storage.sqlite import Storage

log = get_logger(__name__)

# RFC §6.1 default. Comma-separated list of confidence tiers to keep.
DEFAULT_CONFIDENCE = ("EXTRACTED",)

# Set of all valid graphify confidence tiers. Used to validate user input.
_VALID_CONFIDENCE = {"EXTRACTED", "INFERRED", "AMBIGUOUS"}


def _resolve_confidence_filter(env_value: str | None) -> tuple[str, ...]:
    """Parse ``STROPHA_GRAPH_CONFIDENCE`` env var into a tuple of tiers.

    Empty / unset / invalid values fall back to :data:`DEFAULT_CONFIDENCE`.
    """
    if not env_value:
        return DEFAULT_CONFIDENCE
    parts = tuple(p.strip().upper() for p in env_value.split(",") if p.strip())
    valid = tuple(p for p in parts if p in _VALID_CONFIDENCE)
    if not valid:
        log.warning(
            "graphify_loader.confidence_invalid",
            value=env_value,
            falling_back_to=list(DEFAULT_CONFIDENCE),
        )
        return DEFAULT_CONFIDENCE
    return valid


@dataclass(frozen=True)
class LoadResult:
    """Outcome of a :meth:`GraphifyLoader.load` invocation."""

    nodes_loaded: int
    edges_loaded: int
    edges_filtered: int
    communities: int
    duration_ms: int
    graph_path: str
    confidence_filter: tuple[str, ...]

    def as_log_kwargs(self) -> dict[str, object]:
        return {
            "nodes": self.nodes_loaded,
            "edges_loaded": self.edges_loaded,
            "edges_filtered": self.edges_filtered,
            "communities": self.communities,
            "duration_ms": self.duration_ms,
            "graph_path": self.graph_path,
            "confidence_filter": list(self.confidence_filter),
        }


class GraphifyLoader:
    """Mirror ``graphify-out/graph.json`` into the SQLite ``graph_*`` tables.

    The loader is conditional: if no graph file exists the loader simply
    reports ``is_stale() → False`` and ``load()`` is a no-op. This keeps
    stropha working in repos that have never run graphify.
    """

    def __init__(
        self,
        storage: Storage,
        repo_root: Path,
        *,
        graph_path: Path | None = None,
        confidence_filter: tuple[str, ...] | None = None,
    ) -> None:
        self._storage = storage
        # Allow override (tests, --refresh-graph flag, custom out dir).
        self._graph_path = graph_path or self._resolve_graph_path(repo_root)
        self._confidence = confidence_filter or _resolve_confidence_filter(
            os.environ.get("STROPHA_GRAPH_CONFIDENCE")
        )

    # ------------------------------------------------------------------ paths

    @staticmethod
    def _resolve_graph_path(repo_root: Path) -> Path:
        """Default location: ``<repo_root>/graphify-out/graph.json``.

        Honours ``STROPHA_GRAPHIFY_OUT`` for users who keep the graph
        outside the repo (shared workspace, separate volume).
        """
        override = os.environ.get("STROPHA_GRAPHIFY_OUT")
        if override:
            return Path(override).expanduser() / "graph.json"
        return repo_root / "graphify-out" / "graph.json"

    @property
    def graph_path(self) -> Path:
        return self._graph_path

    @property
    def confidence_filter(self) -> tuple[str, ...]:
        return self._confidence

    # --------------------------------------------------------------- staleness

    def exists(self) -> bool:
        """True iff the graph file is present on disk."""
        return self._graph_path.is_file()

    def is_stale(self) -> bool:
        """True iff the graph should be reloaded.

        Compares file mtime with ``graph_meta.last_loaded_mtime``. Returns
        ``False`` when the graph is missing entirely (nothing to load).
        """
        if not self.exists():
            return False
        try:
            file_mtime = self._graph_path.stat().st_mtime
        except OSError:
            return False
        last_str = self._get_meta("last_loaded_mtime")
        if last_str is None:
            return True
        try:
            return float(file_mtime) > float(last_str)
        except (TypeError, ValueError):
            return True

    # --------------------------------------------------------------------- io

    def load(self) -> LoadResult | None:
        """Replace ``graph_nodes`` / ``graph_edges`` from the graph file.

        Returns ``None`` if the graph file does not exist. Otherwise returns
        a :class:`LoadResult` summarising the reload.
        """
        if not self.exists():
            log.info("graphify_loader.missing", path=str(self._graph_path))
            return None
        start = datetime.now(UTC)
        log.info(
            "graphify_loader.start",
            path=str(self._graph_path),
            confidence=list(self._confidence),
        )
        try:
            payload = json.loads(self._graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.error(
                "graphify_loader.parse_failed",
                path=str(self._graph_path),
                error=str(exc),
            )
            return None

        nodes = list(payload.get("nodes") or [])
        # NetworkX node-link uses ``links``; some exports use ``edges``.
        raw_edges = list(payload.get("links") or payload.get("edges") or [])

        # Index communities. Top-level ``communities`` is a dict
        # ``{community_id: [node_id, ...]}`` written by ``graphify.export.to_json``.
        communities = payload.get("communities") or {}
        community_labels = payload.get("community_labels") or {}
        node_to_community: dict[str, int] = {}
        for cid_str, members in communities.items():
            try:
                cid = int(cid_str)
            except (TypeError, ValueError):
                continue
            for nid in members or []:
                node_to_community[str(nid)] = cid

        cur = self._storage._conn.cursor()
        # Wrap the whole thing in a transaction. SQLite already buffers writes
        # in memory but BEGIN ensures atomic visibility.
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute("DELETE FROM graph_edges")
            cur.execute("DELETE FROM graph_nodes")
            cur.execute("DELETE FROM graph_meta")

            cur.executemany(
                """INSERT OR REPLACE INTO graph_nodes
                   (node_id, label, file_type, source_file, source_location,
                    community_id, community_label)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        str(n["id"]),
                        str(n.get("label") or n["id"]),
                        n.get("file_type"),
                        n.get("source_file"),
                        n.get("source_location"),
                        node_to_community.get(str(n["id"])),
                        community_labels.get(
                            str(node_to_community.get(str(n["id"]))), None
                        ) if node_to_community.get(str(n["id"])) is not None else None,
                    )
                    for n in nodes
                    if "id" in n
                ],
            )

            edges_loaded = 0
            edges_filtered = 0
            edge_rows: list[tuple] = []
            for e in raw_edges:
                conf = (e.get("confidence") or "").upper()
                if conf not in self._confidence:
                    edges_filtered += 1
                    continue
                edge_rows.append(
                    (
                        str(e["source"]),
                        str(e["target"]),
                        str(e.get("relation") or ""),
                        conf,
                        e.get("confidence_score"),
                        e.get("context"),
                        e.get("source_file"),
                        e.get("source_location"),
                        e.get("weight"),
                    )
                )
                edges_loaded += 1
            if edge_rows:
                cur.executemany(
                    """INSERT OR REPLACE INTO graph_edges
                       (source, target, relation, confidence, confidence_score,
                        context, source_file, source_location, weight)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    edge_rows,
                )

            now_iso = datetime.now(UTC).isoformat()
            file_mtime = self._graph_path.stat().st_mtime
            for k, v in (
                ("last_loaded_at", now_iso),
                ("last_loaded_mtime", str(file_mtime)),
                ("graph_path", str(self._graph_path)),
                ("confidence_filter", ",".join(self._confidence)),
            ):
                cur.execute(
                    "INSERT OR REPLACE INTO graph_meta(key, value) VALUES(?, ?)",
                    (k, v),
                )
            self._storage.commit()
        except Exception:
            self._storage._conn.rollback()
            raise

        duration_ms = int(
            (datetime.now(UTC) - start).total_seconds() * 1000
        )
        result = LoadResult(
            nodes_loaded=len(nodes),
            edges_loaded=edges_loaded,
            edges_filtered=edges_filtered,
            communities=len(communities),
            duration_ms=duration_ms,
            graph_path=str(self._graph_path),
            confidence_filter=self._confidence,
        )
        log.info("graphify_loader.done", **result.as_log_kwargs())
        return result

    # ------------------------------------------------------------- introspection

    def stats(self) -> dict[str, object] | None:
        """Return a summary suitable for ``Storage.stats()['graph']``.

        ``None`` when the loader has never populated the tables.
        """
        cur = self._storage._conn.cursor()
        last_loaded = self._get_meta("last_loaded_at")
        if last_loaded is None:
            return None
        n_nodes = cur.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()["n"]
        n_edges = cur.execute("SELECT COUNT(*) AS n FROM graph_edges").fetchone()["n"]
        by_conf = {
            row["confidence"]: int(row["n"])
            for row in cur.execute(
                "SELECT confidence, COUNT(*) AS n FROM graph_edges GROUP BY confidence"
            ).fetchall()
        }
        n_communities = cur.execute(
            "SELECT COUNT(DISTINCT community_id) AS n FROM graph_nodes WHERE community_id IS NOT NULL"
        ).fetchone()["n"]
        return {
            "nodes": int(n_nodes),
            "edges_total": int(n_edges),
            "edges_by_confidence": by_conf,
            "communities": int(n_communities),
            "last_loaded_at": last_loaded,
            "graph_path": self._get_meta("graph_path") or str(self._graph_path),
            "confidence_filter": (
                (self._get_meta("confidence_filter") or "").split(",")
                if self._get_meta("confidence_filter")
                else list(self._confidence)
            ),
        }

    # ------------------------------------------------------------------ helpers

    def _get_meta(self, key: str) -> str | None:
        row = self._storage._conn.execute(
            "SELECT value FROM graph_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

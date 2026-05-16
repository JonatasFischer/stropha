"""Graph traversal helpers backing the ``find_*`` MCP tools.

Reads the SQLite mirror populated by :class:`stropha.ingest.graphify_loader.GraphifyLoader`.
Pure SQL — no LLM, no NetworkX in the hot path. All queries are O(degree)
because every traversal direction is indexed
(``idx_graph_edges_target_relation`` / ``idx_graph_edges_source_relation``).

Per RFC §5 (``docs/architecture/stropha-graphify-integration.md``):
- ``find_callers``: who calls X? (``calls`` edges, target = X, BFS depth ≤ 3)
- ``find_related``: anything connected to X (any relation, depth ≤ 2)
- ``get_community``: peer nodes in the same community as X
- ``find_rationale``: which rationale node explains X (``rationale_for`` edges)

All functions:
- return ``[]`` when the graph mirror is empty (no graphify-out loaded)
- enrich each hit with the matching chunk's ``rel_path`` / ``start_line`` /
  ``end_line`` / ``snippet`` when ``source_file`` matches
- never raise — a missing symbol is just an empty list
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..logging import get_logger
from ..storage.sqlite import Storage

log = get_logger(__name__)


# --------------------------------------------------------------------- models


@dataclass(frozen=True)
class GraphNode:
    """Hydrated graphify node, optionally joined with a stropha chunk."""

    node_id: str
    label: str
    file_type: str | None
    source_file: str | None
    source_location: str | None
    community_id: int | None
    community_label: str | None
    repo_id: int | None = None  # schema v7: multi-repo support
    chunk_rel_path: str | None = None
    chunk_start_line: int | None = None
    chunk_end_line: int | None = None
    chunk_snippet: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "file_type": self.file_type,
            "source_file": self.source_file,
            "source_location": self.source_location,
            "repo_id": self.repo_id,
            "community": (
                {"id": self.community_id, "label": self.community_label}
                if self.community_id is not None
                else None
            ),
            "chunk": (
                {
                    "rel_path": self.chunk_rel_path,
                    "start_line": self.chunk_start_line,
                    "end_line": self.chunk_end_line,
                    "snippet": self.chunk_snippet,
                }
                if self.chunk_rel_path
                else None
            ),
        }


@dataclass(frozen=True)
class GraphEdge:
    """One edge in the result subgraph."""

    source: str
    target: str
    relation: str
    confidence: str
    confidence_score: float | None
    source_file: str | None
    source_location: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "confidence": self.confidence,
            "confidence_score": self.confidence_score,
            "source_file": self.source_file,
            "source_location": self.source_location,
        }


# --------------------------------------------------------------------- helpers


def graph_loaded(storage: Storage) -> bool:
    """True iff the graphify mirror tables exist AND have data.

    Used by ``server.py`` to gate tool registration: ``find_*`` tools never
    appear in ``tools/list`` when the graph is absent (RFC §8 MUST NOT).
    """
    try:
        row = storage._conn.execute(
            "SELECT 1 FROM graph_meta WHERE key = 'last_loaded_at' LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return False
    if row is None:
        return False
    n = storage._conn.execute(
        "SELECT COUNT(*) AS n FROM graph_nodes"
    ).fetchone()["n"]
    return int(n) > 0


_LOC_LINE = re.compile(r"L?(\d+)")


def _parse_location_to_line(loc: str | None) -> int | None:
    """Best-effort: convert ``'L42'`` / ``'42'`` / ``'42:5'`` into ``42``."""
    if not loc:
        return None
    m = _LOC_LINE.search(loc)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (ValueError, TypeError):
        return None


def resolve_symbol_to_node(storage: Storage, symbol: str) -> str | None:
    """Find the most likely ``node_id`` for a fully-qualified symbol.

    Strategy (cheapest → most permissive):
      1. exact label match
      2. label ends with ``.symbol`` (handles ``foo.bar.Baz`` → ``Baz``)
      3. label LIKE ``%symbol%``
    Tie-breaker: longest label (most specific).
    """
    if not symbol:
        return None
    cur = storage._conn.cursor()
    # 1: exact
    row = cur.execute(
        "SELECT node_id FROM graph_nodes WHERE label = ? LIMIT 1", (symbol,)
    ).fetchone()
    if row:
        return row["node_id"]
    # 2: dotted suffix (Class.method or module.Class)
    row = cur.execute(
        "SELECT node_id FROM graph_nodes WHERE label LIKE ? ORDER BY LENGTH(label) DESC LIMIT 1",
        (f"%.{symbol}",),
    ).fetchone()
    if row:
        return row["node_id"]
    # 3: substring fallback
    row = cur.execute(
        "SELECT node_id FROM graph_nodes WHERE label LIKE ? ORDER BY LENGTH(label) ASC LIMIT 1",
        (f"%{symbol}%",),
    ).fetchone()
    if row:
        return row["node_id"]
    return None


def _hydrate_node(storage: Storage, row: sqlite3.Row) -> GraphNode:
    """Take a graph_nodes row and (when possible) attach a chunk snippet."""
    line = _parse_location_to_line(row["source_location"])
    rel_path = row["source_file"]
    # Get repo_id if present in the row (schema v7)
    repo_id = row["repo_id"] if "repo_id" in row.keys() else None
    chunk = None
    if rel_path:
        # Pick the chunk whose [start_line, end_line] contains the node's line.
        # When repo_id is available, scope chunk lookup to that repo to avoid
        # cross-repo collisions on identical file paths.
        if line is not None:
            if repo_id is not None:
                chunk = storage._conn.execute(
                    """SELECT rel_path, start_line, end_line, content
                       FROM chunks
                       WHERE rel_path = ? AND repo_id = ?
                         AND start_line <= ? AND end_line >= ?
                       ORDER BY (end_line - start_line) ASC LIMIT 1""",
                    (rel_path, repo_id, line, line),
                ).fetchone()
            else:
                chunk = storage._conn.execute(
                    """SELECT rel_path, start_line, end_line, content
                       FROM chunks
                       WHERE rel_path = ? AND start_line <= ? AND end_line >= ?
                       ORDER BY (end_line - start_line) ASC LIMIT 1""",
                    (rel_path, line, line),
                ).fetchone()
        # Fallback: file-level chunk
        if chunk is None:
            if repo_id is not None:
                chunk = storage._conn.execute(
                    """SELECT rel_path, start_line, end_line, content
                       FROM chunks WHERE rel_path = ? AND repo_id = ?
                       ORDER BY start_line ASC LIMIT 1""",
                    (rel_path, repo_id),
                ).fetchone()
            else:
                chunk = storage._conn.execute(
                    """SELECT rel_path, start_line, end_line, content
                       FROM chunks WHERE rel_path = ? ORDER BY start_line ASC LIMIT 1""",
                    (rel_path,),
                ).fetchone()
    snippet = None
    if chunk and chunk["content"]:
        snippet = chunk["content"][:400]
    return GraphNode(
        node_id=row["node_id"],
        label=row["label"],
        file_type=row["file_type"],
        source_file=row["source_file"],
        source_location=row["source_location"],
        community_id=row["community_id"],
        community_label=row["community_label"],
        repo_id=repo_id,
        chunk_rel_path=chunk["rel_path"] if chunk else None,
        chunk_start_line=chunk["start_line"] if chunk else None,
        chunk_end_line=chunk["end_line"] if chunk else None,
        chunk_snippet=snippet,
    )


def _fetch_node_rows(storage: Storage, node_ids: Sequence[str]) -> dict[str, sqlite3.Row]:
    if not node_ids:
        return {}
    placeholders = ",".join("?" * len(node_ids))
    rows = storage._conn.execute(
        f"SELECT * FROM graph_nodes WHERE node_id IN ({placeholders})",
        tuple(node_ids),
    ).fetchall()
    return {r["node_id"]: r for r in rows}


# --------------------------------------------------------------------- tools


def find_callers(
    storage: Storage,
    symbol: str,
    *,
    depth: int = 1,
    limit: int = 20,
    confidence: tuple[str, ...] = ("EXTRACTED",),
) -> dict[str, Any]:
    """Return code locations that call ``symbol`` (BFS up incoming ``calls`` edges)."""
    depth = max(1, min(depth, 3))
    limit = max(1, min(limit, 100))

    target_id = resolve_symbol_to_node(storage, symbol)
    if target_id is None:
        return {"symbol": symbol, "resolved_node": None, "callers": [], "edges": []}

    visited: set[str] = {target_id}
    frontier: list[str] = [target_id]
    found_ids: list[str] = []
    edges: list[GraphEdge] = []
    conf_placeholders = ",".join("?" * len(confidence))

    for _ in range(depth):
        if not frontier or len(found_ids) >= limit:
            break
        f_placeholders = ",".join("?" * len(frontier))
        rows = storage._conn.execute(
            f"""SELECT source, target, relation, confidence, confidence_score,
                       source_file, source_location
                FROM graph_edges
                WHERE target IN ({f_placeholders})
                  AND relation = 'calls'
                  AND confidence IN ({conf_placeholders})
                LIMIT ?""",
            (*frontier, *confidence, limit * 4),
        ).fetchall()
        new_frontier: list[str] = []
        for r in rows:
            edges.append(
                GraphEdge(
                    source=r["source"], target=r["target"], relation=r["relation"],
                    confidence=r["confidence"],
                    confidence_score=r["confidence_score"],
                    source_file=r["source_file"],
                    source_location=r["source_location"],
                )
            )
            if r["source"] not in visited:
                visited.add(r["source"])
                new_frontier.append(r["source"])
                found_ids.append(r["source"])
                if len(found_ids) >= limit:
                    break
        frontier = new_frontier

    rows_by_id = _fetch_node_rows(storage, found_ids)
    callers = [_hydrate_node(storage, rows_by_id[nid]).as_dict()
               for nid in found_ids if nid in rows_by_id]
    return {
        "symbol": symbol,
        "resolved_node": target_id,
        "depth": depth,
        "callers": callers,
        "edges": [e.as_dict() for e in edges],
        "provenance": "graphify-out/graph.json (confidence={})".format(",".join(confidence)),
    }


def find_related(
    storage: Storage,
    symbol: str,
    *,
    depth: int = 1,
    limit: int = 20,
    relations: Sequence[str] | None = None,
    confidence: tuple[str, ...] = ("EXTRACTED",),
) -> dict[str, Any]:
    """Return any node connected to ``symbol`` regardless of edge direction.

    Useful when the caller doesn't know the relationship type ahead of time.
    Symmetric BFS (in + out edges).
    """
    depth = max(1, min(depth, 3))
    limit = max(1, min(limit, 100))

    src_id = resolve_symbol_to_node(storage, symbol)
    if src_id is None:
        return {"symbol": symbol, "resolved_node": None, "related": [], "edges": []}

    visited: set[str] = {src_id}
    frontier: list[str] = [src_id]
    found_ids: list[str] = []
    edges: list[GraphEdge] = []
    conf_placeholders = ",".join("?" * len(confidence))
    rel_clause = ""
    rel_args: tuple[str, ...] = ()
    if relations:
        rel_placeholders = ",".join("?" * len(relations))
        rel_clause = f" AND relation IN ({rel_placeholders})"
        rel_args = tuple(relations)

    for _ in range(depth):
        if not frontier or len(found_ids) >= limit:
            break
        f_placeholders = ",".join("?" * len(frontier))
        rows = storage._conn.execute(
            f"""SELECT source, target, relation, confidence, confidence_score,
                       source_file, source_location
                FROM graph_edges
                WHERE (source IN ({f_placeholders}) OR target IN ({f_placeholders}))
                  AND confidence IN ({conf_placeholders}) {rel_clause}
                LIMIT ?""",
            (*frontier, *frontier, *confidence, *rel_args, limit * 4),
        ).fetchall()
        new_frontier: list[str] = []
        for r in rows:
            edges.append(
                GraphEdge(
                    source=r["source"], target=r["target"], relation=r["relation"],
                    confidence=r["confidence"],
                    confidence_score=r["confidence_score"],
                    source_file=r["source_file"],
                    source_location=r["source_location"],
                )
            )
            for cand in (r["source"], r["target"]):
                if cand not in visited:
                    visited.add(cand)
                    new_frontier.append(cand)
                    found_ids.append(cand)
                    if len(found_ids) >= limit:
                        break
            if len(found_ids) >= limit:
                break
        frontier = new_frontier

    rows_by_id = _fetch_node_rows(storage, found_ids)
    related = [_hydrate_node(storage, rows_by_id[nid]).as_dict()
               for nid in found_ids if nid in rows_by_id]
    return {
        "symbol": symbol,
        "resolved_node": src_id,
        "depth": depth,
        "relations_filter": list(relations) if relations else None,
        "related": related,
        "edges": [e.as_dict() for e in edges],
        "provenance": "graphify-out/graph.json (confidence={})".format(",".join(confidence)),
    }


def get_community(
    storage: Storage,
    symbol_or_community_id: str | int,
    *,
    limit: int = 50,
) -> dict[str, Any]:
    """Return all members of the community containing ``symbol_or_community_id``."""
    limit = max(1, min(limit, 200))
    cur = storage._conn.cursor()

    cid: int | None = None
    if isinstance(symbol_or_community_id, int):
        cid = symbol_or_community_id
    else:
        # Symbol → resolve → community
        nid = resolve_symbol_to_node(storage, symbol_or_community_id)
        if nid is None:
            return {"query": symbol_or_community_id, "community": None, "members": []}
        row = cur.execute(
            "SELECT community_id, community_label FROM graph_nodes WHERE node_id = ?",
            (nid,),
        ).fetchone()
        if row is None or row["community_id"] is None:
            return {
                "query": symbol_or_community_id,
                "resolved_node": nid,
                "community": None,
                "members": [],
            }
        cid = row["community_id"]

    rows = cur.execute(
        """SELECT * FROM graph_nodes
           WHERE community_id = ? ORDER BY label LIMIT ?""",
        (cid, limit),
    ).fetchall()
    if not rows:
        return {
            "query": symbol_or_community_id,
            "community": {"id": cid, "label": None},
            "members": [],
        }
    label = rows[0]["community_label"]
    members = [_hydrate_node(storage, r).as_dict() for r in rows]
    return {
        "query": symbol_or_community_id,
        "community": {"id": cid, "label": label, "size": len(members)},
        "members": members,
        "provenance": "graphify-out/graph.json",
    }


def find_rationale(
    storage: Storage,
    symbol: str,
    *,
    limit: int = 10,
) -> dict[str, Any]:
    """Return ``rationale``-style nodes that explain ``symbol``.

    Walks ``rationale_for`` edges where ``target = symbol_node``. The
    surfaced nodes are typically docs / ADRs / design comments.
    """
    limit = max(1, min(limit, 50))
    target_id = resolve_symbol_to_node(storage, symbol)
    if target_id is None:
        return {"symbol": symbol, "resolved_node": None, "rationale": []}

    rows = storage._conn.execute(
        """SELECT n.*, e.confidence, e.confidence_score, e.source_file AS edge_source_file,
                  e.source_location AS edge_source_location
           FROM graph_edges e
           JOIN graph_nodes n ON n.node_id = e.source
           WHERE e.target = ? AND e.relation = 'rationale_for'
           ORDER BY COALESCE(e.confidence_score, 0) DESC LIMIT ?""",
        (target_id, limit),
    ).fetchall()
    if not rows:
        return {"symbol": symbol, "resolved_node": target_id, "rationale": []}
    out = []
    for r in rows:
        node = _hydrate_node(storage, r).as_dict()
        node["edge"] = {
            "confidence": r["confidence"],
            "confidence_score": r["confidence_score"],
            "source_file": r["edge_source_file"],
            "source_location": r["edge_source_location"],
        }
        out.append(node)
    return {
        "symbol": symbol,
        "resolved_node": target_id,
        "rationale": out,
        "provenance": "graphify-out/graph.json (rationale_for edges)",
    }


def find_tests_for(
    storage: Storage,
    symbol: str,
    *,
    limit: int = 20,
    confidence: tuple[str, ...] = ("EXTRACTED",),
    test_path_patterns: tuple[str, ...] = ("test_", "_test", "/tests/", "/test/", ".spec.", ".test."),
) -> dict[str, Any]:
    """Return test code that exercises ``symbol``.

    Closes the trio promised by ``stropha-system.md`` §3.5 and the RFC
    intro (``find_callers`` / ``find_tests_for`` / ``trace_feature``).

    Strategy:
      1. Resolve ``symbol`` to a node in the graph.
      2. Find every node that ``calls`` / ``references`` / ``implements``
         the target with ``EXTRACTED`` confidence.
      3. Filter callers whose ``source_file`` matches any of
         ``test_path_patterns`` — that's the test surface.
      4. Hydrate hits with the matching chunk.

    The path-pattern heuristic catches the common conventions
    (``tests/test_foo.py``, ``foo.test.ts``, ``FooSpec.kt``, …).
    Override ``test_path_patterns`` to suit a project's layout.
    """
    limit = max(1, min(limit, 100))
    target_id = resolve_symbol_to_node(storage, symbol)
    if target_id is None:
        return {"symbol": symbol, "resolved_node": None, "tests": []}

    conf_placeholders = ",".join("?" * len(confidence))
    # We accept any edge type that signals "this node touches the target".
    rel_set = ("calls", "references", "implements", "tests")
    rel_placeholders = ",".join("?" * len(rel_set))

    rows = storage._conn.execute(  # noqa: SLF001
        f"""SELECT n.*, e.relation, e.confidence, e.confidence_score
            FROM graph_edges e
            JOIN graph_nodes n ON n.node_id = e.source
            WHERE e.target = ?
              AND e.relation IN ({rel_placeholders})
              AND e.confidence IN ({conf_placeholders})
            ORDER BY COALESCE(e.confidence_score, 0) DESC
            LIMIT ?""",
        (target_id, *rel_set, *confidence, limit * 4),
    ).fetchall()

    tests: list[dict[str, Any]] = []
    seen: set[str] = set()
    pat_lower = tuple(p.lower() for p in test_path_patterns)
    for r in rows:
        sf = (r["source_file"] or "").lower()
        if not any(p in sf for p in pat_lower):
            continue
        if r["node_id"] in seen:
            continue
        seen.add(r["node_id"])
        node = _hydrate_node(storage, r).as_dict()
        node["edge"] = {
            "relation": r["relation"],
            "confidence": r["confidence"],
            "confidence_score": r["confidence_score"],
        }
        tests.append(node)
        if len(tests) >= limit:
            break

    return {
        "symbol": symbol,
        "resolved_node": target_id,
        "tests": tests,
        "test_path_patterns": list(test_path_patterns),
        "provenance": "graphify-out/graph.json (calls/references/implements/tests edges)",
    }


def has_rationale_edges(storage: Storage) -> bool:
    """True iff at least one ``rationale_for`` edge exists.

    Used to conditionally register the ``find_rationale`` tool — RFC §2.2 OBJ-6.
    """
    try:
        row = storage._conn.execute(  # noqa: SLF001
            "SELECT 1 FROM graph_edges WHERE relation = 'rationale_for' LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return False
    return row is not None


# --------------------------------------------------------------------- trace_feature


def trace_feature(
    storage: Storage,
    feature: str,
    *,
    max_paths: int = 5,
    max_depth: int = 6,
    confidence: tuple[str, ...] = ("EXTRACTED",),
) -> dict[str, Any]:
    """Trace a feature description through the call graph.

    Per spec §6.3.5: turn a free-text feature description (typically a
    Gherkin scenario or a behavioural outcome) into a chain of code chunks
    that participate in the feature.

    Strategy:
      1. Pick the top-N graph nodes whose label loosely matches ``feature``
         (substring or token overlap). These are the "entry points" — for
         a BDD codebase they will be step-definition methods; for a plain
         codebase, top-level functions named after the feature.
      2. For each entry point, walk outbound ``calls`` edges DFS up to
         ``max_depth``, collecting unique nodes.
      3. Hydrate each node with the matching stropha chunk (snippet +
         line range) so the caller can render the trace inline.

    The result is a list of paths (rooted at entry points) plus a flat
    set of all unique nodes touched. Cycles are broken at first revisit.
    """
    max_paths = max(1, min(max_paths, 20))
    max_depth = max(1, min(max_depth, 10))

    cur = storage._conn.cursor()  # noqa: SLF001

    # 1. Find candidate entry-point nodes. Token-overlap scoring is more
    #    forgiving than the exact `resolve_symbol_to_node` because feature
    #    descriptions are natural language ("user submits an answer").
    feature_tokens = {
        t.lower() for t in feature.replace("/", " ").split() if len(t) >= 3
    }
    if not feature_tokens:
        return {"feature": feature, "entries": [], "paths": [], "nodes": []}

    # Pull a manageable candidate pool — graph_nodes is small enough to
    # scan in Python for non-trivial scoring.
    rows = cur.execute(
        "SELECT * FROM graph_nodes WHERE label IS NOT NULL"
    ).fetchall()

    def score(label: str) -> int:
        l = label.lower()
        return sum(1 for t in feature_tokens if t in l)

    scored = [(score(r["label"]), r) for r in rows]
    scored = [t for t in scored if t[0] > 0]
    scored.sort(key=lambda t: -t[0])
    entry_rows = [r for _, r in scored[:max_paths]]

    if not entry_rows:
        return {"feature": feature, "entries": [], "paths": [], "nodes": []}

    # 2. DFS each entry point along outbound `calls` edges.
    conf_placeholders = ",".join("?" * len(confidence))
    paths: list[dict[str, Any]] = []
    all_node_ids: set[str] = set()
    for entry in entry_rows:
        entry_id = entry["node_id"]
        all_node_ids.add(entry_id)
        chain: list[dict[str, Any]] = [_hydrate_node(storage, entry).as_dict()]
        visited = {entry_id}
        stack: list[tuple[str, int]] = [(entry_id, 0)]
        while stack:
            node_id, depth = stack.pop()
            if depth >= max_depth:
                continue
            edges = cur.execute(
                f"""SELECT target FROM graph_edges
                    WHERE source = ? AND relation = 'calls'
                      AND confidence IN ({conf_placeholders})
                    LIMIT 8""",
                (node_id, *confidence),
            ).fetchall()
            for er in edges:
                tgt = er["target"]
                if tgt in visited:
                    continue
                visited.add(tgt)
                all_node_ids.add(tgt)
                tgt_row = cur.execute(
                    "SELECT * FROM graph_nodes WHERE node_id = ?", (tgt,)
                ).fetchone()
                if tgt_row is None:
                    continue
                chain.append(_hydrate_node(storage, tgt_row).as_dict())
                stack.append((tgt, depth + 1))
                if len(chain) >= 30:  # cap per entry to keep responses readable
                    break
            if len(chain) >= 30:
                break
        paths.append({
            "entry": _hydrate_node(storage, entry).as_dict(),
            "depth_explored": max_depth,
            "chain": chain,
            "chain_size": len(chain),
        })

    return {
        "feature": feature,
        "entries": [_hydrate_node(storage, e).as_dict() for e in entry_rows],
        "paths": paths,
        "nodes": [
            _hydrate_node(storage, cur.execute(
                "SELECT * FROM graph_nodes WHERE node_id = ?", (nid,)
            ).fetchone()).as_dict()
            for nid in sorted(all_node_ids)
            if cur.execute(
                "SELECT 1 FROM graph_nodes WHERE node_id = ? LIMIT 1", (nid,)
            ).fetchone()
        ],
        "provenance": "graphify-out/graph.json (calls edges, conf={})".format(
            ",".join(confidence)
        ),
    }

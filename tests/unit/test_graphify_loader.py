"""Unit tests for ``stropha.ingest.graphify_loader.GraphifyLoader``.

Covers the contract from RFC §6.1 / §9 (Fase 1.5a):
- idempotência (carregar N vezes não duplica)
- transactional reload (substitui completamente)
- filtro por confidence (default = EXTRACTED only)
- staleness detection via mtime
- no-op quando ``graphify-out/graph.json`` ausente
- ``Storage.stats()['graph']`` reflete estado
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from stropha.ingest.graphify_loader import (
    DEFAULT_CONFIDENCE,
    GraphifyLoader,
    _resolve_confidence_filter,
)
from stropha.storage import Storage

# --------------------------------------------------------------------------- helpers


def _write_graph(path: Path, *, nodes: list[dict], edges: list[dict],
                 communities: dict | None = None,
                 community_labels: dict | None = None) -> None:
    """Write a NetworkX node-link style graph file."""
    payload: dict = {"nodes": nodes, "links": edges}
    if communities is not None:
        payload["communities"] = communities
    if community_labels is not None:
        payload["community_labels"] = community_labels
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "idx.db", embedding_dim=4)
    yield s
    s.close()


# --------------------------------------------------------------------------- env helper


def test_resolve_confidence_default_when_unset() -> None:
    assert _resolve_confidence_filter(None) == DEFAULT_CONFIDENCE
    assert _resolve_confidence_filter("") == DEFAULT_CONFIDENCE


def test_resolve_confidence_parses_csv() -> None:
    assert _resolve_confidence_filter("EXTRACTED,INFERRED") == ("EXTRACTED", "INFERRED")
    assert _resolve_confidence_filter("inferred") == ("INFERRED",)


def test_resolve_confidence_drops_unknown_values() -> None:
    # Unknown tier silently dropped; valid tier preserved.
    assert _resolve_confidence_filter("EXTRACTED, BOGUS") == ("EXTRACTED",)


def test_resolve_confidence_falls_back_when_all_unknown() -> None:
    assert _resolve_confidence_filter("BOGUS,INVALID") == DEFAULT_CONFIDENCE


# --------------------------------------------------------------------------- exists / staleness


def test_no_graph_file_means_no_op(tmp_path: Path, storage: Storage) -> None:
    loader = GraphifyLoader(storage, tmp_path)
    assert loader.exists() is False
    assert loader.is_stale() is False
    assert loader.load() is None
    # stats() returns graph: None
    assert storage.stats()["graph"] is None


def test_stale_when_never_loaded(tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(graph_path, nodes=[], edges=[])
    loader = GraphifyLoader(storage, tmp_path)
    assert loader.exists() is True
    assert loader.is_stale() is True


def test_not_stale_after_load(tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(graph_path, nodes=[{"id": "a", "label": "A"}], edges=[])
    loader = GraphifyLoader(storage, tmp_path)
    loader.load()
    assert loader.is_stale() is False


def test_stale_after_file_touched(tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(graph_path, nodes=[{"id": "a", "label": "A"}], edges=[])
    loader = GraphifyLoader(storage, tmp_path)
    loader.load()
    assert loader.is_stale() is False
    # Bump mtime forward.
    new_mtime = graph_path.stat().st_mtime + 10
    os.utime(graph_path, (new_mtime, new_mtime))
    assert loader.is_stale() is True


# --------------------------------------------------------------------------- load shape


def test_load_populates_nodes_and_edges(tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[
            {"id": "a", "label": "Alpha", "file_type": "code", "source_file": "src/a.py"},
            {"id": "b", "label": "Beta", "file_type": "code", "source_file": "src/b.py"},
        ],
        edges=[
            {
                "source": "a", "target": "b",
                "relation": "calls",
                "confidence": "EXTRACTED",
                "confidence_score": 1.0,
                "source_file": "src/a.py", "source_location": "L42",
                "weight": 1.0,
            }
        ],
    )
    loader = GraphifyLoader(storage, tmp_path)
    res = loader.load()
    assert res is not None
    assert res.nodes_loaded == 2
    assert res.edges_loaded == 1
    assert res.edges_filtered == 0

    # Tables populated
    cur = storage._conn.cursor()
    n_nodes = cur.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()["n"]
    n_edges = cur.execute("SELECT COUNT(*) AS n FROM graph_edges").fetchone()["n"]
    assert n_nodes == 2
    assert n_edges == 1


def test_default_confidence_filter_drops_inferred(tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "a", "target": "b", "relation": "references", "confidence": "INFERRED"},
            {"source": "a", "target": "b", "relation": "calls", "confidence": "AMBIGUOUS"},
        ],
    )
    loader = GraphifyLoader(storage, tmp_path)
    res = loader.load()
    assert res is not None
    assert res.edges_loaded == 1
    assert res.edges_filtered == 2


def test_explicit_confidence_filter_overrides_default(tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[
            {"source": "a", "target": "b", "relation": "x", "confidence": "EXTRACTED"},
            {"source": "a", "target": "b", "relation": "y", "confidence": "INFERRED"},
        ],
    )
    loader = GraphifyLoader(
        storage, tmp_path, confidence_filter=("EXTRACTED", "INFERRED")
    )
    res = loader.load()
    assert res is not None
    assert res.edges_loaded == 2


def test_env_var_drives_confidence_filter(monkeypatch, tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[
            {"source": "a", "target": "b", "relation": "x", "confidence": "EXTRACTED"},
            {"source": "a", "target": "b", "relation": "y", "confidence": "INFERRED"},
        ],
    )
    monkeypatch.setenv("STROPHA_GRAPH_CONFIDENCE", "EXTRACTED,INFERRED")
    loader = GraphifyLoader(storage, tmp_path)
    res = loader.load()
    assert res is not None
    assert res.edges_loaded == 2


# --------------------------------------------------------------------------- communities


def test_community_assignment_persisted(tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[
            {"id": "a", "label": "A"},
            {"id": "b", "label": "B"},
            {"id": "c", "label": "C"},
        ],
        edges=[],
        communities={"0": ["a", "b"], "1": ["c"]},
        community_labels={"0": "Auth", "1": "Storage"},
    )
    loader = GraphifyLoader(storage, tmp_path)
    res = loader.load()
    assert res is not None
    assert res.communities == 2

    cur = storage._conn.cursor()
    rows = {r["node_id"]: r for r in cur.execute(
        "SELECT node_id, community_id, community_label FROM graph_nodes"
    ).fetchall()}
    assert rows["a"]["community_id"] == 0
    assert rows["a"]["community_label"] == "Auth"
    assert rows["b"]["community_id"] == 0
    assert rows["c"]["community_id"] == 1
    assert rows["c"]["community_label"] == "Storage"


# --------------------------------------------------------------------------- idempotência


def test_load_is_idempotent(tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"}
        ],
    )
    loader = GraphifyLoader(storage, tmp_path)
    loader.load()
    loader.load()
    loader.load()
    cur = storage._conn.cursor()
    assert cur.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0] == 2
    assert cur.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 1


def test_load_replaces_previous_graph(tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "old", "label": "Old"}],
        edges=[],
    )
    loader = GraphifyLoader(storage, tmp_path)
    loader.load()

    # Replace the on-disk graph with a totally different one.
    _write_graph(
        graph_path,
        nodes=[{"id": "new1", "label": "New1"}, {"id": "new2", "label": "New2"}],
        edges=[
            {"source": "new1", "target": "new2", "relation": "calls", "confidence": "EXTRACTED"}
        ],
    )
    # Bump mtime so is_stale() returns True
    new_mtime = graph_path.stat().st_mtime + 10
    os.utime(graph_path, (new_mtime, new_mtime))

    loader.load()
    cur = storage._conn.cursor()
    ids = {r["node_id"] for r in cur.execute("SELECT node_id FROM graph_nodes").fetchall()}
    assert ids == {"new1", "new2"}
    assert "old" not in ids
    assert cur.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 1


# --------------------------------------------------------------------------- error handling


def test_invalid_json_returns_none(tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text("{ this is not valid json", encoding="utf-8")
    loader = GraphifyLoader(storage, tmp_path)
    assert loader.load() is None
    # Tables remain empty.
    cur = storage._conn.cursor()
    assert cur.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0] == 0


# --------------------------------------------------------------------------- override path


def test_env_override_path(monkeypatch, tmp_path: Path, storage: Storage) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    _write_graph(
        other / "graph.json",
        nodes=[{"id": "ext", "label": "Ext"}],
        edges=[],
    )
    monkeypatch.setenv("STROPHA_GRAPHIFY_OUT", str(other))
    loader = GraphifyLoader(storage, tmp_path)
    assert loader.exists() is True
    res = loader.load()
    assert res is not None
    assert res.nodes_loaded == 1


# --------------------------------------------------------------------------- Storage.stats integration


def test_storage_stats_graph_field_none_initially(tmp_path: Path, storage: Storage) -> None:
    assert storage.stats()["graph"] is None


def test_storage_stats_graph_field_after_load(tmp_path: Path, storage: Storage) -> None:
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"}
        ],
    )
    GraphifyLoader(storage, tmp_path).load()
    s = storage.stats()
    assert s["graph"] is not None
    g = s["graph"]
    assert g["nodes"] == 2
    assert g["edges_total"] == 1
    assert g["edges_by_confidence"] == {"EXTRACTED": 1}
    assert g["last_loaded_at"]  # truthy ISO string


# --------------------------------------------------------------------------- Phase C — diff-load


def test_diff_load_reports_added_on_first_run(tmp_path: Path, storage: Storage) -> None:
    """Initial load of N nodes reports nodes_added=N, nothing deleted."""
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[],
    )
    res = GraphifyLoader(storage, tmp_path).load()
    assert res is not None
    assert res.nodes_added == 2
    assert res.nodes_deleted == 0


def test_diff_load_idempotent_no_writes(tmp_path: Path, storage: Storage) -> None:
    """Second load with no graph.json change reports zero deltas."""
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[
            {"source": "a", "target": "b", "relation": "calls",
             "confidence": "EXTRACTED", "source_file": "x.py",
             "source_location": "L1"}
        ],
    )
    loader = GraphifyLoader(storage, tmp_path)
    loader.load()
    res2 = loader.load()
    assert res2 is not None
    assert res2.nodes_added == 0
    assert res2.nodes_deleted == 0
    assert res2.edges_added == 0
    assert res2.edges_deleted == 0


def test_diff_load_deletes_removed_nodes(tmp_path: Path, storage: Storage) -> None:
    """Nodes present in the previous load but absent from the new one
    get removed."""
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[
            {"id": "a", "label": "A"},
            {"id": "b", "label": "B"},
            {"id": "c", "label": "C"},
        ],
        edges=[],
    )
    loader = GraphifyLoader(storage, tmp_path)
    loader.load()

    # New version drops "c".
    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[],
    )
    os.utime(graph_path, (1_000_000_000, 1_000_000_000))
    res = loader.load()
    assert res is not None
    assert res.nodes_deleted == 1

    ids = {r[0] for r in storage._conn.execute(
        "SELECT node_id FROM graph_nodes"
    ).fetchall()}
    assert ids == {"a", "b"}


def test_diff_load_adds_only_new_node(tmp_path: Path, storage: Storage) -> None:
    """Initial load + new load with one extra node reports nodes_added=1."""
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}],
        edges=[],
    )
    loader = GraphifyLoader(storage, tmp_path)
    loader.load()

    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}, {"id": "newcomer", "label": "N"}],
        edges=[],
    )
    os.utime(graph_path, (1_000_000_000, 1_000_000_000))
    res = loader.load()
    assert res is not None
    assert res.nodes_added == 1
    assert res.nodes_deleted == 0


def test_diff_load_edges_with_null_source_file_dedupe(
    tmp_path: Path, storage: Storage,
) -> None:
    """Edges without source_file/source_location MUST still dedupe across
    repeat loads (PRIMARY-KEY-allows-NULL caveat; we coerce to '')."""
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[
            # No source_file / source_location → both NULL on the row.
            {"source": "a", "target": "b", "relation": "calls",
             "confidence": "EXTRACTED"},
        ],
    )
    loader = GraphifyLoader(storage, tmp_path)
    loader.load()
    loader.load()
    loader.load()
    n_edges = storage._conn.execute(
        "SELECT COUNT(*) FROM graph_edges"
    ).fetchone()[0]
    assert n_edges == 1  # not 3


# --------------------------------------------------------------------------- multi-repo (schema v7)


def _register_repo(storage: Storage, repo_id_hint: int, normalized_key: str) -> int:
    """Register a fake repo in the storage, returning its ID."""
    from stropha.ingest.git_meta import RepoIdentity
    identity = RepoIdentity(
        normalized_key=normalized_key,
        remote_url=None,
        root_path=f"/fake/{repo_id_hint}",
        default_branch=None,
        head_commit=None,
    )
    repo_id = storage.register_repo(identity)
    storage.commit()  # Commit the transaction so subsequent operations can start fresh
    return repo_id


def test_multi_repo_node_ids_are_prefixed(tmp_path: Path, storage: Storage) -> None:
    """With repo_id, node IDs are prefixed with '{repo_id}:'."""
    repo_id = _register_repo(storage, 1, "test:repo1")
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "foo", "label": "Foo"}],
        edges=[],
    )
    loader = GraphifyLoader(storage, tmp_path, repo_id=repo_id)
    result = loader.load()

    assert result is not None
    assert result.nodes_loaded == 1

    # Check the stored node_id is prefixed
    row = storage._conn.execute(
        "SELECT node_id, repo_id FROM graph_nodes LIMIT 1"
    ).fetchone()
    assert row["node_id"] == f"{repo_id}:foo"
    assert row["repo_id"] == repo_id


def test_multi_repo_edge_endpoints_are_prefixed(tmp_path: Path, storage: Storage) -> None:
    """Edge source and target IDs are also prefixed with repo_id."""
    repo_id = _register_repo(storage, 1, "test:repo1")
    graph_path = tmp_path / "graphify-out" / "graph.json"
    _write_graph(
        graph_path,
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[{"source": "a", "target": "b", "relation": "calls",
                "confidence": "EXTRACTED"}],
    )
    loader = GraphifyLoader(storage, tmp_path, repo_id=repo_id)
    loader.load()

    row = storage._conn.execute(
        "SELECT source, target, repo_id FROM graph_edges LIMIT 1"
    ).fetchone()
    assert row["source"] == f"{repo_id}:a"
    assert row["target"] == f"{repo_id}:b"
    assert row["repo_id"] == repo_id


def test_multi_repo_isolation_different_repos_same_node_id(
    tmp_path: Path, storage: Storage,
) -> None:
    """Two repos can have the same node_id in the graph without collision."""
    repo_a_id = _register_repo(storage, 1, "test:repo_a")
    repo_b_id = _register_repo(storage, 2, "test:repo_b")

    # Repo A: graphify-out-a/graph.json
    repo_a_root = tmp_path / "repo_a"
    repo_a_root.mkdir()
    graph_a = repo_a_root / "graphify-out" / "graph.json"
    _write_graph(
        graph_a,
        nodes=[{"id": "FooClass", "label": "FooClass A"}],
        edges=[],
    )

    # Repo B: graphify-out-b/graph.json
    repo_b_root = tmp_path / "repo_b"
    repo_b_root.mkdir()
    graph_b = repo_b_root / "graphify-out" / "graph.json"
    _write_graph(
        graph_b,
        nodes=[{"id": "FooClass", "label": "FooClass B"}],
        edges=[],
    )

    loader_a = GraphifyLoader(storage, repo_a_root, repo_id=repo_a_id)
    loader_b = GraphifyLoader(storage, repo_b_root, repo_id=repo_b_id)

    loader_a.load()
    loader_b.load()

    # Both nodes exist without collision
    rows = storage._conn.execute(
        "SELECT node_id, label, repo_id FROM graph_nodes ORDER BY repo_id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["node_id"] == f"{repo_a_id}:FooClass"
    assert rows[0]["label"] == "FooClass A"
    assert rows[0]["repo_id"] == repo_a_id
    assert rows[1]["node_id"] == f"{repo_b_id}:FooClass"
    assert rows[1]["label"] == "FooClass B"
    assert rows[1]["repo_id"] == repo_b_id


def test_multi_repo_staleness_is_per_repo(tmp_path: Path, storage: Storage) -> None:
    """Each repo has its own staleness check (per-repo mtime meta key)."""
    repo_a_id = _register_repo(storage, 1, "test:repo_a")
    repo_b_id = _register_repo(storage, 2, "test:repo_b")

    repo_a_root = tmp_path / "repo_a"
    repo_a_root.mkdir()
    graph_a = repo_a_root / "graphify-out" / "graph.json"
    _write_graph(graph_a, nodes=[{"id": "a", "label": "A"}], edges=[])

    repo_b_root = tmp_path / "repo_b"
    repo_b_root.mkdir()
    graph_b = repo_b_root / "graphify-out" / "graph.json"
    _write_graph(graph_b, nodes=[{"id": "b", "label": "B"}], edges=[])

    loader_a = GraphifyLoader(storage, repo_a_root, repo_id=repo_a_id)
    loader_b = GraphifyLoader(storage, repo_b_root, repo_id=repo_b_id)

    # Initially both are stale
    assert loader_a.is_stale() is True
    assert loader_b.is_stale() is True

    # Load repo A
    loader_a.load()
    assert loader_a.is_stale() is False
    assert loader_b.is_stale() is True  # B still stale

    # Load repo B
    loader_b.load()
    assert loader_a.is_stale() is False
    assert loader_b.is_stale() is False


def test_multi_repo_reload_does_not_delete_other_repo_nodes(
    tmp_path: Path, storage: Storage,
) -> None:
    """Reloading one repo's graph does not affect another repo's nodes."""
    repo_a_id = _register_repo(storage, 1, "test:repo_a")
    repo_b_id = _register_repo(storage, 2, "test:repo_b")

    repo_a_root = tmp_path / "repo_a"
    repo_a_root.mkdir()
    graph_a = repo_a_root / "graphify-out" / "graph.json"
    _write_graph(
        graph_a,
        nodes=[{"id": "a1", "label": "A1"}, {"id": "a2", "label": "A2"}],
        edges=[],
    )

    repo_b_root = tmp_path / "repo_b"
    repo_b_root.mkdir()
    graph_b = repo_b_root / "graphify-out" / "graph.json"
    _write_graph(graph_b, nodes=[{"id": "b1", "label": "B1"}], edges=[])

    loader_a = GraphifyLoader(storage, repo_a_root, repo_id=repo_a_id)
    loader_b = GraphifyLoader(storage, repo_b_root, repo_id=repo_b_id)

    loader_a.load()
    loader_b.load()

    # Verify both repos' nodes exist
    assert storage._conn.execute(
        "SELECT COUNT(*) FROM graph_nodes"
    ).fetchone()[0] == 3

    # Now update repo A with only one node (simulating a node deletion)
    _write_graph(graph_a, nodes=[{"id": "a1", "label": "A1 updated"}], edges=[])

    # Recreate loader with fresh mtime detection
    loader_a_fresh = GraphifyLoader(storage, repo_a_root, repo_id=repo_a_id)
    loader_a_fresh.load()

    # Repo A has 1 node (a2 deleted), repo B still has 1 node
    rows = storage._conn.execute(
        "SELECT node_id, label, repo_id FROM graph_nodes ORDER BY repo_id, node_id"
    ).fetchall()
    assert len(rows) == 2  # 1 from A + 1 from B
    assert rows[0]["node_id"] == f"{repo_a_id}:a1"
    assert rows[0]["label"] == "A1 updated"
    assert rows[1]["node_id"] == f"{repo_b_id}:b1"  # B's node untouched


def test_multi_repo_stats_shows_per_repo_counts(tmp_path: Path, storage: Storage) -> None:
    """stats() with repo_id returns counts for that repo only."""
    repo_a_id = _register_repo(storage, 1, "test:repo_a")
    repo_b_id = _register_repo(storage, 2, "test:repo_b")

    repo_a_root = tmp_path / "repo_a"
    repo_a_root.mkdir()
    graph_a = repo_a_root / "graphify-out" / "graph.json"
    _write_graph(
        graph_a,
        nodes=[{"id": "a1", "label": "A1"}, {"id": "a2", "label": "A2"}],
        edges=[{"source": "a1", "target": "a2", "relation": "calls",
                "confidence": "EXTRACTED"}],
    )

    repo_b_root = tmp_path / "repo_b"
    repo_b_root.mkdir()
    graph_b = repo_b_root / "graphify-out" / "graph.json"
    _write_graph(graph_b, nodes=[{"id": "b1", "label": "B1"}], edges=[])

    loader_a = GraphifyLoader(storage, repo_a_root, repo_id=repo_a_id)
    loader_b = GraphifyLoader(storage, repo_b_root, repo_id=repo_b_id)

    loader_a.load()
    loader_b.load()

    # Stats for repo A
    stats_a = loader_a.stats()
    assert stats_a is not None
    assert stats_a["nodes"] == 2
    assert stats_a["edges_total"] == 1
    assert stats_a["repo_id"] == repo_a_id

    # Stats for repo B
    stats_b = loader_b.stats()
    assert stats_b is not None
    assert stats_b["nodes"] == 1
    assert stats_b["edges_total"] == 0
    assert stats_b["repo_id"] == repo_b_id

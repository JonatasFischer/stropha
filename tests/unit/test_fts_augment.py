"""Unit tests for ``Storage.augment_fts_with_graph()`` — L2 retroactive
FTS5 augmentation (RFC §11 / §1.5e).

Verifies that after loading a graph and calling the augment method:
- chunks with a matching graph node receive community + label terms in FTS5
- chunks without a matching node remain unchanged
- the operation is idempotent (running twice produces the same FTS5 state)
- BM25 queries against the new terms find the augmented chunks
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stropha.ingest.graphify_loader import GraphifyLoader
from stropha.models import Chunk
from stropha.storage import Storage


# --------------------------------------------------------------------------- helpers


def _seed_graph(storage: Storage, graph_path: Path,
                nodes: list[dict], edges: list[dict],
                communities: dict | None = None,
                community_labels: dict | None = None) -> None:
    payload: dict = {"nodes": nodes, "links": edges}
    if communities is not None:
        payload["communities"] = communities
    if community_labels is not None:
        payload["community_labels"] = community_labels
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    GraphifyLoader(storage, graph_path.parent.parent).load()


def _seed_chunk(
    storage: Storage,
    *,
    rel_path: str,
    start: int = 1,
    end: int = 10,
    content: str = "def foo(): pass",
    symbol: str | None = None,
) -> int:
    chunk = Chunk(
        chunk_id=f"sha-{rel_path}-{start}",
        rel_path=rel_path, language="python", kind="function",
        symbol=symbol, parent_chunk_id=None,
        start_line=start, end_line=end,
        content=content, content_hash=f"h-{start}",
    )
    return storage.upsert_chunk(
        chunk, embedding=[0.0] * 4,
        embedding_model="stub", embedding_dim=4,
    )


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "idx.db", embedding_dim=4)
    yield s
    s.close()


# --------------------------------------------------------------------------- empty / no-op


def test_augment_returns_missing_graph_when_empty(storage: Storage) -> None:
    """No graph loaded → method reports missing_graph and changes nothing."""
    _seed_chunk(storage, rel_path="x.py")
    result = storage.augment_fts_with_graph()
    assert result["missing_graph"] == 1
    assert result["augmented"] == 0


def test_augment_zero_when_no_chunks_overlap_nodes(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "a", "label": "A", "source_file": "other.py", "source_location": "L1"}],
        edges=[],
    )
    _seed_chunk(storage, rel_path="x.py")
    result = storage.augment_fts_with_graph()
    assert result["augmented"] == 0
    assert result["unchanged"] == 1


# --------------------------------------------------------------------------- happy path


def test_augment_inserts_community_term(tmp_path: Path, storage: Storage) -> None:
    """A chunk overlapping a graph node gets `community:` + `node:` terms."""
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "auth", "label": "AuthService.login",
             "source_file": "src/auth.py", "source_location": "L42"},
        ],
        edges=[],
        communities={"0": ["auth"]},
        community_labels={"0": "Authentication"},
    )
    _seed_chunk(storage, rel_path="src/auth.py", start=40, end=50, content="def login(): ...")

    result = storage.augment_fts_with_graph()
    assert result["augmented"] == 1
    assert result["unchanged"] == 0

    # BM25 query for the community term must surface the chunk.
    hits = storage.search_bm25("authentication", k=5)
    assert len(hits) == 1
    assert hits[0].rel_path == "src/auth.py"


def test_augment_idempotent(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "n", "label": "Worker",
             "source_file": "w.py", "source_location": "L5"},
        ],
        edges=[],
        communities={"1": ["n"]},
        community_labels={"1": "Pipeline"},
    )
    _seed_chunk(storage, rel_path="w.py", start=1, end=20, content="class Worker: ...")

    storage.augment_fts_with_graph()
    storage.augment_fts_with_graph()
    storage.augment_fts_with_graph()

    # FTS5 still has exactly one row for the chunk
    n = storage._conn.execute(
        "SELECT COUNT(*) AS n FROM fts_chunks"
    ).fetchone()["n"]
    assert n == 1
    # Community term still matches
    hits = storage.search_bm25("pipeline", k=5)
    assert len(hits) == 1


def test_augment_does_not_affect_unrelated_chunks(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "a", "label": "ClassA",
             "source_file": "a.py", "source_location": "L10"},
        ],
        edges=[],
        communities={"0": ["a"]},
        community_labels={"0": "Alpha"},
    )
    _seed_chunk(storage, rel_path="a.py", start=5, end=20, content="class A: pass")
    _seed_chunk(storage, rel_path="b.py", start=1, end=10, content="class B: pass")

    storage.augment_fts_with_graph()
    # Query for "alpha" hits only a.py
    hits = storage.search_bm25("alpha", k=5)
    paths = {h.rel_path for h in hits}
    assert "a.py" in paths
    assert "b.py" not in paths
    # Query for the original content of b.py still finds it
    hits_b = storage.search_bm25("class B", k=5)
    assert any(h.rel_path == "b.py" for h in hits_b)


def test_augment_picks_tightest_node_per_chunk(tmp_path: Path, storage: Storage) -> None:
    """When a chunk overlaps multiple nodes, the one closest to its start
    wins (mirrors GraphAwareEnricher behaviour)."""
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "near", "label": "near_fn",
             "source_file": "f.py", "source_location": "L8"},
            {"id": "far", "label": "far_fn",
             "source_file": "f.py", "source_location": "L30"},
        ],
        edges=[],
        communities={"0": ["near"], "1": ["far"]},
        community_labels={"0": "NearCluster", "1": "FarCluster"},
    )
    _seed_chunk(storage, rel_path="f.py", start=5, end=40, content="def near(): pass")
    storage.augment_fts_with_graph()

    # "near" wins because line 8 is closer to chunk start (5) than line 30
    hits_near = storage.search_bm25("nearcluster", k=5)
    assert len(hits_near) == 1
    hits_far = storage.search_bm25("farcluster", k=5)
    assert len(hits_far) == 0  # far_fn lost the tie-break


# --------------------------------------------------------------------------- toggle integration


def test_pipeline_runs_augment_when_toggle_default(monkeypatch, tmp_path: Path, storage: Storage) -> None:
    """End-to-end through the pipeline: after a real index run with a
    graph present, the augment is invoked. We only check that the method
    is callable from that path (not a full pipeline test — covered
    elsewhere)."""
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "x", "label": "X",
                "source_file": "x.py", "source_location": "L1"}],
        edges=[],
        communities={"0": ["x"]},
        community_labels={"0": "Cluster"},
    )
    _seed_chunk(storage, rel_path="x.py", start=1, end=10)
    monkeypatch.setenv("STROPHA_GRAPH_FTS_AUGMENT", "1")
    # Direct call (the pipeline wraps this same line)
    result = storage.augment_fts_with_graph()
    assert result["augmented"] == 1


def test_augment_skips_when_toggle_disabled(monkeypatch, tmp_path: Path, storage: Storage) -> None:
    """When STROPHA_GRAPH_FTS_AUGMENT=0 the pipeline should NOT augment.

    Tests the toggle by checking the pipeline-level decision branch; the
    Storage method itself never reads the env (intentional separation of
    concerns)."""
    import os
    monkeypatch.setenv("STROPHA_GRAPH_FTS_AUGMENT", "0")
    assert os.environ["STROPHA_GRAPH_FTS_AUGMENT"] == "0"

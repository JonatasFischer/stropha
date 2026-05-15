"""Unit tests for ``stropha.adapters.enricher.graph_aware.GraphAwareEnricher``.

L2 augmentation per RFC §0:
- prepend community label + node label to embedding_text
- gracefully fall back when graph absent
- adapter_id changes with config flags (drift detection)
- builder injects storage handle after construction
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stropha.adapters.enricher.graph_aware import (
    GraphAwareEnricher,
    GraphAwareEnricherConfig,
)
from stropha.ingest.graphify_loader import GraphifyLoader
from stropha.models import Chunk
from stropha.pipeline.base import StageContext
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


def _make_chunk(rel_path: str, start: int, end: int, content: str = "code") -> Chunk:
    return Chunk(
        chunk_id=f"sha-{rel_path}-{start}",
        rel_path=rel_path,
        language="python",
        kind="function",
        symbol=None,
        parent_chunk_id=None,
        start_line=start,
        end_line=end,
        content=content,
        content_hash=f"hash-{start}",
    )


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "idx.db", embedding_dim=4)
    yield s
    s.close()


# --------------------------------------------------------------------------- adapter_id (drift)


def test_adapter_id_default() -> None:
    e = GraphAwareEnricher()
    assert e.adapter_id == "graph-aware:c:n:p"


def test_adapter_id_changes_with_flags() -> None:
    e1 = GraphAwareEnricher(GraphAwareEnricherConfig(include_community=False))
    e2 = GraphAwareEnricher(GraphAwareEnricherConfig(include_community=True))
    assert e1.adapter_id != e2.adapter_id


def test_adapter_id_drops_disabled_flags() -> None:
    e = GraphAwareEnricher(
        GraphAwareEnricherConfig(
            include_community=False,
            include_node_label=True,
            include_parent_skeleton=False,
        )
    )
    assert e.adapter_id == "graph-aware:n"


# --------------------------------------------------------------------------- health


def test_health_warning_without_storage() -> None:
    e = GraphAwareEnricher()
    h = e.health()
    assert h.status == "warning"
    assert "storage" in h.message.lower()


def test_health_warning_when_graph_empty(tmp_path: Path, storage: Storage) -> None:
    e = GraphAwareEnricher(storage=storage)
    h = e.health()
    # Graph tables exist (schema v4) but empty
    assert h.status == "warning"


def test_health_ready_when_graph_loaded(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "a", "label": "Alpha", "source_file": "a.py", "source_location": "L1"}],
        edges=[],
    )
    e = GraphAwareEnricher(storage=storage)
    h = e.health()
    assert h.status == "ready"
    assert "1 nodes" in h.message


# --------------------------------------------------------------------------- enrich


def test_enrich_falls_back_to_raw_when_no_storage() -> None:
    e = GraphAwareEnricher()
    chunk = _make_chunk("foo.py", 1, 10, "def foo(): pass")
    out = e.enrich(chunk, StageContext())
    assert out == "def foo(): pass"


def test_enrich_falls_back_to_raw_when_no_match(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "a", "label": "Alpha", "source_file": "other.py", "source_location": "L1"}],
        edges=[],
    )
    e = GraphAwareEnricher(storage=storage)
    chunk = _make_chunk("notfound.py", 1, 10, "raw content")
    out = e.enrich(chunk, StageContext())
    assert out == "raw content"


def test_enrich_prepends_community_and_label(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {
                "id": "auth_login", "label": "AuthService.login",
                "source_file": "src/auth.py", "source_location": "L42",
            }
        ],
        edges=[],
        communities={"3": ["auth_login"]},
        community_labels={"3": "Authentication"},
    )
    e = GraphAwareEnricher(storage=storage)
    chunk = _make_chunk("src/auth.py", 40, 50, "def login(): ...")
    out = e.enrich(chunk, StageContext())
    assert "# community: Authentication" in out
    assert "# node: AuthService.login" in out
    assert out.endswith("def login(): ...")


def test_enrich_combines_with_parent_skeleton(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {
                "id": "n1", "label": "Worker.tick",
                "source_file": "src/w.py", "source_location": "L20",
            }
        ],
        edges=[],
        communities={"0": ["n1"]},
        community_labels={"0": "Pipeline"},
    )
    e = GraphAwareEnricher(storage=storage)
    parent = Chunk(
        chunk_id="sha-parent", rel_path="src/w.py", language="python",
        kind="class", symbol="Worker", parent_chunk_id=None,
        start_line=10, end_line=80, content="...", content_hash="h",
    )
    chunk = _make_chunk("src/w.py", 18, 25, "def tick(): pass")
    ctx = StageContext(parent_chunk=parent)
    out = e.enrich(chunk, ctx)
    assert "# in class Worker" in out
    assert "# community: Pipeline" in out
    assert "# node: Worker.tick" in out


def test_enrich_can_disable_community(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "x", "label": "X", "source_file": "x.py", "source_location": "L1"}
        ],
        edges=[],
        communities={"0": ["x"]},
        community_labels={"0": "Cluster"},
    )
    cfg = GraphAwareEnricherConfig(include_community=False, include_parent_skeleton=False)
    e = GraphAwareEnricher(cfg, storage=storage)
    chunk = _make_chunk("x.py", 1, 5, "src")
    out = e.enrich(chunk, StageContext())
    assert "# community" not in out
    assert "# node: X" in out


def test_enrich_picks_tightest_node_match(tmp_path: Path, storage: Storage) -> None:
    """When multiple nodes share a file, prefer the one whose line is closest."""
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "early", "label": "early_fn",
             "source_file": "f.py", "source_location": "L5"},
            {"id": "late", "label": "late_fn",
             "source_file": "f.py", "source_location": "L40"},
        ],
        edges=[],
    )
    e = GraphAwareEnricher(storage=storage)
    chunk = _make_chunk("f.py", 30, 50, "code")  # contains line 40
    out = e.enrich(chunk, StageContext())
    assert "late_fn" in out
    assert "early_fn" not in out


# --------------------------------------------------------------------------- builder integration


def test_builder_injects_storage(tmp_path: Path) -> None:
    """End-to-end: build_stages wires storage into the graph-aware enricher."""
    from stropha.pipeline import build_stages

    cfg = {
        "walker": {"adapter": "git-ls-files", "config": {}},
        "chunker": {"adapter": "tree-sitter-dispatch", "config": {}},
        "enricher": {"adapter": "graph-aware", "config": {}},
        "embedder": {
            "adapter": "local",
            "config": {"model": "BAAI/bge-small-en-v1.5"},
        },
        "storage": {"adapter": "sqlite-vec", "config": {"path": str(tmp_path / "idx.db")}},
        "retrieval": {"adapter": "hybrid-rrf", "config": {}},
    }
    built = build_stages(cfg)
    try:
        # Storage was injected
        assert built.enricher._storage is not None  # type: ignore[attr-defined]
        # Health is now warning (graph empty) not "no storage"
        h = built.enricher.health()
        assert "storage" not in h.message.lower() or "graph empty" in h.message.lower()
    finally:
        built.storage.close()

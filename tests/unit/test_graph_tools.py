"""Unit tests for ``stropha.retrieval.graph`` — find_callers / find_related /
get_community / find_rationale.

Per RFC §9 Fase 1.5b critério de saída: ``find_callers("StudyService.submitAnswer")``
returns ≥1 caller correctly. We replicate the structure with a synthetic graph
seeded directly into the SQLite mirror (no graphify CLI required for tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stropha.ingest.graphify_loader import GraphifyLoader
from stropha.retrieval.graph import (
    find_callers,
    find_rationale,
    find_related,
    get_community,
    graph_loaded,
    has_rationale_edges,
    resolve_symbol_to_node,
)
from stropha.storage import Storage

# --------------------------------------------------------------------------- helpers


def _seed_graph(storage: Storage, graph_path: Path,
                nodes: list[dict], edges: list[dict],
                communities: dict | None = None,
                community_labels: dict | None = None) -> None:
    """Write a graph file and load it into the storage mirror."""
    payload: dict = {"nodes": nodes, "links": edges}
    if communities is not None:
        payload["communities"] = communities
    if community_labels is not None:
        payload["community_labels"] = community_labels
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    GraphifyLoader(storage, graph_path.parent.parent).load()


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "idx.db", embedding_dim=4)
    yield s
    s.close()


# --------------------------------------------------------------------------- gating


def test_graph_loaded_false_initially(storage: Storage) -> None:
    assert graph_loaded(storage) is False


def test_graph_loaded_true_after_load(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "a", "label": "A"}], edges=[],
    )
    assert graph_loaded(storage) is True


def test_has_rationale_edges_false_when_none(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"}
        ],
    )
    assert has_rationale_edges(storage) is False


def test_has_rationale_edges_true_when_present(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "doc", "label": "ADR-001"}, {"id": "code", "label": "MyClass"}],
        edges=[
            {"source": "doc", "target": "code", "relation": "rationale_for", "confidence": "EXTRACTED"}
        ],
    )
    assert has_rationale_edges(storage) is True


# --------------------------------------------------------------------------- symbol resolution


def test_resolve_returns_none_when_not_loaded(storage: Storage) -> None:
    assert resolve_symbol_to_node(storage, "anything") is None


def test_resolve_exact_match(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "n1", "label": "FsrsCalculator"}],
        edges=[],
    )
    assert resolve_symbol_to_node(storage, "FsrsCalculator") == "n1"


def test_resolve_dotted_suffix(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "n1", "label": "stropha.fsrs.FsrsCalculator"}],
        edges=[],
    )
    assert resolve_symbol_to_node(storage, "FsrsCalculator") == "n1"


def test_resolve_substring_fallback(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "n1", "label": "OldFsrsCalculatorImpl"}],
        edges=[],
    )
    assert resolve_symbol_to_node(storage, "Fsrs") == "n1"


def test_resolve_returns_none_when_no_match(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "n1", "label": "Foo"}], edges=[],
    )
    assert resolve_symbol_to_node(storage, "Bar") is None


# --------------------------------------------------------------------------- find_callers


def test_find_callers_returns_empty_when_unknown(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "x", "label": "Existing"}], edges=[],
    )
    result = find_callers(storage, "Missing")
    assert result["callers"] == []
    assert result["resolved_node"] is None


def test_find_callers_depth_1(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "target", "label": "submitAnswer"},
            {"id": "caller1", "label": "doSubmit"},
            {"id": "caller2", "label": "handleSubmit"},
            {"id": "unrelated", "label": "draw"},
        ],
        edges=[
            {"source": "caller1", "target": "target", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "caller2", "target": "target", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "target", "target": "unrelated", "relation": "calls", "confidence": "EXTRACTED"},
        ],
    )
    result = find_callers(storage, "submitAnswer")
    assert result["resolved_node"] == "target"
    caller_ids = {c["node_id"] for c in result["callers"]}
    assert caller_ids == {"caller1", "caller2"}
    assert "unrelated" not in caller_ids


def test_find_callers_respects_limit(tmp_path: Path, storage: Storage) -> None:
    nodes = [{"id": "target", "label": "T"}]
    edges = []
    for i in range(50):
        nodes.append({"id": f"c{i}", "label": f"Caller{i}"})
        edges.append({
            "source": f"c{i}", "target": "target",
            "relation": "calls", "confidence": "EXTRACTED",
        })
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=nodes, edges=edges,
    )
    result = find_callers(storage, "T", limit=5)
    assert len(result["callers"]) == 5


def test_find_callers_filters_non_calls_relation(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "target", "label": "T"},
            {"id": "ref", "label": "Ref"},
            {"id": "call", "label": "Call"},
        ],
        edges=[
            {"source": "ref", "target": "target", "relation": "references", "confidence": "EXTRACTED"},
            {"source": "call", "target": "target", "relation": "calls", "confidence": "EXTRACTED"},
        ],
    )
    result = find_callers(storage, "T")
    ids = {c["node_id"] for c in result["callers"]}
    assert ids == {"call"}


def test_find_callers_filters_non_extracted(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "target", "label": "T"},
            {"id": "ext", "label": "Ext"},
            {"id": "inf", "label": "Inf"},  # INFERRED - filtered by default loader
        ],
        edges=[
            {"source": "ext", "target": "target", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "inf", "target": "target", "relation": "calls", "confidence": "INFERRED"},
        ],
    )
    # Default loader filters INFERRED, so only `ext` survives.
    result = find_callers(storage, "T")
    ids = {c["node_id"] for c in result["callers"]}
    assert ids == {"ext"}


# --------------------------------------------------------------------------- find_related


def test_find_related_returns_neighbors(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "x", "label": "X"},
            {"id": "neigh1", "label": "Neigh1"},
            {"id": "neigh2", "label": "Neigh2"},
            {"id": "far", "label": "Far"},
        ],
        edges=[
            {"source": "x", "target": "neigh1", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "neigh2", "target": "x", "relation": "imports", "confidence": "EXTRACTED"},
            {"source": "far", "target": "neigh1", "relation": "calls", "confidence": "EXTRACTED"},
        ],
    )
    result = find_related(storage, "X", depth=1)
    ids = {n["node_id"] for n in result["related"]}
    assert {"neigh1", "neigh2"} <= ids
    assert "far" not in ids  # depth=1, far is depth-2


def test_find_related_depth_2_reaches_far(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "x", "label": "X"},
            {"id": "mid", "label": "Mid"},
            {"id": "far", "label": "Far"},
        ],
        edges=[
            {"source": "x", "target": "mid", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "mid", "target": "far", "relation": "calls", "confidence": "EXTRACTED"},
        ],
    )
    result = find_related(storage, "X", depth=2)
    ids = {n["node_id"] for n in result["related"]}
    assert "far" in ids


def test_find_related_relations_filter(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "x", "label": "X"},
            {"id": "called", "label": "Called"},
            {"id": "implemented", "label": "Implemented"},
        ],
        edges=[
            {"source": "x", "target": "called", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "x", "target": "implemented", "relation": "implements", "confidence": "EXTRACTED"},
        ],
    )
    result = find_related(storage, "X", relations=("calls",))
    ids = {n["node_id"] for n in result["related"]}
    assert ids == {"called"}


# --------------------------------------------------------------------------- get_community


def test_get_community_returns_peers(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "a", "label": "A"},
            {"id": "b", "label": "B"},
            {"id": "c", "label": "C"},
            {"id": "d", "label": "D"},
        ],
        edges=[],
        communities={"0": ["a", "b"], "1": ["c", "d"]},
        community_labels={"0": "Auth", "1": "Storage"},
    )
    result = get_community(storage, "A")
    assert result["community"]["id"] == 0
    assert result["community"]["label"] == "Auth"
    ids = {m["node_id"] for m in result["members"]}
    assert ids == {"a", "b"}


def test_get_community_by_id(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "a", "label": "A"},
            {"id": "b", "label": "B"},
        ],
        edges=[],
        communities={"5": ["a", "b"]},
        community_labels={"5": "Cluster5"},
    )
    result = get_community(storage, 5)
    assert result["community"]["id"] == 5
    assert {m["node_id"] for m in result["members"]} == {"a", "b"}


def test_get_community_unknown_returns_empty(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "a", "label": "A"}], edges=[],
        communities={"0": ["a"]},
    )
    result = get_community(storage, 999)
    assert result["members"] == []


# --------------------------------------------------------------------------- find_rationale


def test_find_rationale_returns_explaining_docs(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "code", "label": "MyClass", "file_type": "code"},
            {"id": "adr", "label": "ADR-001", "file_type": "document"},
            {"id": "spec", "label": "system-spec", "file_type": "document"},
        ],
        edges=[
            {"source": "adr", "target": "code", "relation": "rationale_for", "confidence": "EXTRACTED", "confidence_score": 0.9},
            {"source": "spec", "target": "code", "relation": "rationale_for", "confidence": "EXTRACTED", "confidence_score": 0.7},
        ],
    )
    result = find_rationale(storage, "MyClass")
    ids = [n["node_id"] for n in result["rationale"]]
    # Sorted by confidence_score DESC
    assert ids == ["adr", "spec"]


def test_find_rationale_empty_when_no_edges(tmp_path: Path, storage: Storage) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "code", "label": "X"}], edges=[],
    )
    result = find_rationale(storage, "X")
    assert result["rationale"] == []

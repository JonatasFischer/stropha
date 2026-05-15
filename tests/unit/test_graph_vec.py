"""Unit tests for ``GraphVecLoader`` (Trilha A L3) + ``GraphVecStream``.

Uses a deterministic stub embedder so the tests don't require network
access nor a live model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from stropha.adapters.retrieval.streams.graph_vec import (
    GraphVecConfig,
    GraphVecStream,
    _cosine,
)
from stropha.embeddings.base import Embedder
from stropha.ingest.graph_vec_loader import GraphVecLoader, _pack_vec, _unpack_vec
from stropha.ingest.graphify_loader import GraphifyLoader
from stropha.models import Chunk
from stropha.storage import Storage


# --------------------------------------------------------------------------- helpers


class _DeterministicEmbedder:
    """Hash-based 4-dim embedder. Same input -> same output. No network."""

    def __init__(self, dim: int = 4, model: str = "stub-v1") -> None:
        self._dim = dim
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def batch_size(self) -> int:
        return 32

    def _vec_from_text(self, text: str) -> list[float]:
        # Simple deterministic per-character bucket. Keeps tests stable.
        v = [0.0] * self._dim
        for i, ch in enumerate(text or "x"):
            v[i % self._dim] += (ord(ch) % 17) / 16.0
        # Normalize to unit length so cosine == dot product.
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec_from_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec_from_text(text)


def _seed_graph(storage: Storage, graph_path: Path,
                nodes: list[dict], edges: list[dict]) -> None:
    payload = {"nodes": nodes, "links": edges}
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    GraphifyLoader(storage, graph_path.parent.parent).load()


def _seed_chunk(storage: Storage, *, rel_path: str, start: int, end: int,
                content: str = "code", embedding_dim: int = 4) -> None:
    """Insert a chunk row so the GraphVec stream can hydrate hits from it."""
    chunk = Chunk(
        chunk_id=f"sha-{rel_path}-{start}",
        rel_path=rel_path, language="python", kind="function",
        symbol=None, parent_chunk_id=None,
        start_line=start, end_line=end,
        content=content, content_hash=f"h-{start}",
    )
    embedding = [0.0] * embedding_dim
    storage.upsert_chunk(
        chunk, embedding=embedding,
        embedding_model="stub-v1", embedding_dim=embedding_dim,
    )


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "idx.db", embedding_dim=4)
    yield s
    s.close()


@pytest.fixture
def embedder() -> _DeterministicEmbedder:
    return _DeterministicEmbedder()


# --------------------------------------------------------------------------- pack / unpack


def test_pack_unpack_roundtrip() -> None:
    vec = [0.1, -0.2, 0.3, 0.0, 0.5]
    packed = _pack_vec(vec)
    assert isinstance(packed, bytes)
    out = _unpack_vec(packed)
    assert all(abs(a - b) < 1e-6 for a, b in zip(vec, out, strict=True))


def test_cosine_unit_vectors_orthogonal() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine(a, b) == 0.0


def test_cosine_identical_vectors() -> None:
    a = [0.5, 0.3, -0.1]
    assert _cosine(a, a) == pytest.approx(1.0)


def test_cosine_handles_zero_vector() -> None:
    a = [0.0, 0.0, 0.0]
    b = [1.0, 1.0, 1.0]
    assert _cosine(a, b) == 0.0


# --------------------------------------------------------------------------- GraphVecLoader


def test_loader_needs_run_when_no_embeddings(tmp_path: Path,
                                              storage: Storage,
                                              embedder: _DeterministicEmbedder) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "a", "label": "Alpha"}], edges=[],
    )
    assert GraphVecLoader(storage, embedder).needs_run() is True


def test_loader_embeds_all_nodes(tmp_path: Path, storage: Storage,
                                  embedder: _DeterministicEmbedder) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "a", "label": "Alpha"},
            {"id": "b", "label": "Beta"},
            {"id": "c", "label": "Gamma"},
        ],
        edges=[],
    )
    res = GraphVecLoader(storage, embedder).load()
    assert res.embedded == 3
    assert res.failed == 0
    n = storage._conn.execute(
        "SELECT COUNT(*) FROM graph_nodes WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    assert n == 3


def test_loader_skips_already_current_model(tmp_path: Path, storage: Storage,
                                             embedder: _DeterministicEmbedder) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[],
    )
    loader = GraphVecLoader(storage, embedder)
    loader.load()
    # Second run must be a no-op.
    res2 = loader.load()
    assert res2.embedded == 0


def test_loader_re_embeds_when_model_changes(tmp_path: Path, storage: Storage,
                                              embedder: _DeterministicEmbedder) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "a", "label": "A"}], edges=[],
    )
    GraphVecLoader(storage, embedder).load()
    other = _DeterministicEmbedder(model="stub-v2")
    res = GraphVecLoader(storage, other).load()
    assert res.embedded == 1
    row = storage._conn.execute(
        "SELECT embedding_model FROM graph_nodes WHERE node_id = 'a'"
    ).fetchone()
    assert row["embedding_model"] == "stub-v2"


# --------------------------------------------------------------------------- GraphVecStream


def test_stream_returns_empty_when_no_embeddings(storage: Storage) -> None:
    stream = GraphVecStream(storage=storage)
    hits = stream.search("anything", [0.5, 0.5, 0.5, 0.5])
    assert hits == []


def test_stream_returns_empty_when_query_vec_none(tmp_path: Path,
                                                    storage: Storage,
                                                    embedder: _DeterministicEmbedder) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "a", "label": "A", "source_file": "a.py", "source_location": "L1"}],
        edges=[],
    )
    GraphVecLoader(storage, embedder).load()
    _seed_chunk(storage, rel_path="a.py", start=1, end=10)
    stream = GraphVecStream(storage=storage)
    assert stream.search("q", None) == []


def test_stream_finds_most_similar_node(tmp_path: Path, storage: Storage,
                                          embedder: _DeterministicEmbedder) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "auth", "label": "AuthService.login",
             "source_file": "src/auth.py", "source_location": "L42"},
            {"id": "math", "label": "compute_pi",
             "source_file": "src/math.py", "source_location": "L10"},
        ],
        edges=[],
    )
    GraphVecLoader(storage, embedder).load()
    _seed_chunk(storage, rel_path="src/auth.py", start=40, end=50, content="login body")
    _seed_chunk(storage, rel_path="src/math.py", start=5, end=15, content="math body")

    stream = GraphVecStream(GraphVecConfig(k=5, min_similarity=0.0), storage=storage)
    qvec = embedder.embed_query("AuthService.login")
    hits = stream.search("AuthService.login", qvec)

    # The query vector matches the auth label exactly, so auth must rank
    # higher than math.
    assert len(hits) >= 1
    assert hits[0].rel_path == "src/auth.py"


def test_stream_drops_below_min_similarity(tmp_path: Path, storage: Storage) -> None:
    """Manually plant an embedding orthogonal to the query vector so the
    cosine is provably 0 and below ``min_similarity``."""
    import struct
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[
            {"id": "x", "label": "X", "source_file": "x.py", "source_location": "L1"}
        ],
        edges=[],
    )
    _seed_chunk(storage, rel_path="x.py", start=1, end=5)
    # Plant a known unit vector for node "x".
    node_vec = [1.0, 0.0, 0.0, 0.0]
    storage._conn.execute(
        """UPDATE graph_nodes
           SET embedding = ?, embedding_model = 'manual', embedding_dim = 4
           WHERE node_id = 'x'""",
        (struct.pack("<4f", *node_vec),),
    )
    storage.commit()

    stream = GraphVecStream(
        GraphVecConfig(k=5, min_similarity=0.5), storage=storage,
    )
    # Orthogonal query → cosine = 0 → filtered out.
    qvec_orthogonal = [0.0, 1.0, 0.0, 0.0]
    assert stream.search("anything", qvec_orthogonal) == []
    # Aligned query → cosine = 1 → returned.
    qvec_aligned = [1.0, 0.0, 0.0, 0.0]
    hits = stream.search("anything", qvec_aligned)
    assert len(hits) == 1
    assert hits[0].rel_path == "x.py"


def test_stream_health_warning_when_empty(storage: Storage) -> None:
    stream = GraphVecStream(storage=storage)
    h = stream.health()
    assert h.status == "warning"


def test_stream_health_ready_after_loading(tmp_path: Path, storage: Storage,
                                              embedder: _DeterministicEmbedder) -> None:
    _seed_graph(
        storage, tmp_path / "graphify-out" / "graph.json",
        nodes=[{"id": "a", "label": "A"}], edges=[],
    )
    GraphVecLoader(storage, embedder).load()
    stream = GraphVecStream(storage=storage)
    h = stream.health()
    assert h.status == "ready"
    assert "1 embedded nodes" in h.message


def test_stream_requires_storage() -> None:
    with pytest.raises(ValueError, match="storage="):
        GraphVecStream()


def test_stream_adapter_id_includes_config() -> None:
    stream = GraphVecStream(GraphVecConfig(k=42, min_similarity=0.5),
                             storage=None.__class__)  # type: ignore[arg-type]
    assert "k=42" in stream.adapter_id
    assert "min=0.5" in stream.adapter_id

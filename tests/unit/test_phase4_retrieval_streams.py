"""Tests for Phase 4 retrieval sub-pipeline (streams as adapters)."""

from __future__ import annotations

from typing import Any

import pytest

from stropha import adapters  # noqa: F401
from stropha.adapters.retrieval.hybrid_rrf import (
    HybridRrfConfig,
    HybridRrfRetrieval,
)
from stropha.adapters.retrieval.streams.fts5_bm25 import (
    Fts5Bm25Config,
    Fts5Bm25Stream,
)
from stropha.adapters.retrieval.streams.like_tokens import (
    LikeTokensConfig,
    LikeTokensStream,
)
from stropha.adapters.retrieval.streams.vec_cosine import (
    VecCosineConfig,
    VecCosineStream,
)
from stropha.errors import ConfigError
from stropha.models import SearchHit
from stropha.pipeline.registry import all_adapters

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _RecordingStorage:
    """Records calls + returns deterministic synthetic hits."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    @staticmethod
    def _hit(rank: int, chunk_id: str, score: float = 1.0) -> SearchHit:
        return SearchHit(
            rank=rank,
            score=score,
            rel_path=f"file_{chunk_id}.py",
            language="python",
            kind="file",
            symbol=None,
            start_line=1,
            end_line=10,
            snippet=f"snippet {chunk_id}",
            chunk_id=chunk_id,
        )

    def search_dense(self, vec: list[float], k: int) -> list[SearchHit]:
        self.calls.append(("dense", k))
        return [self._hit(i + 1, f"d{i}") for i in range(k)]

    def search_bm25(self, query: str, k: int) -> list[SearchHit]:
        self.calls.append(("bm25", k))
        return [self._hit(i + 1, f"b{i}") for i in range(k)]

    def search_symbol_tokens(self, query: str, k: int) -> list[SearchHit]:
        self.calls.append(("symbol", k))
        return [self._hit(i + 1, f"s{i}") for i in range(k)]


class _StubEmbedder:
    @property
    def stage_name(self) -> str: return "embedder"
    @property
    def adapter_name(self) -> str: return "stub"
    @property
    def adapter_id(self) -> str: return "stub"
    @property
    def model_name(self) -> str: return "stub"
    @property
    def dim(self) -> int: return 4
    @property
    def batch_size(self) -> int: return 16
    @property
    def config_schema(self):
        from pydantic import BaseModel

        class _C(BaseModel):
            pass
        return _C
    def health(self):
        from stropha.pipeline.base import StageHealth
        return StageHealth(status="ready", message="stub")
    def embed_documents(self, texts):
        return [[0.0] * 4 for _ in texts]
    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_streams_registered() -> None:
    reg = all_adapters()
    assert set(reg["retrieval-stream"]) == {"vec-cosine", "fts5-bm25", "like-tokens"}


# ---------------------------------------------------------------------------
# Individual streams
# ---------------------------------------------------------------------------


def test_vec_cosine_stream_routes_to_storage_search_dense() -> None:
    storage = _RecordingStorage()
    s = VecCosineStream(VecCosineConfig(k=7), storage=storage)
    out = s.search("query", [0.1, 0.2, 0.3, 0.4])
    assert len(out) == 7
    assert storage.calls == [("dense", 7)]
    assert s.adapter_id == "vec-cosine:k=7"


def test_vec_cosine_stream_returns_empty_when_no_query_vec() -> None:
    s = VecCosineStream(VecCosineConfig(), storage=_RecordingStorage())
    assert s.search("query", None) == []


def test_fts5_bm25_stream_routes_to_storage_search_bm25() -> None:
    storage = _RecordingStorage()
    s = Fts5Bm25Stream(Fts5Bm25Config(k=12), storage=storage)
    out = s.search("query", None)
    assert len(out) == 12
    assert storage.calls == [("bm25", 12)]


def test_like_tokens_stream_routes_to_storage_search_symbol() -> None:
    storage = _RecordingStorage()
    s = LikeTokensStream(LikeTokensConfig(k=5), storage=storage)
    out = s.search("query", None)
    assert len(out) == 5
    assert storage.calls == [("symbol", 5)]


def test_streams_require_storage() -> None:
    for cls in (VecCosineStream, Fts5Bm25Stream, LikeTokensStream):
        with pytest.raises(ValueError):
            cls()


# ---------------------------------------------------------------------------
# Hybrid coordinator
# ---------------------------------------------------------------------------


def test_hybrid_rrf_calls_all_default_streams() -> None:
    storage = _RecordingStorage()
    embedder = _StubEmbedder()
    r = HybridRrfRetrieval(HybridRrfConfig(), storage=storage, embedder=embedder)
    hits = r.search("hello world")
    # All three streams produced data; RRF fused them.
    assert hits
    seen = {c[0] for c in storage.calls}
    assert seen == {"dense", "bm25", "symbol"}


def test_hybrid_rrf_disables_stream_set_to_null() -> None:
    storage = _RecordingStorage()
    embedder = _StubEmbedder()
    r = HybridRrfRetrieval(
        HybridRrfConfig(streams={"symbol": None}),
        storage=storage, embedder=embedder,
    )
    hits = r.search("x")
    seen = {c[0] for c in storage.calls}
    # symbol must NOT be in the call set.
    assert "symbol" not in seen
    assert {"dense", "bm25"} <= seen
    assert hits


def test_hybrid_rrf_skips_embedder_when_no_dense_stream() -> None:
    """Performance: no dense stream → don't pay embed_query cost."""
    storage = _RecordingStorage()
    embedder = _StubEmbedder()

    class _CountingEmbedder(_StubEmbedder):
        def __init__(self) -> None:
            self.calls = 0

        def embed_query(self, text):
            self.calls += 1
            return super().embed_query(text)

    embedder = _CountingEmbedder()
    r = HybridRrfRetrieval(
        HybridRrfConfig(streams={"dense": None}),
        storage=storage, embedder=embedder,
    )
    r.search("xyz")
    assert embedder.calls == 0


def test_hybrid_rrf_adapter_id_changes_with_stream_swap() -> None:
    storage = _RecordingStorage()
    embedder = _StubEmbedder()
    a = HybridRrfRetrieval(HybridRrfConfig(), storage=storage, embedder=embedder).adapter_id
    b = HybridRrfRetrieval(
        HybridRrfConfig(streams={"dense": {"adapter": "vec-cosine", "config": {"k": 100}}}),
        storage=storage, embedder=embedder,
    ).adapter_id
    assert a != b


def test_hybrid_rrf_unknown_sub_adapter_raises() -> None:
    storage = _RecordingStorage()
    embedder = _StubEmbedder()
    with pytest.raises(ConfigError):
        HybridRrfRetrieval(
            HybridRrfConfig(streams={"dense": {"adapter": "no-such"}}),
            storage=storage, embedder=embedder,
        )


def test_hybrid_rrf_returns_empty_for_blank_query() -> None:
    storage = _RecordingStorage()
    embedder = _StubEmbedder()
    r = HybridRrfRetrieval(HybridRrfConfig(), storage=storage, embedder=embedder)
    assert r.search("") == []

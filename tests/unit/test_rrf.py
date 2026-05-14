"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

from stropha.models import SearchHit
from stropha.retrieval.rrf import rrf_fuse


def _hit(chunk_id: str, rank: int) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=1.0,
        rel_path="x.py",
        language="python",
        kind="method",
        start_line=1,
        end_line=2,
        snippet="…",
        chunk_id=chunk_id,
    )


def test_rrf_prefers_chunks_in_both_lists() -> None:
    dense = [_hit("a", 1), _hit("b", 2), _hit("c", 3)]
    sparse = [_hit("b", 1), _hit("a", 2), _hit("d", 3)]
    fused = rrf_fuse(dense, sparse, k=60, top_k=4)
    # `a` is rank 1 in dense + rank 2 in sparse  → 1/61 + 1/62
    # `b` is rank 2 in dense + rank 1 in sparse  → 1/62 + 1/61  (same as a)
    # `c` only in dense, `d` only in sparse.
    ids = [h.chunk_id for h in fused]
    assert ids[0] in {"a", "b"}
    assert ids[1] in {"a", "b"}
    assert set(ids[:2]) == {"a", "b"}
    assert "c" in ids and "d" in ids


def test_rrf_handles_empty_list() -> None:
    dense = [_hit("a", 1), _hit("b", 2)]
    fused = rrf_fuse(dense, [], k=60, top_k=5)
    assert [h.chunk_id for h in fused] == ["a", "b"]


def test_rrf_assigns_consecutive_ranks() -> None:
    dense = [_hit(c, i + 1) for i, c in enumerate(["a", "b", "c", "d"])]
    sparse = [_hit(c, i + 1) for i, c in enumerate(["e", "f", "g"])]
    fused = rrf_fuse(dense, sparse, k=60, top_k=10)
    assert [h.rank for h in fused] == list(range(1, len(fused) + 1))


def test_rrf_score_monotonic() -> None:
    dense = [_hit("a", 1), _hit("b", 5)]
    fused = rrf_fuse(dense, [], k=60, top_k=2)
    assert fused[0].score > fused[1].score

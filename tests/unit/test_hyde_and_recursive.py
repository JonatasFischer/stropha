"""Unit tests for HyDE query rewrite and recursive retrieval.

Both are query-path-only features — they don't mutate stored data. So
tests focus on the contract: enable/disable toggles, graceful fallback
on failures, and the merge semantics for adjacency / parent.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from stropha.models import Chunk, SearchHit
from stropha.retrieval.hyde import maybe_hyde_rewrite
from stropha.retrieval.recursive import (
    DEFAULT_ADJACENCY_LINES,
    _adjacent_or_overlap,
    merge_hits,
)
from stropha.storage import Storage


# --------------------------------------------------------------------------- HyDE


def test_hyde_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("STROPHA_HYDE_ENABLED", raising=False)
    assert maybe_hyde_rewrite("how does login work") is None


def test_hyde_returns_text_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("STROPHA_HYDE_ENABLED", "1")
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "def login(user, password): ..."
        out = maybe_hyde_rewrite("how does login work")
    assert out is not None
    assert "def login" in out


def test_hyde_fallback_on_http_error(monkeypatch) -> None:
    monkeypatch.setenv("STROPHA_HYDE_ENABLED", "1")
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = None  # Simulate failure
        assert maybe_hyde_rewrite("how does login work") is None


def test_hyde_fallback_on_empty_response(monkeypatch) -> None:
    monkeypatch.setenv("STROPHA_HYDE_ENABLED", "1")
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = ""  # Empty string triggers fallback
        assert maybe_hyde_rewrite("q") is None


def test_hyde_empty_query_short_circuits(monkeypatch) -> None:
    monkeypatch.setenv("STROPHA_HYDE_ENABLED", "1")
    assert maybe_hyde_rewrite("") is None
    assert maybe_hyde_rewrite("   ") is None


def test_hyde_caps_long_output(monkeypatch) -> None:
    monkeypatch.setenv("STROPHA_HYDE_ENABLED", "1")
    long = "x" * 5000
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = long
        out = maybe_hyde_rewrite("q")
    assert out is not None
    assert len(out) <= 2000


# --------------------------------------------------------------------------- recursive merge


def _hit(*, chunk_id: str, rel_path: str, start: int, end: int,
         score: float = 1.0, content: str = "code", rank: int = 1) -> SearchHit:
    return SearchHit(
        rank=rank, score=score, rel_path=rel_path,
        language="python", kind="function", symbol=None,
        start_line=start, end_line=end,
        snippet=content, chunk_id=chunk_id, repo=None,
    )


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "idx.db", embedding_dim=4)
    yield s
    s.close()


def test_recursive_disabled_pass_through(storage: Storage) -> None:
    hits = [_hit(chunk_id="a", rel_path="x.py", start=1, end=5)]
    out = merge_hits(hits, storage, enabled=False)
    assert len(out) == 1
    assert out[0].chunk_id == "a"


def test_recursive_empty_input(storage: Storage) -> None:
    assert merge_hits([], storage, enabled=True) == []


def test_adjacent_hits_in_same_file_merge(storage: Storage) -> None:
    """Two hits 3 lines apart on the same file should collapse into one."""
    # Seed two adjacent chunks so the merge has DB rows to read.
    for i, (cid, s, e) in enumerate([("a", 1, 5), ("b", 9, 15)]):
        c = Chunk(
            chunk_id=cid, rel_path="x.py", language="python",
            kind="function", symbol=None, parent_chunk_id=None,
            start_line=s, end_line=e,
            content=f"body {i}", content_hash=cid,
        )
        storage.upsert_chunk(c, embedding=[0.0] * 4,
                             embedding_model="stub", embedding_dim=4)

    hits = [
        _hit(chunk_id="a", rel_path="x.py", start=1, end=5,
             score=0.8, content="snippet A"),
        _hit(chunk_id="b", rel_path="x.py", start=9, end=15,
             score=0.6, content="snippet B"),
    ]
    out = merge_hits(hits, storage, enabled=True, adjacency_lines=5)
    assert len(out) == 1
    merged = out[0]
    assert merged.start_line == 1
    assert merged.end_line == 15
    assert "snippet A" in merged.snippet
    assert "snippet B" in merged.snippet
    assert merged.score == 0.8  # max of the two


def test_distant_hits_not_merged(storage: Storage) -> None:
    """Two hits >adjacency apart stay distinct."""
    for cid, s, e in [("a", 1, 5), ("b", 200, 220)]:
        c = Chunk(
            chunk_id=cid, rel_path="x.py", language="python",
            kind="function", symbol=None, parent_chunk_id=None,
            start_line=s, end_line=e,
            content="x", content_hash=cid,
        )
        storage.upsert_chunk(c, embedding=[0.0] * 4,
                             embedding_model="stub", embedding_dim=4)

    hits = [
        _hit(chunk_id="a", rel_path="x.py", start=1, end=5),
        _hit(chunk_id="b", rel_path="x.py", start=200, end=220),
    ]
    out = merge_hits(hits, storage, enabled=True, adjacency_lines=5)
    assert len(out) == 2


def test_different_files_never_merge(storage: Storage) -> None:
    for cid, rel in [("a", "x.py"), ("b", "y.py")]:
        c = Chunk(
            chunk_id=cid, rel_path=rel, language="python",
            kind="function", symbol=None, parent_chunk_id=None,
            start_line=1, end_line=5,
            content="x", content_hash=cid,
        )
        storage.upsert_chunk(c, embedding=[0.0] * 4,
                             embedding_model="stub", embedding_dim=4)
    hits = [
        _hit(chunk_id="a", rel_path="x.py", start=1, end=5),
        _hit(chunk_id="b", rel_path="y.py", start=1, end=5),
    ]
    out = merge_hits(hits, storage, enabled=True, adjacency_lines=100)
    assert len(out) == 2


def test_parent_chunk_promotion(storage: Storage) -> None:
    """When two children of the same parent are retrieved, the parent
    chunk is used in place of the first child."""
    # Parent
    parent = Chunk(
        chunk_id="parent", rel_path="x.py", language="python",
        kind="class", symbol="MyClass", parent_chunk_id=None,
        start_line=1, end_line=50,
        content="class MyClass body", content_hash="parent",
    )
    storage.upsert_chunk(parent, embedding=[0.0] * 4,
                         embedding_model="stub", embedding_dim=4)
    # Two children pointing at it
    for cid, s, e in [("m1", 5, 15), ("m2", 20, 35)]:
        c = Chunk(
            chunk_id=cid, rel_path="x.py", language="python",
            kind="function", symbol=None,
            parent_chunk_id="parent",
            start_line=s, end_line=e,
            content="method body", content_hash=cid,
        )
        storage.upsert_chunk(c, embedding=[0.0] * 4,
                             embedding_model="stub", embedding_dim=4)

    hits = [
        _hit(chunk_id="m1", rel_path="x.py", start=5, end=15, score=0.9),
        _hit(chunk_id="m2", rel_path="x.py", start=20, end=35, score=0.7),
    ]
    out = merge_hits(hits, storage, enabled=True, adjacency_lines=0)
    # m1 should have been promoted to parent.
    assert any(h.chunk_id == "parent" for h in out)


# --------------------------------------------------------------------------- _adjacent_or_overlap


def test_adjacent_or_overlap_with_overlap() -> None:
    a = _hit(chunk_id="a", rel_path="f.py", start=10, end=20)
    b = _hit(chunk_id="b", rel_path="f.py", start=15, end=25)
    assert _adjacent_or_overlap(a, b, 0) is True


def test_adjacent_or_overlap_with_close_gap() -> None:
    a = _hit(chunk_id="a", rel_path="f.py", start=1, end=5)
    b = _hit(chunk_id="b", rel_path="f.py", start=8, end=12)
    assert _adjacent_or_overlap(a, b, 5) is True  # gap = 3, threshold = 5


def test_adjacent_or_overlap_with_large_gap() -> None:
    a = _hit(chunk_id="a", rel_path="f.py", start=1, end=5)
    b = _hit(chunk_id="b", rel_path="f.py", start=100, end=120)
    assert _adjacent_or_overlap(a, b, 5) is False


def test_default_adjacency_constant_is_sane() -> None:
    assert 1 <= DEFAULT_ADJACENCY_LINES <= 20  # avoid runaway merging

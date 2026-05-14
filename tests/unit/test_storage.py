"""Tests for SQLite + sqlite-vec + FTS5 layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from mimoria_rag.errors import StorageError
from mimoria_rag.models import Chunk
from mimoria_rag.storage import Storage


def _make_chunk(
    chunk_id: str,
    rel_path: str,
    content: str,
    symbol: str | None = None,
    kind: str = "method",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        rel_path=rel_path,
        language="java",
        kind=kind,
        symbol=symbol,
        start_line=1,
        end_line=10,
        content=content,
        content_hash=chunk_id,  # use chunk_id as a stable proxy
    )


def _vec(value: float, dim: int = 4) -> list[float]:
    """Return a unit vector aligned to one axis."""
    v = [0.0] * dim
    v[int(value) % dim] = 1.0
    return v


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    db = tmp_path / "test.db"
    return Storage(db, embedding_dim=4)


def test_upsert_and_dense_search(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk("sha256:a", "Foo.java", "class Foo {}"),
        _vec(0), "test-model", 4,
    )
    storage.upsert_chunk(
        _make_chunk("sha256:b", "Bar.java", "class Bar {}"),
        _vec(1), "test-model", 4,
    )
    storage.commit()

    hits = storage.search_dense(_vec(0), k=2)
    assert len(hits) == 2
    assert hits[0].chunk_id == "sha256:a"  # exact match wins
    assert hits[0].score > hits[1].score


def test_bm25_search_returns_lexical_matches(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk(
            "sha256:a", "FsrsCalculator.java",
            "public class FsrsCalculator { void calculateStability() {} }",
            symbol="FsrsCalculator",
        ),
        _vec(0), "test-model", 4,
    )
    storage.upsert_chunk(
        _make_chunk(
            "sha256:b", "Unrelated.java",
            "public class Other { void doSomething() {} }",
        ),
        _vec(1), "test-model", 4,
    )
    storage.commit()

    hits = storage.search_bm25("FsrsCalculator", k=5)
    assert len(hits) >= 1
    assert hits[0].chunk_id == "sha256:a"


def test_bm25_camelcase_split(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk(
            "sha256:a", "x.java",
            "the method submitAnswer is here",
        ),
        _vec(0), "test-model", 4,
    )
    storage.commit()
    # Query uses different casing — sanitizer should split CamelCase.
    hits = storage.search_bm25("submitAnswer", k=5)
    assert any(h.chunk_id == "sha256:a" for h in hits)


def test_chunk_is_fresh(storage: Storage) -> None:
    chunk = _make_chunk("sha256:a", "x.java", "...")
    storage.upsert_chunk(chunk, _vec(0), "test-model", 4)
    storage.commit()
    assert storage.chunk_is_fresh("sha256:a", chunk.content_hash, "test-model")
    assert not storage.chunk_is_fresh("sha256:a", "other-hash", "test-model")
    assert not storage.chunk_is_fresh("sha256:a", chunk.content_hash, "other-model")
    assert not storage.chunk_is_fresh("sha256:nope", chunk.content_hash, "test-model")


def test_delete_by_paths(storage: Storage) -> None:
    storage.upsert_chunk(_make_chunk("sha256:a", "x.java", "a"), _vec(0), "m", 4)
    storage.upsert_chunk(_make_chunk("sha256:b", "x.java", "b"), _vec(1), "m", 4)
    storage.upsert_chunk(_make_chunk("sha256:c", "y.java", "c"), _vec(2), "m", 4)
    storage.commit()
    deleted = storage.delete_by_paths(["x.java"])
    assert deleted == 2
    storage.commit()
    assert storage.stats()["chunks"] == 1


def test_lookup_by_symbol_prefers_exact(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk(
            "sha256:a", "x.java", "...", symbol="FsrsCalculator.calculateStability",
        ),
        _vec(0), "m", 4,
    )
    storage.upsert_chunk(
        _make_chunk(
            "sha256:b", "y.java", "...", symbol="FsrsCalculator",
        ),
        _vec(1), "m", 4,
    )
    storage.commit()
    hits = storage.lookup_by_symbol("FsrsCalculator")
    assert hits[0].chunk_id == "sha256:b"


def test_file_outline(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk("sha256:a", "F.java", "...", symbol="F", kind="class"),
        _vec(0), "m", 4,
    )
    storage.upsert_chunk(
        _make_chunk("sha256:b", "F.java", "...", symbol="F.x", kind="method"),
        _vec(1), "m", 4,
    )
    storage.commit()
    outline = storage.file_outline("F.java")
    assert [o["symbol"] for o in outline] == ["F", "F.x"]


def test_meta_roundtrip(storage: Storage) -> None:
    storage.set_meta("foo", "bar")
    storage.commit()
    assert storage.get_meta("foo") == "bar"
    storage.set_meta("foo", "baz")
    storage.commit()
    assert storage.get_meta("foo") == "baz"


def test_dimension_mismatch_rejected(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)
    s.upsert_chunk(_make_chunk("sha256:a", "x.java", "x"), _vec(0, 4), "m", 4)
    s.commit()
    s.close()
    with pytest.raises(StorageError):
        Storage(db, embedding_dim=8)

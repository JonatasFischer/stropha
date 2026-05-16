"""Tests for SQLite + sqlite-vec + FTS5 layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from stropha.errors import StorageError
from stropha.ingest.git_meta import RepoIdentity
from stropha.models import Chunk
from stropha.storage import Storage


def _identity(key: str = "github.com/example/repo", url: str | None = None) -> RepoIdentity:
    return RepoIdentity(
        normalized_key=key,
        remote_url=url or f"https://{key}.git",
        root_path=Path("/tmp/example/repo"),
        default_branch="main",
        head_commit="abc123",
    )


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


# ---- repos table & SearchHit.repo ----------------------------------------

def test_register_repo_is_idempotent(storage: Storage) -> None:
    rid1 = storage.register_repo(_identity("github.com/a/b"))
    rid2 = storage.register_repo(_identity("github.com/a/b"))
    storage.commit()
    assert rid1 == rid2
    repos = storage.list_repos()
    assert len(repos) == 1


def test_register_repo_distinct_keys_get_distinct_ids(storage: Storage) -> None:
    a = storage.register_repo(_identity("github.com/a/b"))
    b = storage.register_repo(_identity("github.com/c/d"))
    storage.commit()
    assert a != b
    assert {r.normalized_key for r in storage.list_repos()} == {
        "github.com/a/b",
        "github.com/c/d",
    }


def test_search_hit_carries_repo(storage: Storage) -> None:
    rid = storage.register_repo(_identity("github.com/foo/bar"))
    storage.upsert_chunk(
        _make_chunk("sha256:r", "Foo.java", "class Foo {}", symbol="Foo"),
        _vec(0), "m", 4, repo_id=rid,
    )
    storage.commit()

    hits = storage.search_dense(_vec(0), k=1)
    assert hits and hits[0].repo is not None
    assert hits[0].repo.normalized_key == "github.com/foo/bar"
    assert hits[0].repo.url == "https://github.com/foo/bar.git"
    assert hits[0].repo.default_branch == "main"

    # Same field surfaces through bm25 and symbol-lookup streams.
    bm = storage.search_bm25("Foo", k=1)
    assert bm and bm[0].repo and bm[0].repo.normalized_key == "github.com/foo/bar"

    sym = storage.lookup_by_symbol("Foo", limit=1)
    assert sym and sym[0].repo and sym[0].repo.normalized_key == "github.com/foo/bar"


def test_chunk_without_repo_id_yields_none_repo(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk("sha256:n", "Bare.java", "class Bare {}"),
        _vec(0), "m", 4, repo_id=None,
    )
    storage.commit()
    hits = storage.search_dense(_vec(0), k=1)
    assert hits and hits[0].repo is None


def test_clear_preserves_repos(storage: Storage) -> None:
    rid = storage.register_repo(_identity("github.com/keep/me"))
    storage.upsert_chunk(
        _make_chunk("sha256:k", "x.java", "x"),
        _vec(0), "m", 4, repo_id=rid,
    )
    storage.commit()
    storage.clear()
    # Chunks gone, repo preserved.
    assert storage.stats()["chunks"] == 0
    assert len(storage.list_repos()) == 1


def test_backfill_with_sanity_check(tmp_path: Path) -> None:
    """Auto-backfill assigns orphan chunks when target_repo paths match."""
    repo_root = tmp_path / "fake-repo"
    repo_root.mkdir()
    real_file = repo_root / "Hello.java"
    real_file.write_text("class Hello {}")

    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)
    # Insert orphan chunk pointing at a path that EXISTS under repo_root.
    s.upsert_chunk(
        _make_chunk("sha256:o", "Hello.java", "class Hello {}"),
        _vec(0), "m", 4, repo_id=None,
    )
    s.commit()

    assert s.count_chunks_without_repo() == 1
    rid = s.register_repo(_identity("github.com/q/r"))
    n = s.backfill_chunks_to_repo(rid, sample_root=repo_root)
    s.commit()
    assert n == 1
    assert s.count_chunks_without_repo() == 0


def test_backfill_refuses_when_paths_dont_match(tmp_path: Path) -> None:
    """Auto-backfill is a no-op when the target_repo changed since indexing."""
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)
    s.upsert_chunk(
        _make_chunk("sha256:o", "GhostFile.java", "x"),
        _vec(0), "m", 4, repo_id=None,
    )
    s.commit()
    rid = s.register_repo(_identity("github.com/x/y"))
    # tmp_path has no GhostFile.java → sanity check fails.
    n = s.backfill_chunks_to_repo(rid, sample_root=tmp_path)
    assert n == 0
    assert s.count_chunks_without_repo() == 1


# ---- faceted search tests ----


def _make_chunk_with_lang(
    chunk_id: str,
    rel_path: str,
    content: str,
    language: str = "java",
    kind: str = "method",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        rel_path=rel_path,
        language=language,
        kind=kind,
        symbol=None,
        start_line=1,
        end_line=10,
        content=content,
        content_hash=chunk_id,
    )


def test_compute_facets_full_index(tmp_path: Path) -> None:
    """compute_facets(None) returns counts for entire index."""
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)

    # Insert chunks with different languages and kinds
    s.upsert_chunk(
        _make_chunk_with_lang("sha256:a", "Foo.java", "class Foo", "java", "class"),
        _vec(0), "m", 4,
    )
    s.upsert_chunk(
        _make_chunk_with_lang("sha256:b", "Bar.java", "void bar()", "java", "method"),
        _vec(1), "m", 4,
    )
    s.upsert_chunk(
        _make_chunk_with_lang("sha256:c", "baz.py", "def baz():", "python", "function"),
        _vec(2), "m", 4,
    )
    s.commit()

    facets = s.compute_facets(None)

    assert facets["language"] == {"java": 2, "python": 1}
    assert facets["kind"] == {"class": 1, "method": 1, "function": 1}
    assert "(local)" in facets["repo"]  # no repo assigned


def test_compute_facets_for_specific_chunks(tmp_path: Path) -> None:
    """compute_facets(chunk_ids) returns counts only for those chunks."""
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)

    s.upsert_chunk(
        _make_chunk_with_lang("sha256:a", "Foo.java", "class Foo", "java", "class"),
        _vec(0), "m", 4,
    )
    s.upsert_chunk(
        _make_chunk_with_lang("sha256:b", "Bar.java", "void bar()", "java", "method"),
        _vec(1), "m", 4,
    )
    s.upsert_chunk(
        _make_chunk_with_lang("sha256:c", "baz.py", "def baz():", "python", "function"),
        _vec(2), "m", 4,
    )
    s.commit()

    # Only get facets for Java chunks
    facets = s.compute_facets(["sha256:a", "sha256:b"])

    assert facets["language"] == {"java": 2}
    assert facets["kind"] == {"class": 1, "method": 1}


def test_compute_facets_empty_list(tmp_path: Path) -> None:
    """compute_facets([]) returns empty dicts."""
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)

    s.upsert_chunk(
        _make_chunk_with_lang("sha256:a", "Foo.java", "class Foo", "java", "class"),
        _vec(0), "m", 4,
    )
    s.commit()

    facets = s.compute_facets([])

    assert facets == {"language": {}, "kind": {}, "repo": {}}

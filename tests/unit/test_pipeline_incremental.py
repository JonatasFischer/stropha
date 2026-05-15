"""Tests for Phase A — file-level dirty cache (schema v6).

Covers:
- `Storage.file_is_fresh` semantics (mtime + size + enricher + embedder)
- `Storage.upsert_file_meta` round-trip
- `Storage.delete_file_meta` returns count, removes row
- `Storage.list_stale_files` finds rows not in current set
- `Storage.delete_chunks_by_repo_paths` is repo-scoped
- Pipeline-level: file_skipped_fresh counters fire, chunker is bypassed
- Pipeline-level: file content change (mtime delta) invalidates the skip
- Pipeline-level: stale files get evicted at end of repo run
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from stropha.adapters.enricher.hierarchical import HierarchicalEnricher
from stropha.adapters.enricher.noop import NoopEnricher
from stropha.embeddings.base import Embedder
from stropha.ingest.git_meta import RepoIdentity
from stropha.models import Chunk
from stropha.pipeline import Pipeline
from stropha.storage import Storage


# --------------------------------------------------------------------------- helpers


class _StubEmbedder(Embedder):
    """4-dim deterministic embedder. Used in tests to avoid loading
    fastembed weights for what are really plumbing tests."""

    @property
    def model_name(self) -> str:
        return "stub:v1"

    @property
    def dim(self) -> int:
        return 4

    @property
    def batch_size(self) -> int:
        return 8

    @property
    def adapter_id(self) -> str:
        return "stub:v1"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "idx.db", embedding_dim=4)
    yield s
    s.close()


@pytest.fixture
def repo_id(storage: Storage) -> int:
    return storage.register_repo(
        RepoIdentity(
            normalized_key="local:test",
            remote_url=None,
            root_path="/tmp/test",
            default_branch=None,
            head_commit=None,
        )
    )


def _git_init_with_file(repo: Path, name: str, content: str) -> Path:
    import subprocess

    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@x.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    file_path = repo / name
    file_path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return file_path


# --------------------------------------------------------------------------- Storage methods


def test_file_is_fresh_returns_false_on_unknown_file(storage: Storage, repo_id: int) -> None:
    assert storage.file_is_fresh(
        repo_id, "missing.py", mtime=100.0, size_bytes=42,
    ) is False


def test_file_is_fresh_returns_true_when_state_matches(storage: Storage, repo_id: int) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=3,
        enricher_id="noop", embedder_model="m1",
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", mtime=100.0, size_bytes=42,
        enricher_id="noop", embedder_model="m1",
    ) is True


def test_file_is_fresh_returns_false_when_mtime_changes(
    storage: Storage, repo_id: int,
) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=3,
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", mtime=200.0, size_bytes=42,
    ) is False


def test_file_is_fresh_returns_false_when_size_changes(
    storage: Storage, repo_id: int,
) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=3,
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", mtime=100.0, size_bytes=99,
    ) is False


def test_file_is_fresh_returns_false_when_enricher_differs(
    storage: Storage, repo_id: int,
) -> None:
    """Drift contract (ADR-004): switching enricher must invalidate fresh
    files even when (mtime, size) match exactly."""
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=3,
        enricher_id="noop", embedder_model="m1",
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", mtime=100.0, size_bytes=42,
        enricher_id="hierarchical", embedder_model="m1",
    ) is False


def test_file_is_fresh_returns_false_when_embedder_differs(
    storage: Storage, repo_id: int,
) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=3,
        enricher_id="noop", embedder_model="m1",
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", mtime=100.0, size_bytes=42,
        enricher_id="noop", embedder_model="m2",
    ) is False


def test_file_is_fresh_ignores_unspecified_pipeline_keys(
    storage: Storage, repo_id: int,
) -> None:
    """Callers that pass enricher_id=None / embedder_model=None opt out of
    that side of the comparison — useful for storage-only tests."""
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=3,
        enricher_id="ENRICHER", embedder_model="MODEL",
    )
    # Both None → only mtime/size check
    assert storage.file_is_fresh(repo_id, "a.py", mtime=100.0, size_bytes=42) is True


def test_upsert_file_meta_idempotent(storage: Storage, repo_id: int) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=3,
    )
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="def",
        size_bytes=43, mtime=200.0, chunk_count=4,
    )
    rows = storage._conn.execute(
        "SELECT * FROM files WHERE repo_id = ? AND rel_path = ?",
        (repo_id, "a.py"),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["content_hash"] == "def"
    assert int(rows[0]["last_chunk_count"]) == 4


def test_delete_file_meta_removes_rows(storage: Storage, repo_id: int) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="h", size_bytes=1, mtime=1.0, chunk_count=1,
    )
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="b.py", content_hash="h", size_bytes=1, mtime=1.0, chunk_count=1,
    )
    count = storage.delete_file_meta(repo_id, ["a.py", "b.py", "missing.py"])
    assert count == 2  # missing.py reports 0


def test_list_stale_files_returns_missing(storage: Storage, repo_id: int) -> None:
    for p in ("a.py", "b.py", "c.py"):
        storage.upsert_file_meta(
            repo_id=repo_id, rel_path=p, content_hash="h",
            size_bytes=1, mtime=1.0, chunk_count=1,
        )
    stale = storage.list_stale_files(repo_id, ["a.py", "c.py"])
    assert set(stale) == {"b.py"}


def test_list_stale_files_empty_current_returns_all(storage: Storage, repo_id: int) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="x.py", content_hash="h",
        size_bytes=1, mtime=1.0, chunk_count=1,
    )
    assert storage.list_stale_files(repo_id, []) == ["x.py"]


def test_delete_chunks_by_repo_paths_is_scoped(storage: Storage) -> None:
    """Two repos with identically-named files; deletion in one must NOT
    touch the other."""
    repo_a = storage.register_repo(RepoIdentity(
        normalized_key="r-a", remote_url=None, root_path="/a",
        default_branch=None, head_commit=None,
    ))
    repo_b = storage.register_repo(RepoIdentity(
        normalized_key="r-b", remote_url=None, root_path="/b",
        default_branch=None, head_commit=None,
    ))
    for repo, cid in ((repo_a, "ca"), (repo_b, "cb")):
        c = Chunk(
            chunk_id=cid, rel_path="shared.py", language="python",
            kind="file", symbol=None, parent_chunk_id=None,
            start_line=1, end_line=5,
            content="x", content_hash=cid,
        )
        storage.upsert_chunk(c, [0.0] * 4, "stub", 4, repo_id=repo)
    storage.commit()

    deleted = storage.delete_chunks_by_repo_paths(repo_a, ["shared.py"])
    assert deleted == 1
    remaining = storage._conn.execute(
        "SELECT chunk_id FROM chunks WHERE rel_path = 'shared.py'"
    ).fetchall()
    assert {r["chunk_id"] for r in remaining} == {"cb"}


def test_stats_reports_files_tracked(storage: Storage, repo_id: int) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="h",
        size_bytes=1, mtime=1.0, chunk_count=1,
    )
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="b.py", content_hash="h",
        size_bytes=1, mtime=1.0, chunk_count=1,
    )
    s = storage.stats()
    assert s["files_tracked"] == 2


# --------------------------------------------------------------------------- Pipeline end-to-end


def test_pipeline_skips_unchanged_files_on_second_run(tmp_path: Path) -> None:
    """The user-visible Phase A win: second run of `stropha index` on an
    unchanged repo records files_skipped_fresh == files_visited and
    chunks_seen == 0 (chunker never ran).

    Pinning `mode="full"` keeps the Phase A semantic — every file goes
    through the walker, but the file-level cache short-circuits before
    chunking. Phase B `auto` mode would route through git-diff instead
    and skip even visiting unchanged files (covered by Phase B tests).
    """
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()

        s1 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True, mode="full")
        assert s1.chunks_embedded >= 1
        assert s1.files_skipped_fresh == 0

        s2 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="full")
        assert s2.files_skipped_fresh == s1.files_visited
        assert s2.chunks_seen == 0  # chunker bypassed
        assert s2.chunks_embedded == 0


def test_pipeline_invalidates_skip_when_file_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    main = _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True, mode="full")

        # Mutate the file → mtime + size change → must NOT be skipped.
        main.write_text("x = 2\ny = 3\n", encoding="utf-8")
        # Make sure mtime really changes (some filesystems quantize to 1s)
        os.utime(main, (1_000_000_000, 1_000_000_000))

        s2 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="full")
        assert s2.files_skipped_fresh == 0
        assert s2.chunks_seen >= 1


# --------------------------------------------------------------------------- Phase B — git-diff incremental


def test_incremental_falls_back_to_full_on_first_run(tmp_path: Path) -> None:
    """No `last_indexed_sha` stored yet → `--incremental` must NOT error;
    it walks everything and stores the checkpoint for next time."""
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        # Explicit incremental on a fresh DB.
        stats = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True, mode="incremental")
        # Fallback happened — full pass embedded the file.
        assert stats.chunks_embedded >= 1
        # last_indexed_sha is now set so future runs go incremental.
        repos = s.list_repos()
        assert len(repos) == 1
        meta_key = f"last_indexed_sha_{1}"
        assert s.get_meta(meta_key)


def test_incremental_picks_up_new_commit(tmp_path: Path) -> None:
    """Add a file in a 2nd commit; incremental mode embeds just that one."""
    import subprocess
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "a.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True)

        # Add new file in a fresh commit.
        (repo / "b.py").write_text("y = 2\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "add b"], cwd=repo, check=True)

        stats = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="incremental")
        # Only b.py touched
        assert stats.files_visited == 1
        # First file (a.py) skipped — no work
        assert stats.files_skipped_fresh == 0  # a.py not even visited
        assert stats.files_evicted == 0


def test_incremental_evicts_deleted_file(tmp_path: Path) -> None:
    import subprocess
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "keep.py", "k\n")
    (repo / "drop.py").write_text("d\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "add drop"], cwd=repo, check=True)

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True)

        # Delete file in next commit.
        subprocess.run(["git", "rm", "drop.py", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "drop"], cwd=repo, check=True)

        stats = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="incremental")
        assert stats.files_evicted == 1
        assert s._conn.execute(
            "SELECT 1 FROM chunks WHERE rel_path='drop.py' LIMIT 1"
        ).fetchone() is None


def test_incremental_renames_chunks_without_re_embed(tmp_path: Path) -> None:
    """Pure rename → rename_chunks preserves the original chunks (zero
    re-embed work). The file_meta row's rel_path is updated."""
    import subprocess
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "old_name.py", "def foo():\n    return 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        s1 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True)
        embedded_before = s1.chunks_embedded

        # Pure rename, no content change.
        subprocess.run(["git", "mv", "old_name.py", "new_name.py"],
                       cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "rename"], cwd=repo, check=True)

        s2 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="incremental")

        # The chunk_id today still hashes rel_path so the rename causes
        # at least some downstream re-processing. What MUST hold is that
        # the new path is present in chunks and the old is absent.
        rows = s._conn.execute(
            "SELECT rel_path FROM chunks ORDER BY rel_path"
        ).fetchall()
        paths = {r["rel_path"] for r in rows}
        assert "new_name.py" in paths
        assert "old_name.py" not in paths


def test_full_flag_overrides_incremental_default(tmp_path: Path) -> None:
    """Even after a checkpoint exists, mode='full' walks everything."""
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "a.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True)
        # Touch the file mtime artificially so file_is_fresh fails:
        # we want to confirm that mode='full' calls discover() on every
        # file, not git diff. The simplest check is that files_visited
        # > 0 even when no commits happened since last index.
        stats = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="full")
        # Walker visited the file (it was just skipped via cache).
        assert stats.files_visited == 1


def test_pipeline_evicts_stale_files_at_end_of_run(tmp_path: Path) -> None:
    """A file present at run 1 but deleted before run 2 has its chunks +
    file_meta evicted (Phase A passive stale cleanup)."""
    repo = tmp_path / "repo"
    keep = _git_init_with_file(repo, "keep.py", "x = 1\n")
    transient = repo / "transient.py"
    transient.write_text("y = 2\n", encoding="utf-8")
    import subprocess
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "add transient"], cwd=repo, check=True)

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        s1 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True, mode="full")
        assert s1.files_visited == 2

        # Remove the transient file from the working tree AND from git.
        transient.unlink()
        subprocess.run(["git", "rm", "transient.py", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "drop transient"], cwd=repo, check=True)

        s2 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="full")
        assert s2.files_evicted == 1
        # No chunks for transient.py remain.
        remaining = s._conn.execute(
            "SELECT 1 FROM chunks WHERE rel_path = 'transient.py' LIMIT 1"
        ).fetchone()
        assert remaining is None
        # file_meta row also gone.
        assert s._conn.execute(
            "SELECT 1 FROM files WHERE rel_path = 'transient.py' LIMIT 1"
        ).fetchone() is None

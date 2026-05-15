"""Multi-repo indexing & chunk_id namespacing (Phase 4)."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from stropha.embeddings.base import Embedder
from stropha.ingest.chunker import Chunker
from stropha.ingest.chunkers.base import make_chunk_id
from stropha.ingest.pipeline import IndexPipeline
from stropha.models import SourceFile
from stropha.storage import Storage

# ---- a deterministic stub embedder ---------------------------------------

class _StubEmbedder(Embedder):
    """Hashing pseudo-embedder — deterministic and dependency-free."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def batch_size(self) -> int:
        return 4

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = hash(t)
            v = [((h >> (i * 4)) & 0xF) / 16.0 for i in range(self._dim)]
            out.append(v)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _init_repo(path: Path, *, remote: str, file_name: str, file_content: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    (path / file_name).write_text(file_content)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-q", "-m", "seed"],
        cwd=path, check=True,
    )
    subprocess.run(["git", "remote", "add", "origin", remote], cwd=path, check=True)


# ---- chunk_id namespacing -------------------------------------------------

def test_chunk_id_namespacing_distinguishes_repos() -> None:
    """Two repos with identical (path, range, content) get distinct ids."""
    a = make_chunk_id("README.md", 1, 10, "hash", repo_key="github.com/x/a")
    b = make_chunk_id("README.md", 1, 10, "hash", repo_key="github.com/x/b")
    assert a != b


def test_chunk_id_without_repo_key_is_backward_compatible() -> None:
    """Default (no repo_key) matches the v0.1 derivation."""
    legacy = make_chunk_id("README.md", 1, 10, "hash")
    explicit = make_chunk_id("README.md", 1, 10, "hash", repo_key=None)
    assert legacy == explicit


def test_chunk_id_is_deterministic_under_namespace() -> None:
    a = make_chunk_id("x.py", 1, 5, "h", repo_key="key")
    b = make_chunk_id("x.py", 1, 5, "h", repo_key="key")
    assert a == b


# ---- dispatcher applies repo_key -----------------------------------------

def test_chunker_respects_repo_key(tmp_path: Path) -> None:
    """Identical files in two repos yield distinct chunk_ids when repo_key differs."""
    f = tmp_path / "small.py"
    f.write_text("print('hi')\n")
    sf = SourceFile(path=f, rel_path="small.py", language="python", size_bytes=12)

    c = Chunker()
    chunks_a = list(c.chunk([sf], repo_key="github.com/foo/a"))
    chunks_b = list(c.chunk([sf], repo_key="github.com/foo/b"))
    chunks_none = list(c.chunk([sf]))

    assert len(chunks_a) == 1
    assert chunks_a[0].chunk_id != chunks_b[0].chunk_id
    assert chunks_a[0].chunk_id != chunks_none[0].chunk_id
    # Content hash is unchanged — only the id is namespaced.
    assert chunks_a[0].content_hash == chunks_b[0].content_hash


# ---- full pipeline over two repos ----------------------------------------

@pytest.fixture
def two_repos_with_collision(tmp_path: Path) -> tuple[Path, Path]:
    """Build two real git repos that share the same README content + name."""
    repo_a = tmp_path / "alpha"
    repo_b = tmp_path / "beta"
    _init_repo(
        repo_a,
        remote="https://github.com/example/alpha.git",
        file_name="README.md",
        file_content="# Same Title\n\nIdentical body across repos for collision testing.\n",
    )
    _init_repo(
        repo_b,
        remote="https://github.com/example/beta.git",
        file_name="README.md",
        file_content="# Same Title\n\nIdentical body across repos for collision testing.\n",
    )
    return repo_a, repo_b


def test_pipeline_indexes_two_repos_without_collision(
    tmp_path: Path, two_repos_with_collision: tuple[Path, Path]
) -> None:
    repo_a, repo_b = two_repos_with_collision
    db = tmp_path / "multi.db"
    embedder = _StubEmbedder()

    with Storage(db, embedding_dim=embedder.dim) as storage:
        pipeline = IndexPipeline(
            storage=storage, embedder=embedder, repos=[repo_a, repo_b]
        )
        stats = pipeline.run(rebuild=True)

        # Both repos registered.
        repos = storage.list_repos()
        keys = {r.normalized_key for r in repos}
        assert keys == {"github.com/example/alpha", "github.com/example/beta"}

        # README from each repo lives in the index as a distinct chunk row.
        row_a = next(r for r in repos if r.normalized_key == "github.com/example/alpha")
        row_b = next(r for r in repos if r.normalized_key == "github.com/example/beta")
        assert row_a.files >= 1
        assert row_b.files >= 1
        assert row_a.chunks >= 1 and row_b.chunks >= 1

    # Stats reflect 2 repos.
    assert len(stats.repos) == 2
    assert stats.files_visited >= 2


def test_search_returns_correct_repo_for_each_hit(
    tmp_path: Path, two_repos_with_collision: tuple[Path, Path]
) -> None:
    repo_a, repo_b = two_repos_with_collision
    db = tmp_path / "multi.db"
    embedder = _StubEmbedder()

    with Storage(db, embedding_dim=embedder.dim) as storage:
        IndexPipeline(
            storage=storage, embedder=embedder, repos=[repo_a, repo_b]
        ).run(rebuild=True)

        # BM25 search hits both copies (different chunk_ids, same content).
        hits = storage.search_bm25("identical body", k=10)
        assert len(hits) >= 2

        # Every hit carries a non-null repo with one of the two keys.
        keys = {h.repo.normalized_key for h in hits if h.repo}
        assert keys == {"github.com/example/alpha", "github.com/example/beta"}


def test_pipeline_rebuild_clears_only_chunks_not_repos(
    tmp_path: Path, two_repos_with_collision: tuple[Path, Path]
) -> None:
    repo_a, repo_b = two_repos_with_collision
    db = tmp_path / "multi.db"
    embedder = _StubEmbedder()

    with Storage(db, embedding_dim=embedder.dim) as storage:
        # First pass: registers both repos + indexes.
        IndexPipeline(
            storage=storage, embedder=embedder, repos=[repo_a, repo_b]
        ).run(rebuild=True)
        first_repos = {r.normalized_key for r in storage.list_repos()}
        assert first_repos == {"github.com/example/alpha", "github.com/example/beta"}

        # Second pass with rebuild=True: chunks cleared then repopulated.
        IndexPipeline(
            storage=storage, embedder=embedder, repos=[repo_a]
        ).run(rebuild=True)
        second_repos = {r.normalized_key for r in storage.list_repos()}
        # Beta is still in `repos` (we don't garbage-collect repos table)
        # but has 0 chunks because the rebuild scope was alpha-only.
        assert "github.com/example/beta" in second_repos
        beta_row = next(
            r for r in storage.list_repos()
            if r.normalized_key == "github.com/example/beta"
        )
        assert beta_row.chunks == 0


def test_pipeline_rejects_no_target() -> None:
    """Constructing with neither `repo` nor `repos` is a programmer error."""
    embedder = _StubEmbedder()
    storage = Storage(Path("/tmp/never-opened.db"), embedding_dim=embedder.dim)
    try:
        with pytest.raises(ValueError):
            IndexPipeline(storage=storage, embedder=embedder)
        with pytest.raises(ValueError):
            IndexPipeline(
                storage=storage, embedder=embedder,
                repo=Path("/x"), repos=[Path("/y")],
            )
    finally:
        storage.close()
        Path("/tmp/never-opened.db").unlink(missing_ok=True)

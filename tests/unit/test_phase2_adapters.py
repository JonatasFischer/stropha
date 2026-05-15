"""Tests for Phase 2 adapters (walker, storage, retrieval) + builder wiring."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from stropha import adapters  # noqa: F401
from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
from stropha.adapters.storage.sqlite_vec import SqliteVecStorage, SqliteVecStorageConfig
from stropha.adapters.walker.git_ls_files import (
    GitLsFilesWalker,
    GitLsFilesWalkerConfig,
)
from stropha.errors import ConfigError
from stropha.models import Chunk
from stropha.pipeline import build_stages, load_pipeline_config
from stropha.pipeline.registry import all_adapters


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubEmbedder:
    @property
    def stage_name(self) -> str: return "embedder"
    @property
    def adapter_name(self) -> str: return "stub"
    @property
    def adapter_id(self) -> str: return "stub:v1"
    @property
    def model_name(self) -> str: return "stub-v1"
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

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """Init a tiny git repo with a Python and a Markdown file."""
    repo = tmp_path / "tinyrepo"
    repo.mkdir()
    (repo / "main.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    (repo / "README.md").write_text("# Tiny\n\nA tiny project.\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------


def test_phase2_stages_all_registered() -> None:
    reg = all_adapters()
    assert "git-ls-files" in reg["walker"]
    assert "sqlite-vec" in reg["storage"]
    assert "hybrid-rrf" in reg["retrieval"]


# ---------------------------------------------------------------------------
# Walker adapter
# ---------------------------------------------------------------------------


def test_walker_discovers_files_in_git_repo(tiny_repo: Path) -> None:
    walker = GitLsFilesWalker()
    found = list(walker.discover(tiny_repo))
    rel = sorted(sf.rel_path for sf in found)
    assert rel == ["README.md", "main.py"]


def test_walker_adapter_id_includes_max_bytes() -> None:
    a = GitLsFilesWalker(GitLsFilesWalkerConfig(max_file_bytes=2048))
    b = GitLsFilesWalker(GitLsFilesWalkerConfig(max_file_bytes=4096))
    assert a.adapter_id != b.adapter_id


def test_walker_health_is_ready() -> None:
    assert GitLsFilesWalker().health().status == "ready"


# ---------------------------------------------------------------------------
# Storage adapter
# ---------------------------------------------------------------------------


def test_storage_adapter_requires_embedding_dim(tmp_path: Path) -> None:
    cfg = SqliteVecStorageConfig(path=str(tmp_path / "x.db"))
    with pytest.raises(ValueError):
        SqliteVecStorage(cfg)


def test_storage_adapter_basic_lifecycle(tmp_path: Path) -> None:
    cfg = SqliteVecStorageConfig(path=str(tmp_path / "x.db"))
    s = SqliteVecStorage(cfg, embedding_dim=4)
    try:
        assert s.adapter_name == "sqlite-vec"
        assert s.adapter_id == "sqlite-vec:dim=4"
        h = s.health()
        assert h.status == "ready"
        assert "0" in h.detail["chunks"]
    finally:
        s.close()


def test_storage_adapter_persists_chunk(tmp_path: Path) -> None:
    cfg = SqliteVecStorageConfig(path=str(tmp_path / "x.db"))
    s = SqliteVecStorage(cfg, embedding_dim=4)
    try:
        chunk = Chunk(
            chunk_id="c1",
            rel_path="x.py",
            language="python",
            kind="file",
            start_line=1,
            end_line=2,
            content="x = 1",
            content_hash="hc1",
        )
        s.upsert_chunk(chunk, [1.0, 0.0, 0.0, 0.0], "stub-v1", 4, enricher_id="noop")
        s.commit()
        assert s.stats()["chunks"] == 1
    finally:
        s.close()


def test_storage_adapter_path_expansion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MY_DIR", str(tmp_path))
    cfg = SqliteVecStorageConfig(path="$MY_DIR/x.db")
    s = SqliteVecStorage(cfg, embedding_dim=4)
    try:
        # Round-trip: stats report the resolved path.
        assert str(tmp_path) in s.stats()["db_path"]
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Retrieval adapter
# ---------------------------------------------------------------------------


def test_retrieval_adapter_id_reflects_config() -> None:
    a = HybridRrfRetrieval(
        HybridRrfConfig(rrf_k=60, candidate_k=50),
        storage=_DummyStorage(),
        embedder=_StubEmbedder(),
    )
    b = HybridRrfRetrieval(
        HybridRrfConfig(rrf_k=120, candidate_k=50),
        storage=_DummyStorage(),
        embedder=_StubEmbedder(),
    )
    assert a.adapter_id != b.adapter_id


def test_retrieval_adapter_requires_storage_and_embedder() -> None:
    with pytest.raises(ValueError):
        HybridRrfRetrieval(HybridRrfConfig())


def test_retrieval_adapter_returns_empty_for_blank_query() -> None:
    r = HybridRrfRetrieval(
        HybridRrfConfig(),
        storage=_DummyStorage(),
        embedder=_StubEmbedder(),
    )
    assert r.search("") == []
    assert r.search("   ") == []


class _DummyStorage:
    """Storage stub: returns empty hit lists for all read methods."""

    def search_dense(self, *_args, **_kw): return []
    def search_bm25(self, *_args, **_kw): return []
    def search_symbol_tokens(self, *_args, **_kw): return []


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_builder_assembles_full_pipeline(tmp_path: Path) -> None:
    resolved = load_pipeline_config(
        project_root=tmp_path,
        environ={
            "STROPHA_INDEX_PATH": str(tmp_path / "x.db"),
            "STROPHA_LOCAL_EMBED_MODEL": "BAAI/bge-small-en-v1.5",
        },
    )
    # Skip storage open to avoid loading fastembed in this assertion-focused test.
    built = build_stages(resolved, open_storage=False)
    assert built.walker.adapter_name == "git-ls-files"
    assert built.enricher.adapter_name == "noop"
    # embedder is built (loads ONNX) — that's OK for a single shot.
    assert built.embedder.adapter_name in ("local", "voyage")
    assert built.storage is None
    assert built.retrieval is None


def test_builder_storage_path_alias_propagates(tmp_path: Path) -> None:
    """STROPHA_INDEX_PATH must route into pipeline.storage.config.path."""
    resolved = load_pipeline_config(
        project_root=tmp_path,
        environ={"STROPHA_INDEX_PATH": str(tmp_path / "alias.db")},
    )
    assert resolved["storage"]["config"]["path"] == str(tmp_path / "alias.db")


def test_builder_unknown_storage_adapter_raises(tmp_path: Path) -> None:
    resolved = load_pipeline_config(project_root=tmp_path, environ={})
    resolved["storage"]["adapter"] = "no-such-storage"
    with pytest.raises(ConfigError):
        build_stages(resolved, open_storage=True)

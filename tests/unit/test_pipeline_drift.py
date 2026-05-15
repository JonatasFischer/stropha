"""Tests for adapter-drift detection (ADR-004) and enrichment cache.

Drift = stored ``enricher_id`` differs from the active adapter's id. The
pipeline must re-enrich + re-embed the affected chunk transparently, no
``--rebuild`` required. The enrichments cache (PRIMARY KEY = content_hash,
enricher_id) avoids redoing the enricher's work when the same source body
is touched twice within a single DB.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from stropha.adapters.enricher.hierarchical import HierarchicalEnricher
from stropha.adapters.enricher.noop import NoopEnricher
from stropha.models import Chunk
from stropha.pipeline.pipeline import Pipeline
from stropha.storage import Storage

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Deterministic embedder. Vector = hash of text mod a small int."""

    @property
    def stage_name(self) -> str:
        return "embedder"

    @property
    def adapter_name(self) -> str:
        return "stub"

    @property
    def adapter_id(self) -> str:
        return "stub:v1"

    @property
    def model_name(self) -> str:
        return "stub-v1"

    @property
    def dim(self) -> int:
        return 8

    @property
    def batch_size(self) -> int:
        return 16

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
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        h = abs(hash(text)) % 1000
        return [(h % 7) / 7.0] * 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_with_one_file(tmp_path: Path) -> Path:
    """Init a tiny git repo with one Python file."""
    import subprocess

    repo = tmp_path / "tinyrepo"
    repo.mkdir()
    (repo / "main.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def test_chunk_is_fresh_treats_legacy_null_as_noop(tmp_path: Path) -> None:
    """Upgrading from v0.1.0 (NULL enricher_id) MUST NOT trigger re-embed for noop."""
    db = tmp_path / "test.db"
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
    with Storage(db, embedding_dim=8) as s:
        # Insert as a legacy v0.1.0 row would: no enricher_id.
        s.upsert_chunk(chunk, [0.0] * 8, "stub-v1", 8)
        s.commit()
        # Active enricher is noop -> stored NULL must compare as noop.
        assert s.chunk_is_fresh("c1", "hc1", "stub-v1", enricher_id="noop") is True
        # Different enricher -> drift detected.
        assert s.chunk_is_fresh("c1", "hc1", "stub-v1", enricher_id="hierarchical:p") is False


def test_chunk_is_fresh_detects_content_change(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
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
    with Storage(db, embedding_dim=8) as s:
        s.upsert_chunk(chunk, [0.0] * 8, "m", 8, enricher_id="noop")
        s.commit()
        assert s.chunk_is_fresh("c1", "DIFFERENT", "m", enricher_id="noop") is False


def test_chunk_is_fresh_detects_model_change(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
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
    with Storage(db, embedding_dim=8) as s:
        s.upsert_chunk(chunk, [0.0] * 8, "m1", 8, enricher_id="noop")
        s.commit()
        assert s.chunk_is_fresh("c1", "hc1", "m2", enricher_id="noop") is False


# ---------------------------------------------------------------------------
# Enrichment cache
# ---------------------------------------------------------------------------


def test_enrichment_cache_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with Storage(db, embedding_dim=8) as s:
        assert s.get_enrichment("h", "noop") is None
        s.put_enrichment("h", "noop", "hello world")
        assert s.get_enrichment("h", "noop") == "hello world"
        # Different enricher_id is a cache miss.
        assert s.get_enrichment("h", "hierarchical:p") is None


def test_enrichment_cache_overwrite_on_repeat_put(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with Storage(db, embedding_dim=8) as s:
        s.put_enrichment("h", "noop", "v1")
        s.put_enrichment("h", "noop", "v2")
        assert s.get_enrichment("h", "noop") == "v2"


# ---------------------------------------------------------------------------
# Pipeline end-to-end with stub embedder
# ---------------------------------------------------------------------------


def test_pipeline_switching_enricher_triggers_re_embed(
    tmp_path: Path, repo_with_one_file: Path
) -> None:
    db = tmp_path / "drift.db"
    embedder = _StubEmbedder()

    with Storage(db, embedding_dim=embedder.dim) as storage:
        # Run 1: noop enricher.
        p1 = Pipeline(
            storage=storage,
            embedder=embedder,
            enricher=NoopEnricher(),
            repos=[repo_with_one_file],
        )
        stats1 = p1.run(rebuild=True)
        assert stats1.chunks_embedded >= 1
        assert stats1.chunks_skipped_fresh == 0
        assert stats1.files_skipped_fresh == 0

        # Run 2: same enricher → Phase A file cache short-circuits before
        # the chunker. The user-visible invariant is "no new embedding
        # work"; chunks_seen will be 0 because the chunker never ran.
        p2 = Pipeline(
            storage=storage,
            embedder=embedder,
            enricher=NoopEnricher(),
            repos=[repo_with_one_file],
        )
        stats2 = p2.run()
        assert stats2.chunks_embedded == 0
        assert stats2.files_skipped_fresh == stats1.files_visited

        # Run 3: switch enricher → file_is_fresh sees enricher_id drift
        # and returns False, so the chunker runs and every chunk re-embeds.
        p3 = Pipeline(
            storage=storage,
            embedder=embedder,
            enricher=HierarchicalEnricher(),
            repos=[repo_with_one_file],
        )
        stats3 = p3.run()
        assert stats3.chunks_embedded == stats1.chunks_seen
        assert stats3.chunks_skipped_fresh == 0
        assert stats3.files_skipped_fresh == 0

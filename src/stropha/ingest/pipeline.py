"""Indexing pipeline orchestration.

Composition: Walker → Chunker → Embedder → Storage.

Phase 1 (MVP) semantics:
- `index` → walk everything, upsert. The per-chunk freshness check
  (`Storage.chunk_is_fresh`) makes a no-op re-run near-instant — no API
  calls, no DB writes. Stale chunks for deleted source are NOT cleaned up
  in this mode.
- `index --rebuild` → drop the index, then full walk. Use this when the
  source tree changed (file deletions, method renames) and you want a
  clean state.

True git-diff-based incremental (post-commit hooks, soft index for
working tree) is deferred to Phase 2/3 — see spec §8.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..embeddings.base import Embedder
from ..logging import get_logger
from ..models import Chunk
from ..storage import Storage
from .chunker import Chunker
from .git_meta import detect as detect_repo
from .walker import Walker

log = get_logger(__name__)


@dataclass
class IndexStats:
    files_visited: int = 0
    chunks_seen: int = 0
    chunks_embedded: int = 0
    chunks_skipped_fresh: int = 0
    repo_normalized_key: str | None = None
    repo_url: str | None = None
    chunks_backfilled: int = 0


class IndexPipeline:
    def __init__(
        self,
        repo: Path,
        storage: Storage,
        embedder: Embedder,
        chunker: Chunker | None = None,
        walker: Walker | None = None,
    ) -> None:
        self._repo = repo.resolve()
        self._storage = storage
        self._embedder = embedder
        self._chunker = chunker or Chunker()
        self._walker = walker or Walker(self._repo)

    def run(self, *, rebuild: bool = False) -> IndexStats:
        stats = IndexStats()

        # 1. Identify + register the repository BEFORE clearing/indexing so
        #    every chunk carries the correct repo_id.
        identity = detect_repo(self._repo)
        repo_id = self._storage.register_repo(identity)
        stats.repo_normalized_key = identity.normalized_key
        stats.repo_url = identity.remote_url
        log.info(
            "pipeline.repo_registered",
            repo=identity.normalized_key,
            url=identity.remote_url,
            branch=identity.default_branch,
            commit=identity.head_commit,
        )

        # 2. Backfill orphan chunks (schema v1 → v2 upgrade path).
        if not rebuild and self._storage.count_chunks_without_repo() > 0:
            n = self._storage.backfill_chunks_to_repo(repo_id, self._repo)
            stats.chunks_backfilled = n
            if n > 0:
                log.info("pipeline.chunks_backfilled", count=n, repo_id=repo_id)
                self._storage.commit()

        # 3. Optional clean slate.
        if rebuild:
            log.info("pipeline.rebuild_clear")
            self._storage.clear()

        # 4. Walk → chunk → embed → upsert.
        self._index_files(self._walker.discover(), stats, repo_id=repo_id)
        self._storage.commit()
        return stats

    # ----- internals -----

    def _index_files(
        self, files: Iterable, stats: IndexStats, *, repo_id: int
    ) -> None:
        batch_size = self._embedder.batch_size
        batch_texts: list[str] = []
        batch_chunks: list[Chunk] = []
        seen_paths: set[str] = set()

        def flush() -> None:
            if not batch_texts:
                return
            vectors = self._embedder.embed_documents(batch_texts)
            for chunk, vec in zip(batch_chunks, vectors, strict=True):
                self._storage.upsert_chunk(
                    chunk,
                    vec,
                    self._embedder.model_name,
                    self._embedder.dim,
                    repo_id=repo_id,
                )
                stats.chunks_embedded += 1
            self._storage.commit()
            batch_texts.clear()
            batch_chunks.clear()

        for sf in files:
            seen_paths.add(sf.rel_path)
            for chunk in self._chunker.chunk([sf]):
                stats.chunks_seen += 1
                if self._storage.chunk_is_fresh(
                    chunk.chunk_id, chunk.content_hash, self._embedder.model_name
                ):
                    stats.chunks_skipped_fresh += 1
                    continue
                batch_texts.append(chunk.content)
                batch_chunks.append(chunk)
                if len(batch_texts) >= batch_size:
                    flush()
        flush()
        stats.files_visited = len(seen_paths)

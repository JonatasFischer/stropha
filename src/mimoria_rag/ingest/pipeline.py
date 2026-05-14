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
from .walker import Walker

log = get_logger(__name__)


@dataclass
class IndexStats:
    files_visited: int = 0
    chunks_seen: int = 0
    chunks_embedded: int = 0
    chunks_skipped_fresh: int = 0


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
        if rebuild:
            log.info("pipeline.rebuild_clear")
            self._storage.clear()
        stats = IndexStats()
        self._index_files(self._walker.discover(), stats)
        self._storage.commit()
        return stats

    # ----- internals -----

    def _index_files(self, files: Iterable, stats: IndexStats) -> None:
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
                    chunk, vec, self._embedder.model_name, self._embedder.dim
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

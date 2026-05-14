"""Indexing pipeline orchestration.

Composition: Walker → Chunker → Embedder → Storage.

Phase 1 (MVP) semantics — single repo:
- `index` → walk everything, upsert. The per-chunk freshness check
  (`Storage.chunk_is_fresh`) makes a no-op re-run near-instant — no API
  calls, no DB writes. Stale chunks for deleted source are NOT cleaned up
  in this mode.
- `index --rebuild` → drop the index, then full walk. Use this when the
  source tree changed (file deletions, method renames) and you want a
  clean state.

Phase 4 (multi-repo) semantics — list of repos:
- The pipeline accepts ``repos: list[Path]``. Each repo is walked and
  indexed sequentially, sharing the same Storage and Embedder.
- ``chunk_id`` is namespaced by repo (via ``Chunker.chunk(repo_key=…)``)
  so identical files in distinct repos do not collide.
- ``--rebuild`` clears chunks but keeps the ``repos`` table; identities
  survive rebuilds so FK references stay stable.

True git-diff-based incremental (post-commit hooks, soft index for
working tree) is deferred to Phase 2/3 — see spec §8.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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
class RepoStats:
    """Per-repo counters reported back from a pipeline run."""

    normalized_key: str
    url: str | None
    files_visited: int = 0
    chunks_seen: int = 0
    chunks_embedded: int = 0
    chunks_skipped_fresh: int = 0
    chunks_backfilled: int = 0


@dataclass
class IndexStats:
    """Aggregate result of an `IndexPipeline.run()`."""

    repos: list[RepoStats] = field(default_factory=list)

    @property
    def files_visited(self) -> int:
        return sum(r.files_visited for r in self.repos)

    @property
    def chunks_seen(self) -> int:
        return sum(r.chunks_seen for r in self.repos)

    @property
    def chunks_embedded(self) -> int:
        return sum(r.chunks_embedded for r in self.repos)

    @property
    def chunks_skipped_fresh(self) -> int:
        return sum(r.chunks_skipped_fresh for r in self.repos)

    @property
    def chunks_backfilled(self) -> int:
        return sum(r.chunks_backfilled for r in self.repos)

    # Back-compat single-repo accessors (used by CLI display and tests
    # written against the v0.1.0 single-repo API).
    @property
    def repo_normalized_key(self) -> str | None:
        return self.repos[0].normalized_key if self.repos else None

    @property
    def repo_url(self) -> str | None:
        return self.repos[0].url if self.repos else None


class IndexPipeline:
    """Walks one or more repos, chunks each file, embeds, persists."""

    def __init__(
        self,
        storage: Storage,
        embedder: Embedder,
        *,
        repo: Path | None = None,
        repos: Sequence[Path] | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        if repo is None and not repos:
            raise ValueError("IndexPipeline requires `repo` or `repos`.")
        if repo is not None and repos:
            raise ValueError("Pass exactly one of `repo` or `repos`.")

        targets = [repo] if repo is not None else list(repos or [])
        self._repos: list[Path] = [p.resolve() for p in targets]
        self._storage = storage
        self._embedder = embedder
        self._chunker = chunker or Chunker()

    def run(self, *, rebuild: bool = False) -> IndexStats:
        stats = IndexStats()

        # Auto-backfill orphans (schema v1 → v2 upgrade). Only meaningful in
        # single-repo mode where we know which repo the orphans belong to.
        # In multi-repo mode, the user should run `index --rebuild` to get
        # a clean state because there's no way to attribute orphans correctly.
        if not rebuild and self._storage.count_chunks_without_repo() > 0:
            if len(self._repos) == 1:
                # Single-repo: best-effort backfill with sanity check.
                identity = detect_repo(self._repos[0])
                repo_id = self._storage.register_repo(identity)
                n = self._storage.backfill_chunks_to_repo(repo_id, self._repos[0])
                if n > 0:
                    log.info(
                        "pipeline.chunks_backfilled", count=n, repo_id=repo_id
                    )
                    self._storage.commit()
            else:
                log.warning(
                    "pipeline.legacy_chunks_present_multi_repo",
                    advice="run `stropha index --rebuild` to clean up",
                )

        # Clear in the rebuild case BEFORE registering, so the repos table is
        # preserved but every chunk row goes away.
        if rebuild:
            log.info("pipeline.rebuild_clear")
            self._storage.clear()

        # Walk + index each repo in sequence.
        for repo_path in self._repos:
            repo_stats = self._index_one_repo(repo_path)
            stats.repos.append(repo_stats)
            self._storage.commit()

        # Record current chunk_id derivation version so future migrations
        # know whether existing rows have repo-namespaced ids.
        self._storage.set_meta("chunk_id_version", "2")
        self._storage.commit()
        return stats

    # ----- internals -----

    def _index_one_repo(self, repo_path: Path) -> RepoStats:
        identity = detect_repo(repo_path)
        repo_id = self._storage.register_repo(identity)
        log.info(
            "pipeline.repo_registered",
            repo=identity.normalized_key,
            url=identity.remote_url,
            branch=identity.default_branch,
            commit=identity.head_commit,
        )
        rstats = RepoStats(
            normalized_key=identity.normalized_key,
            url=identity.remote_url,
        )
        walker = Walker(repo_path)
        self._index_files(
            walker.discover(),
            rstats,
            repo_id=repo_id,
            repo_key=identity.normalized_key,
        )
        log.info(
            "pipeline.repo_done",
            repo=identity.normalized_key,
            files=rstats.files_visited,
            chunks=rstats.chunks_seen,
            embedded=rstats.chunks_embedded,
            reused=rstats.chunks_skipped_fresh,
        )
        return rstats

    def _index_files(
        self,
        files: Iterable,
        stats: RepoStats,
        *,
        repo_id: int,
        repo_key: str,
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
            for chunk in self._chunker.chunk([sf], repo_key=repo_key):
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

"""Pipeline orchestrator — adapter-aware indexing.

Phase 2 wires walker, enricher, embedder, and storage as adapter
instances. Chunker remains the legacy ``stropha.ingest.chunker.Chunker``
until Phase 3 ships its dispatcher adapter.

Drift detection: a chunk is fresh only when its stored
``(content_hash, embedding_model, enricher_id)`` matches the active
triplet. Changing any of those triggers transparent re-processing. The
``enrichments`` cache table avoids redoing the enricher's work when the
same content+adapter combo has been seen before in this DB.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..ingest.git_meta import detect as detect_repo
from ..logging import get_logger
from ..models import Chunk
from ..stages.chunker import ChunkerStage
from ..stages.embedder import EmbedderStage
from ..stages.enricher import EnricherStage
from ..stages.storage import StorageStage
from ..stages.walker import WalkerStage
from .base import StageContext

log = get_logger(__name__)


@dataclass
class RepoStats:
    """Per-repo counters returned by :meth:`Pipeline.run`."""

    normalized_key: str
    url: str | None
    files_visited: int = 0
    chunks_seen: int = 0
    chunks_embedded: int = 0
    chunks_skipped_fresh: int = 0
    chunks_enriched_from_cache: int = 0
    chunks_enriched_fresh: int = 0


@dataclass
class PipelineStats:
    """Aggregate result of a :meth:`Pipeline.run`."""

    repos: list[RepoStats] = field(default_factory=list)
    enricher_id: str = ""
    embedder_id: str = ""

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
    def chunks_enriched_from_cache(self) -> int:
        return sum(r.chunks_enriched_from_cache for r in self.repos)

    @property
    def chunks_enriched_fresh(self) -> int:
        return sum(r.chunks_enriched_fresh for r in self.repos)


class Pipeline:
    """Adapter-aware walker → chunker → enricher → embedder → storage."""

    def __init__(
        self,
        *,
        storage: StorageStage,
        embedder: EmbedderStage,
        enricher: EnricherStage,
        repos: Sequence[Path],
        walker: WalkerStage | None = None,
        chunker: ChunkerStage | None = None,
    ) -> None:
        if not repos:
            raise ValueError("Pipeline requires at least one repo")
        self._storage = storage
        self._embedder = embedder
        self._enricher = enricher
        self._repos = [p.resolve() for p in repos]
        # Lazy import — keeps the framework decoupled from a specific
        # adapter until the caller passes one or we fall back here.
        if walker is None:
            from ..adapters.walker.git_ls_files import GitLsFilesWalker

            walker = GitLsFilesWalker()
        if chunker is None:
            from ..adapters.chunker.tree_sitter_dispatch import (
                TreeSitterDispatchChunker,
            )

            chunker = TreeSitterDispatchChunker()
        self._walker = walker
        self._chunker = chunker

    # ------------------------------------------------------------------ run
    def run(self, *, rebuild: bool = False) -> PipelineStats:
        stats = PipelineStats(
            enricher_id=self._enricher.adapter_id,
            embedder_id=self._embedder.adapter_id,
        )
        # Auto-backfill orphans (v1→v2 upgrade path).
        if not rebuild and self._storage.count_chunks_without_repo() > 0:
            if len(self._repos) == 1:
                identity = detect_repo(self._repos[0])
                repo_id = self._storage.register_repo(identity)
                n = self._storage.backfill_chunks_to_repo(repo_id, self._repos[0])
                if n > 0:
                    log.info("pipeline.chunks_backfilled", count=n)
                    self._storage.commit()
            else:
                log.warning(
                    "pipeline.legacy_chunks_present_multi_repo",
                    advice="run `stropha index --rebuild` to clean up",
                )

        if rebuild:
            log.info("pipeline.rebuild_clear")
            self._storage.clear()

        for repo_path in self._repos:
            rstats = self._index_one_repo(repo_path)
            stats.repos.append(rstats)
            self._storage.commit()

        self._storage.set_meta("chunk_id_version", "2")
        self._storage.set_meta("active_enricher_id", self._enricher.adapter_id)
        self._storage.set_meta("active_embedder_id", self._embedder.adapter_id)
        self._storage.commit()
        return stats

    # --------------------------------------------------------------- internals
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
        self._index_files(
            self._walker.discover(repo_path),
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
            enricher_cache_hits=rstats.chunks_enriched_from_cache,
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
        batch_embed_texts: list[str] = []
        seen_paths: set[str] = set()
        enricher_id = self._enricher.adapter_id
        embedder_model = self._embedder.model_name

        def flush() -> None:
            if not batch_texts:
                return
            vectors = self._embedder.embed_documents(batch_texts)
            for chunk, vec, embedded_text in zip(
                batch_chunks, vectors, batch_embed_texts, strict=True
            ):
                self._storage.upsert_chunk(
                    chunk,
                    vec,
                    embedder_model,
                    self._embedder.dim,
                    repo_id=repo_id,
                    embedding_text=embedded_text,
                    enricher_id=enricher_id,
                )
                stats.chunks_embedded += 1
            self._storage.commit()
            batch_texts.clear()
            batch_chunks.clear()
            batch_embed_texts.clear()

        for sf in files:
            seen_paths.add(sf.rel_path)
            # Build a per-file index of chunks for parent-chunk lookup in
            # the hierarchical enricher. Cheap — chunker emits ≤ ~tens per file.
            file_chunks: list[Chunk] = list(
                self._chunker.chunk([sf], repo_key=repo_key)
            )
            by_id: dict[str, Chunk] = {c.chunk_id: c for c in file_chunks}
            for chunk in file_chunks:
                stats.chunks_seen += 1
                if self._storage.chunk_is_fresh(
                    chunk.chunk_id,
                    chunk.content_hash,
                    embedder_model,
                    enricher_id=enricher_id,
                ):
                    stats.chunks_skipped_fresh += 1
                    continue

                # Enrich (cache-aware).
                cached = self._storage.get_enrichment(chunk.content_hash, enricher_id)
                if cached is not None:
                    embed_text = cached
                    stats.chunks_enriched_from_cache += 1
                else:
                    parent = (
                        by_id.get(chunk.parent_chunk_id)
                        if chunk.parent_chunk_id
                        else None
                    )
                    ctx = StageContext(repo_key=repo_key, parent_chunk=parent)
                    embed_text = self._enricher.enrich(chunk, ctx)
                    self._storage.put_enrichment(
                        chunk.content_hash, enricher_id, embed_text
                    )
                    stats.chunks_enriched_fresh += 1

                batch_texts.append(embed_text)
                batch_chunks.append(chunk)
                batch_embed_texts.append(embed_text)
                if len(batch_texts) >= batch_size:
                    flush()
        flush()
        stats.files_visited = len(seen_paths)

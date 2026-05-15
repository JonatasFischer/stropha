"""StorageStage protocol — persist chunks + vectors and serve read APIs.

A storage adapter encapsulates everything the index needs:

- writes: ``upsert_chunk``, ``register_repo``, ``commit``, ``clear``
- reads: ``search_dense``, ``search_bm25``, ``search_symbol_tokens``,
        ``lookup_by_symbol``, ``file_outline``, ``list_repos``, ``stats``
- introspection: drift cache hooks (``get_enrichment``, ``put_enrichment``,
        ``chunk_is_fresh``)

Phase 2 ships a single adapter, ``sqlite-vec``, that wraps the existing
``stropha.storage.sqlite.Storage``. Future adapters (Phase 4 spec §10.5):
``qdrant``, ``pgvector``, ``lancedb``.

Concrete adapters MAY add backend-specific helpers; the Pipeline / CLI
must depend only on the protocol surface listed above.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ..ingest.git_meta import RepoIdentity
from ..models import Chunk, RepoStats, SearchHit
from ..pipeline.base import StageHealth


@runtime_checkable
class StorageStage(Protocol):
    """Persistence + read backend for the index."""

    # ----- Stage contract --------------------------------------------------

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    @property
    def dim(self) -> int:
        """Vector dimensionality the backend was provisioned with."""

    # ----- Lifecycle ------------------------------------------------------

    def commit(self) -> None: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...

    # ----- Writes ---------------------------------------------------------

    def register_repo(self, identity: RepoIdentity) -> int: ...

    def upsert_chunk(
        self,
        chunk: Chunk,
        embedding: list[float],
        embedding_model: str,
        embedding_dim: int,
        repo_id: int | None = None,
        embedding_text: str | None = None,
        enricher_id: str | None = None,
    ) -> int: ...

    def chunk_is_fresh(
        self,
        chunk_id: str,
        content_hash: str,
        embedding_model: str,
        enricher_id: str | None = None,
    ) -> bool: ...

    def get_enrichment(self, content_hash: str, enricher_id: str) -> str | None: ...

    def put_enrichment(
        self, content_hash: str, enricher_id: str, embedding_text: str
    ) -> None: ...

    def count_chunks_without_repo(self) -> int: ...

    def backfill_chunks_to_repo(self, repo_id: int, sample_root: Any) -> int: ...

    def set_meta(self, key: str, value: str) -> None: ...

    def get_meta(self, key: str) -> str | None: ...

    # ----- Reads ----------------------------------------------------------

    def search_dense(self, query_vec: list[float], k: int = 10) -> list[SearchHit]: ...

    def search_bm25(self, query: str, k: int = 50) -> list[SearchHit]: ...

    def search_symbol_tokens(self, query: str, k: int = 20) -> list[SearchHit]: ...

    def lookup_by_symbol(self, symbol: str, limit: int = 10) -> list[SearchHit]: ...

    def file_outline(self, rel_path: str) -> list[dict[str, Any]]: ...

    def list_repos(self) -> list[RepoStats]: ...

    def stats(self) -> dict[str, Any]: ...

"""Sparse retrieval stream — SQLite FTS5 BM25."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ....models import SearchHit
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter
from ....stages.storage import StorageStage


class Fts5Bm25Config(BaseModel):
    k: int = Field(default=50, ge=1, le=500)


@register_adapter(stage="retrieval-stream", name="fts5-bm25")
class Fts5Bm25Stream:
    Config = Fts5Bm25Config

    def __init__(
        self,
        config: Fts5Bm25Config | None = None,
        *,
        storage: StorageStage | None = None,
    ) -> None:
        if storage is None:
            raise ValueError("Fts5Bm25Stream requires storage=")
        self._config = config or Fts5Bm25Config()
        self._storage = storage

    @property
    def stage_name(self) -> str: return "retrieval-stream"

    @property
    def adapter_name(self) -> str: return "fts5-bm25"

    @property
    def adapter_id(self) -> str: return f"fts5-bm25:k={self._config.k}"

    @property
    def config_schema(self) -> type[BaseModel]: return Fts5Bm25Config

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message=f"fts5-bm25 (k={self._config.k})")

    @property
    def k(self) -> int:
        return self._config.k

    def search(self, query: str, query_vec: list[float] | None) -> list[SearchHit]:
        return self._storage.search_bm25(query, k=self._config.k)

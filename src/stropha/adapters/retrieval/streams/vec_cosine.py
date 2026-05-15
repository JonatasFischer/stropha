"""Dense ANN retrieval stream — sqlite-vec virtual table."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ....models import SearchHit
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter
from ....stages.storage import StorageStage


class VecCosineConfig(BaseModel):
    k: int = Field(default=50, ge=1, le=500)


@register_adapter(stage="retrieval-stream", name="vec-cosine")
class VecCosineStream:
    Config = VecCosineConfig

    def __init__(
        self,
        config: VecCosineConfig | None = None,
        *,
        storage: StorageStage | None = None,
    ) -> None:
        if storage is None:
            raise ValueError("VecCosineStream requires storage=")
        self._config = config or VecCosineConfig()
        self._storage = storage

    @property
    def stage_name(self) -> str: return "retrieval-stream"

    @property
    def adapter_name(self) -> str: return "vec-cosine"

    @property
    def adapter_id(self) -> str: return f"vec-cosine:k={self._config.k}"

    @property
    def config_schema(self) -> type[BaseModel]: return VecCosineConfig

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message=f"vec-cosine (k={self._config.k})")

    @property
    def k(self) -> int:
        return self._config.k

    def search(self, query: str, query_vec: list[float] | None) -> list[SearchHit]:
        if query_vec is None:
            return []
        return self._storage.search_dense(query_vec, k=self._config.k)

"""Identifier-token retrieval stream — direct LIKE match on the symbol column.

Cheap query routing per ``docs/architecture/stropha-system.md`` §6.3.5.
Used to lift exact symbol-name queries into the top of the fused result.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ....models import SearchHit
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter
from ....stages.storage import StorageStage


class LikeTokensConfig(BaseModel):
    k: int = Field(default=20, ge=1, le=200)


@register_adapter(stage="retrieval-stream", name="like-tokens")
class LikeTokensStream:
    Config = LikeTokensConfig

    def __init__(
        self,
        config: LikeTokensConfig | None = None,
        *,
        storage: StorageStage | None = None,
    ) -> None:
        if storage is None:
            raise ValueError("LikeTokensStream requires storage=")
        self._config = config or LikeTokensConfig()
        self._storage = storage

    @property
    def stage_name(self) -> str: return "retrieval-stream"

    @property
    def adapter_name(self) -> str: return "like-tokens"

    @property
    def adapter_id(self) -> str: return f"like-tokens:k={self._config.k}"

    @property
    def config_schema(self) -> type[BaseModel]: return LikeTokensConfig

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message=f"like-tokens (k={self._config.k})")

    @property
    def k(self) -> int:
        return self._config.k

    def search(self, query: str, query_vec: list[float] | None) -> list[SearchHit]:
        return self._storage.search_symbol_tokens(query, k=self._config.k)

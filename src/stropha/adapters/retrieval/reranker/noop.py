"""Noop reranker — pass-through, preserves RRF order.

Default reranker when no cross-encoder is configured. Zero latency overhead.
"""

from __future__ import annotations

from pydantic import BaseModel

from ....models import SearchHit
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter


class NoopRerankerConfig(BaseModel):
    """Empty config. The noop reranker has no tunables."""
    pass


@register_adapter(stage="reranker", name="noop")
class NoopReranker:
    """Pass-through reranker. Returns hits unchanged, truncated to top_k."""

    Config = NoopRerankerConfig

    def __init__(self, config: NoopRerankerConfig | None = None) -> None:
        self._config = config or NoopRerankerConfig()

    @property
    def stage_name(self) -> str:
        return "reranker"

    @property
    def adapter_name(self) -> str:
        return "noop"

    @property
    def adapter_id(self) -> str:
        return "reranker:noop"

    @property
    def config_schema(self) -> type[BaseModel]:
        return NoopRerankerConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message="noop reranker (pass-through)",
        )

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Return hits unchanged, truncated to top_k."""
        return hits[:top_k]

"""Noop enricher — identity. Default; preserves Phase 0/1 behavior exactly."""

from __future__ import annotations

from pydantic import BaseModel

from ...models import Chunk
from ...pipeline.base import StageContext, StageHealth
from ...pipeline.registry import register_adapter
from ...stages.enricher import EnricherStage


class NoopEnricherConfig(BaseModel):
    """Empty config. The noop enricher has no tunables."""


@register_adapter(stage="enricher", name="noop")
class NoopEnricher(EnricherStage):
    """Returns ``chunk.content`` unchanged."""

    Config = NoopEnricherConfig

    def __init__(self, config: NoopEnricherConfig | None = None) -> None:
        self._config = config or NoopEnricherConfig()

    @property
    def stage_name(self) -> str:
        return "enricher"

    @property
    def adapter_name(self) -> str:
        return "noop"

    @property
    def adapter_id(self) -> str:
        return "noop"

    @property
    def config_schema(self) -> type[BaseModel]:
        return NoopEnricherConfig

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message="noop enricher")

    def enrich(self, chunk: Chunk, ctx: StageContext) -> str:
        return chunk.content

"""Vue Single-File Component split sub-adapter."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel

from ....ingest.chunkers.vue import VueChunker as _LegacyVue
from ....models import Chunk, SourceFile
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter


class SfcSplitConfig(BaseModel):
    pass


@register_adapter(stage="language-chunker", name="sfc-split")
class SfcSplitLanguageChunker:
    Config = SfcSplitConfig

    def __init__(self, config: SfcSplitConfig | None = None) -> None:
        self._config = config or SfcSplitConfig()
        self._impl = _LegacyVue()

    @property
    def stage_name(self) -> str: return "language-chunker"

    @property
    def adapter_name(self) -> str: return "sfc-split"

    @property
    def adapter_id(self) -> str: return "sfc-split"

    @property
    def config_schema(self) -> type[BaseModel]: return SfcSplitConfig

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message="vue SFC split")

    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        return self._impl.chunk(file, content)

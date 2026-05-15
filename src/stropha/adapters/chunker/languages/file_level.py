"""File-level fallback sub-adapter — one chunk per file.

Used as the dispatcher's ``_fallback`` entry. Per spec §3.3.4, when no
language-specific chunker is available we still index the file as a
single chunk so search can find it lexically/semantically.
"""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel

from ....ingest.chunkers.fallback import FallbackChunker as _LegacyFallback
from ....models import Chunk, SourceFile
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter


class FileLevelConfig(BaseModel):
    pass


@register_adapter(stage="language-chunker", name="file-level")
class FileLevelLanguageChunker:
    Config = FileLevelConfig

    def __init__(self, config: FileLevelConfig | None = None) -> None:
        self._config = config or FileLevelConfig()
        self._impl = _LegacyFallback()

    @property
    def stage_name(self) -> str: return "language-chunker"

    @property
    def adapter_name(self) -> str: return "file-level"

    @property
    def adapter_id(self) -> str: return "file-level"

    @property
    def config_schema(self) -> type[BaseModel]: return FileLevelConfig

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message="single-chunk fallback")

    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        return self._impl.chunk(file, content)

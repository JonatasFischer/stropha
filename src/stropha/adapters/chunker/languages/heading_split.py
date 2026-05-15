"""Markdown heading-split sub-adapter."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel

from ....ingest.chunkers.markdown import MarkdownChunker as _LegacyMarkdown
from ....models import Chunk, SourceFile
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter


class HeadingSplitConfig(BaseModel):
    """Empty for now — the legacy chunker uses fixed heading detection."""


@register_adapter(stage="language-chunker", name="heading-split")
class HeadingSplitLanguageChunker:
    Config = HeadingSplitConfig

    def __init__(self, config: HeadingSplitConfig | None = None) -> None:
        self._config = config or HeadingSplitConfig()
        self._impl = _LegacyMarkdown()

    @property
    def stage_name(self) -> str: return "language-chunker"

    @property
    def adapter_name(self) -> str: return "heading-split"

    @property
    def adapter_id(self) -> str: return "heading-split"

    @property
    def config_schema(self) -> type[BaseModel]: return HeadingSplitConfig

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message="markdown heading split")

    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        return self._impl.chunk(file, content)

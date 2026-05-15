"""Gherkin (Cucumber) feature/scenario regex split."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel

from ....ingest.chunkers.gherkin import GherkinChunker as _LegacyGherkin
from ....models import Chunk, SourceFile
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter


class RegexFeatureScenarioConfig(BaseModel):
    pass


@register_adapter(stage="language-chunker", name="regex-feature-scenario")
class RegexFeatureScenarioLanguageChunker:
    Config = RegexFeatureScenarioConfig

    def __init__(self, config: RegexFeatureScenarioConfig | None = None) -> None:
        self._config = config or RegexFeatureScenarioConfig()
        self._impl = _LegacyGherkin()

    @property
    def stage_name(self) -> str: return "language-chunker"

    @property
    def adapter_name(self) -> str: return "regex-feature-scenario"

    @property
    def adapter_id(self) -> str: return "regex-feature-scenario"

    @property
    def config_schema(self) -> type[BaseModel]: return RegexFeatureScenarioConfig

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message="gherkin regex split")

    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        return self._impl.chunk(file, content)

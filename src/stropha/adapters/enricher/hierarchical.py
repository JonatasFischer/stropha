"""Hierarchical enricher — prepend a small skeleton of the parent chunk.

Rationale: a method chunk in isolation loses the class context. Embedding
just the method body misses cues like the surrounding class name and
sibling methods. Prepending a one-line skeleton recovers most of that
signal for ~30 extra tokens.

Design constraints (per ADR-001):
- No external services, no LLM calls — must be deterministic + cheap.
- Falls back to plain content when no parent is available.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...models import Chunk
from ...pipeline.base import StageContext, StageHealth
from ...pipeline.registry import register_adapter
from ...stages.enricher import EnricherStage


class HierarchicalEnricherConfig(BaseModel):
    include_parent_skeleton: bool = Field(
        default=True,
        description="Prepend `parent.kind parent.symbol` when available.",
    )
    include_repo_url: bool = Field(
        default=False,
        description="Prepend the repo URL line. Useful when the index spans many repos.",
    )
    separator: str = Field(default="\n", description="String between prefix and content.")


@register_adapter(stage="enricher", name="hierarchical")
class HierarchicalEnricher(EnricherStage):
    Config = HierarchicalEnricherConfig

    def __init__(self, config: HierarchicalEnricherConfig | None = None) -> None:
        self._config = config or HierarchicalEnricherConfig()

    @property
    def stage_name(self) -> str:
        return "enricher"

    @property
    def adapter_name(self) -> str:
        return "hierarchical"

    @property
    def adapter_id(self) -> str:
        # adapter_id includes the flags that affect output — toggling them
        # MUST invalidate caches per ADR-004/008.
        parts = ["hierarchical"]
        if self._config.include_parent_skeleton:
            parts.append("p")
        if self._config.include_repo_url:
            parts.append("r")
        return ":".join(parts)

    @property
    def config_schema(self) -> type[BaseModel]:
        return HierarchicalEnricherConfig

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message="hierarchical enricher")

    def enrich(self, chunk: Chunk, ctx: StageContext) -> str:
        prefix_lines: list[str] = []
        if self._config.include_repo_url and ctx.repo_key:
            prefix_lines.append(f"# repo: {ctx.repo_key}")
        if self._config.include_parent_skeleton and ctx.parent_chunk is not None:
            parent: Chunk = ctx.parent_chunk
            symbol = parent.symbol or "<anon>"
            prefix_lines.append(
                f"# in {parent.kind} {symbol} ({parent.rel_path}:{parent.start_line})"
            )
        if not prefix_lines:
            return chunk.content
        return self._config.separator.join(prefix_lines) + self._config.separator + chunk.content

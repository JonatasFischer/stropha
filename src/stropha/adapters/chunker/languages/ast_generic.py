"""AST chunker sub-adapter (tree-sitter-language-pack ``process()`` based).

Covers Java, TypeScript, JavaScript, Python, Rust, Go, Kotlin via
``tree_sitter_language_pack``. The actual splitting logic lives in
``stropha.ingest.chunkers.ast_generic.AstGenericChunker``; this module
only adds the ``Stage`` contract and registry side-effect.
"""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, Field

from ....ingest.chunkers.ast_generic import AstGenericChunker as _LegacyAstGeneric
from ....models import Chunk, SourceFile
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter


class AstGenericConfig(BaseModel):
    language: str = Field(
        description=(
            "Identifier passed to tree-sitter-language-pack "
            "(e.g. 'java', 'typescript', 'python')."
        ),
    )


@register_adapter(stage="language-chunker", name="ast-generic")
class AstGenericLanguageChunker:
    """Stage adapter for ``AstGenericChunker``."""

    Config = AstGenericConfig

    def __init__(self, config: AstGenericConfig) -> None:
        self._config = config
        self._impl = _LegacyAstGeneric(config.language)

    @property
    def stage_name(self) -> str:
        return "language-chunker"

    @property
    def adapter_name(self) -> str:
        return "ast-generic"

    @property
    def adapter_id(self) -> str:
        return f"ast-generic:{self._config.language}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return AstGenericConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message=f"tree-sitter ({self._config.language})",
        )

    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        return self._impl.chunk(file, content)

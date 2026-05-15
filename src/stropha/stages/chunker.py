"""ChunkerStage protocol — split files into indexable chunks.

The default adapter (``tree-sitter-dispatch``) is a sub-pipeline: it routes
each file to a per-language sub-adapter (``ast-generic``, ``heading-split``,
``sfc-split``, …). Per ADR-006 of ``stropha-pipeline-adapters.md``, the
sub-pipeline IS an adapter — the framework treats it identically to a
single-implementation adapter.

The ``LanguageChunkerStage`` protocol below covers the sub-adapters. They
register under their own stage name (``language-chunker``) so the dispatcher
can resolve them by language.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import Chunk, SourceFile
from ..pipeline.base import StageHealth


@runtime_checkable
class ChunkerStage(Protocol):
    """Split an iterable of ``SourceFile`` into ``Chunk`` records."""

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    def chunk(
        self,
        files: Iterable[SourceFile],
        *,
        repo_key: str | None = None,
    ) -> Iterator[Chunk]:
        """Yield every chunk produced by every file."""


@runtime_checkable
class LanguageChunkerStage(Protocol):
    """Per-language chunker (sub-adapter of the dispatcher).

    The minimal contract is one method, ``chunk(file, content) -> Iterator[Chunk]``,
    plus the standard ``Stage`` introspection properties.
    """

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        """Split ONE file's content into chunks."""

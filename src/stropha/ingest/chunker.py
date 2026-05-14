"""Chunker dispatcher.

For each `SourceFile`, picks a language-specific chunker and yields `Chunk`s.
Languages without an AST chunker fall back to file-level chunking
(`FallbackChunker`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..errors import ChunkerError
from ..logging import get_logger
from ..models import Chunk, SourceFile
from .chunkers import FallbackChunker, LanguageChunker
from .chunkers.ast_generic import AstGenericChunker
from .chunkers.gherkin import GherkinChunker
from .chunkers.markdown import MarkdownChunker
from .chunkers.vue import VueChunker

log = get_logger(__name__)


def _build_registry() -> dict[str, LanguageChunker]:
    return {
        # Languages with tree-sitter-language-pack structure extraction:
        "java": AstGenericChunker("java"),
        "typescript": AstGenericChunker("typescript"),
        "javascript": AstGenericChunker("javascript"),
        "python": AstGenericChunker("python"),
        "rust": AstGenericChunker("rust"),
        "go": AstGenericChunker("go"),
        "kotlin": AstGenericChunker("kotlin"),
        # Custom chunkers:
        "vue": VueChunker(),
        "markdown": MarkdownChunker(),
        "gherkin": GherkinChunker(),
    }


class Chunker:
    """Public entry point. Replaces Phase 0's FileChunker."""

    def __init__(self) -> None:
        self._registry = _build_registry()
        self._fallback = FallbackChunker()

    def chunk(self, files: Iterable[SourceFile]) -> Iterator[Chunk]:
        for sf in files:
            try:
                content = sf.path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise ChunkerError(f"Cannot read {sf.rel_path}: {exc}") from exc
            if not content.strip():
                continue
            chunker = self._registry.get(sf.language, self._fallback)
            try:
                yield from chunker.chunk(sf, content)
            except Exception as exc:
                # Per the spec's "graceful failure" principle: log + fallback,
                # never poison the whole index because of one weird file.
                log.warning(
                    "chunker.error_fallback",
                    path=sf.rel_path,
                    language=sf.language,
                    error=str(exc),
                )
                yield from self._fallback.chunk(sf, content)

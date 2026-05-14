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
from .chunkers.base import make_chunk_id
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

    def chunk(
        self,
        files: Iterable[SourceFile],
        *,
        repo_key: str | None = None,
    ) -> Iterator[Chunk]:
        """Chunk every file. When ``repo_key`` is provided, every emitted
        chunk's ``chunk_id`` is re-derived to include the repo discriminator
        so two repos with identical files do not collide on the global
        ``chunks.chunk_id`` UNIQUE constraint.
        """
        for sf in files:
            try:
                content = sf.path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise ChunkerError(f"Cannot read {sf.rel_path}: {exc}") from exc
            if not content.strip():
                continue
            chunker = self._registry.get(sf.language, self._fallback)
            try:
                emitted = chunker.chunk(sf, content)
            except Exception as exc:
                # Per the spec's "graceful failure" principle: log + fallback,
                # never poison the whole index because of one weird file.
                log.warning(
                    "chunker.error_fallback",
                    path=sf.rel_path,
                    language=sf.language,
                    error=str(exc),
                )
                emitted = self._fallback.chunk(sf, content)
            for chunk in emitted:
                yield self._with_repo_key(chunk, repo_key)

    @staticmethod
    def _with_repo_key(chunk: Chunk, repo_key: str | None) -> Chunk:
        """Re-derive chunk_id with repo discriminator (no-op when repo_key is None)."""
        if not repo_key:
            return chunk
        new_id = make_chunk_id(
            chunk.rel_path,
            chunk.start_line,
            chunk.end_line,
            chunk.content_hash,
            repo_key=repo_key,
        )
        if new_id == chunk.chunk_id:
            return chunk
        return chunk.model_copy(update={"chunk_id": new_id})

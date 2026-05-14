"""Embedder protocol — the only contract callers should depend on."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Synchronous embedder. Async wrappers (when justified) live in subclasses.

    Per CLAUDE.md hygiene rule #2: never bypass this abstraction.
    """

    @property
    def model_name(self) -> str:
        """Identifier persisted alongside vectors (e.g. 'voyage-code-3')."""

    @property
    def dim(self) -> int:
        """Vector dimensionality. Used to size the sqlite-vec virtual table."""

    @property
    def batch_size(self) -> int:
        """Max items per call. The pipeline batches according to this."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents (chunks during indexing)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query. Providers may use a different input_type."""

"""EmbedderStage protocol — text → vector.

Same surface the pre-pipeline ``stropha.embeddings.Embedder`` exposed, plus
the ``Stage`` introspection properties (``stage_name``, ``adapter_id``, …)
so the framework can swap implementations and detect drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..pipeline.base import StageHealth


@runtime_checkable
class EmbedderStage(Protocol):
    """Synchronous embedder. Async wrappers, if ever needed, live in subclasses.

    Per CLAUDE.md hygiene rule #2 — callers MUST depend on this protocol,
    never on a concrete client (``voyageai.Client``, ``fastembed.TextEmbedding``).
    """

    # ----- Stage contract --------------------------------------------------

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    # ----- Embedder-specific surface --------------------------------------

    @property
    def model_name(self) -> str:
        """Identifier persisted alongside vectors (e.g. ``'voyage-code-3'``)."""

    @property
    def dim(self) -> int:
        """Vector dimensionality. Sizes the sqlite-vec virtual table."""

    @property
    def batch_size(self) -> int:
        """Max items per ``embed_documents`` call. Pipeline batches accordingly."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents (chunks during indexing)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query. Providers may use a distinct input_type."""

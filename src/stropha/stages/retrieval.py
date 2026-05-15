"""RetrievalStage protocol — query string → ranked SearchHits.

Phase 2 ships ``hybrid-rrf`` (the existing 3-stream + RRF fusion).
Phase 4 will add ``dense-only``, ``bm25-only``, and the sub-pipeline
form where each stream is its own sub-adapter.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import SearchHit
from ..pipeline.base import StageHealth


@runtime_checkable
class RetrievalStage(Protocol):
    """Read-time component. Constructed with refs to embedder + storage."""

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        """Top-k hits for a free-text query."""

"""``RetrievalStreamStage`` — one ranked-list source consumed by RRF.

A retrieval adapter (``hybrid-rrf``) coordinates one or more streams. Each
stream is itself a registered adapter under stage ``retrieval-stream``:

- ``vec-cosine``  — sqlite-vec ANN over the embedder
- ``fts5-bm25``   — SQLite FTS5 BM25 over the augmented document
- ``like-tokens`` — direct LIKE match on the ``symbol`` column

Splitting the streams into sub-adapters lets the user replace one (e.g.
swap ``fts5-bm25`` for SPLADE or LanceDB Reranker) without touching
``hybrid-rrf`` itself.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import SearchHit
from ..pipeline.base import StageHealth


@runtime_checkable
class RetrievalStreamStage(Protocol):
    """Returns a ranked list of ``SearchHit`` for a query."""

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    @property
    def k(self) -> int:
        """Per-stream candidate budget. Coordinator passes to the backend."""

    def search(self, query: str, query_vec: list[float] | None) -> list[SearchHit]:
        """Return ranked hits. ``query_vec`` may be None for non-dense streams."""

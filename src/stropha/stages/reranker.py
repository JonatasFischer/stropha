"""``RerankerStage`` — cross-encoder reranking of retrieval candidates.

Rerankers take the fused candidate list from RRF and re-score using a
cross-encoder model. Cross-encoders encode (query, document) pairs jointly,
providing higher quality scores at the cost of O(n) forward passes.

Typical pipeline flow:
    streams (vec-cosine, fts5-bm25, like-tokens)
        → RRF fusion (top 50-100 candidates)
        → Reranker (cross-encoder, ~200ms for 50 docs)
        → top_k final results

Available adapters:
- ``noop``         — pass-through, preserves RRF order (default)
- ``mxbai-rerank`` — mixedbread mxbai-rerank-large-v1 (ONNX local)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import SearchHit
from ..pipeline.base import StageHealth


@runtime_checkable
class RerankerStage(Protocol):
    """Re-scores candidates using a cross-encoder model."""

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Re-score hits using cross-encoder and return top_k.
        
        Args:
            query: The user query string.
            hits: Candidate hits from RRF fusion (typically 50-100).
            top_k: Number of results to return after reranking.
            
        Returns:
            Reranked hits, sorted by cross-encoder score, truncated to top_k.
        """

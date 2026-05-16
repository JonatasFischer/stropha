"""Cross-encoder reranker using fastembed's TextCrossEncoder.

Uses ONNX-based cross-encoder models for local, zero-network reranking.
Default model: BAAI/bge-reranker-base (~1GB, good quality/speed tradeoff).

Alternative models supported by fastembed:
- Xenova/ms-marco-MiniLM-L-6-v2  (80MB, fastest, lower quality)
- Xenova/ms-marco-MiniLM-L-12-v2 (120MB, fast)
- BAAI/bge-reranker-base         (1GB, recommended)
- jinaai/jina-reranker-v1-turbo-en (150MB, good for code)
- jinaai/jina-reranker-v2-base-multilingual (1.1GB, multilingual)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ....errors import ConfigError
from ....logging import get_logger
from ....models import SearchHit
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter

log = get_logger(__name__)


class CrossEncoderRerankerConfig(BaseModel):
    """Config for cross-encoder reranker."""

    model: str = Field(
        default="BAAI/bge-reranker-base",
        description="HuggingFace model id supported by fastembed TextCrossEncoder.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for reranking. Lower = less memory, higher = faster.",
    )


@register_adapter(stage="reranker", name="cross-encoder")
class CrossEncoderReranker:
    """Cross-encoder reranker using fastembed ONNX models.
    
    Reranks candidates by computing (query, document) pair scores using a
    cross-encoder model. More accurate than bi-encoder similarity but O(n)
    in the number of candidates.
    
    Typical latency: ~200ms for 50 candidates on CPU.
    """

    Config = CrossEncoderRerankerConfig

    def __init__(self, config: CrossEncoderRerankerConfig | None = None) -> None:
        self._config = config or CrossEncoderRerankerConfig()
        self._model = None  # Lazy load
        self._model_loaded = False

    def _ensure_model(self) -> None:
        """Lazy-load the cross-encoder model on first use."""
        if self._model_loaded:
            return
        
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise ConfigError(
                "fastembed package is required for cross-encoder reranking. "
                "It should already be installed as a stropha dependency."
            ) from exc

        try:
            log.info(
                "cross_encoder_reranker.loading",
                model=self._config.model,
            )
            self._model = TextCrossEncoder(model_name=self._config.model)
            self._model_loaded = True
            log.info(
                "cross_encoder_reranker.ready",
                model=self._config.model,
            )
        except Exception as exc:
            raise ConfigError(
                f"Failed to load cross-encoder model {self._config.model!r}: {exc}"
            ) from exc

    @property
    def stage_name(self) -> str:
        return "reranker"

    @property
    def adapter_name(self) -> str:
        return "cross-encoder"

    @property
    def adapter_id(self) -> str:
        return f"reranker:cross-encoder:{self._config.model}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return CrossEncoderRerankerConfig

    def health(self) -> StageHealth:
        if not self._model_loaded:
            return StageHealth(
                status="ready",
                message=f"cross-encoder reranker (lazy, model={self._config.model})",
                detail={"model": self._config.model, "loaded": False},
            )
        return StageHealth(
            status="ready",
            message=f"cross-encoder reranker (model={self._config.model})",
            detail={"model": self._config.model, "loaded": True},
        )

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Rerank hits using cross-encoder scores.
        
        Args:
            query: The user query string.
            hits: Candidate hits from RRF fusion.
            top_k: Number of results to return after reranking.
            
        Returns:
            Reranked hits sorted by cross-encoder score, truncated to top_k.
        """
        if not hits:
            return []
        
        if len(hits) <= 1:
            return hits[:top_k]

        self._ensure_model()

        # Extract document texts for reranking
        documents = [hit.snippet for hit in hits]

        # Get cross-encoder scores
        try:
            scores = list(self._model.rerank(
                query=query,
                documents=documents,
                batch_size=self._config.batch_size,
            ))
        except Exception as exc:
            log.warning(
                "cross_encoder_reranker.failed",
                error=str(exc),
                fallback="returning original order",
            )
            return hits[:top_k]

        # Sort hits by score (descending)
        scored_hits = list(zip(scores, hits))
        scored_hits.sort(key=lambda x: x[0], reverse=True)

        # Return top_k reranked hits
        return [hit for _, hit in scored_hits[:top_k]]

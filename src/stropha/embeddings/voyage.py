"""Voyage AI embedder. Active when VOYAGE_API_KEY is set.

Spec §4.1: `voyage-code-3` is SOTA for code (CodeSearchNet, CoIR). The model
emits Matryoshka vectors; we truncate to `dim` (default 512) per spec §15.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import EmbeddingError
from ..logging import get_logger
from .base import Embedder

log = get_logger(__name__)


class VoyageEmbedder(Embedder):
    """Wraps the official `voyageai` SDK."""

    # Per Voyage rate limits + payload size guidance.
    _BATCH = 128

    def __init__(self, api_key: str, model: str = "voyage-code-3", dim: int = 512) -> None:
        try:
            import voyageai  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "voyageai package is not installed. Run `uv sync`."
            ) from exc
        if not api_key:
            raise EmbeddingError("VOYAGE_API_KEY is empty.")
        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._dim = dim

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def batch_size(self) -> int:
        return self._BATCH

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        try:
            result = self._client.embed(
                list(texts),
                model=self._model,
                input_type=input_type,
                output_dimension=self._dim,
            )
        except Exception as exc:  # voyageai raises various; normalize
            raise EmbeddingError(f"Voyage embed failed: {exc}") from exc
        embeddings: list[list[float]] = list(result.embeddings)
        if embeddings and len(embeddings[0]) != self._dim:
            log.warning(
                "voyage.dim_mismatch",
                expected=self._dim,
                actual=len(embeddings[0]),
                model=self._model,
            )
        return embeddings

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]

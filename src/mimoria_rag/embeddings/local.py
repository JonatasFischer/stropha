"""Local fastembed embedder. Used when no VOYAGE_API_KEY is provided.

Spec §15 acknowledges the quality gap; this exists so Phase 0 runs zero-cost
and zero-network. Phase 4 may add a stronger open-source model (bge-m3).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import EmbeddingError
from ..logging import get_logger
from .base import Embedder

log = get_logger(__name__)


class LocalEmbedder(Embedder):
    """ONNX-based, runs on CPU. Default model: BAAI/bge-small-en-v1.5 (384 dim)."""

    _BATCH = 32

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "fastembed package is not installed. Run `uv sync`."
            ) from exc
        try:
            self._model_obj = TextEmbedding(model_name=model)
        except Exception as exc:
            raise EmbeddingError(f"fastembed failed to load {model!r}: {exc}") from exc
        self._model = model
        # fastembed exposes model metadata; infer dim from a probe to stay robust.
        probe = next(iter(self._model_obj.embed(["probe"])))
        self._dim = len(list(probe))
        log.info("local_embedder.ready", model=model, dim=self._dim)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def batch_size(self) -> int:
        return self._BATCH

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return [list(v) for v in self._model_obj.embed(list(texts))]
        except Exception as exc:
            raise EmbeddingError(f"Local embed failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return list(next(iter(self._model_obj.query_embed([text]))))
        except Exception as exc:
            raise EmbeddingError(f"Local query embed failed: {exc}") from exc

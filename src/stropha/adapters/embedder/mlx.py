"""MLX-based embedder for Apple Silicon GPU acceleration.

Uses ``mlx-embedding-models`` to run BERT/RoBERTa-based embedding models
natively on Apple Silicon GPU via MLX framework.

Supported models (from registry):
- mixedbread-large: mxbai-embed-large-v1, 1024-dim (same as local default)
- bge-m3: BAAI/bge-m3, 1024-dim, multilingual, 8K context
- bge-large: BAAI/bge-large-en-v1.5, 1024-dim
- nomic-text-v1.5: nomic-ai/nomic-embed-text-v1.5, 768-dim, 2K context
- bge-small: BAAI/bge-small-en-v1.5, 384-dim (fast)

Install dependency:
    uv add mlx-embedding-models
    # or
    pip install mlx-embedding-models

Performance: ~3-5x faster than fastembed ONNX on M1/M2/M3 Macs.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ...errors import EmbeddingError
from ...logging import get_logger
from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...stages.embedder import EmbedderStage

log = get_logger(__name__)

# Models from mlx-embedding-models registry with their dimensions
_MLX_MODEL_REGISTRY = {
    "mixedbread-large": 1024,
    "bge-m3": 1024,
    "bge-large": 1024,
    "snowflake-lg": 1024,
    "nomic-text-v1.5": 768,
    "nomic-text-v1": 768,
    "bge-base": 768,
    "bge-small": 384,
    "minilm-l12": 384,
    "minilm-l6": 384,
    "gte-tiny": 384,
    "bge-micro": 384,
}


class MlxEmbedderConfig(BaseModel):
    """Config schema for the MLX embedder adapter."""

    model: str = Field(
        default="mixedbread-large",
        description=(
            "Model name from mlx-embedding-models registry. "
            "Options: mixedbread-large, bge-m3, bge-large, nomic-text-v1.5, bge-small, etc."
        ),
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Batch size for embedding. MLX can handle larger batches efficiently.",
    )


def _is_mlx_available() -> bool:
    """Check if mlx-embedding-models is installed and MLX is available."""
    try:
        import mlx.core  # noqa: F401
        from mlx_embedding_models.embedding import EmbeddingModel  # noqa: F401
        return True
    except ImportError:
        return False


@register_adapter(stage="embedder", name="mlx")
class MlxEmbedder(EmbedderStage):
    """MLX-based embedder for Apple Silicon GPU acceleration.

    Uses the mlx-embedding-models library to run embedding models natively
    on Apple Silicon GPU. Significantly faster than ONNX-based alternatives.

    Requires Apple Silicon Mac with mlx-embedding-models installed:
        pip install mlx-embedding-models
    """

    Config = MlxEmbedderConfig

    def __init__(self, config: MlxEmbedderConfig | None = None) -> None:
        self._config = config or MlxEmbedderConfig()
        self._model_obj = None
        self._dim: int | None = None

        # Validate model name
        if self._config.model not in _MLX_MODEL_REGISTRY:
            available = ", ".join(sorted(_MLX_MODEL_REGISTRY.keys()))
            raise EmbeddingError(
                f"Unknown MLX model {self._config.model!r}. "
                f"Available: {available}"
            )

        # Lazy load the model
        self._load_model()

    def _load_model(self) -> None:
        """Load the MLX embedding model."""
        if self._model_obj is not None:
            return

        try:
            from mlx_embedding_models.embedding import EmbeddingModel
        except ImportError as exc:
            raise EmbeddingError(
                "mlx-embedding-models package is not installed. "
                "Install with: pip install mlx-embedding-models"
            ) from exc

        try:
            self._model_obj = EmbeddingModel.from_registry(self._config.model)
            self._dim = _MLX_MODEL_REGISTRY[self._config.model]
            log.info(
                "mlx_embedder.ready",
                model=self._config.model,
                dim=self._dim,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load MLX model {self._config.model!r}: {exc}"
            ) from exc

    # ----- Stage contract -------------------------------------------------

    @property
    def stage_name(self) -> str:
        return "embedder"

    @property
    def adapter_name(self) -> str:
        return "mlx"

    @property
    def adapter_id(self) -> str:
        return f"mlx:{self._config.model}:{self._dim}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return MlxEmbedderConfig

    def health(self) -> StageHealth:
        if not _is_mlx_available():
            return StageHealth(
                status="error",
                message="mlx-embedding-models not installed",
                detail={"install": "pip install mlx-embedding-models"},
            )

        if self._model_obj is None:
            return StageHealth(
                status="error",
                message="MLX model not loaded",
            )

        return StageHealth(
            status="ready",
            message=f"MLX embedder ready ({self._config.model}, dim={self._dim})",
            detail={"model": self._config.model, "dim": str(self._dim)},
        )

    # ----- Embedder surface ----------------------------------------------

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise EmbeddingError("MLX model not loaded")
        return self._dim

    @property
    def batch_size(self) -> int:
        return self._config.batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple documents using MLX GPU acceleration."""
        if self._model_obj is None:
            self._load_model()

        try:
            # mlx_embedding_models returns numpy arrays
            embeddings = self._model_obj.encode(list(texts))
            return [list(emb) for emb in embeddings]
        except Exception as exc:
            raise EmbeddingError(f"MLX embed failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query using MLX GPU acceleration."""
        if self._model_obj is None:
            self._load_model()

        try:
            embeddings = self._model_obj.encode([text])
            return list(embeddings[0])
        except Exception as exc:
            raise EmbeddingError(f"MLX query embed failed: {exc}") from exc

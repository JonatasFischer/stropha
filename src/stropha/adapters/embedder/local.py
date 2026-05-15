"""Local fastembed embedder (ONNX, no torch, no network).

Default model per ADR-008 of ``docs/architecture/stropha-graphify-integration.md``:
``mixedbread-ai/mxbai-embed-large-v1`` — top open-source English MTEB at
the ~1 GB scale, 1024-dim, stable on macOS aarch64.

Avoid ``jinaai/jina-embeddings-v2-base-code`` here — it has an ONNX-runtime
instability on aarch64 that hangs the process on the second consecutive
embed call.
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


class LocalEmbedderConfig(BaseModel):
    """Config schema for the local fastembed adapter."""

    model: str = Field(
        default="mixedbread-ai/mxbai-embed-large-v1",
        description="HuggingFace model id supported by fastembed.",
    )
    batch_size: int = Field(default=32, ge=1, le=512)


@register_adapter(stage="embedder", name="local")
class LocalEmbedder(EmbedderStage):
    """ONNX-based embedder, runs on CPU."""

    Config = LocalEmbedderConfig

    def __init__(self, config: LocalEmbedderConfig | None = None, *, model: str | None = None) -> None:
        # Allow legacy keyword form ``LocalEmbedder(model="...")`` for back-compat
        # with code that hasn't migrated to passing a Config yet.
        if config is None:
            config = LocalEmbedderConfig(
                model=model or LocalEmbedderConfig.model_fields["model"].default
            )
        elif model is not None:
            raise TypeError("Pass either config= or model=, not both")
        self._config = config

        try:
            from fastembed import TextEmbedding  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "fastembed package is not installed. Run `uv sync`."
            ) from exc
        try:
            self._model_obj = TextEmbedding(model_name=config.model)
        except Exception as exc:
            raise EmbeddingError(
                f"fastembed failed to load {config.model!r}: {exc}"
            ) from exc
        # fastembed exposes model metadata; probe to be robust.
        probe = next(iter(self._model_obj.embed(["probe"])))
        self._dim = len(list(probe))
        log.info("local_embedder.ready", model=config.model, dim=self._dim)

    # ----- Stage contract -------------------------------------------------

    @property
    def stage_name(self) -> str:
        return "embedder"

    @property
    def adapter_name(self) -> str:
        return "local"

    @property
    def adapter_id(self) -> str:
        return f"local:{self._config.model}:{self._dim}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return LocalEmbedderConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message=f"fastembed model loaded ({self._config.model}, dim={self._dim})",
            detail={"model": self._config.model, "dim": str(self._dim)},
        )

    # ----- Embedder surface ----------------------------------------------

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def batch_size(self) -> int:
        return self._config.batch_size

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

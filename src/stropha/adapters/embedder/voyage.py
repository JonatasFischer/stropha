"""Voyage AI embedder adapter.

Active when ``VOYAGE_API_KEY`` is set. ``voyage-code-3`` is SOTA for code
per ``docs/architecture/stropha-system.md`` §4.1; the model emits
Matryoshka vectors which we truncate to ``dim`` (default 512).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from pydantic import BaseModel, Field

from ...errors import EmbeddingError
from ...logging import get_logger
from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...stages.embedder import EmbedderStage

log = get_logger(__name__)


class VoyageEmbedderConfig(BaseModel):
    """Config schema for the Voyage adapter."""

    model: str = Field(default="voyage-code-3")
    dim: int = Field(default=512, ge=64, le=4096)
    batch_size: int = Field(default=128, ge=1, le=512)
    api_key_env: str = Field(
        default="VOYAGE_API_KEY",
        description="Name of the env var that holds the API key.",
    )


@register_adapter(stage="embedder", name="voyage")
class VoyageEmbedder(EmbedderStage):
    """Wraps the official ``voyageai`` SDK."""

    Config = VoyageEmbedderConfig

    def __init__(
        self,
        config: VoyageEmbedderConfig | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        dim: int | None = None,
    ) -> None:
        # Back-compat positional form: ``VoyageEmbedder(api_key=..., model=..., dim=...)``.
        if config is None:
            config = VoyageEmbedderConfig(
                model=model or VoyageEmbedderConfig.model_fields["model"].default,
                dim=dim if dim is not None else VoyageEmbedderConfig.model_fields["dim"].default,
            )
        elif api_key is not None or model is not None or dim is not None:
            raise TypeError("Pass either config= or legacy kwargs, not both")
        self._config = config

        try:
            import voyageai  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "voyageai package is not installed. Run `uv sync`."
            ) from exc
        resolved_key = api_key if api_key is not None else os.environ.get(config.api_key_env, "")
        if not resolved_key:
            raise EmbeddingError(
                f"Voyage API key is empty (env var {config.api_key_env!r} not set)."
            )
        self._client = voyageai.Client(api_key=resolved_key)

    # ----- Stage contract -------------------------------------------------

    @property
    def stage_name(self) -> str:
        return "embedder"

    @property
    def adapter_name(self) -> str:
        return "voyage"

    @property
    def adapter_id(self) -> str:
        return f"voyage:{self._config.model}:{self._config.dim}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return VoyageEmbedderConfig

    def health(self) -> StageHealth:
        # No-op probe — instantiation already validated the API key shape.
        return StageHealth(
            status="ready",
            message=f"voyageai client ready ({self._config.model}, dim={self._config.dim})",
            detail={"model": self._config.model, "dim": str(self._config.dim)},
        )

    # ----- Embedder surface ----------------------------------------------

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def dim(self) -> int:
        return self._config.dim

    @property
    def batch_size(self) -> int:
        return self._config.batch_size

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        try:
            result = self._client.embed(
                list(texts),
                model=self._config.model,
                input_type=input_type,
                output_dimension=self._config.dim,
            )
        except Exception as exc:
            raise EmbeddingError(f"Voyage embed failed: {exc}") from exc
        embeddings: list[list[float]] = list(result.embeddings)
        if embeddings and len(embeddings[0]) != self._config.dim:
            log.warning(
                "voyage.dim_mismatch",
                expected=self._config.dim,
                actual=len(embeddings[0]),
                model=self._config.model,
            )
        return embeddings

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]

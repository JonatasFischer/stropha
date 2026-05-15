"""bge-m3 embedder adapter — pre-configured local fastembed backend.

``BAAI/bge-m3`` is a multilingual, multi-functional embedder (dense +
sparse + colbert) widely cited as the top open-source choice for code
retrieval beyond English-only models like ``mxbai-embed-large-v1``.

This adapter exists as a thin pre-configured wrapper around
:class:`stropha.adapters.embedder.local.LocalEmbedder` so users can opt
in via ``--embedder bge-m3`` (or ``pipeline.embedder.adapter: bge-m3``)
without remembering to flip the model config. All behaviour is
inherited from the local fastembed backend; the only difference is the
default model + a slightly bigger default batch (bge-m3 tolerates it).

Per spec §15 / Phase 4: "Modelo de embedding self-hosted (bge-m3) como
fallback offline." This adapter delivers that surface.

Cost: same as any local embedder — ~30-60s one-shot model download on
first use, then ~ms per query.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...pipeline.registry import register_adapter
from .local import LocalEmbedder


class BgeM3Config(BaseModel):
    """bge-m3 is multilingual + multi-function; defaults tuned for code RAG."""

    model: str = Field(
        default="BAAI/bge-m3",
        description=(
            "Fastembed-supported model id. Override only if you want to test "
            "a quantised variant (e.g. `BAAI/bge-m3-onnx-int8`)."
        ),
    )
    batch_size: int = Field(
        default=24, ge=1, le=256,
        description=(
            "bge-m3 is bigger than mxbai — smaller batches keep memory "
            "bounded on machines without a discrete GPU."
        ),
    )


@register_adapter(stage="embedder", name="bge-m3")
class BgeM3Embedder(LocalEmbedder):
    """Same fastembed backend as ``LocalEmbedder``, pinned to ``BAAI/bge-m3``."""

    Config = BgeM3Config  # type: ignore[assignment]

    def __init__(self, config: BgeM3Config | None = None) -> None:
        from .local import LocalEmbedderConfig

        cfg = config or BgeM3Config()
        # Reuse LocalEmbedder's __init__ via the LocalEmbedderConfig shape.
        super().__init__(
            LocalEmbedderConfig(model=cfg.model, batch_size=cfg.batch_size)
        )

    @property
    def adapter_name(self) -> str:
        return "bge-m3"

    @property
    def adapter_id(self) -> str:
        # Keep `bge-m3:` prefix so drift detection treats a switch from
        # `local:...` to `bge-m3:...` (even with the same underlying model
        # weights) as a real adapter change.
        return f"bge-m3:{self._config.model}:{self._dim}"

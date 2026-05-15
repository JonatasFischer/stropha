"""Hybrid retrieval (dense + BM25 + symbol-token) fused via RRF.

Phase 4 refactors this into a sub-pipeline whose streams are themselves
registered adapters under stage ``retrieval-stream``. The legacy default
(3 streams: ``vec-cosine`` + ``fts5-bm25`` + ``like-tokens``) is preserved
when no ``streams`` block is supplied in YAML, so behavior is unchanged
for existing setups.

YAML example with overrides::

    retrieval:
      adapter: hybrid-rrf
      config:
        top_k: 10
        rrf_k: 60
        streams:
          dense:  { adapter: vec-cosine,  config: { k: 100 } }
          sparse: { adapter: fts5-bm25,   config: { k: 100 } }
          symbol: { adapter: like-tokens, config: { k: 30  } }

A user can also drop a stream by setting it to ``null`` (e.g. to disable
the symbol stream while keeping dense + sparse).
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from ...errors import ConfigError
from ...models import SearchHit
from ...pipeline.base import StageHealth
from ...pipeline.registry import lookup_adapter, register_adapter
from ...retrieval.rrf import DEFAULT_K, rrf_fuse
from ...stages.embedder import EmbedderStage
from ...stages.retrieval import RetrievalStage
from ...stages.retrieval_stream import RetrievalStreamStage
from ...stages.storage import StorageStage


# Legacy single-stream defaults — preserves Phase 1/2 behavior.
_DEFAULT_STREAMS: dict[str, dict[str, Any] | None] = {
    "dense":  {"adapter": "vec-cosine",  "config": {"k": 50}},
    "sparse": {"adapter": "fts5-bm25",   "config": {"k": 50}},
    "symbol": {"adapter": "like-tokens", "config": {"k": 20}},
}


class HybridRrfConfig(BaseModel):
    top_k: int = Field(default=10, ge=1, le=200)
    rrf_k: int = Field(
        default=DEFAULT_K,
        ge=1,
        le=1000,
        description="RRF smoothing constant (paper recommends 60).",
    )
    streams: dict[str, dict[str, Any] | None] = Field(
        default_factory=dict,
        description=(
            "Sub-pipeline: each named stream → {adapter, config}. "
            "Set a value to null to disable. Keys omitted inherit defaults."
        ),
    )


@register_adapter(stage="retrieval", name="hybrid-rrf")
class HybridRrfRetrieval(RetrievalStage):
    """RRF over N retrieval streams, each itself a registered adapter."""

    Config = HybridRrfConfig

    def __init__(
        self,
        config: HybridRrfConfig | None = None,
        *,
        storage: StorageStage | None = None,
        embedder: EmbedderStage | None = None,
    ) -> None:
        if config is None:
            config = HybridRrfConfig()
        if storage is None or embedder is None:
            raise ValueError(
                "HybridRrfRetrieval requires storage= and embedder= (built by "
                "the pipeline builder; do not instantiate manually)."
            )
        self._config = config
        self._storage = storage
        self._embedder = embedder
        self._streams = self._build_streams(storage)

    # ----- Stage contract -------------------------------------------------

    @property
    def stage_name(self) -> str:
        return "retrieval"

    @property
    def adapter_name(self) -> str:
        return "hybrid-rrf"

    @property
    def adapter_id(self) -> str:
        canonical = sorted(
            (name, s.adapter_id) for name, s in self._streams.items()
        )
        digest = hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()[:8]
        return f"hybrid-rrf:k={self._config.rrf_k}:streams={digest}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return HybridRrfConfig

    def health(self) -> StageHealth:
        if not self._streams:
            return StageHealth(
                status="warning",
                message="hybrid-rrf has no enabled streams",
            )
        return StageHealth(
            status="ready",
            message=f"hybrid-rrf with {len(self._streams)} streams",
            detail={name: s.adapter_id for name, s in self._streams.items()},
        )

    # ----- search ---------------------------------------------------------

    def search(self, query: str, *, top_k: int | None = None) -> list[SearchHit]:
        if not query.strip() or not self._streams:
            return []
        k = top_k or self._config.top_k
        # Single embed for the dense streams (avoids paying twice if the
        # user enables multiple dense backends in the future).
        needs_vec = any(
            s.adapter_name in ("vec-cosine",) for s in self._streams.values()
        )
        query_vec = self._embedder.embed_query(query) if needs_vec else None

        ranked: list[list[SearchHit]] = []
        for stream in self._streams.values():
            hits = stream.search(query, query_vec)
            if hits:
                ranked.append(hits)
        if not ranked:
            return []
        if len(ranked) == 1:
            return ranked[0][:k]
        return rrf_fuse(*ranked, k=self._config.rrf_k, top_k=k)

    # ----- helpers --------------------------------------------------------

    def _build_streams(
        self, storage: StorageStage
    ) -> dict[str, RetrievalStreamStage]:
        merged: dict[str, dict[str, Any] | None] = {**_DEFAULT_STREAMS, **self._config.streams}
        out: dict[str, RetrievalStreamStage] = {}
        for name, spec in merged.items():
            if spec is None:
                continue  # explicit disable
            adapter_name = spec.get("adapter")
            if not adapter_name:
                raise ConfigError(
                    f"retrieval.streams.{name}: missing `adapter` key"
                )
            cls = lookup_adapter("retrieval-stream", adapter_name)
            try:
                cfg_obj = cls.Config(**(spec.get("config") or {}))
            except Exception as exc:
                raise ConfigError(
                    f"Invalid config for retrieval.streams.{name} "
                    f"(adapter={adapter_name!r}): {exc}"
                ) from exc
            out[name] = cls(cfg_obj, storage=storage)
        return out

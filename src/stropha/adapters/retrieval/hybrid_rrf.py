"""Hybrid retrieval (dense + BM25 + symbol-token) fused via RRF.

Phase 4 refactors this into a sub-pipeline whose streams are themselves
registered adapters under stage ``retrieval-stream``. The legacy default
(3 streams: ``vec-cosine`` + ``fts5-bm25`` + ``like-tokens``) is preserved
when no ``streams`` block is supplied in YAML, so behavior is unchanged
for existing setups.

Optional reranker support: after RRF fusion, a cross-encoder can rescore
the top candidates for improved precision.

YAML example with overrides::

    retrieval:
      adapter: hybrid-rrf
      config:
        top_k: 10
        rrf_k: 60
        candidate_k: 50  # candidates passed to reranker
        streams:
          dense:  { adapter: vec-cosine,  config: { k: 100 } }
          sparse: { adapter: fts5-bm25,   config: { k: 100 } }
          symbol: { adapter: like-tokens, config: { k: 30  } }
        reranker:
          adapter: cross-encoder
          config:
            model: BAAI/bge-reranker-base

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
from ...stages.reranker import RerankerStage
from ...stages.storage import StorageStage

# Legacy single-stream defaults — preserves Phase 1/2 behavior.
_DEFAULT_STREAMS: dict[str, dict[str, Any] | None] = {
    "dense":  {"adapter": "vec-cosine",  "config": {"k": 50}},
    "sparse": {"adapter": "fts5-bm25",   "config": {"k": 50}},
    "symbol": {"adapter": "like-tokens", "config": {"k": 20}},
}


class HybridRrfConfig(BaseModel):
    top_k: int = Field(default=10, ge=1, le=200)
    candidate_k: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Number of RRF candidates to pass to reranker.",
    )
    rrf_k: int = Field(
        default=DEFAULT_K,
        ge=1,
        le=1000,
        description="RRF smoothing constant (paper recommends 60).",
    )
    hyde_enabled: bool = Field(
        default=False,
        description=(
            "Enable HyDE (Hypothetical Document Embeddings). "
            "Generates a hypothetical code snippet via LLM before dense embedding. "
            "Requires Ollama running locally. Set STROPHA_HYDE_ENABLED=1 or this flag."
        ),
    )
    query_rewrite_enabled: bool = Field(
        default=False,
        description=(
            "Enable query rewriting. LLM expands natural language into code terms. "
            "Affects ALL streams (dense, sparse, symbol). "
            "Requires Ollama. Set STROPHA_QUERY_REWRITE_ENABLED=1 or this flag."
        ),
    )
    multi_query_enabled: bool = Field(
        default=False,
        description=(
            "Enable multi-query expansion. Generates 3-5 paraphrases of the query, "
            "searches with each, and fuses results via RRF. Increases recall for "
            "conceptual queries. Requires Ollama. Set STROPHA_MULTI_QUERY_ENABLED=1."
        ),
    )
    multi_query_count: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of paraphrases to generate for multi-query expansion.",
    )
    recursive_enabled: bool = Field(
        default=False,
        description=(
            "Enable recursive retrieval / auto-merging. When multiple sibling "
            "chunks from the same parent match, promotes to the parent chunk. "
            "Also merges adjacent chunks on the same file. "
            "Requires STROPHA_RECURSIVE_RETRIEVAL=1 or this flag."
        ),
    )
    recursive_adjacency: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum line gap to consider chunks adjacent for merging.",
    )
    streams: dict[str, dict[str, Any] | None] = Field(
        default_factory=dict,
        description=(
            "Sub-pipeline: each named stream → {adapter, config}. "
            "Set a value to null to disable. Keys omitted inherit defaults."
        ),
    )
    reranker: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional reranker: {adapter, config}. "
            "If null or omitted, RRF output is returned directly."
        ),
    )


@register_adapter(stage="retrieval", name="hybrid-rrf")
class HybridRrfRetrieval(RetrievalStage):
    """RRF over N retrieval streams, with optional cross-encoder reranking."""

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
        self._reranker = self._build_reranker()

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
        reranker_id = self._reranker.adapter_id if self._reranker else "none"
        digest = hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()[:8]
        return f"hybrid-rrf:k={self._config.rrf_k}:streams={digest}:reranker={reranker_id}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return HybridRrfConfig

    def health(self) -> StageHealth:
        if not self._streams:
            return StageHealth(
                status="warning",
                message="hybrid-rrf has no enabled streams",
            )
        detail = {name: s.adapter_id for name, s in self._streams.items()}
        if self._reranker:
            detail["reranker"] = self._reranker.adapter_id
        reranker_msg = f" + reranker" if self._reranker else ""
        return StageHealth(
            status="ready",
            message=f"hybrid-rrf with {len(self._streams)} streams{reranker_msg}",
            detail=detail,
        )

    # ----- search ---------------------------------------------------------

    def search(self, query: str, *, top_k: int | None = None) -> list[SearchHit]:
        if not query.strip() or not self._streams:
            return []
        k = top_k or self._config.top_k
        
        import os
        
        # Check if multi-query expansion is enabled
        multi_query_enabled = (
            self._config.multi_query_enabled
            or os.environ.get("STROPHA_MULTI_QUERY_ENABLED", "0") == "1"
        )
        
        if multi_query_enabled:
            return self._search_multi_query(query, k)
        else:
            return self._search_single_query(query, k)

    def _search_single_query(
        self, query: str, k: int, *, search_query: str | None = None
    ) -> list[SearchHit]:
        """Execute search with a single query (the standard path)."""
        import os
        
        # Query rewriting: expand natural language to code terms (affects ALL streams)
        if search_query is None:
            search_query = query
            rewrite_enabled = (
                self._config.query_rewrite_enabled
                or os.environ.get("STROPHA_QUERY_REWRITE_ENABLED", "0") == "1"
            )
            if rewrite_enabled:
                from ...retrieval.query_rewrite import maybe_rewrite_query
                rewritten = maybe_rewrite_query(query, force_enabled=True)
                if rewritten:
                    search_query = rewritten
        
        # Single embed for the dense streams (avoids paying twice if the
        # user enables multiple dense backends in the future).
        needs_vec = any(
            s.adapter_name in ("vec-cosine",) for s in self._streams.values()
        )
        
        # HyDE: generate hypothetical document for dense embedding only
        # Enabled via config.hyde_enabled OR env var STROPHA_HYDE_ENABLED=1
        dense_query = search_query
        if needs_vec:
            hyde_enabled = (
                self._config.hyde_enabled
                or os.environ.get("STROPHA_HYDE_ENABLED", "0") == "1"
            )
            if hyde_enabled:
                from ...retrieval.hyde import maybe_hyde_rewrite
                rewritten = maybe_hyde_rewrite(query, force_enabled=True)  # Use original query for HyDE
                if rewritten:
                    dense_query = rewritten
        
        query_vec = self._embedder.embed_query(dense_query) if needs_vec else None

        ranked: list[list[SearchHit]] = []
        for stream in self._streams.values():
            # Use search_query for sparse/symbol streams (may be rewritten)
            # query_vec already uses dense_query (may be HyDE-rewritten)
            hits = stream.search(search_query, query_vec)
            if hits:
                ranked.append(hits)
        if not ranked:
            return []
        
        # RRF fusion
        if len(ranked) == 1:
            fused = ranked[0]
        else:
            # Get more candidates if we have a reranker
            fusion_k = self._config.candidate_k if self._reranker else k
            fused = rrf_fuse(*ranked, k=self._config.rrf_k, top_k=fusion_k)
        
        # Optional reranking
        if self._reranker and fused:
            fused = self._reranker.rerank(query, fused, top_k=k)
        else:
            fused = fused[:k]
        
        # Recursive retrieval / auto-merging
        return self._maybe_recursive_merge(fused)

    def _search_multi_query(self, query: str, k: int) -> list[SearchHit]:
        """Execute search with multiple paraphrases, fusing results via RRF.
        
        Multi-query expansion generates N paraphrases of the query,
        searches with each, and fuses all results via RRF for better recall.
        """
        from ...retrieval.multi_query import generate_paraphrases
        
        # Generate paraphrases (includes original query as first element)
        paraphrases = generate_paraphrases(
            query,
            force_enabled=True,
            num_paraphrases=self._config.multi_query_count,
        )
        
        if len(paraphrases) <= 1:
            # Fallback to single query if paraphrase generation failed
            return self._search_single_query(query, k)
        
        # Search with each paraphrase
        all_results: list[list[SearchHit]] = []
        for paraphrase in paraphrases:
            # Use larger candidate set since we'll fuse multiple result sets
            cand_k = max(k * 2, self._config.candidate_k)
            hits = self._search_single_query(
                query,  # Original query for HyDE/reranking context
                cand_k,
                search_query=paraphrase,  # Paraphrase for actual search
            )
            if hits:
                all_results.append(hits)
        
        if not all_results:
            return []
        
        if len(all_results) == 1:
            fused = all_results[0]
        else:
            # Fuse results from all paraphrases via RRF
            # Use more candidates since we're combining multiple searches
            fusion_k = self._config.candidate_k if self._reranker else k
            fused = rrf_fuse(*all_results, k=self._config.rrf_k, top_k=fusion_k)
        
        # Optional reranking on the fused results
        if self._reranker and fused:
            fused = self._reranker.rerank(query, fused, top_k=k)
        else:
            fused = fused[:k]
        
        # Recursive retrieval / auto-merging
        return self._maybe_recursive_merge(fused)

    def _maybe_recursive_merge(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Apply recursive retrieval / auto-merging if enabled."""
        import os
        
        recursive_enabled = (
            self._config.recursive_enabled
            or os.environ.get("STROPHA_RECURSIVE_RETRIEVAL", "0") == "1"
        )
        
        if not recursive_enabled or not hits:
            return hits
        
        from ...retrieval.recursive import merge_hits
        
        adjacency = self._config.recursive_adjacency
        env_adjacency = os.environ.get("STROPHA_RECURSIVE_ADJACENCY")
        if env_adjacency:
            try:
                adjacency = int(env_adjacency)
            except ValueError:
                pass
        
        return merge_hits(
            hits,
            self._storage,  # type: ignore[arg-type]
            adjacency_lines=adjacency,
            enabled=True,  # Already checked above
        )

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

    def _build_reranker(self) -> RerankerStage | None:
        """Build optional reranker from config."""
        spec = self._config.reranker
        if spec is None:
            return None
        
        adapter_name = spec.get("adapter")
        if not adapter_name:
            raise ConfigError("retrieval.reranker: missing `adapter` key")
        
        cls = lookup_adapter("reranker", adapter_name)
        try:
            cfg_obj = cls.Config(**(spec.get("config") or {}))
        except Exception as exc:
            raise ConfigError(
                f"Invalid config for retrieval.reranker "
                f"(adapter={adapter_name!r}): {exc}"
            ) from exc
        return cls(cfg_obj)

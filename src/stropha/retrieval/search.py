"""Hybrid retrieval (spec §6.1): dense (vec) + sparse (FTS5) fused via RRF.

Phase 1: no reranker yet — spec §16 puts Voyage `rerank-2.5` in Phase 2.
"""

from __future__ import annotations

from ..embeddings.base import Embedder
from ..logging import get_logger
from ..models import SearchHit
from ..storage import Storage
from .rrf import DEFAULT_K, rrf_fuse

log = get_logger(__name__)

# Spec §6.1 pipeline default: top 50 from each side, fuse, return top 10.
CANDIDATE_K = 50


class SearchEngine:
    """Hybrid dense+BM25 + RRF. Falls back gracefully if one side is empty."""

    def __init__(self, storage: Storage, embedder: Embedder) -> None:
        self._storage = storage
        self._embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_k: int = CANDIDATE_K,
        rrf_k: int = DEFAULT_K,
    ) -> list[SearchHit]:
        """Three-stream hybrid retrieval, fused with RRF.

        Streams:
        - dense  — sqlite-vec ANN over the embedder (semantic similarity).
        - sparse — FTS5 BM25 over content + path + symbol (lexical recall).
        - symbol — direct match on the `symbol` column for identifier tokens
                   in the query (spec §6.3.5 query routing).
        Empty streams are dropped before fusion.
        """
        if not query.strip():
            return []
        # HyDE — only the dense embedding sees the hypothetical doc; BM25
        # and the symbol-token lane keep the literal query. Falls back
        # to the raw query if disabled or the LLM call fails.
        from .hyde import maybe_hyde_rewrite

        dense_query = maybe_hyde_rewrite(query) or query
        if dense_query is not query:
            log.info("search.hyde_rewrite", original_len=len(query),
                     rewritten_len=len(dense_query))
        query_vec = self._embedder.embed_query(dense_query)
        dense_hits = self._storage.search_dense(query_vec, k=candidate_k)
        sparse_hits = self._storage.search_bm25(query, k=candidate_k)
        symbol_hits = self._storage.search_symbol_tokens(query, k=20)

        streams = [s for s in (dense_hits, sparse_hits, symbol_hits) if s]
        if not streams:
            return []
        if len(streams) == 1:
            fused = streams[0][:top_k]
        else:
            fused = rrf_fuse(*streams, k=rrf_k, top_k=top_k)
        # Recursive retrieval / auto-merge (Phase 3 §6.4). No-op when
        # STROPHA_RECURSIVE_RETRIEVAL is unset / 0 — back-compat preserved.
        from .recursive import merge_hits
        return merge_hits(fused, self._storage)

    def search_dense_only(self, query: str, top_k: int = 10) -> list[SearchHit]:
        """Escape hatch for debugging or A/B comparison."""
        query_vec = self._embedder.embed_query(query)
        return self._storage.search_dense(query_vec, k=top_k)

    def search_sparse_only(self, query: str, top_k: int = 10) -> list[SearchHit]:
        return self._storage.search_bm25(query, k=top_k)

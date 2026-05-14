"""Retrieval pipeline. Phase 1: hybrid dense+BM25 + RRF. Phase 2 will add rerank."""

from .rrf import rrf_fuse
from .search import SearchEngine

__all__ = ["SearchEngine", "rrf_fuse"]

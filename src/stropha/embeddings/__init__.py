"""Embedding providers. Single abstraction (`Embedder`), swappable backends."""

from .base import Embedder
from .factory import build_embedder

__all__ = ["Embedder", "build_embedder"]

"""Per-language chunkers. See `chunker.py` for the dispatcher."""

from .base import LanguageChunker
from .fallback import FallbackChunker

__all__ = ["FallbackChunker", "LanguageChunker"]

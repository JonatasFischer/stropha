"""Stage-specific protocols.

Each module defines the contract a specific stage's adapters must satisfy.
Concrete implementations live under ``stropha.adapters.<stage>``.

Phase 1 shipped ``embedder`` and ``enricher``. Phase 2 adds ``walker``,
``storage``, ``retrieval``. Phase 3 adds ``chunker``.
"""

from .chunker import ChunkerStage, LanguageChunkerStage
from .embedder import EmbedderStage
from .enricher import EnricherStage
from .retrieval import RetrievalStage
from .retrieval_stream import RetrievalStreamStage
from .storage import StorageStage
from .walker import WalkerStage

__all__ = [
    "ChunkerStage",
    "EmbedderStage",
    "EnricherStage",
    "LanguageChunkerStage",
    "RetrievalStage",
    "RetrievalStreamStage",
    "StorageStage",
    "WalkerStage",
]

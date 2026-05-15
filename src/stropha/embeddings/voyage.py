"""Back-compat shim — re-exports the adapter implementation.

The canonical class lives at ``stropha.adapters.embedder.voyage``. This
module is kept so existing imports (``from stropha.embeddings.voyage
import VoyageEmbedder``) continue to resolve unchanged.
"""

from __future__ import annotations

from ..adapters.embedder.voyage import VoyageEmbedder, VoyageEmbedderConfig

__all__ = ["VoyageEmbedder", "VoyageEmbedderConfig"]

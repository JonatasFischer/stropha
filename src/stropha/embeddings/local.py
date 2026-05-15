"""Back-compat shim — re-exports the adapter implementation.

The canonical class lives at ``stropha.adapters.embedder.local``. This
module is kept so existing imports (``from stropha.embeddings.local
import LocalEmbedder``) continue to resolve unchanged.
"""

from __future__ import annotations

from ..adapters.embedder.local import LocalEmbedder, LocalEmbedderConfig

__all__ = ["LocalEmbedder", "LocalEmbedderConfig"]

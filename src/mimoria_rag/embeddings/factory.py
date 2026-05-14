"""Builds the right Embedder based on Config."""

from __future__ import annotations

from ..config import Config
from ..logging import get_logger
from .base import Embedder

log = get_logger(__name__)


def build_embedder(config: Config) -> Embedder:
    """Return Voyage when key is set, else local fastembed."""
    if config.use_voyage:
        from .voyage import VoyageEmbedder

        log.info(
            "embedder.selected",
            provider="voyage",
            model=config.voyage_embed_model,
            dim=config.voyage_embed_dim,
        )
        return VoyageEmbedder(
            api_key=config.voyage_api_key or "",
            model=config.voyage_embed_model,
            dim=config.voyage_embed_dim,
        )
    from .local import LocalEmbedder

    log.info(
        "embedder.selected",
        provider="local",
        model=config.local_embed_model,
        reason="no_voyage_api_key",
    )
    return LocalEmbedder(model=config.local_embed_model)

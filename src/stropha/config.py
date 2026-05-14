"""Configuration loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_index_path() -> Path:
    return Path.home() / ".stropha" / "index.db"


def _default_target_repo() -> Path:
    """Default target = current working directory."""
    return Path.cwd()


class Config(BaseSettings):
    """Runtime configuration. All product env vars are prefixed `STROPHA_`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    # API keys (provider names, no STROPHA_ prefix).
    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Paths.
    target_repo: Path = Field(default_factory=_default_target_repo, alias="STROPHA_TARGET_REPO")
    index_path: Path = Field(default_factory=_default_index_path, alias="STROPHA_INDEX_PATH")

    # Models.
    voyage_embed_model: str = Field(
        default="voyage-code-3", alias="STROPHA_VOYAGE_EMBED_MODEL"
    )
    voyage_embed_dim: int = Field(default=512, alias="STROPHA_VOYAGE_EMBED_DIM")
    # Default per ADR-008 (docs/architecture/stropha-graphify-integration.md):
    # mxbai-embed-large-v1 — top open-source English MTEB at this scale,
    # 1024-dim, ~0.64 GB. Chosen over jina-embeddings-v2-base-code because
    # the latter has an ONNX-runtime instability on macOS aarch64 that
    # causes hang/crash on the second consecutive embed call.
    local_embed_model: str = Field(
        default="mixedbread-ai/mxbai-embed-large-v1", alias="STROPHA_LOCAL_EMBED_MODEL"
    )

    # Behavior.
    log_level: str = Field(default="INFO", alias="STROPHA_LOG_LEVEL")
    max_file_bytes: int = Field(default=524_288, alias="STROPHA_MAX_FILE_BYTES")

    @property
    def use_voyage(self) -> bool:
        return bool(self.voyage_api_key and self.voyage_api_key.strip())

    def resolve_index_path(self) -> Path:
        """Expand ~ and env vars in the configured index path."""
        return Path(os.path.expandvars(str(self.index_path))).expanduser()

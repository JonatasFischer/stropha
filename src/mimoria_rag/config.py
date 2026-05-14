"""Configuration loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_index_path() -> Path:
    return Path.home() / ".mimoria-rag" / "index.db"


def _default_target_repo() -> Path:
    """Default target = sibling 'mimoria' if it exists, else cwd."""
    sibling = Path.cwd().parent / "mimoria"
    return sibling if sibling.exists() else Path.cwd()


class Config(BaseSettings):
    """Runtime configuration. All env vars are prefixed `RAG_` except API keys."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    # API keys (no RAG_ prefix; standard provider names).
    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Paths.
    target_repo: Path = Field(default_factory=_default_target_repo, alias="RAG_TARGET_REPO")
    index_path: Path = Field(default_factory=_default_index_path, alias="RAG_INDEX_PATH")

    # Models.
    voyage_embed_model: str = Field(
        default="voyage-code-3", alias="RAG_VOYAGE_EMBED_MODEL"
    )
    voyage_embed_dim: int = Field(default=512, alias="RAG_VOYAGE_EMBED_DIM")
    # Default per ADR-008 (docs/architecture/rag-graphify-integration.md):
    # jina-embeddings-v2-base-code is code-specialized (30 PLs incl. Java/TS/JS/Py/Go/Rust/Kotlin),
    # 768-dim, ~0.64 GB — single biggest single-hop quality win for local-only setups.
    local_embed_model: str = Field(
        default="jinaai/jina-embeddings-v2-base-code", alias="RAG_LOCAL_EMBED_MODEL"
    )

    # Behavior.
    log_level: str = Field(default="INFO", alias="RAG_LOG_LEVEL")
    max_file_bytes: int = Field(default=524_288, alias="RAG_MAX_FILE_BYTES")

    @property
    def use_voyage(self) -> bool:
        return bool(self.voyage_api_key and self.voyage_api_key.strip())

    def resolve_index_path(self) -> Path:
        """Expand ~ and env vars in the configured index path."""
        return Path(os.path.expandvars(str(self.index_path))).expanduser()

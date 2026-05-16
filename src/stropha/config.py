"""Configuration loaded from environment / .env via pydantic-settings.

This module provides a centralized configuration singleton that all parts of
the system should use. The configuration is loaded once and cached, ensuring
consistent behavior across CLI, MCP server, and pipeline components.

Usage:
    from stropha.config import get_config
    cfg = get_config()  # Returns the singleton instance

    # For debugging / MCP tool:
    from stropha.config import get_config_info
    info = get_config_info()  # Returns dict with config + source info
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_index_path() -> Path:
    """Default index path: .stropha/index.db relative to cwd, or ~/.stropha/index.db."""
    local = Path.cwd() / ".stropha" / "index.db"
    if local.parent.exists():
        return local
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

    # MCP server watch integration (auto-reindex on file changes).
    mcp_watch: bool = Field(default=True, alias="STROPHA_MCP_WATCH")
    mcp_watch_interval: float = Field(default=1.0, alias="STROPHA_MCP_WATCH_INTERVAL")
    mcp_watch_debounce: float = Field(default=2.0, alias="STROPHA_MCP_WATCH_DEBOUNCE")

    @property
    def use_voyage(self) -> bool:
        return bool(self.voyage_api_key and self.voyage_api_key.strip())

    def resolve_index_path(self) -> Path:
        """Expand ~ and env vars in the configured index path."""
        return Path(os.path.expandvars(str(self.index_path))).expanduser()


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_config_instance: Config | None = None
_config_env_snapshot: dict[str, str | None] = {}


def get_config(*, reload: bool = False) -> Config:
    """Get the singleton Config instance.
    
    The config is loaded once from environment variables (with optional .env
    fallback). Subsequent calls return the cached instance unless reload=True.
    
    Environment variables take precedence over .env file values. The .env file
    is only consulted if STROPHA_INDEX_PATH is not already set in the environment
    (this prevents the stropha project's .env from overriding MCP client config).
    
    Args:
        reload: Force reload from environment. Use sparingly — typically only
            in tests or after programmatically changing os.environ.
    
    Returns:
        The singleton Config instance.
    """
    global _config_instance, _config_env_snapshot
    
    if _config_instance is None or reload:
        # Snapshot the env vars BEFORE loading .env so we can report sources
        _config_env_snapshot = {
            "STROPHA_INDEX_PATH": os.environ.get("STROPHA_INDEX_PATH"),
            "STROPHA_TARGET_REPO": os.environ.get("STROPHA_TARGET_REPO"),
            "STROPHA_LOG_LEVEL": os.environ.get("STROPHA_LOG_LEVEL"),
            "STROPHA_LOCAL_EMBED_MODEL": os.environ.get("STROPHA_LOCAL_EMBED_MODEL"),
            "VOYAGE_API_KEY": "(set)" if os.environ.get("VOYAGE_API_KEY") else None,
        }
        
        # Only load .env if critical env vars are not already set.
        # This ensures MCP clients (opencode) can override via env without
        # the stropha project's .env clobbering the values.
        if not os.environ.get("STROPHA_INDEX_PATH"):
            from dotenv import load_dotenv
            load_dotenv(override=False)
        
        _config_instance = Config()  # type: ignore[call-arg]
    
    return _config_instance


def get_config_info() -> dict[str, Any]:
    """Get configuration details for debugging.
    
    Returns a dict with:
    - All resolved config values
    - Source information (which env vars were set vs defaulted)
    - Current working directory
    
    This is used by the MCP `get_config` tool and can be used for CLI debugging.
    """
    cfg = get_config()
    
    def _source(key: str, value: Any) -> str:
        """Determine the source of a config value."""
        env_val = _config_env_snapshot.get(key)
        if env_val is not None:
            return "env"
        # Check if .env might have provided it
        current_env = os.environ.get(key)
        if current_env is not None:
            return "dotenv"
        return "default"
    
    return {
        "config": {
            "index_path": str(cfg.resolve_index_path()),
            "target_repo": str(cfg.target_repo),
            "log_level": cfg.log_level,
            "max_file_bytes": cfg.max_file_bytes,
            "use_voyage": cfg.use_voyage,
            "local_embed_model": cfg.local_embed_model,
            "voyage_embed_model": cfg.voyage_embed_model,
            "voyage_embed_dim": cfg.voyage_embed_dim,
        },
        "sources": {
            "STROPHA_INDEX_PATH": _source("STROPHA_INDEX_PATH", cfg.index_path),
            "STROPHA_TARGET_REPO": _source("STROPHA_TARGET_REPO", cfg.target_repo),
            "STROPHA_LOG_LEVEL": _source("STROPHA_LOG_LEVEL", cfg.log_level),
            "STROPHA_LOCAL_EMBED_MODEL": _source("STROPHA_LOCAL_EMBED_MODEL", cfg.local_embed_model),
        },
        "env_values": {
            k: v if v != "(set)" else "(redacted)"
            for k, v in _config_env_snapshot.items()
        },
        "runtime": {
            "cwd": str(Path.cwd()),
            "index_exists": cfg.resolve_index_path().exists(),
            "target_exists": cfg.target_repo.exists(),
        },
    }


def reset_config() -> None:
    """Reset the singleton (for tests). Forces reload on next get_config()."""
    global _config_instance, _config_env_snapshot
    _config_instance = None
    _config_env_snapshot = {}

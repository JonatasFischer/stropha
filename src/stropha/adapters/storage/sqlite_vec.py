"""Storage adapter for sqlite-vec.

Subclasses the existing ``stropha.storage.sqlite.Storage`` so the entire
read/write API surface is inherited unchanged. Adds:

- ``Config`` (path) — user-facing knobs in YAML/env
- ``Stage`` introspection properties (``stage_name``, ``adapter_id``, …)
- ``health()`` probe — confirms the SQLite file is openable

The file path expansion mirrors ``stropha.config.Config.resolve_index_path``:
``~`` and ``$VARS`` are expanded.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...storage.sqlite import Storage


class SqliteVecStorageConfig(BaseModel):
    path: str = Field(
        default="~/.stropha/index.db",
        description="Path to the SQLite DB. ~ and $VARS are expanded.",
    )


@register_adapter(stage="storage", name="sqlite-vec")
class SqliteVecStorage(Storage):
    """Storage stage backed by SQLite + sqlite-vec + FTS5."""

    Config = SqliteVecStorageConfig

    def __init__(
        self,
        config: SqliteVecStorageConfig | None = None,
        *,
        embedding_dim: int | None = None,
    ) -> None:
        if config is None:
            config = SqliteVecStorageConfig()
        if embedding_dim is None:
            raise ValueError(
                "SqliteVecStorage requires `embedding_dim=` (propagate from "
                "the active embedder)."
            )
        self._adapter_config = config
        resolved = Path(os.path.expandvars(config.path)).expanduser()
        super().__init__(resolved, embedding_dim=embedding_dim)

    # ----- Stage contract -------------------------------------------------

    @property
    def stage_name(self) -> str:
        return "storage"

    @property
    def adapter_name(self) -> str:
        return "sqlite-vec"

    @property
    def adapter_id(self) -> str:
        return f"sqlite-vec:dim={self._dim}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return SqliteVecStorageConfig

    def health(self) -> StageHealth:
        # The connection is open by the time we're here (super().__init__
        # raises on failure). Report DB size + chunk count as detail.
        try:
            stats = self.stats()
            return StageHealth(
                status="ready",
                message=f"sqlite-vec ready ({stats['chunks']} chunks)",
                detail={
                    "db_path": stats["db_path"],
                    "chunks": str(stats["chunks"]),
                    "dim": str(stats["index_dim"]),
                },
            )
        except Exception as exc:  # pragma: no cover  defensive
            return StageHealth(
                status="error",
                message=f"sqlite-vec probe failed: {exc}",
            )

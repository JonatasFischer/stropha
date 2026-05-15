"""WalkerStage protocol — discover indexable files in a repository.

Phase 2 introduces a single adapter (``git-ls-files``) that wraps the
existing ``stropha.ingest.walker.Walker``. Future adapters (Phase 2/3):
``filesystem`` (no git required), ``nested-git`` (per-file toplevel
detection for monorepos with submodules).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import SourceFile
from ..pipeline.base import StageHealth


@runtime_checkable
class WalkerStage(Protocol):
    """Discover ``SourceFile`` records under a repo root."""

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    def discover(self, repo: Path) -> Iterable[SourceFile]:
        """Yield every file the indexer should consider for chunking."""

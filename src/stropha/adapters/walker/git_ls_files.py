"""Walker adapter that delegates to the existing ``Walker`` class.

The legacy class already does git-ls-files-with-filesystem-fallback. This
adapter is a thin Stage wrapper around it; the heavy lifting code stays
in ``stropha.ingest.walker`` to avoid churning a well-tested module.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from ...ingest.walker import Walker
from ...models import SourceFile
from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...stages.walker import WalkerStage


class GitLsFilesWalkerConfig(BaseModel):
    max_file_bytes: int = Field(
        default=524_288,
        ge=1024,
        description="Skip files larger than this. Default 512 KiB.",
    )
    respect_strophaignore: bool = Field(
        default=True,
        description="Honor `.strophaignore` patterns alongside `.gitignore`.",
    )


@register_adapter(stage="walker", name="git-ls-files")
class GitLsFilesWalker(WalkerStage):
    """Stage adapter wrapping ``stropha.ingest.walker.Walker``."""

    Config = GitLsFilesWalkerConfig

    def __init__(self, config: GitLsFilesWalkerConfig | None = None) -> None:
        self._config = config or GitLsFilesWalkerConfig()

    @property
    def stage_name(self) -> str:
        return "walker"

    @property
    def adapter_name(self) -> str:
        return "git-ls-files"

    @property
    def adapter_id(self) -> str:
        return f"git-ls-files:max={self._config.max_file_bytes}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return GitLsFilesWalkerConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message="git-ls-files walker (filesystem fallback)",
        )

    def discover(self, repo: Path) -> Iterable[SourceFile]:
        return Walker(repo, max_file_bytes=self._config.max_file_bytes).discover()

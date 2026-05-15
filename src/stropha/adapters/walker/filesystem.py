"""Filesystem walker — works for non-git directories.

Use when the target is a plain folder (no `.git/`) — typically downloaded
docs, vendored deps, or a snapshot. Skips a fixed list of cache/build
directories (``.venv``, ``node_modules``, ``__pycache__``, …) so the
output is reasonable without requiring a `.gitignore`.
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

# Directory names that are almost always noise. Added on top of any
# `.strophaignore` / `.gitignore` rules the legacy Walker honours.
_DEFAULT_SKIP_DIRS = (
    ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "dist", "build", ".tox", ".cache",
    "target", ".gradle", ".idea", ".vscode",
)


class FilesystemWalkerConfig(BaseModel):
    max_file_bytes: int = Field(default=524_288, ge=1024)
    skip_dirs: tuple[str, ...] = Field(
        default=_DEFAULT_SKIP_DIRS,
        description="Directory names skipped during traversal.",
    )


@register_adapter(stage="walker", name="filesystem")
class FilesystemWalker(WalkerStage):
    """Pure filesystem traversal — does NOT require a git repo."""

    Config = FilesystemWalkerConfig

    def __init__(self, config: FilesystemWalkerConfig | None = None) -> None:
        self._config = config or FilesystemWalkerConfig()

    @property
    def stage_name(self) -> str:
        return "walker"

    @property
    def adapter_name(self) -> str:
        return "filesystem"

    @property
    def adapter_id(self) -> str:
        return f"filesystem:max={self._config.max_file_bytes}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return FilesystemWalkerConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message=(
                "filesystem walker — works for non-git dirs "
                f"(skipping {len(self._config.skip_dirs)} default cache dirs)"
            ),
        )

    def discover(self, repo: Path) -> Iterable[SourceFile]:
        # Reuse the legacy Walker's filtering logic but force the
        # filesystem path. We do this by NOT having a `.git/` and letting
        # the walker fall back to its filesystem branch.
        # If `.git/` exists we still bypass git-ls-files by passing a custom
        # prune set via subclass-style override — the legacy class doesn't
        # expose that hook directly, so we monkey-walk here.
        skip = set(self._config.skip_dirs)
        for path in repo.rglob("*"):
            if not path.is_file():
                continue
            # Skip if any ancestor matches a configured skip dir
            if any(part in skip for part in path.relative_to(repo).parts):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > self._config.max_file_bytes:
                continue
            # Reuse the legacy Walker for the "is this a real source file?"
            # decisions (binary detection, language detection, .strophaignore).
            walker = Walker(repo, max_file_bytes=self._config.max_file_bytes)
            rel = str(path.relative_to(repo))
            yield from walker.discover_paths([rel])

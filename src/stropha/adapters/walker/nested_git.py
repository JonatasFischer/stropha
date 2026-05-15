"""Nested-git walker — discovers nested ``.git/`` directories under a root.

Use for monorepos that vendor multiple git submodules / vendored deps,
or for an `~/sources/` umbrella directory containing many independent
repos. Each nested git repo is processed by the standard ``git-ls-files``
walker, so per-repo `.gitignore` is honoured automatically.

Output ``SourceFile.rel_path`` is *relative to the umbrella root* so the
caller still sees a single coherent file tree. The actual repo identity
is preserved in the chunk's ``repo_id`` foreign key (the pipeline calls
``git_meta.detect`` on each nested repo independently).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from ...ingest.walker import Walker
from ...logging import get_logger
from ...models import SourceFile
from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...stages.walker import WalkerStage

log = get_logger(__name__)


class NestedGitWalkerConfig(BaseModel):
    max_file_bytes: int = Field(default=524_288, ge=1024)
    max_depth: int = Field(
        default=4, ge=1, le=10,
        description="Maximum directory depth to scan for nested .git/ folders.",
    )
    follow_root_first: bool = Field(
        default=True,
        description=(
            "When the umbrella root is itself a git repo, walk it before "
            "descending into nested submodules."
        ),
    )


@register_adapter(stage="walker", name="nested-git")
class NestedGitWalker(WalkerStage):
    """Discover every nested ``.git/`` and walk each as its own repo."""

    Config = NestedGitWalkerConfig

    def __init__(self, config: NestedGitWalkerConfig | None = None) -> None:
        self._config = config or NestedGitWalkerConfig()

    @property
    def stage_name(self) -> str:
        return "walker"

    @property
    def adapter_name(self) -> str:
        return "nested-git"

    @property
    def adapter_id(self) -> str:
        return f"nested-git:depth={self._config.max_depth}:max={self._config.max_file_bytes}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return NestedGitWalkerConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message=f"nested-git walker (max depth={self._config.max_depth})",
        )

    def discover(self, repo: Path) -> Iterable[SourceFile]:
        seen: set[Path] = set()
        nested = list(self._find_nested_repos(repo, max_depth=self._config.max_depth))
        log.info(
            "nested_git_walker.discovered",
            root=str(repo),
            nested_count=len(nested),
            paths=[str(p.relative_to(repo)) for p in nested[:10]],
        )

        # Optionally walk the umbrella root first
        if self._config.follow_root_first and (repo / ".git").is_dir():
            yield from self._walk_one(repo, root=repo, seen=seen)

        for sub in nested:
            yield from self._walk_one(sub, root=repo, seen=seen)

    # ------------------------------------------------------------------ helpers

    def _find_nested_repos(self, root: Path, *, max_depth: int) -> Iterable[Path]:
        """BFS for ``<dir>/.git`` directories under ``root``.

        Skips the root's own ``.git`` (handled by ``follow_root_first``).
        """
        from collections import deque

        queue: deque[tuple[Path, int]] = deque([(root, 0)])
        while queue:
            d, depth = queue.popleft()
            if depth >= max_depth:
                continue
            try:
                children = list(d.iterdir())
            except (OSError, PermissionError):
                continue
            for child in children:
                if not child.is_dir():
                    continue
                if child.name == ".git":
                    if d != root:
                        yield d
                    # Either way, don't descend into .git
                    continue
                # Skip massive caches
                if child.name in (".venv", "node_modules", "target", "build", "dist"):
                    continue
                queue.append((child, depth + 1))

    def _walk_one(
        self, sub_repo: Path, *, root: Path, seen: set[Path],
    ) -> Iterable[SourceFile]:
        """Walk one nested repo, rebasing rel_path against the umbrella root."""
        try:
            walker = Walker(sub_repo, max_file_bytes=self._config.max_file_bytes)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("nested_git_walker.walker_init_failed",
                        sub=str(sub_repo), error=str(exc))
            return
        for sf in walker.discover():
            # sf.rel_path is relative to sub_repo; rebase to umbrella root.
            abs_path = (sub_repo / sf.rel_path).resolve()
            if abs_path in seen:
                continue
            seen.add(abs_path)
            try:
                rebased = str(abs_path.relative_to(root.resolve()))
            except ValueError:
                rebased = sf.rel_path  # paths outside root: keep original
            # Reconstruct SourceFile with the rebased path.
            yield sf.model_copy(update={"rel_path": rebased, "path": abs_path})

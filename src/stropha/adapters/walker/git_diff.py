"""Git-diff aware walker — Phase B incremental ingestion.

Walks only the files that changed between two git SHAs. Output is a
stream of :class:`stropha.models.FileDelta` records — one per add /
modify / delete / rename. The pipeline uses these to skip the
chunker + embedder for unchanged files, evict chunks of deleted files,
and rename chunks atomically (zero re-embed when content is unchanged).

This adapter does NOT implement the legacy ``discover(repo) -> [SourceFile]``
surface because the contract is fundamentally different (delta vs full
list). Callers that resolve a walker for a non-incremental pass must
pick ``git-ls-files`` or ``filesystem`` instead.

Per RFC §8 (stropha-system.md): this is the foundation for the "soft
index" + "post-commit hook in background" stories. The hook installer
v=4 invokes ``stropha index --incremental`` which resolves the SHA
delta against ``meta['last_indexed_sha_<repo_id>']`` automatically.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from ...errors import WalkerError
from ...ingest.walker import _LANGUAGE_BY_EXT
from ...logging import get_logger
from ...models import FileDelta
from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter

log = get_logger(__name__)

_GIT_TIMEOUT_S = 30


def _default_rename_threshold() -> int:
    """`STROPHA_RENAME_THRESHOLD` overrides; default 80%.

    Lower = more aggressive rename detection (more matches, more false
    positives). Higher = stricter (more renames missed, treated as
    add+delete). 80 is conservative and tracks git's own heuristics
    well for typical refactors.
    """
    try:
        val = int(os.environ.get("STROPHA_RENAME_THRESHOLD", "80"))
    except ValueError:
        return 80
    return max(0, min(100, val))


class GitDiffWalkerConfig(BaseModel):
    rename_threshold: int = Field(
        default_factory=_default_rename_threshold,
        ge=0, le=100,
        description=(
            "Minimum similarity (0-100) for git to call a path pair a "
            "rename instead of (delete, add). Lower is more aggressive."
        ),
    )


@register_adapter(stage="walker", name="git-diff")
class GitDiffWalker:
    """Stage walker that produces ``FileDelta`` records for an SHA range.

    Note: this adapter does NOT satisfy the same protocol as
    ``WalkerStage.discover`` — incremental walks have a different shape.
    Callers route through :meth:`discover_deltas` directly.
    """

    Config = GitDiffWalkerConfig

    def __init__(self, config: GitDiffWalkerConfig | None = None) -> None:
        self._config = config or GitDiffWalkerConfig()

    @property
    def stage_name(self) -> str:
        return "walker"

    @property
    def adapter_name(self) -> str:
        return "git-diff"

    @property
    def adapter_id(self) -> str:
        return f"git-diff:rename>={self._config.rename_threshold}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return GitDiffWalkerConfig

    def health(self) -> StageHealth:
        # Light probe: ensure git is on PATH.
        try:
            r = subprocess.run(
                ["git", "--version"],
                capture_output=True, text=True, check=False, timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            return StageHealth(
                status="error",
                message=f"git not callable: {exc}",
            )
        if r.returncode != 0:
            return StageHealth(
                status="error",
                message=f"git --version exited {r.returncode}",
            )
        return StageHealth(
            status="ready",
            message=f"git-diff (rename threshold {self._config.rename_threshold})",
        )

    # ---- discovery surface ------------------------------------------------

    def discover(self, repo: Path) -> Iterable:  # noqa: ARG002
        """Not supported — use :meth:`discover_deltas` with a since SHA."""
        raise WalkerError(
            "GitDiffWalker.discover() is delta-only; "
            "call discover_deltas(repo, since_sha=...) instead."
        )

    def discover_deltas(
        self, repo: Path, *, since_sha: str, until_sha: str = "HEAD",
    ) -> list[FileDelta]:
        """Return all file deltas between ``since_sha`` and ``until_sha``.

        Uses ``git diff --name-status --find-renames=<threshold>`` and
        filters out paths whose extension is not in the supported
        language map (binary files, lockfiles, etc.). Submodule updates
        appear as bare directory entries — we drop those.

        Raises :class:`WalkerError` when git fails (e.g. bad SHA, repo
        not initialised). Callers can catch and fall back to a full walk.
        """
        repo = repo.resolve()
        cmd = [
            "git", "diff",
            f"--find-renames={self._config.rename_threshold}",
            "--name-status",
            "-z",  # null-terminated so paths with spaces / quotes are safe
            f"{since_sha}..{until_sha}",
        ]
        try:
            result = subprocess.run(
                cmd, cwd=repo, capture_output=True, check=False,
                timeout=_GIT_TIMEOUT_S,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            raise WalkerError(f"git diff failed: {exc}") from exc
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace").strip()
            raise WalkerError(
                f"git diff exited {result.returncode}: {err or '(no stderr)'}"
            )

        return list(self._parse_name_status(result.stdout))

    # ---- parsing ----------------------------------------------------------

    def _parse_name_status(self, stdout: bytes) -> Iterable[FileDelta]:
        """Parse ``git diff --name-status -z`` output.

        Format (NUL-separated):
            STATUS\\0path\\0           for A/M/D
            R<score>\\0old\\0new\\0    for renames
            C<score>\\0old\\0new\\0    for copies (treated as add)

        We iterate tokens with an index rather than a generator to handle
        the variable-length rename triples cleanly.
        """
        text = stdout.decode("utf-8", errors="replace")
        tokens = text.split("\x00")
        # Drop the trailing empty token from the final NUL
        if tokens and tokens[-1] == "":
            tokens.pop()

        i = 0
        while i < len(tokens):
            status = tokens[i]
            i += 1
            if not status:
                continue
            code = status[0]
            if code in ("R", "C") and i + 1 < len(tokens):
                old, new = tokens[i], tokens[i + 1]
                i += 2
                if not self._is_supported(new):
                    continue
                try:
                    similarity = int(status[1:]) if len(status) > 1 else None
                except ValueError:
                    similarity = None
                if code == "R":
                    yield FileDelta(
                        action="rename",
                        rel_path=new,
                        old_rel_path=old,
                        similarity=similarity,
                    )
                else:  # C — copy treated as add
                    yield FileDelta(
                        action="add",
                        rel_path=new,
                        similarity=similarity,
                    )
            elif code in ("A", "M", "D", "T") and i < len(tokens):
                path = tokens[i]
                i += 1
                if not self._is_supported(path):
                    continue
                action = (
                    "delete" if code == "D"
                    else ("add" if code in ("A", "T") else "modify")
                )
                yield FileDelta(action=action, rel_path=path)
            else:
                # Unknown status code — skip the matching path token
                # defensively to avoid a parse desync.
                if i < len(tokens):
                    i += 1

    def _is_supported(self, rel_path: str) -> bool:
        """Filter to file types stropha can chunk.

        Submodule updates show as directory paths (no extension or end
        with ``/``); we drop those. Generated files (``.lock``, etc.)
        slip through here but are caught downstream by the
        ``.strophaignore`` check in the actual chunker invocation.
        """
        if not rel_path or rel_path.endswith("/"):
            return False
        # Allow markdown / docs explicitly even when not in the AST table.
        suffix = Path(rel_path).suffix.lower()
        # Walker's language table covers the languages we tree-sitter chunk;
        # accept anything in it. For renames/deletes we accept ANY path so
        # the pipeline can evict chunks that exist for files outside this
        # set (the walker that originally indexed them may have been a
        # different adapter). Add/modify still filter to supported types.
        return suffix in _LANGUAGE_BY_EXT or suffix in {".md", ".rst", ".txt"}

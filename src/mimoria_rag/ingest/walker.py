"""Source file discovery.

Per spec §3.1:
- Use `git ls-files` when target is a git repo (respects .gitignore for free).
- Apply additional `.ragignore` patterns.
- Skip binaries (null-byte heuristic) and files over max_file_bytes.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import pathspec

from ..errors import WalkerError
from ..logging import get_logger
from ..models import SourceFile

log = get_logger(__name__)

# Mapping of file extensions to language tags used downstream. Phase 0 indexes
# everything textual; Phase 1 will gate per-language tree-sitter grammars off
# this same table.
_LANGUAGE_BY_EXT: dict[str, str] = {
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".groovy": "groovy",
    ".vue": "vue",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".py": "python",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".jsonc": "json",
    ".md": "markdown",
    ".mdx": "markdown",
    ".feature": "gherkin",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".properties": "properties",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".xml": "xml",
}

# Filenames (no extension) we still want to index.
_FILENAME_LANGUAGE: dict[str, str] = {
    "Dockerfile": "dockerfile",
    "Makefile": "make",
    "Justfile": "just",
    "Procfile": "procfile",
}


def detect_language(path: Path) -> str | None:
    """Return language tag for a path, or None if unsupported."""
    if path.name in _FILENAME_LANGUAGE:
        return _FILENAME_LANGUAGE[path.name]
    return _LANGUAGE_BY_EXT.get(path.suffix.lower())


def _is_binary(path: Path, sniff_bytes: int = 8192) -> bool:
    """Null-byte heuristic — fast, no libmagic dependency."""
    try:
        with path.open("rb") as f:
            chunk = f.read(sniff_bytes)
    except OSError:
        return True
    return b"\x00" in chunk


def _git_ls_files(repo: Path) -> list[str] | None:
    """Return tracked files via git, or None if not a git repo / git missing."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=repo,
            capture_output=True,
            check=False,
            text=False,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return [p for p in result.stdout.decode("utf-8", errors="replace").split("\0") if p]


def _filesystem_walk(repo: Path) -> Iterator[str]:
    """Fallback walker for non-git directories."""
    for path in repo.rglob("*"):
        if path.is_file():
            try:
                yield str(path.relative_to(repo))
            except ValueError:
                continue


def _load_ragignore(repo: Path) -> pathspec.PathSpec:
    """Load .ragignore patterns from the target repo and from this project."""
    patterns: list[str] = []
    for candidate in (repo / ".ragignore", Path(__file__).resolve().parents[3] / ".ragignore"):
        if candidate.is_file():
            patterns.extend(candidate.read_text().splitlines())
    return pathspec.PathSpec.from_lines("gitignore", patterns)


class Walker:
    """Discovers indexable files under a target repository."""

    def __init__(self, repo: Path, max_file_bytes: int = 524_288) -> None:
        if not repo.exists() or not repo.is_dir():
            raise WalkerError(f"Target repo not found or not a directory: {repo}")
        self.repo = repo.resolve()
        self.max_file_bytes = max_file_bytes
        self._spec = _load_ragignore(self.repo)

    def discover(self) -> Iterable[SourceFile]:
        """Yield every file that survives all filters."""
        rel_paths = _git_ls_files(self.repo)
        source: Iterator[str]
        if rel_paths is None:
            log.info("walker.fallback", reason="not_a_git_repo_or_no_git", repo=str(self.repo))
            source = _filesystem_walk(self.repo)
        else:
            source = iter(rel_paths)

        kept = skipped_lang = skipped_size = skipped_binary = skipped_ignore = 0
        for rel in source:
            sf = self.evaluate_path(rel)
            if sf is None:
                # Reason already counted inside evaluate_path via this branch.
                # Reuse the same check to bump the right counter.
                if not rel or self._spec.match_file(rel):
                    skipped_ignore += 1
                else:
                    abs_path = self.repo / rel
                    if not abs_path.is_file():
                        continue
                    if detect_language(abs_path) is None:
                        skipped_lang += 1
                    else:
                        try:
                            size = abs_path.stat().st_size
                        except OSError:
                            continue
                        if size > self.max_file_bytes:
                            skipped_size += 1
                        elif size == 0:
                            continue
                        elif _is_binary(abs_path):
                            skipped_binary += 1
                continue
            kept += 1
            yield sf

        log.info(
            "walker.done",
            kept=kept,
            skipped_language=skipped_lang,
            skipped_size=skipped_size,
            skipped_binary=skipped_binary,
            skipped_ignore=skipped_ignore,
        )

    def discover_paths(self, rel_paths: Iterable[str]) -> Iterator[SourceFile]:
        """Yield SourceFile only for the explicit subset of paths given."""
        for rel in rel_paths:
            sf = self.evaluate_path(rel)
            if sf is not None:
                yield sf

    def evaluate_path(self, rel: str) -> SourceFile | None:
        """Apply all filters to a single rel_path. Returns None if filtered out."""
        if not rel or self._spec.match_file(rel):
            return None
        abs_path = self.repo / rel
        if not abs_path.is_file():
            return None
        language = detect_language(abs_path)
        if language is None:
            return None
        try:
            size = abs_path.stat().st_size
        except OSError:
            return None
        if size == 0 or size > self.max_file_bytes:
            return None
        if _is_binary(abs_path):
            return None
        return SourceFile(
            path=abs_path,
            rel_path=rel,
            language=language,
            size_bytes=size,
        )

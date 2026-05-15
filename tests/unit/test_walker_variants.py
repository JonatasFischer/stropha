"""Unit tests for the new walker adapters: filesystem + nested-git.

Both reuse the legacy ``stropha.ingest.walker.Walker`` for the actual
file filtering — the adapters are responsible for *which directories to
hand to it*. Tests focus on that orchestration layer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from stropha.adapters.walker.filesystem import FilesystemWalker, FilesystemWalkerConfig
from stropha.adapters.walker.nested_git import NestedGitWalker, NestedGitWalkerConfig


# --------------------------------------------------------------------------- helpers


def _git_init(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-q"], cwd=path, check=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
             "GIT_TERMINAL_PROMPT": "0"},
    )


# --------------------------------------------------------------------------- FilesystemWalker


def test_filesystem_walker_finds_text_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "b.md").write_text("# header", encoding="utf-8")

    walker = FilesystemWalker()
    out = list(walker.discover(tmp_path))
    paths = {sf.rel_path for sf in out}
    assert "a.py" in paths
    assert "b.md" in paths


def test_filesystem_walker_skips_default_caches(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("x=1", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk.py").write_text("y=2", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("z=3", encoding="utf-8")

    walker = FilesystemWalker()
    paths = {sf.rel_path for sf in walker.discover(tmp_path)}
    assert "main.py" in paths
    assert ".venv/junk.py" not in paths
    assert "node_modules/lib.js" not in paths


def test_filesystem_walker_respects_size_limit(tmp_path: Path) -> None:
    big = tmp_path / "huge.py"
    big.write_text("x" * 2_000_000, encoding="utf-8")  # 2 MB
    small = tmp_path / "small.py"
    small.write_text("x", encoding="utf-8")

    walker = FilesystemWalker(FilesystemWalkerConfig(max_file_bytes=1024))
    paths = {sf.rel_path for sf in walker.discover(tmp_path)}
    assert "small.py" in paths
    assert "huge.py" not in paths


def test_filesystem_walker_health_ready() -> None:
    h = FilesystemWalker().health()
    assert h.status == "ready"


def test_filesystem_walker_adapter_id_includes_size() -> None:
    w = FilesystemWalker(FilesystemWalkerConfig(max_file_bytes=4096))
    assert "max=4096" in w.adapter_id


# --------------------------------------------------------------------------- NestedGitWalker


def test_nested_git_walker_finds_no_repos_in_empty_dir(tmp_path: Path) -> None:
    walker = NestedGitWalker()
    out = list(walker.discover(tmp_path))
    assert out == []


def test_nested_git_walker_finds_nested_repos(tmp_path: Path) -> None:
    """A monorepo umbrella with two sibling git repos inside."""
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    _git_init(repo_a)
    _git_init(repo_b)
    (repo_a / "a.py").write_text("a=1", encoding="utf-8")
    (repo_b / "b.py").write_text("b=2", encoding="utf-8")

    walker = NestedGitWalker()
    paths = {sf.rel_path for sf in walker.discover(tmp_path)}
    # rel_paths are rebased to the umbrella root
    assert "repo_a/a.py" in paths
    assert "repo_b/b.py" in paths


def test_nested_git_walker_walks_root_repo_too(tmp_path: Path) -> None:
    """Umbrella root that is itself a git repo with a nested submodule."""
    _git_init(tmp_path)
    (tmp_path / "main.py").write_text("m=1", encoding="utf-8")

    sub = tmp_path / "vendored"
    sub.mkdir()
    _git_init(sub)
    (sub / "lib.py").write_text("lib=1", encoding="utf-8")

    walker = NestedGitWalker(NestedGitWalkerConfig(follow_root_first=True))
    paths = {sf.rel_path for sf in walker.discover(tmp_path)}
    # Both root file and nested submodule file present
    assert "main.py" in paths
    assert "vendored/lib.py" in paths


def test_nested_git_walker_dedupes_overlapping_paths(tmp_path: Path) -> None:
    """When a file would be discovered both via the root walker and a
    nested one, only one entry should come out."""
    _git_init(tmp_path)
    sub = tmp_path / "submod"
    sub.mkdir()
    _git_init(sub)
    # Same logical file, only present in the submodule
    (sub / "shared.py").write_text("s=1", encoding="utf-8")

    walker = NestedGitWalker(NestedGitWalkerConfig(follow_root_first=True))
    paths = [sf.rel_path for sf in walker.discover(tmp_path)]
    # shared.py appears at most once
    assert paths.count("submod/shared.py") <= 1


def test_nested_git_walker_respects_max_depth(tmp_path: Path) -> None:
    """Nested git repo deeper than max_depth should be ignored."""
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    _git_init(deep)
    (deep / "deep.py").write_text("d=1", encoding="utf-8")

    walker = NestedGitWalker(NestedGitWalkerConfig(max_depth=2))
    paths = {sf.rel_path for sf in walker.discover(tmp_path)}
    assert "a/b/c/d/deep.py" not in paths


def test_nested_git_walker_health_ready() -> None:
    h = NestedGitWalker().health()
    assert h.status == "ready"


def test_nested_git_walker_adapter_id_includes_depth() -> None:
    w = NestedGitWalker(NestedGitWalkerConfig(max_depth=7))
    assert "depth=7" in w.adapter_id


# --------------------------------------------------------------------------- registry


def test_walkers_registered() -> None:
    from stropha.pipeline.registry import all_adapters

    walkers = set(all_adapters()["walker"])
    assert {"git-ls-files", "filesystem", "nested-git"} <= walkers

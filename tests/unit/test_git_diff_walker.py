"""Unit tests for ``GitDiffWalker`` — Phase B incremental walker.

Uses real ``git init`` + commits so the parsing logic runs against
output from the actual git binary (parsing is fragile enough to deserve
end-to-end coverage; mocks would hide real bugs in null-separator
handling, rename score formats, etc.).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from stropha.adapters.walker.git_diff import (
    GitDiffWalker,
    GitDiffWalkerConfig,
    _default_rename_threshold,
)
from stropha.errors import WalkerError


# --------------------------------------------------------------------------- helpers


def _git(repo: Path, *args: str) -> str:
    """Run git, return stdout, raise on non-zero exit."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@x.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@x.com",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", *args], cwd=repo, env=env,
        capture_output=True, text=True, check=True,
    )
    return result.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@x.com")
    _git(r, "config", "user.name", "t")
    return r


def _commit(repo: Path, msg: str = "c") -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)
    return _git(repo, "rev-parse", "HEAD").strip()


# --------------------------------------------------------------------------- registry / config


def test_walker_registered_in_registry() -> None:
    from stropha.pipeline.registry import lookup_adapter
    cls = lookup_adapter("walker", "git-diff")
    assert cls is GitDiffWalker


def test_default_rename_threshold_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("STROPHA_RENAME_THRESHOLD", raising=False)
    assert _default_rename_threshold() == 80


def test_rename_threshold_env_override(monkeypatch) -> None:
    monkeypatch.setenv("STROPHA_RENAME_THRESHOLD", "55")
    assert _default_rename_threshold() == 55


def test_rename_threshold_clamps_to_range(monkeypatch) -> None:
    monkeypatch.setenv("STROPHA_RENAME_THRESHOLD", "999")
    assert _default_rename_threshold() == 100
    monkeypatch.setenv("STROPHA_RENAME_THRESHOLD", "-5")
    assert _default_rename_threshold() == 0


def test_adapter_id_includes_threshold() -> None:
    w = GitDiffWalker(GitDiffWalkerConfig(rename_threshold=72))
    assert "rename>=72" in w.adapter_id


def test_health_reports_git_version() -> None:
    h = GitDiffWalker().health()
    assert h.status == "ready"
    assert "git-diff" in h.message


# --------------------------------------------------------------------------- discover_deltas


def test_discover_raises_on_unsupported_full_discover(repo: Path) -> None:
    """Legacy `discover` shape is intentionally not supported."""
    with pytest.raises(WalkerError):
        list(GitDiffWalker().discover(repo))


def test_detects_add(repo: Path) -> None:
    (repo / "a.py").write_text("x = 1\n")
    base = _commit(repo, "init")
    (repo / "b.py").write_text("y = 2\n")
    _commit(repo, "add b")

    deltas = GitDiffWalker().discover_deltas(repo, since_sha=base)
    actions = {(d.action, d.rel_path) for d in deltas}
    assert ("add", "b.py") in actions


def test_detects_modify(repo: Path) -> None:
    (repo / "a.py").write_text("x = 1\n")
    base = _commit(repo, "init")
    (repo / "a.py").write_text("x = 2\n")
    _commit(repo, "modify a")

    deltas = GitDiffWalker().discover_deltas(repo, since_sha=base)
    actions = {(d.action, d.rel_path) for d in deltas}
    assert ("modify", "a.py") in actions


def test_detects_delete(repo: Path) -> None:
    (repo / "a.py").write_text("x = 1\n")
    (repo / "b.py").write_text("y = 2\n")
    base = _commit(repo, "init")
    (repo / "b.py").unlink()
    _commit(repo, "remove b")

    deltas = GitDiffWalker().discover_deltas(repo, since_sha=base)
    actions = {(d.action, d.rel_path) for d in deltas}
    assert ("delete", "b.py") in actions


def test_detects_pure_rename(repo: Path) -> None:
    """File renamed with NO content change → rename delta, similarity 100."""
    (repo / "a.py").write_text("def foo():\n    return 42\n")
    base = _commit(repo, "init")
    _git(repo, "mv", "a.py", "b.py")
    _commit(repo, "rename")

    deltas = GitDiffWalker().discover_deltas(repo, since_sha=base)
    renames = [d for d in deltas if d.action == "rename"]
    assert len(renames) == 1
    r = renames[0]
    assert r.old_rel_path == "a.py"
    assert r.rel_path == "b.py"
    assert r.similarity == 100


def test_detects_rename_with_partial_edit(repo: Path) -> None:
    """Same file renamed AND lightly edited — similarity below 100 but
    still above the default threshold (80) → still a rename."""
    body = "def foo():\n    return 42\n" + "# filler\n" * 30
    (repo / "a.py").write_text(body)
    base = _commit(repo, "init")
    _git(repo, "mv", "a.py", "b.py")
    # Tweak one line in the new path.
    (repo / "b.py").write_text(body.replace("return 42", "return 43"))
    _commit(repo, "rename + edit")

    deltas = GitDiffWalker().discover_deltas(repo, since_sha=base)
    renames = [d for d in deltas if d.action == "rename"]
    assert len(renames) == 1
    assert renames[0].similarity is not None
    assert renames[0].similarity >= 80


def test_filters_unsupported_extensions(repo: Path) -> None:
    (repo / "a.py").write_text("x = 1\n")
    base = _commit(repo, "init")
    (repo / "binary.dat").write_text("BINARY\n")
    (repo / "image.png").write_text("PNG\n")
    (repo / "code.py").write_text("y = 2\n")
    _commit(repo, "add binary + code")

    deltas = GitDiffWalker().discover_deltas(repo, since_sha=base)
    paths = {d.rel_path for d in deltas}
    assert "code.py" in paths
    assert "binary.dat" not in paths
    assert "image.png" not in paths


def test_handles_paths_with_spaces(repo: Path) -> None:
    (repo / "weird name.py").write_text("x = 1\n")
    base = _commit(repo, "init")
    (repo / "weird name.py").write_text("x = 2\n")
    _commit(repo, "edit weird name")

    deltas = GitDiffWalker().discover_deltas(repo, since_sha=base)
    assert any(d.rel_path == "weird name.py" for d in deltas)


def test_raises_on_bad_since_sha(repo: Path) -> None:
    (repo / "a.py").write_text("x\n")
    _commit(repo)
    with pytest.raises(WalkerError):
        GitDiffWalker().discover_deltas(repo, since_sha="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")


def test_empty_diff_returns_empty_list(repo: Path) -> None:
    """No commits between since and HEAD → empty result, not error."""
    (repo / "a.py").write_text("x\n")
    sha = _commit(repo, "init")
    deltas = GitDiffWalker().discover_deltas(repo, since_sha=sha)
    assert deltas == []


def test_multiple_actions_in_one_diff(repo: Path) -> None:
    (repo / "keep.py").write_text("k\n")
    (repo / "remove.py").write_text("r\n")
    (repo / "modify.py").write_text("m\n")
    base = _commit(repo, "init")

    (repo / "remove.py").unlink()
    (repo / "modify.py").write_text("m2\n")
    (repo / "add.py").write_text("a\n")
    _commit(repo, "mixed")

    deltas = GitDiffWalker().discover_deltas(repo, since_sha=base)
    actions = {(d.action, d.rel_path) for d in deltas}
    assert ("delete", "remove.py") in actions
    assert ("modify", "modify.py") in actions
    assert ("add", "add.py") in actions
    # keep.py untouched → no delta
    assert not any(d.rel_path == "keep.py" for d in deltas)

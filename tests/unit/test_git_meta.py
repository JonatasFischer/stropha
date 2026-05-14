"""Tests for the git identity helper."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from stropha.ingest.git_meta import RepoIdentity, detect, normalize_url


# ---- normalize_url --------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_key,expected_url,expected_stripped",
    [
        # SSH short — standard git@ user is not "stripped auth".
        (
            "git@github.com:Foo/Bar.git",
            "github.com/foo/bar",
            "https://github.com/foo/bar.git",
            False,
        ),
        # HTTPS without .git, mixed case → lowercased + .git appended.
        (
            "https://github.com/Foo/Bar",
            "github.com/foo/bar",
            "https://github.com/foo/bar.git",
            False,
        ),
        # HTTPS canonical — round-trip.
        (
            "https://github.com/foo/bar.git",
            "github.com/foo/bar",
            "https://github.com/foo/bar.git",
            False,
        ),
        # PAT-leaked URL → stripped + flagged.
        (
            "https://x-token:ghp_abc123@github.com/foo/bar.git",
            "github.com/foo/bar",
            "https://github.com/foo/bar.git",
            True,
        ),
        # ssh:// scheme.
        (
            "ssh://git@gitlab.example.com/group/repo.git",
            "gitlab.example.com/group/repo",
            "https://gitlab.example.com/group/repo.git",
            False,
        ),
        # Trailing slash gets stripped.
        (
            "https://github.com/foo/bar/",
            "github.com/foo/bar",
            "https://github.com/foo/bar.git",
            False,
        ),
    ],
)
def test_normalize_url(
    raw: str,
    expected_key: str,
    expected_url: str,
    expected_stripped: bool,
) -> None:
    key, url, stripped = normalize_url(raw)
    assert key == expected_key
    assert url == expected_url
    assert stripped is expected_stripped


def test_normalize_url_rejects_empty() -> None:
    with pytest.raises(ValueError):
        normalize_url("")


def test_normalize_url_rejects_bare_path() -> None:
    with pytest.raises(ValueError):
        normalize_url("just-a-name")


# ---- detect on filesystem fixtures ---------------------------------------

def _init_repo(path: Path, remote: str | None = None) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-q", "--allow-empty", "-m", "init"],
        cwd=path, check=True,
    )
    if remote:
        subprocess.run(
            ["git", "remote", "add", "origin", remote], cwd=path, check=True
        )


def test_detect_non_git(tmp_path: Path) -> None:
    ident = detect(tmp_path)
    assert ident.normalized_key.startswith("path:")
    assert ident.remote_url is None
    assert ident.head_commit is None


def test_detect_local_only(tmp_path: Path) -> None:
    repo = tmp_path / "local"
    repo.mkdir()
    _init_repo(repo)
    ident = detect(repo)
    assert ident.normalized_key.startswith("local:")
    assert ident.remote_url is None
    # Even without remote we still have HEAD.
    assert ident.head_commit is not None and len(ident.head_commit) == 40


def test_detect_with_remote(tmp_path: Path) -> None:
    repo = tmp_path / "remote"
    repo.mkdir()
    _init_repo(repo, remote="git@github.com:Foo/Bar.git")
    ident = detect(repo)
    assert ident.normalized_key == "github.com/foo/bar"
    assert ident.remote_url == "https://github.com/foo/bar.git"
    assert ident.head_commit is not None


def test_detect_strips_auth_token(tmp_path: Path) -> None:
    repo = tmp_path / "leaky"
    repo.mkdir()
    _init_repo(repo, remote="https://x-tok:ghp_secret@github.com/foo/bar.git")
    ident = detect(repo)
    # Secret never appears in the persisted URL.
    assert "ghp_secret" not in (ident.remote_url or "")
    assert "x-tok" not in (ident.remote_url or "")
    assert ident.remote_url == "https://github.com/foo/bar.git"


def test_repo_identity_is_frozen() -> None:
    ident = RepoIdentity(
        normalized_key="github.com/a/b",
        remote_url="https://github.com/a/b.git",
        root_path=Path("/tmp/x"),
        default_branch="main",
        head_commit="abc",
    )
    with pytest.raises((AttributeError, TypeError)):
        ident.normalized_key = "mutated"  # type: ignore[misc]

"""Unit tests for ``stropha.ingest.manifest.load_manifest``.

Coverage matrix:
- valid manifest with absolute and relative paths
- enabled flag respected
- bare-string entries
- invalid YAML / missing keys / wrong types raise ManifestError
- expanduser
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stropha.ingest.manifest import (
    ManifestEntry,
    ManifestError,
    load_manifest,
)


def _write_manifest(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- happy path


def test_loads_absolute_paths(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    manifest = _write_manifest(
        tmp_path / "repos.yaml",
        f"repos:\n  - path: {repo_a}\n  - path: {repo_b}\n",
    )
    entries = load_manifest(manifest)
    assert [e.path for e in entries] == [repo_a.resolve(), repo_b.resolve()]
    assert all(e.enabled for e in entries)


def test_loads_relative_paths_resolved_from_manifest_dir(tmp_path: Path) -> None:
    sub = tmp_path / "subdir"
    sub.mkdir()
    relative_target = sub / "child"
    relative_target.mkdir()
    manifest = _write_manifest(
        sub / "manifest.yaml",
        "repos:\n  - path: ./child\n",
    )
    entries = load_manifest(manifest)
    assert entries[0].path == relative_target.resolve()


def test_bare_string_entries(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    manifest = _write_manifest(
        tmp_path / "repos.yaml",
        f"repos:\n  - {repo}\n",
    )
    entries = load_manifest(manifest)
    assert entries == [ManifestEntry(path=repo.resolve(), enabled=True)]


def test_disabled_entries_are_skipped(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    manifest = _write_manifest(
        tmp_path / "repos.yaml",
        f"repos:\n  - path: {a}\n  - path: {b}\n    enabled: false\n",
    )
    entries = load_manifest(manifest)
    assert len(entries) == 1
    assert entries[0].path == a.resolve()


def test_expanduser(tmp_path: Path, monkeypatch) -> None:
    """Tilde paths are expanded."""
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    repo = fake_home / "myrepo"
    repo.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    manifest = _write_manifest(
        tmp_path / "repos.yaml",
        "repos:\n  - path: ~/myrepo\n",
    )
    entries = load_manifest(manifest)
    assert entries[0].path == repo.resolve()


# --------------------------------------------------------------------------- failure modes


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "bad.yaml", "this: is: bad: yaml:::")
    with pytest.raises(ManifestError, match="not valid YAML"):
        load_manifest(manifest)


def test_empty_file_raises(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "empty.yaml", "")
    with pytest.raises(ManifestError, match="empty"):
        load_manifest(manifest)


def test_missing_repos_key_raises(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "wrong.yaml", "settings: {}\n")
    with pytest.raises(ManifestError, match="`repos:`"):
        load_manifest(manifest)


def test_repos_must_be_list(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "wrong.yaml", "repos: hello\n")
    with pytest.raises(ManifestError, match="must be a list"):
        load_manifest(manifest)


def test_repo_entry_missing_path_raises(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "wrong.yaml",
        "repos:\n  - enabled: true\n",
    )
    with pytest.raises(ManifestError, match="missing required `path:`"):
        load_manifest(manifest)


def test_repo_entry_wrong_type_raises(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "wrong.yaml",
        "repos:\n  - 42\n",
    )
    with pytest.raises(ManifestError, match="must be a string or mapping"):
        load_manifest(manifest)

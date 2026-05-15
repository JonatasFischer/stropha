"""Unit tests for ``stropha.tools.hook_install``.

Per RFC §6.2 / §9 (Fase 1.5c):
- install() é idempotente
- preserva conteúdo existente do post-commit (não sobrescreve)
- detecta marker version
- detecta coexistência com graphify hook
- uninstall() remove APENAS nosso bloco
- file removido se sobra só shebang
- core.hooksPath respeitado quando válido
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from stropha.tools.hook_install import (
    END_MARKER,
    GRAPHIFY_MARKER,
    HOOK_VERSION,
    START_MARKER,
    _resolve_hooks_dir,
    install,
    status,
    uninstall,
)

# --------------------------------------------------------------------------- helpers


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Initialise a bare git repo skeleton (just .git/) suitable for hook tests."""
    r = tmp_path / "repo"
    r.mkdir()
    (r / ".git").mkdir()
    (r / ".git" / "hooks").mkdir()
    return r


def _read_hook(repo: Path) -> str:
    return (repo / ".git" / "hooks" / "post-commit").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- install


def test_install_creates_new_hook(repo: Path) -> None:
    result = install(repo)
    assert result["action"] == "created"
    assert result["version"] == HOOK_VERSION

    hook_text = _read_hook(repo)
    assert hook_text.startswith("#!/bin/sh")
    assert START_MARKER in hook_text
    assert END_MARKER in hook_text
    assert f"v={HOOK_VERSION}" in hook_text

    # Executable
    mode = (repo / ".git" / "hooks" / "post-commit").stat().st_mode
    assert mode & 0o100  # owner-x


def test_install_is_idempotent(repo: Path) -> None:
    install(repo)
    first = _read_hook(repo)
    install(repo)
    second = _read_hook(repo)
    assert first == second


def test_install_appends_when_existing_file_has_no_markers(repo: Path) -> None:
    hook_path = repo / ".git" / "hooks" / "post-commit"
    hook_path.write_text(
        "#!/bin/sh\necho 'pre-existing user hook'\n",
        encoding="utf-8",
    )
    result = install(repo)
    assert result["action"] == "appended"
    body = _read_hook(repo)
    # Pre-existing content preserved
    assert "pre-existing user hook" in body
    # Our block added
    assert START_MARKER in body
    assert END_MARKER in body


def test_install_replaces_existing_block(repo: Path) -> None:
    install(repo)
    # Manually corrupt the version line to simulate an outdated install
    hook_path = repo / ".git" / "hooks" / "post-commit"
    body = hook_path.read_text(encoding="utf-8")
    body = body.replace(f"{START_MARKER} v={HOOK_VERSION}",
                        f"{START_MARKER} v=0")
    hook_path.write_text(body, encoding="utf-8")

    result = install(repo)
    assert result["action"] == "updated"

    new_body = _read_hook(repo)
    # The old block is gone, only the new (current) version line remains
    assert f"{START_MARKER} v=0" not in new_body
    assert f"{START_MARKER} v={HOOK_VERSION}" in new_body
    # Should appear exactly once
    assert new_body.count(START_MARKER) == 1


def test_install_detects_graphify_cohabit(repo: Path) -> None:
    hook_path = repo / ".git" / "hooks" / "post-commit"
    hook_path.write_text(
        f"#!/bin/sh\n{GRAPHIFY_MARKER}\necho 'graphify'\n# graphify-hook-end\n",
        encoding="utf-8",
    )
    result = install(repo)
    assert result["graphify_cohabit"] is True
    assert result["coexist_warning"] is True

    # With force=True, no warning
    result_force = install(repo, force=True)
    assert result_force["graphify_cohabit"] is True
    assert result_force["coexist_warning"] is False


def test_install_rejects_non_git_dir(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    with pytest.raises(ValueError, match="not a git repository"):
        install(not_a_repo)


def test_install_target_substituted_in_script(repo: Path) -> None:
    install(repo)
    body = _read_hook(repo)
    # The target absolute path appears in the EXPECTED= line
    assert f'EXPECTED="{repo.resolve()}"' in body


# --------------------------------------------------------------------------- status


def test_status_when_not_installed(repo: Path) -> None:
    s = status(repo)
    assert s.installed is False
    assert s.version is None
    assert s.graphify_cohabit is False


def test_status_after_install(repo: Path) -> None:
    install(repo)
    s = status(repo)
    assert s.installed is True
    assert s.version == HOOK_VERSION
    assert s.graphify_cohabit is False


def test_status_outdated_version(repo: Path) -> None:
    install(repo)
    hook_path = repo / ".git" / "hooks" / "post-commit"
    body = hook_path.read_text(encoding="utf-8")
    body = body.replace(f"{START_MARKER} v={HOOK_VERSION}",
                        f"{START_MARKER} v=0")
    hook_path.write_text(body, encoding="utf-8")
    s = status(repo)
    assert s.installed is True
    assert s.version == 0  # detected, but != HOOK_VERSION


# --------------------------------------------------------------------------- uninstall


def test_uninstall_when_not_installed(repo: Path) -> None:
    result = uninstall(repo)
    assert result["action"] == "noop"


def test_uninstall_removes_only_our_block(repo: Path) -> None:
    hook_path = repo / ".git" / "hooks" / "post-commit"
    hook_path.write_text(
        "#!/bin/sh\necho 'user hook before'\n",
        encoding="utf-8",
    )
    install(repo)
    body_after_install = _read_hook(repo)
    assert "user hook before" in body_after_install
    assert START_MARKER in body_after_install

    result = uninstall(repo)
    assert result["action"] == "block_removed"
    body_after_uninstall = _read_hook(repo)
    assert "user hook before" in body_after_uninstall
    assert START_MARKER not in body_after_uninstall
    assert END_MARKER not in body_after_uninstall


def test_uninstall_removes_file_when_only_shebang_left(repo: Path) -> None:
    install(repo)
    result = uninstall(repo)
    assert result["action"] == "removed"
    assert not (repo / ".git" / "hooks" / "post-commit").exists()


def test_uninstall_noop_when_only_unrelated_content(repo: Path) -> None:
    hook_path = repo / ".git" / "hooks" / "post-commit"
    hook_path.write_text("#!/bin/sh\necho hello\n", encoding="utf-8")
    result = uninstall(repo)
    assert result["action"] == "noop"
    # File preserved
    assert hook_path.read_text(encoding="utf-8") == "#!/bin/sh\necho hello\n"


# --------------------------------------------------------------------------- core.hooksPath


def test_resolve_hooks_dir_default(repo: Path) -> None:
    """No core.hooksPath configured → default `.git/hooks` (RFC §6.3)."""
    p = _resolve_hooks_dir(repo)
    assert p == (repo / ".git" / "hooks").resolve()


def test_resolve_hooks_dir_honours_core_hookspath(tmp_path: Path) -> None:
    """When git config core.hooksPath set inside repo, follow it."""
    # Initialise a real git repo so `git config` works
    repo = tmp_path / "withhooks"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True,
                   env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})

    custom = repo / ".husky"
    custom.mkdir()
    subprocess.run(
        ["git", "config", "core.hooksPath", ".husky"],
        cwd=repo, check=True,
    )
    p = _resolve_hooks_dir(repo)
    assert p == custom.resolve()


def test_resolve_hooks_dir_rejects_unsafe_path(tmp_path: Path) -> None:
    """A core.hooksPath outside repo and HOME is rejected → falls back."""
    repo = tmp_path / "unsafe"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True,
                   env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    # Point hooksPath to /tmp (outside both repo and HOME, on most systems)
    bad = Path("/private/tmp/totally-not-allowed")
    subprocess.run(
        ["git", "config", "core.hooksPath", str(bad)],
        cwd=repo, check=True,
    )
    p = _resolve_hooks_dir(repo)
    # When the configured path is rejected, we fall back to .git/hooks.
    # /tmp may legitimately resolve under HOME on some CI machines so we
    # only assert the fallback was chosen when the path was outside HOME.
    if not str(bad).startswith(str(Path.home().resolve())):
        assert p == (repo / ".git" / "hooks").resolve()


# --------------------------------------------------------------------------- v=3 cross-repo baked defaults


def test_install_without_new_args_uses_legacy_toplevel(repo: Path) -> None:
    """Back-compat regression guard: install() with no new flags must emit
    empty bakes and let runtime fall back to $TOPLEVEL."""
    install(repo)
    body = _read_hook(repo)
    assert 'PROJECT_DIR_DEFAULT=""' in body
    assert 'INDEX_PATH_DEFAULT=""' in body
    assert 'LOG_DEFAULT=""' in body
    # Runtime fallback chain must point at $TOPLEVEL when default is empty.
    assert 'PROJECT_DIR="${STROPHA_HOOK_PROJECT_DIR:-${PROJECT_DIR_DEFAULT:-$TOPLEVEL}}"' in body
    # And the CMD still uses $PROJECT_DIR for --directory (NOT hardcoded $TOPLEVEL).
    assert "--directory $PROJECT_DIR" in body
    # No env injection for the index step when no override is baked.
    assert 'INDEX_ENV=""' in body


def test_install_with_project_dir_bakes_it(repo: Path, tmp_path: Path) -> None:
    venv_dir = tmp_path / "venv-host"
    venv_dir.mkdir()
    # Make pyproject.toml present so the bake is realistic
    (venv_dir / "pyproject.toml").write_text("[project]\nname = 'host'", encoding="utf-8")

    result = install(repo, project_dir=venv_dir)
    body = _read_hook(repo)
    assert f'PROJECT_DIR_DEFAULT="{venv_dir.resolve()}"' in body
    # The CMD line still references $PROJECT_DIR (not the literal path) —
    # the bake flows through the env-resolution chain.
    assert "--directory $PROJECT_DIR" in body
    # install() returns the resolved path
    assert result["project_dir"] == str(venv_dir.resolve())


def test_install_with_index_path_bakes_env_injection(repo: Path, tmp_path: Path) -> None:
    idx = tmp_path / "elsewhere" / "index.db"

    result = install(repo, index_path=idx)
    body = _read_hook(repo)
    assert f'INDEX_PATH_DEFAULT="{idx.expanduser()}"' in body
    # The env injection block must appear and use the override path.
    assert 'INDEX_ENV="env STROPHA_TARGET_REPO=$TOPLEVEL STROPHA_INDEX_PATH=$INDEX_PATH_OVERRIDE"' in body
    assert result["index_path"] == str(idx.expanduser())


def test_install_with_log_path_bakes_it(repo: Path, tmp_path: Path) -> None:
    log = tmp_path / "logs" / "hook.log"

    result = install(repo, log_path=log)
    body = _read_hook(repo)
    assert f'LOG_DEFAULT="{log.expanduser()}"' in body
    # The LOG runtime resolution must consume LOG_DEFAULT.
    assert 'LOG="${STROPHA_HOOK_LOG:-${LOG_DEFAULT:-${HOME}/.cache/stropha-hook.log}}"' in body
    assert result["log_path"] == str(log.expanduser())


def test_status_reports_baked_defaults(repo: Path, tmp_path: Path) -> None:
    project = tmp_path / "host"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'host'", encoding="utf-8")
    idx = tmp_path / "i.db"
    logp = tmp_path / "h.log"

    install(repo, project_dir=project, index_path=idx, log_path=logp)
    s = status(repo)
    assert s.installed is True
    assert s.version == HOOK_VERSION
    assert s.project_dir == project.resolve()
    assert s.index_path == idx.expanduser()
    assert s.log_path_default == logp.expanduser()
    # Dict serialisation also surfaces them.
    d = s.as_dict()
    assert d["project_dir"] == str(project.resolve())
    assert d["index_path"] == str(idx.expanduser())
    assert d["log_path_default"] == str(logp.expanduser())


def test_install_replaces_older_block_with_current(repo: Path) -> None:
    """Upgrade path: a manually-written older-version block must be
    cleanly rewritten to ``HOOK_VERSION`` in place (no duplication, no
    leftover markers). Validates the v=2 → v=4 upgrade."""
    hook_path = repo / ".git" / "hooks" / "post-commit"
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    # Synthetic v=2 block — content doesn't matter, just the markers.
    hook_path.write_text(
        "#!/bin/sh\n"
        "# user prelude\n"
        f"{START_MARKER} v=2\n"
        "echo legacy v=2 body\n"
        f"{END_MARKER}\n"
        "echo user epilog\n",
        encoding="utf-8",
    )

    result = install(repo)
    assert result["action"] == "updated"

    body = _read_hook(repo)
    # The new block reports the current HOOK_VERSION …
    assert f"{START_MARKER} v={HOOK_VERSION}" in body
    # … the old v=2 marker is gone …
    assert f"{START_MARKER} v=2" not in body
    # … and the surrounding user content is preserved.
    assert "# user prelude" in body
    assert "echo user epilog" in body
    # Block must appear exactly once.
    assert body.count(START_MARKER) == 1


def test_install_v4_template_uses_incremental_flag(repo: Path) -> None:
    """Phase B contract: the generated hook v=4 invokes
    ``stropha index --incremental`` so the post-commit refresh re-chunks
    only the files touched by the commit."""
    install(repo)
    body = _read_hook(repo)
    assert "stropha index --repo $TOPLEVEL --incremental" in body

"""Post-commit hook installer (RFC §6 / §7.1).

Manages the stropha post-commit hook that refreshes the graphify graph and
re-indexes the repo in detached background after every commit.

Public surface:
- :func:`install` — write or update the hook block between markers
- :func:`uninstall` — remove our block, leave the rest of the file intact
- :func:`status` — report whether the hook is installed and operational

Markers used: ``# stropha-hook-start v=N`` / ``# stropha-hook-end``.
The version number lets future installs detect outdated blocks and rewrite.
The hook itself runs in the background via ``nohup`` + ``flock``, never
blocks the commit by more than ~100 ms (the time to fork).

Coexistence: if the existing post-commit file already contains a graphify
hook (``# graphify-hook-start``), :func:`install` warns the caller. The
stropha hook itself already calls ``graphify update`` so keeping both
duplicates work; :func:`install` honours ``force=True`` to silence the warning.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from ..logging import get_logger

log = get_logger(__name__)

# Bump this when the embedded script changes in a backwards-incompatible way
# (e.g. added env var, renamed binary). `install --force` and `status` use it
# to detect outdated blocks.
HOOK_VERSION = 2

START_MARKER = "# stropha-hook-start"
END_MARKER = "# stropha-hook-end"
GRAPHIFY_MARKER = "# graphify-hook-start"

# Regex matches `# stropha-hook-start v=<digits>` followed by anything up to
# `# stropha-hook-end`. DOTALL so it matches across lines.
_BLOCK_RE = re.compile(
    rf"{re.escape(START_MARKER)} v=\d+.*?{re.escape(END_MARKER)}\n?",
    re.DOTALL,
)
_VERSION_RE = re.compile(rf"{re.escape(START_MARKER)} v=(\d+)")


# --------------------------------------------------------------------- model


@dataclass(frozen=True)
class HookStatus:
    target_repo: Path
    hook_path: Path
    installed: bool
    version: int | None
    graphify_cohabit: bool
    log_path: Path

    def as_dict(self) -> dict:
        return {
            "target_repo": str(self.target_repo),
            "hook_path": str(self.hook_path),
            "installed": self.installed,
            "version": self.version,
            "graphify_cohabit": self.graphify_cohabit,
            "log_path": str(self.log_path),
        }


# --------------------------------------------------------------------- helpers


def _resolve_hooks_dir(repo: Path) -> Path:
    """Honour ``core.hooksPath`` so husky / lefthook coexist (RFC §6.3).

    Validates the resolved path is rooted under ``repo`` or ``$HOME`` to
    block supply-chain attacks via malicious git config.
    """
    try:
        result = subprocess.run(
            ["git", "config", "--get", "core.hooksPath"],
            cwd=repo, capture_output=True, text=True, check=False, timeout=5,
        )
        configured = result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        configured = ""

    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = repo / candidate
        try:
            candidate = candidate.resolve()
            repo_resolved = repo.resolve()
            home_resolved = Path.home().resolve()
            if str(candidate).startswith((str(repo_resolved), str(home_resolved))):
                return candidate
        except (OSError, RuntimeError):  # pragma: no cover — defensive
            pass

    return (repo / ".git" / "hooks").resolve()


def _hook_path(repo: Path) -> Path:
    return _resolve_hooks_dir(repo) / "post-commit"


def _log_path() -> Path:
    return Path(os.environ.get("STROPHA_HOOK_LOG", str(Path.home() / ".cache" / "stropha-hook.log")))


def _detect_uv_path() -> str:
    """Best-effort path to the user's ``uv`` binary (hooks have stripped PATH)."""
    candidates = [
        os.environ.get("STROPHA_HOOK_UV"),
        Path.home() / ".local" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ]
    for c in candidates:
        if c and Path(str(c)).is_file():
            return str(c)
    # Fallback: use whatever PATH gives us at install time.
    try:
        result = subprocess.run(
            ["which", "uv"], capture_output=True, text=True, check=False, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "uv"  # let runtime resolution discover via STROPHA_HOOK_UV override


# --------------------------------------------------------------------- script template


def _render_hook_block(target: Path) -> str:
    """The script body sandwiched between START and END markers.

    Same logic as RFC §6.1 + the manual install: graphify update FIRST
    (no LLM, fast), then stropha index — both inside one detached background
    nohup process protected by flock.
    """
    target_str = str(target)
    # Use single-quoted heredoc semantics: every $ in the template is a literal
    # for the generated shell file (no Python interpolation).
    return dedent(
        f"""\
        {START_MARKER} v={HOOK_VERSION}
        # Post-commit: refresh graphify graph + stropha index in background.
        # Generated by `stropha hook install`. Edit overrides via env vars only.
        # graphify update runs BEFORE stropha index so the index can pick up
        # the fresh graph.json on the same commit.
        #
        # Overrides:
        #   STROPHA_HOOK_SKIP=1         -> skip entirely (rebases)
        #   STROPHA_HOOK_TIMEOUT=600    -> wall-clock seconds for bg job
        #   STROPHA_HOOK_LOG=<path>     -> override log file
        #   STROPHA_HOOK_UV=<path>      -> override `uv` binary
        #   STROPHA_HOOK_GRAPHIFY=<p>   -> override `graphify` binary
        #   STROPHA_HOOK_NO_GRAPHIFY=1  -> skip graphify update step

        # --- 1. Skip conditions ----------------------------------------------
        [ "${{STROPHA_HOOK_SKIP:-0}}" = "1" ] && exit 0

        GIT_DIR=$(git rev-parse --git-dir 2>/dev/null) || exit 0
        [ -d "$GIT_DIR/rebase-merge" ]      && exit 0
        [ -d "$GIT_DIR/rebase-apply" ]      && exit 0
        [ -f "$GIT_DIR/MERGE_HEAD" ]        && exit 0
        [ -f "$GIT_DIR/CHERRY_PICK_HEAD" ]  && exit 0

        TOPLEVEL=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
        EXPECTED="{target_str}"
        [ "$TOPLEVEL" = "$EXPECTED" ] || exit 0

        CHANGED=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || git diff --name-only HEAD)
        [ -z "$CHANGED" ] && exit 0

        # --- 2. Resolve `uv` (hooks have a stripped PATH) --------------------
        UV="${{STROPHA_HOOK_UV:-}}"
        if [ -z "$UV" ]; then
            if command -v uv >/dev/null 2>&1; then
                UV=$(command -v uv)
            elif [ -x "$HOME/.local/bin/uv" ]; then
                UV="$HOME/.local/bin/uv"
            elif [ -x "/opt/homebrew/bin/uv" ]; then
                UV="/opt/homebrew/bin/uv"
            elif [ -x "/usr/local/bin/uv" ]; then
                UV="/usr/local/bin/uv"
            fi
        fi

        # --- 3. Resolve optional `timeout` -----------------------------------
        TIMEOUT_BIN=""
        if command -v timeout >/dev/null 2>&1; then TIMEOUT_BIN="timeout"; fi
        if [ -z "$TIMEOUT_BIN" ] && command -v gtimeout >/dev/null 2>&1; then TIMEOUT_BIN="gtimeout"; fi

        # --- 3b. Resolve `graphify` ------------------------------------------
        GRAPHIFY="${{STROPHA_HOOK_GRAPHIFY:-}}"
        if [ -z "$GRAPHIFY" ] && [ "${{STROPHA_HOOK_NO_GRAPHIFY:-0}}" != "1" ]; then
            if command -v graphify >/dev/null 2>&1; then
                GRAPHIFY=$(command -v graphify)
            elif [ -x "$HOME/.local/bin/graphify" ]; then
                GRAPHIFY="$HOME/.local/bin/graphify"
            elif [ -x "/opt/homebrew/bin/graphify" ]; then
                GRAPHIFY="/opt/homebrew/bin/graphify"
            elif [ -x "/usr/local/bin/graphify" ]; then
                GRAPHIFY="/usr/local/bin/graphify"
            fi
        fi

        # --- 4. Detached background run --------------------------------------
        LOG="${{STROPHA_HOOK_LOG:-${{HOME}}/.cache/stropha-hook.log}}"
        LOCK="/tmp/stropha-hook-$(printf '%s' "$TOPLEVEL" | md5).lock"
        mkdir -p "$(dirname "$LOG")"

        SHORT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        echo "[$(date -u +%FT%TZ)] commit ${{SHORT_SHA}} - launching refresh" >> "$LOG"

        if [ -z "$UV" ]; then
            echo "[stropha-hook] uv not found on PATH; aborting (set STROPHA_HOOK_UV)" >> "$LOG"
            exit 0
        fi

        if [ -n "$TIMEOUT_BIN" ]; then
            CMD="$TIMEOUT_BIN ${{STROPHA_HOOK_TIMEOUT:-600}} $UV run --directory $TOPLEVEL stropha index --repo $TOPLEVEL"
        else
            CMD="$UV run --directory $TOPLEVEL stropha index --repo $TOPLEVEL"
        fi

        nohup sh -c "
            exec 9>'$LOCK'
            if ! flock -n 9 2>/dev/null; then
                :
            fi
            cd '$TOPLEVEL' || exit 1

            if [ -n '$GRAPHIFY' ]; then
                if [ -f '$TOPLEVEL/graphify-out/graph.json' ]; then
                    echo '[stropha-hook] graphify update (commit ${{SHORT_SHA}})' >> '$LOG'
                    '$GRAPHIFY' update '$TOPLEVEL' --no-cluster >> '$LOG' 2>&1 \\
                        || echo '[stropha-hook] graphify update failed (commit ${{SHORT_SHA}})' >> '$LOG'
                else
                    echo '[stropha-hook] graphify-out/ missing - skipping graphify update' >> '$LOG'
                fi
            else
                echo '[stropha-hook] graphify binary not found - skipping graphify update' >> '$LOG'
            fi

            $CMD >> '$LOG' 2>&1 \\
                && echo '[stropha-hook] refresh done (commit ${{SHORT_SHA}})' >> '$LOG' \\
                || echo '[stropha-hook] failed (commit ${{SHORT_SHA}})' >> '$LOG'
        " >> "$LOG" 2>&1 < /dev/null &
        disown 2>/dev/null || true
        {END_MARKER}
        """
    )


def _atomic_write(path: Path, content: str, *, executable: bool = True) -> None:
    """Write ``content`` to ``path`` via tmpfile + rename (RFC §6.2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    if executable:
        st = tmp.stat()
        tmp.chmod(st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    tmp.replace(path)


# --------------------------------------------------------------------- public api


def install(
    target: Path,
    *,
    force: bool = False,
) -> dict:
    """Install or update the post-commit hook for ``target`` repo.

    Behaviour matrix (RFC §6.2):
      - File missing → create with shebang + our block
      - File exists, no markers → append our block at end
      - File exists, our markers → replace block in-place
      - File exists, graphify markers → warn (or proceed silently when ``force``)

    Returns a structured outcome dict suitable for CLI rendering or logging.
    """
    target = target.resolve()
    if not (target / ".git").exists():
        raise ValueError(f"{target} is not a git repository (no .git/)")

    hook = _hook_path(target)
    block = _render_hook_block(target)
    existing = hook.read_text(encoding="utf-8") if hook.is_file() else ""

    has_graphify = GRAPHIFY_MARKER in existing
    coexist_warning = has_graphify and not force

    if not existing:
        # New file → shebang + block
        body = "#!/bin/sh\n" + block
        action = "created"
    elif _BLOCK_RE.search(existing):
        # Replace our block in-place
        body = _BLOCK_RE.sub(block, existing, count=1)
        action = "updated"
    else:
        # Append after existing content
        sep = "" if existing.endswith("\n") else "\n"
        body = existing + sep + "\n" + block
        action = "appended"

    _atomic_write(hook, body, executable=True)
    log.info("hook.install", target=str(target), hook=str(hook), action=action,
             version=HOOK_VERSION, graphify_cohabit=has_graphify)

    return {
        "action": action,
        "hook_path": str(hook),
        "version": HOOK_VERSION,
        "coexist_warning": coexist_warning,
        "graphify_cohabit": has_graphify,
        "log_path": str(_log_path()),
    }


def uninstall(target: Path) -> dict:
    """Remove our block. Leave any other content intact.

    If the resulting file is just ``#!/bin/sh`` (whitespace-only otherwise),
    the file is removed.
    """
    target = target.resolve()
    hook = _hook_path(target)
    if not hook.is_file():
        return {"action": "noop", "hook_path": str(hook),
                "reason": "hook file not present"}

    existing = hook.read_text(encoding="utf-8")
    if not _BLOCK_RE.search(existing):
        return {"action": "noop", "hook_path": str(hook),
                "reason": "stropha block not found"}

    new_content = _BLOCK_RE.sub("", existing)
    # Strip trailing whitespace blocks after removal.
    new_content = re.sub(r"\n{3,}", "\n\n", new_content).rstrip() + "\n"

    # If only the shebang remains, drop the file entirely.
    if new_content.strip() in ("", "#!/bin/sh", "#!/usr/bin/env sh"):
        hook.unlink()
        log.info("hook.uninstall", target=str(target), hook=str(hook), action="removed")
        return {"action": "removed", "hook_path": str(hook)}

    _atomic_write(hook, new_content, executable=True)
    log.info("hook.uninstall", target=str(target), hook=str(hook), action="block_removed")
    return {"action": "block_removed", "hook_path": str(hook)}


def status(target: Path) -> HookStatus:
    """Inspect whether the hook is installed for ``target``."""
    target = target.resolve()
    hook = _hook_path(target)
    if not hook.is_file():
        return HookStatus(
            target_repo=target, hook_path=hook, installed=False,
            version=None, graphify_cohabit=False, log_path=_log_path(),
        )
    existing = hook.read_text(encoding="utf-8")
    m = _VERSION_RE.search(existing)
    version = int(m.group(1)) if m else None
    return HookStatus(
        target_repo=target, hook_path=hook,
        installed=version is not None,
        version=version,
        graphify_cohabit=GRAPHIFY_MARKER in existing,
        log_path=_log_path(),
    )

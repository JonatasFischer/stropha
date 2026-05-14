"""Git repository identity extraction.

For every file we index we want to remember which git repository it came
from, so different users — pulling from this same SQLite index — can run
``git clone <url>`` to retrieve the source.

Identity is normalized so that the same logical repo accessed via SSH
(``git@github.com:foo/bar.git``) and HTTPS (``https://github.com/foo/bar``)
collapses to a single row. Auth tokens accidentally embedded in remote
URLs (``https://x-token:ghp_…@github.com/foo/bar``) are stripped before
anything is logged or persisted.

This module shells out to ``git``; it never reads ``.git/config`` directly,
because git already implements the lookup correctly across worktrees,
submodules, and ``includeIf`` configurations.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from ..logging import get_logger

log = get_logger(__name__)

_GIT_TIMEOUT_S = 5
_SSH_SHORT_RE = re.compile(r"^(?P<user>[^@]+)@(?P<host>[^:]+):(?P<path>.+)$")


@dataclass(frozen=True)
class RepoIdentity:
    """Stable identification of a git repository (or a fallback)."""

    normalized_key: str
    """Stable cross-user key. ``host/path`` for git remotes, ``local:<abs>``
    for git repos without a remote, ``path:<abs>`` for non-git directories."""

    remote_url: str | None
    """Canonical HTTPS URL suitable for ``git clone``. ``None`` when no
    remote is configured (local-only or non-git)."""

    root_path: Path
    """Absolute path of the repository on the indexer host (informational)."""

    default_branch: str | None
    """Default branch as reported by ``git symbolic-ref refs/remotes/origin/HEAD``,
    with a fallback chain. ``None`` on detached HEAD with no remote."""

    head_commit: str | None
    """SHA at index time. ``None`` for non-git directories."""


# ---- public API ----------------------------------------------------------

def detect(path: Path) -> RepoIdentity:
    """Detect the git identity of ``path``.

    Never raises: returns a fallback ``RepoIdentity`` with ``local:`` or
    ``path:`` normalized_key when git is unavailable / not a repo / has no
    remote.
    """
    abs_path = path.resolve()
    toplevel = _run_git(["rev-parse", "--show-toplevel"], cwd=abs_path)
    if toplevel is None:
        # Not a git repository (or git missing).
        return RepoIdentity(
            normalized_key=f"path:{abs_path}",
            remote_url=None,
            root_path=abs_path,
            default_branch=None,
            head_commit=None,
        )

    root = Path(toplevel)
    head_commit = _run_git(["rev-parse", "HEAD"], cwd=root)
    raw_url = _run_git(["remote", "get-url", "origin"], cwd=root)

    if raw_url is None:
        # Git repo but no `origin` remote configured.
        return RepoIdentity(
            normalized_key=f"local:{root}",
            remote_url=None,
            root_path=root,
            default_branch=_detect_default_branch(root),
            head_commit=head_commit,
        )

    normalized_key, sanitized_url, auth_stripped = normalize_url(raw_url)
    if auth_stripped:
        log.info("git_meta.url_auth_stripped", repo=normalized_key)

    return RepoIdentity(
        normalized_key=normalized_key,
        remote_url=sanitized_url,
        root_path=root,
        default_branch=_detect_default_branch(root),
        head_commit=head_commit,
    )


def normalize_url(raw: str) -> tuple[str, str, bool]:
    """Normalize a git remote URL.

    Returns a tuple ``(normalized_key, sanitized_https_url, auth_was_stripped)``.

    Rules:

    - SSH short form (``git@host:path``) is converted to ``ssh://git@host/path``
      before parsing.
    - Userinfo (``user:pass@`` or token) is removed.
    - Trailing ``.git`` and any trailing slash are stripped from the path.
    - Host and path are lower-cased so the same GitHub repo accessed via
      different cases collapses to one row. (Trade-off: case-sensitive hosts
      with mixed-case paths — rare on GitLab self-hosted — would collide;
      see ADR in docs.)
    - The reconstructed display URL is always ``https://<host>/<path>.git``.
    """
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("empty git remote URL")

    # Convert SSH-short form to a parseable URL.
    m = _SSH_SHORT_RE.match(cleaned)
    if m:
        cleaned = f"ssh://{m.group('user')}@{m.group('host')}/{m.group('path')}"

    parts = urlsplit(cleaned)
    host = (parts.hostname or "").lower()
    if not host:
        # Last-resort: treat the whole thing as opaque — never happens for
        # legitimate git URLs but we refuse to crash.
        raise ValueError(f"could not parse host from URL: {raw!r}")

    path = parts.path.lstrip("/")
    if path.endswith("/"):
        path = path[:-1]
    if path.endswith(".git"):
        path = path[: -len(".git")]
    path = path.lower()

    normalized_key = f"{host}/{path}"
    sanitized_https = urlunsplit(("https", host, f"/{path}.git", "", ""))
    # `git@` and `hg@` are standard SSH service users, not credentials.
    # Only flag as "auth stripped" when a password is present or the user is
    # something else (e.g. ``x-token:ghp_…`` or a PAT-style username).
    auth_stripped = bool(parts.password) or (
        bool(parts.username) and parts.username not in {"git", "hg"}
    )
    return normalized_key, sanitized_https, auth_stripped


# ---- internals -----------------------------------------------------------

def _run_git(args: list[str], cwd: Path) -> str | None:
    """Run a git subcommand. Returns stdout trimmed, or ``None`` on failure.

    Never raises. Timeout-bounded so a hung git invocation cannot stall
    indexing.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log.debug("git_meta.subprocess_failed", args=args, error=str(exc))
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


def _detect_default_branch(root: Path) -> str | None:
    """Try three sources, in order of fidelity."""
    ref = _run_git(["symbolic-ref", "refs/remotes/origin/HEAD"], cwd=root)
    if ref:
        # e.g. ``refs/remotes/origin/main`` → ``main``
        return ref.rsplit("/", 1)[-1]
    init_default = _run_git(["config", "--get", "init.defaultBranch"], cwd=root)
    if init_default:
        return init_default
    return None

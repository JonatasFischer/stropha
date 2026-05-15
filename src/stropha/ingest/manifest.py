"""Repo manifest loader — declarative multi-repo lists (Phase 4).

Lets users replace ``stropha index --repo /a --repo /b --repo /c`` with
a single YAML file that documents the corpus shape. Useful for:

- Multi-repo umbrella projects ("our entire stack lives in these 8 repos")
- Reproducible CI runs (the manifest is the contract)
- Onboarding ("here's the file — clone these and `stropha index --manifest`")

Schema:

```yaml
# repos.yaml
repos:
  - path: /Users/jonatas/sources/stropha
  - path: ~/sources/another-repo
    enabled: true            # default true; set false to skip
  - path: ../sibling-repo    # relative paths resolved from the manifest dir
```

Anything beyond ``path`` and ``enabled`` is ignored today (forward-compat
slot for branch overrides, alias names, per-repo enricher config, …).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..errors import StrophaError
from ..logging import get_logger

log = get_logger(__name__)


class ManifestError(StrophaError):
    """Failure parsing or validating a repo manifest."""


@dataclass(frozen=True)
class ManifestEntry:
    path: Path
    enabled: bool = True


def load_manifest(path: Path) -> list[ManifestEntry]:
    """Parse a repo manifest. Returns the **enabled** entries in declaration order.

    Raises :class:`ManifestError` for unreadable or malformed files. Each
    entry's ``path`` is resolved against the manifest's directory so the
    file can use relative paths.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ManifestError(f"Manifest not found: {path}")

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"Manifest is not valid YAML: {exc}") from exc

    if payload is None:
        raise ManifestError(f"Manifest is empty: {path}")

    if not isinstance(payload, dict) or "repos" not in payload:
        raise ManifestError(
            f"Manifest must be a mapping with a top-level `repos:` list ({path})"
        )

    raw_repos = payload.get("repos") or []
    if not isinstance(raw_repos, list):
        raise ManifestError(f"`repos:` must be a list (got {type(raw_repos).__name__})")

    base = path.parent
    entries: list[ManifestEntry] = []
    for i, raw in enumerate(raw_repos, start=1):
        # Two shapes: bare string or mapping with `path:`.
        if isinstance(raw, str):
            repo_path = Path(raw).expanduser()
            enabled = True
        elif isinstance(raw, dict):
            if "path" not in raw:
                raise ManifestError(
                    f"Repo entry #{i} in {path} missing required `path:` key"
                )
            repo_path = Path(str(raw["path"])).expanduser()
            enabled = bool(raw.get("enabled", True))
        else:
            raise ManifestError(
                f"Repo entry #{i} in {path} must be a string or mapping "
                f"(got {type(raw).__name__})"
            )
        if not repo_path.is_absolute():
            repo_path = (base / repo_path).resolve()
        else:
            repo_path = repo_path.resolve()
        entries.append(ManifestEntry(path=repo_path, enabled=enabled))

    enabled_entries = [e for e in entries if e.enabled]
    log.info(
        "manifest.loaded",
        path=str(path),
        total=len(entries),
        enabled=len(enabled_entries),
        skipped=len(entries) - len(enabled_entries),
    )
    return enabled_entries

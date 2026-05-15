"""Pipeline configuration loader.

Cascading precedence (per ``docs/architecture/stropha-pipeline-adapters.md`` §5):

    CLI overrides  >  Env vars  >  YAML file(s)  >  Built-in defaults

YAML lookup order: ``./stropha.yaml`` (project) then ``~/.stropha/config.yaml``
(user). The project file wins on key conflict. Both are optional — no YAML
means we fall back to env-only behavior identical to v0.1.0.

Env-var convention: ``STROPHA_<STAGE>__<KEY>__<SUBKEY>`` (``__`` separates
levels). Legacy v0.1.0 vars (``STROPHA_INDEX_PATH``, ``STROPHA_LOCAL_EMBED_MODEL``,
…) keep working as aliases per ADR-005.

Phase 1 scope: only the ``embedder`` and ``enricher`` sections are
adapter-aware. Other stages still come from ``stropha.config.Config``
until Phases 2/3 migrate them.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from ..errors import ConfigError
from ..logging import get_logger

log = get_logger(__name__)

# Top-level config skeleton. Adapter sections that omit ``adapter`` fall back
# to these defaults. Adding a stage here in a future phase is the only place
# the framework needs touching.
_DEFAULTS: dict[str, dict[str, Any]] = {
    "walker": {"adapter": "git-ls-files", "config": {}},
    "chunker": {"adapter": "tree-sitter-dispatch", "config": {}},
    "enricher": {"adapter": "noop", "config": {}},
    "embedder": {"adapter": None, "config": {}},  # `None` => auto (voyage if key, else local)
    "storage": {"adapter": "sqlite-vec", "config": {}},
    "retrieval": {"adapter": "hybrid-rrf", "config": {}},
}

# Env var → (stage, dotted key inside `config`). When dotted key is empty
# the env var sets the adapter name itself.
_LEGACY_ALIASES: dict[str, tuple[str, str]] = {
    "STROPHA_LOCAL_EMBED_MODEL": ("embedder", "model"),  # only honored when adapter=local
    "STROPHA_VOYAGE_EMBED_MODEL": ("embedder", "model"),  # only honored when adapter=voyage
    "STROPHA_VOYAGE_EMBED_DIM": ("embedder", "dim"),     # only honored when adapter=voyage
    "STROPHA_INDEX_PATH": ("storage", "path"),
    "STROPHA_MAX_FILE_BYTES": ("walker", "max_file_bytes"),
}


def load_pipeline_config(
    *,
    project_root: Path | None = None,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the full pipeline config dict.

    Args:
        project_root: dir to look for ``stropha.yaml`` (defaults to cwd).
        overrides: CLI-level overrides, deep-merged last (highest priority).
        environ: env-var snapshot (defaults to ``os.environ``); injected to
                 make tests deterministic.

    Returns the resolved config in the form ``{"embedder": {...}, "enricher": {...}}``,
    where each section has ``adapter`` (str|None) and ``config`` (dict).
    """
    env = environ if environ is not None else dict(os.environ)
    cfg = copy.deepcopy(_DEFAULTS)
    sources: list[str] = []

    # 1. User YAML (~/.stropha/config.yaml)
    user_yaml = Path.home() / ".stropha" / "config.yaml"
    if user_yaml.is_file():
        _merge(cfg, _read_yaml(user_yaml))
        sources.append(str(user_yaml))

    # 2. Project YAML (./stropha.yaml)
    project = project_root or Path.cwd()
    project_yaml = project / "stropha.yaml"
    if project_yaml.is_file():
        _merge(cfg, _read_yaml(project_yaml))
        sources.append(str(project_yaml))

    # 3. Env vars (new format)
    env_overlay = _env_to_dict(env)
    if env_overlay:
        _merge(cfg, env_overlay)
        sources.append("env")

    # 4. CLI overrides (applied early so they win the adapter-auto-resolution).
    if overrides:
        _merge(cfg, overrides)
        sources.append("cli")

    # 5. Resolve embedder=None (auto pick voyage vs local). Must happen BEFORE
    # legacy alias resolution because the aliases pick a target based on the
    # active adapter.
    if cfg["embedder"].get("adapter") is None:
        if env.get("VOYAGE_API_KEY", "").strip():
            cfg["embedder"]["adapter"] = "voyage"
        else:
            cfg["embedder"]["adapter"] = "local"

    # 6. Legacy env aliases — applied LAST so they only patch the now-known
    # adapter (e.g. STROPHA_VOYAGE_EMBED_MODEL only fires when adapter='voyage').
    legacy_overlay = _legacy_env_to_dict(env, cfg)
    if legacy_overlay:
        _merge(cfg, legacy_overlay)
        sources.append("env(legacy)")

    log.debug("pipeline.config.load", sources=sources, resolved=cfg)
    return cfg


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    # Accept either {"pipeline": {...}} (canonical) or the bare sections.
    inner = raw.get("pipeline", raw)
    if not isinstance(inner, dict):
        raise ConfigError(f"{path}: `pipeline` must be a mapping")
    return inner


def _env_to_dict(env: dict[str, str]) -> dict[str, Any]:
    """Translate ``STROPHA_<STAGE>__<KEY>__...`` into nested dict overlay.

    ``STROPHA_<STAGE>`` (no double underscore) sets the adapter name.
    """
    out: dict[str, Any] = {}
    for raw_key, raw_val in env.items():
        if not raw_key.startswith("STROPHA_"):
            continue
        body = raw_key[len("STROPHA_"):]
        if not body:
            continue
        # Reject the legacy single-underscore form by checking if the body
        # looks like a known top-level stage with __ separator.
        if "__" not in body:
            # Could be `STROPHA_EMBEDDER` (adapter name) or any unrelated env.
            stage = body.lower()
            if stage in _DEFAULTS:
                out.setdefault(stage, {})["adapter"] = raw_val
            continue
        parts = body.split("__")
        stage = parts[0].lower()
        if stage not in _DEFAULTS:
            continue  # not ours — skip silently
        # Remaining parts become the dotted path into `config`.
        node = out.setdefault(stage, {}).setdefault("config", {})
        for p in parts[1:-1]:
            node = node.setdefault(p.lower(), {})
        last = parts[-1].lower()
        node[last] = _coerce(raw_val)
    return out


def _legacy_env_to_dict(env: dict[str, str], current: dict[str, Any]) -> dict[str, Any]:
    """Map deprecated env vars onto the new shape. Logs a warning.

    Caller MUST resolve adapter auto-selection before invoking this — the
    aliases are routed strictly by ``current[stage]['adapter']`` to avoid
    cross-adapter contamination (e.g. ``STROPHA_VOYAGE_EMBED_MODEL`` MUST
    NOT leak into a ``local`` adapter's config).
    """
    out: dict[str, Any] = {}
    for legacy_var, (stage, dotted) in _LEGACY_ALIASES.items():
        if legacy_var not in env:
            continue
        active = current.get(stage, {}).get("adapter")
        if legacy_var == "STROPHA_LOCAL_EMBED_MODEL" and active != "local":
            continue
        if legacy_var in ("STROPHA_VOYAGE_EMBED_MODEL", "STROPHA_VOYAGE_EMBED_DIM"):
            if active != "voyage":
                continue
        node = out.setdefault(stage, {}).setdefault("config", {})
        for p in dotted.split(".")[:-1]:
            node = node.setdefault(p, {})
        node[dotted.split(".")[-1]] = _coerce(env[legacy_var])
        log.info(
            "pipeline.config.legacy_alias",
            from_var=legacy_var,
            to_path=f"pipeline.{stage}.config.{dotted}",
        )
    return out


def _coerce(value: str) -> Any:
    """Cheap coercion for env-var strings to int/float/bool/str."""
    lo = value.strip().lower()
    if lo in ("true", "yes", "on"):
        return True
    if lo in ("false", "no", "off"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Deep merge ``src`` into ``dst`` in-place."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v

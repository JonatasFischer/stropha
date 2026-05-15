"""Adapter registry — populated at import-time via the ``@register_adapter`` decorator.

Per ADR-003 (``docs/architecture/stropha-pipeline-adapters.md``): adding a new
adapter is adding a single file under ``src/stropha/adapters/<stage>/``. The
package's ``__init__`` walks every submodule on import, which triggers the
decorator side-effect that populates ``_REGISTRY``.

Imports are intentionally cheap: heavy work (ONNX model load, HTTP clients)
happens in the adapter's ``__init__``, not at module import.
"""

from __future__ import annotations

from typing import Any

from ..errors import AdapterError, ConfigError

# (stage_name, adapter_name) → adapter class
_REGISTRY: dict[tuple[str, str], type[Any]] = {}


def register_adapter(*, stage: str, name: str):
    """Class decorator. Registers ``cls`` under ``(stage, name)``.

    The adapter class MUST expose a ``Config`` class attribute (a pydantic
    ``BaseModel`` subtype) so the builder can validate user-supplied config.
    """

    def deco(cls: type[Any]) -> type[Any]:
        if not hasattr(cls, "Config"):
            raise AdapterError(
                f"{cls.__module__}.{cls.__name__} is missing the `Config` "
                f"class attribute required by register_adapter"
            )
        key = (stage, name)
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise AdapterError(
                f"Duplicate adapter registration for {stage!r}/{name!r}: "
                f"{_REGISTRY[key].__module__} vs {cls.__module__}"
            )
        _REGISTRY[key] = cls
        return cls

    return deco


def lookup_adapter(stage: str, name: str) -> type[Any]:
    """Return the adapter class registered for ``(stage, name)``.

    Raises ``ConfigError`` with the list of valid alternatives when the
    requested adapter does not exist.
    """
    cls = _REGISTRY.get((stage, name))
    if cls is None:
        available = available_for_stage(stage)
        hint = f"Available {stage} adapters: {available}" if available else (
            f"No {stage} adapters registered. Did you forget to import "
            f"`stropha.adapters` to trigger auto-registration?"
        )
        raise ConfigError(f"Unknown {stage} adapter {name!r}. {hint}")
    return cls


def available_for_stage(stage: str) -> list[str]:
    """Return sorted adapter names registered for ``stage``."""
    return sorted(n for s, n in _REGISTRY if s == stage)


def all_adapters() -> dict[str, list[str]]:
    """Snapshot of the registry, grouped by stage. Used by ``adapters list``."""
    out: dict[str, list[str]] = {}
    for s, n in _REGISTRY:
        out.setdefault(s, []).append(n)
    for v in out.values():
        v.sort()
    return out


def _reset_for_tests() -> None:
    """Test helper: clear the registry. Do NOT call from production code."""
    _REGISTRY.clear()

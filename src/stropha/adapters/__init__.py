"""Concrete adapters for each pipeline stage.

Per ADR-003 (``docs/architecture/stropha-pipeline-adapters.md``): importing
this package walks every submodule below, which triggers the
``@register_adapter`` decorator side-effect that populates the registry.

Doing the walk here (rather than at app start) makes the registry available
to any caller that imports ``stropha.adapters``, including tests that
exercise the registry directly.

Import is intentionally cheap — adapter modules MUST NOT eagerly load
ONNX models, open HTTP clients, or read disk in their top-level code.
"""

from __future__ import annotations

import importlib
import pkgutil


def _autoload() -> None:
    for _, modname, _ in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
        importlib.import_module(modname)


_autoload()

"""Pipeline framework.

Exports the small surface that callers (CLI, server, tests) depend on. The
concrete adapters live under ``stropha.adapters`` — see
``docs/architecture/stropha-pipeline-adapters.md`` §7 for the layout.
"""

from .base import Stage, StageContext, StageHealth
from .builder import BuiltStages, build_stages
from .config import load_pipeline_config
from .pipeline import Pipeline, PipelineStats, RepoStats
from .registry import (
    all_adapters,
    available_for_stage,
    lookup_adapter,
    register_adapter,
)

__all__ = [
    "Stage",
    "StageContext",
    "StageHealth",
    "BuiltStages",
    "build_stages",
    "load_pipeline_config",
    "Pipeline",
    "PipelineStats",
    "RepoStats",
    "register_adapter",
    "lookup_adapter",
    "available_for_stage",
    "all_adapters",
]

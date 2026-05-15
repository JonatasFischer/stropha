"""Stage protocol + shared types.

Per ``docs/architecture/stropha-pipeline-adapters.md`` §3: every pipeline
stage (walker, chunker, enricher, embedder, storage, retrieval) implements
a small protocol so adapters can be swapped via configuration without
touching callers.

This module is intentionally framework-only: no concrete adapters here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel


@dataclass
class StageContext:
    """Cross-cutting context every stage may consult.

    Stages NEVER mutate the context. The pipeline (``pipeline/pipeline.py``)
    builds a fresh context per item where appropriate (e.g. one per chunk
    when calling the enricher).
    """

    repo_key: str | None = None
    """Normalized repo key of the current item (for adapters that vary per repo)."""

    parent_chunk: Any | None = None
    """The chunk whose ``chunk_id`` equals ``current_chunk.parent_chunk_id``.

    Used by the hierarchical enricher to prepend a parent skeleton. None when
    the chunk has no parent or the parent could not be located.
    """

    file_content: str | None = None
    """Full original file body. Provided to LLM-based enrichers that need to
    look beyond the chunk window."""

    pipeline_meta: dict[str, str] = field(default_factory=dict)
    """Free-form key/value bag for inter-stage communication."""


@dataclass
class StageHealth:
    """Result of a lightweight readiness probe.

    ``status='warning'`` is non-fatal — the adapter is reachable but a
    dependency (model file, API key, daemon) is in a degraded state.
    ``status='error'`` blocks pipeline construction.
    """

    status: Literal["ready", "warning", "error"]
    message: str
    detail: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class Stage(Protocol):
    """One responsibility in the pipeline. Exactly one adapter active per run.

    Adapters declare ``Config = SomePydanticModel`` as a class attribute and
    are registered via ``@register_adapter(stage=..., name=...)``.

    The instance properties below are the only public surface the pipeline
    and CLI rely on. Concrete adapters add stage-specific methods (e.g.
    ``EnricherStage.enrich``, ``EmbedderStage.embed_documents``).
    """

    @property
    def stage_name(self) -> str:
        """``'walker'`` | ``'chunker'`` | ``'enricher'`` | ``'embedder'`` | ``'storage'`` | ``'retrieval'``."""
        ...

    @property
    def adapter_name(self) -> str:
        """Short identifier, unique per stage (``'noop'``, ``'hierarchical'``, …)."""
        ...

    @property
    def adapter_id(self) -> str:
        """Stable, fully-qualified id including model/config used for cache keys.

        Examples: ``'noop'``, ``'hierarchical'``,
        ``'ollama:qwen2.5-coder:14b'``, ``'voyage:voyage-code-3:512'``.

        Per ADR-008: changing the model component of an adapter_id is a
        cache miss — it forces re-processing of affected rows.
        """
        ...

    @property
    def config_schema(self) -> type[BaseModel]:
        """Pydantic model describing this adapter's config section."""
        ...

    def health(self) -> StageHealth:
        """Lightweight readiness probe, ≤2 s, non-blocking.

        Called by ``stropha pipeline validate`` and (warning-level) before
        ``stropha index``. MAY perform a tiny network call (e.g. Ollama
        ``GET /api/tags``) but MUST timeout fast.
        """
        ...

"""Builder: resolved config dict → instantiated adapter objects.

Importing this module triggers ``stropha.adapters`` auto-registration so
``lookup_adapter`` resolves immediately.

Build order (Phase 2):
    walker → enricher → embedder → storage(embedder.dim) → retrieval(storage, embedder)

The chunker stage (Phase 3) is built lazily by the Pipeline using the
legacy ``stropha.ingest.chunker.Chunker`` until §10.3 ships its adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import adapters  # noqa: F401  — side-effect: populate registry
from ..errors import ConfigError
from ..logging import get_logger
from ..stages.chunker import ChunkerStage
from ..stages.embedder import EmbedderStage
from ..stages.enricher import EnricherStage
from ..stages.retrieval import RetrievalStage
from ..stages.storage import StorageStage
from ..stages.walker import WalkerStage
from .registry import lookup_adapter

log = get_logger(__name__)


@dataclass
class BuiltStages:
    """Set of constructed adapters returned by :func:`build_stages`.

    All 6 stages are adapter-aware after Phase 3. The Pipeline / CLI
    consumes this dataclass directly — never reaches into the registry.
    """

    walker: WalkerStage
    chunker: ChunkerStage
    enricher: EnricherStage
    embedder: EmbedderStage
    storage: StorageStage
    retrieval: RetrievalStage
    resolved: dict[str, Any]


def build_stages(
    resolved_config: dict[str, Any],
    *,
    open_storage: bool = True,
) -> BuiltStages:
    """Instantiate every adapter named in ``resolved_config``.

    ``open_storage=False`` skips storage + retrieval construction and
    returns ``None`` for those slots. Useful for ``stropha pipeline show``
    when the user has not yet populated an index file (avoids opening a
    SQLite connection just to print config).
    """
    walker = _build_simple(resolved_config, "walker")
    chunker = _build_simple(resolved_config, "chunker")
    enricher = _build_simple(resolved_config, "enricher")
    embedder = _build_simple(resolved_config, "embedder")

    storage: StorageStage | None = None
    retrieval: RetrievalStage | None = None
    if open_storage:
        storage_section = resolved_config.get("storage") or {}
        storage_name = storage_section.get("adapter") or "sqlite-vec"
        storage_cls = lookup_adapter("storage", storage_name)
        try:
            storage_cfg = storage_cls.Config(**(storage_section.get("config") or {}))
        except Exception as exc:
            raise ConfigError(
                f"Invalid config for pipeline.storage adapter={storage_name!r}: {exc}"
            ) from exc
        storage = storage_cls(storage_cfg, embedding_dim=embedder.dim)
        log.info(
            "pipeline.adapter.built",
            stage="storage",
            adapter=storage_name,
            adapter_id=storage.adapter_id,
        )

        retrieval_section = resolved_config.get("retrieval") or {}
        retrieval_name = retrieval_section.get("adapter") or "hybrid-rrf"
        retrieval_cls = lookup_adapter("retrieval", retrieval_name)
        try:
            retrieval_cfg = retrieval_cls.Config(**(retrieval_section.get("config") or {}))
        except Exception as exc:
            raise ConfigError(
                f"Invalid config for pipeline.retrieval adapter={retrieval_name!r}: {exc}"
            ) from exc
        retrieval = retrieval_cls(retrieval_cfg, storage=storage, embedder=embedder)
        log.info(
            "pipeline.adapter.built",
            stage="retrieval",
            adapter=retrieval_name,
            adapter_id=retrieval.adapter_id,
        )

    return BuiltStages(
        walker=walker,
        chunker=chunker,
        enricher=enricher,
        embedder=embedder,
        storage=storage,  # type: ignore[arg-type]
        retrieval=retrieval,  # type: ignore[arg-type]
        resolved=resolved_config,
    )


def _build_simple(resolved_config: dict[str, Any], stage: str) -> Any:
    """Build a stage that needs no cross-stage dependency injection."""
    section = resolved_config.get(stage) or {}
    adapter_name = section.get("adapter")
    if not adapter_name:
        raise ConfigError(f"pipeline.{stage}.adapter is required")
    cls = lookup_adapter(stage, adapter_name)
    config_payload = section.get("config") or {}
    try:
        cfg_obj = cls.Config(**config_payload)
    except Exception as exc:
        raise ConfigError(
            f"Invalid config for pipeline.{stage} adapter={adapter_name!r}: {exc}"
        ) from exc
    try:
        instance = cls(cfg_obj)
    except TypeError:
        instance = cls()
    log.info(
        "pipeline.adapter.built",
        stage=stage,
        adapter=adapter_name,
        adapter_id=getattr(instance, "adapter_id", "?"),
    )
    return instance

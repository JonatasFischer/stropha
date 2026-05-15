"""EnricherStage protocol — transform a chunk before embedding.

Phase 1 introduces two adapters:

- ``noop``         — identity (default; preserves current behavior).
- ``hierarchical`` — prepends a small skeleton of the parent chunk
                     (class header for a method, file header for a class, …).

Future adapters (Phase 2/3): ``ollama``, ``anthropic``, ``mlx``, ``openai`` —
LLM-generated contextual prefixes per the Anthropic "Contextual Retrieval"
technique (``docs/architecture/stropha-system.md`` §3.4).

Drift is detected via the ``adapter_id`` stored on each chunk row
(``chunks.enricher_id``). When the active enricher's id differs from the
stored id, the pipeline re-embeds the chunk transparently — no
``--rebuild`` required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import Chunk
from ..pipeline.base import StageContext, StageHealth


@runtime_checkable
class EnricherStage(Protocol):
    """Returns the text that should be fed to the embedder for ``chunk``.

    The original ``chunk.content`` is preserved on the chunk row (so a
    ``Read`` of a search hit always shows source-truth). Only the
    embedded representation changes.
    """

    # ----- Stage contract --------------------------------------------------

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    # ----- Enricher-specific surface --------------------------------------

    def enrich(self, chunk: Chunk, ctx: StageContext) -> str:
        """Return the text to embed.

        MUST be deterministic for a given ``(chunk, ctx, adapter_config)``
        triplet — non-determinism breaks the drift-detection cache.

        MAY consult ``ctx.parent_chunk`` and ``ctx.file_content``. MUST
        tolerate both being ``None``.
        """

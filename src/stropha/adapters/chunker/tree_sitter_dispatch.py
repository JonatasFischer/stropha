"""Chunker dispatcher — picks the right per-language sub-adapter for each file.

This is the canonical example of a "sub-pipeline as adapter" per
``docs/architecture/stropha-pipeline-adapters.md`` §4.2:

```yaml
chunker:
  adapter: tree-sitter-dispatch
  config:
    languages:
      java:           { adapter: ast-generic, config: { language: java } }
      typescript:     { adapter: ast-generic, config: { language: typescript } }
      python:         { adapter: ast-generic, config: { language: python } }
      markdown:       { adapter: heading-split }
      vue:            { adapter: sfc-split }
      gherkin:        { adapter: regex-feature-scenario }
      _fallback:      { adapter: file-level }
```

When a section is omitted from YAML the dispatcher falls back to the
built-in registry below — same shape stropha shipped before Phase 3.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from typing import Any

from pydantic import BaseModel, Field

from ...errors import ChunkerError
from ...ingest.chunkers.base import make_chunk_id
from ...logging import get_logger
from ...models import Chunk, SourceFile
from ...pipeline.base import StageHealth
from ...pipeline.registry import lookup_adapter, register_adapter
from ...stages.chunker import LanguageChunkerStage

log = get_logger(__name__)


# Default per-language wiring. Mirrors `ingest/chunker.py:_build_registry`
# so behavior with no config is unchanged.
_DEFAULT_LANGUAGES: dict[str, dict[str, Any]] = {
    "java":       {"adapter": "ast-generic", "config": {"language": "java"}},
    "typescript": {"adapter": "ast-generic", "config": {"language": "typescript"}},
    "javascript": {"adapter": "ast-generic", "config": {"language": "javascript"}},
    "python":     {"adapter": "ast-generic", "config": {"language": "python"}},
    "rust":       {"adapter": "ast-generic", "config": {"language": "rust"}},
    "go":         {"adapter": "ast-generic", "config": {"language": "go"}},
    "kotlin":     {"adapter": "ast-generic", "config": {"language": "kotlin"}},
    "vue":        {"adapter": "sfc-split"},
    "markdown":   {"adapter": "heading-split"},
    "gherkin":    {"adapter": "regex-feature-scenario"},
    "_fallback":  {"adapter": "file-level"},
}


class TreeSitterDispatchConfig(BaseModel):
    """Sub-pipeline config: each language → ``{adapter, config}``."""

    languages: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Map of source language → sub-adapter spec. "
            "Use the special key '_fallback' to override the catch-all. "
            "Keys omitted here inherit the built-in defaults."
        ),
    )


@register_adapter(stage="chunker", name="tree-sitter-dispatch")
class TreeSitterDispatchChunker:
    """Stage adapter — dispatches per file to a language sub-adapter."""

    Config = TreeSitterDispatchConfig

    def __init__(self, config: TreeSitterDispatchConfig | None = None) -> None:
        self._config = config or TreeSitterDispatchConfig()
        # Merge defaults with user overrides.
        merged: dict[str, dict[str, Any]] = {**_DEFAULT_LANGUAGES, **self._config.languages}
        self._language_specs = merged
        self._registry = self._instantiate_sub_adapters(merged)

    # ----- Stage contract -------------------------------------------------

    @property
    def stage_name(self) -> str:
        return "chunker"

    @property
    def adapter_name(self) -> str:
        return "tree-sitter-dispatch"

    @property
    def adapter_id(self) -> str:
        # Encode the resolved language map so changing any sub-adapter
        # surfaces as a different adapter_id.
        canonical = sorted(
            (lang, sub.get("adapter", "?")) for lang, sub in self._language_specs.items()
        )
        digest = hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()[:8]
        return f"tree-sitter-dispatch:{digest}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return TreeSitterDispatchConfig

    def health(self) -> StageHealth:
        n = len(self._registry)
        return StageHealth(
            status="ready",
            message=f"dispatcher with {n} language sub-adapters",
            detail={lang: a.adapter_id for lang, a in self._registry.items()},
        )

    # ----- Chunker surface -----------------------------------------------

    def chunk(
        self,
        files: Iterable[SourceFile],
        *,
        repo_key: str | None = None,
    ) -> Iterator[Chunk]:
        fallback = self._registry.get("_fallback")
        for sf in files:
            try:
                content = sf.path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise ChunkerError(f"Cannot read {sf.rel_path}: {exc}") from exc
            if not content.strip():
                continue
            sub = self._registry.get(sf.language) or fallback
            if sub is None:
                log.warning("chunker.no_fallback", path=sf.rel_path)
                continue
            try:
                emitted = sub.chunk(sf, content)
                emitted = list(emitted)
            except Exception as exc:
                log.warning(
                    "chunker.error_fallback",
                    path=sf.rel_path,
                    language=sf.language,
                    error=str(exc),
                )
                if fallback is None:
                    continue
                emitted = list(fallback.chunk(sf, content))
            for chunk in emitted:
                yield self._with_repo_key(chunk, repo_key)

    # ----- helpers --------------------------------------------------------

    def _instantiate_sub_adapters(
        self, mapping: dict[str, dict[str, Any]]
    ) -> dict[str, LanguageChunkerStage]:
        out: dict[str, LanguageChunkerStage] = {}
        for lang, spec in mapping.items():
            adapter_name = spec.get("adapter")
            if not adapter_name:
                continue
            cls = lookup_adapter("language-chunker", adapter_name)
            cfg_payload = spec.get("config") or {}
            try:
                cfg_obj = cls.Config(**cfg_payload)
            except Exception as exc:
                raise ChunkerError(
                    f"Invalid config for chunker.languages.{lang} "
                    f"(adapter={adapter_name!r}): {exc}"
                ) from exc
            try:
                instance = cls(cfg_obj)
            except TypeError:
                instance = cls()
            out[lang] = instance
            log.debug(
                "chunker.sub_adapter.built",
                language=lang,
                adapter=adapter_name,
                adapter_id=instance.adapter_id,
            )
        return out

    @staticmethod
    def _with_repo_key(chunk: Chunk, repo_key: str | None) -> Chunk:
        if not repo_key:
            return chunk
        new_id = make_chunk_id(
            chunk.rel_path,
            chunk.start_line,
            chunk.end_line,
            chunk.content_hash,
            repo_key=repo_key,
        )
        if new_id == chunk.chunk_id:
            return chunk
        return chunk.model_copy(update={"chunk_id": new_id})

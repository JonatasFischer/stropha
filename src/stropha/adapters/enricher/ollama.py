"""Ollama LLM enricher — calls a local model to generate a one-line summary
prepended to ``embedding_text``.

Status: **alpha**. The protocol contract and registry wiring are complete;
expect to tune the prompt + model defaults on real workloads. Drift detection
already covers it (``adapter_id`` digests model + prompt + temperature) so
config tweaks force re-enrichment.

Why local Ollama: no API key, no per-token cost, runs on the developer's
machine. Default model ``qwen2.5-coder:1.5b`` is small enough to enrich
tens of thousands of chunks in minutes on Apple Silicon. Bigger models
(``qwen2.5-coder:7b``, ``llama3.1:8b``) are a config swap.

Failure semantics: any HTTP error, timeout, or model-unavailable returns
the raw ``chunk.content`` — the index keeps building, the warning surfaces
in the log + ``stropha pipeline validate``. Never blocks indexing.
"""

from __future__ import annotations

import hashlib
import json
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import BaseModel, Field

from ...models import Chunk
from ...pipeline.base import StageContext, StageHealth
from ...pipeline.registry import register_adapter
from ...stages.enricher import EnricherStage

_DEFAULT_PROMPT = (
    "Summarise the following code in ONE concise sentence (≤ 25 words). "
    "Focus on what it does, not how. Return only the sentence, no preamble.\n\n"
    "```\n{content}\n```"
)


class OllamaEnricherConfig(BaseModel):
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama HTTP endpoint. ``OLLAMA_HOST`` env var is the usual override.",
    )
    model: str = Field(
        default="qwen2.5-coder:1.5b",
        description="Ollama model tag. Run `ollama pull <model>` first.",
    )
    prompt_template: str = Field(
        default=_DEFAULT_PROMPT,
        description="Format string with `{content}` placeholder.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_s: float = Field(
        default=15.0, ge=1.0, le=120.0,
        description="Per-call timeout. Hits during indexing fall back to raw content.",
    )
    include_summary: bool = Field(
        default=True,
        description="Prepend `# summary: <one line>` to the chunk content.",
    )


@register_adapter(stage="enricher", name="ollama")
class OllamaEnricher(EnricherStage):
    """One-line LLM summary prepended to the chunk for richer embeddings.

    Pure stdlib HTTP — no extra dependency to vendor. The Ollama HTTP API
    is documented at https://github.com/ollama/ollama/blob/main/docs/api.md.
    """

    Config = OllamaEnricherConfig

    def __init__(self, config: OllamaEnricherConfig | None = None) -> None:
        self._config = config or OllamaEnricherConfig()

    @property
    def stage_name(self) -> str:
        return "enricher"

    @property
    def adapter_name(self) -> str:
        return "ollama"

    @property
    def adapter_id(self) -> str:
        # Hash the prompt so prompt edits invalidate the enrichment cache.
        prompt_hash = hashlib.sha256(self._config.prompt_template.encode()).hexdigest()[:8]
        return (
            f"ollama:{self._config.model}"
            f":t={self._config.temperature}"
            f":p={prompt_hash}"
        )

    @property
    def config_schema(self) -> type[BaseModel]:
        return OllamaEnricherConfig

    def health(self) -> StageHealth:
        """Probe ``GET {base_url}/api/tags`` to confirm the daemon and model."""
        try:
            req = urllib_request.Request(
                f"{self._config.base_url.rstrip('/')}/api/tags",
                method="GET",
            )
            with urllib_request.urlopen(req, timeout=2.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib_error.URLError, OSError, json.JSONDecodeError) as exc:
            return StageHealth(
                status="warning",
                message=(
                    f"Ollama unreachable at {self._config.base_url} ({exc!s}). "
                    "Run `ollama serve` and `ollama pull "
                    f"{self._config.model}`."
                ),
            )
        models = {m.get("name") for m in payload.get("models") or []}
        if self._config.model not in models:
            return StageHealth(
                status="warning",
                message=(
                    f"Model {self._config.model!r} not pulled. "
                    f"Run `ollama pull {self._config.model}`. "
                    f"Available: {sorted(models)[:5]}"
                ),
            )
        return StageHealth(
            status="ready",
            message=f"ollama @ {self._config.base_url} model={self._config.model}",
        )

    def enrich(self, chunk: Chunk, ctx: StageContext) -> str:
        if not self._config.include_summary:
            return chunk.content
        summary = self._summarise(chunk.content)
        if not summary:
            return chunk.content
        return f"# summary: {summary}\n{chunk.content}"

    # ------------------------------------------------------------------ http

    def _summarise(self, content: str) -> str | None:
        """POST ``/api/generate``. Returns None on any failure (graceful)."""
        prompt = self._config.prompt_template.format(content=content[:8000])
        body = json.dumps(
            {
                "model": self._config.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self._config.temperature},
            }
        ).encode("utf-8")
        try:
            req = urllib_request.Request(
                f"{self._config.base_url.rstrip('/')}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=self._config.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib_error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            return None
        text = (payload.get("response") or "").strip()
        # First non-empty line; strip code fences just in case.
        for line in text.splitlines():
            cleaned = line.strip().strip("`").strip()
            if cleaned:
                return cleaned[:250]
        return None

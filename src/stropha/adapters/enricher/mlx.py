"""MLX LLM enricher — native Apple Silicon inference via ``mlx-lm``.

Same contract as :class:`stropha.adapters.enricher.ollama.OllamaEnricher`
(one-line summary prepended to ``embedding_text``), but generates locally
through Apple's MLX framework instead of HTTP-ing to the Ollama daemon.

**Why use this over Ollama:**
- Direct in-process inference — no daemon, no HTTP overhead
- Uses Apple's unified memory + Metal — typically 1.5-2× faster than
  Ollama on M-series for the same quantised model
- No port conflicts, no service to manage

**Why use Ollama instead:**
- Cross-platform (Linux, Windows, Intel Mac)
- Easier model management (`ollama pull` vs hand-converting weights)
- Single daemon shared across many tools

**Status: alpha.** ``mlx-lm`` is an optional dependency (``pip install
'stropha[mlx]'``). When the package isn't installed the adapter still
loads — it just reports a warning health and falls back to raw content
on every enrich call (never blocks indexing).

Models are pulled lazily by mlx-lm on first call. Recommended starter:
``mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`` (~1 GB, top quality
for one-line summaries on M-series).
"""

from __future__ import annotations

import hashlib

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


class MlxEnricherConfig(BaseModel):
    model: str = Field(
        default="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        description="HF model id pre-converted to MLX format. Pulled by mlx-lm on first call.",
    )
    prompt_template: str = Field(
        default=_DEFAULT_PROMPT,
        description="Format string with `{content}` placeholder.",
    )
    max_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    include_summary: bool = Field(default=True)


@register_adapter(stage="enricher", name="mlx")
class MlxEnricher(EnricherStage):
    """In-process MLX inference. ``mlx-lm`` import is lazy + soft."""

    Config = MlxEnricherConfig

    def __init__(self, config: MlxEnricherConfig | None = None) -> None:
        self._config = config or MlxEnricherConfig()
        # Lazy state — populated on first successful generate.
        self._loaded: bool = False
        self._mlx_load = None  # type: ignore[assignment]
        self._mlx_generate = None  # type: ignore[assignment]
        self._model = None
        self._tokenizer = None

    @property
    def stage_name(self) -> str:
        return "enricher"

    @property
    def adapter_name(self) -> str:
        return "mlx"

    @property
    def adapter_id(self) -> str:
        prompt_hash = hashlib.sha256(self._config.prompt_template.encode()).hexdigest()[:8]
        return (
            f"mlx:{self._config.model}"
            f":t={self._config.temperature}"
            f":p={prompt_hash}"
        )

    @property
    def config_schema(self) -> type[BaseModel]:
        return MlxEnricherConfig

    def health(self) -> StageHealth:
        try:
            import mlx_lm  # noqa: F401  — presence check
        except ImportError:
            return StageHealth(
                status="warning",
                message=(
                    "mlx-lm not installed. Install with `uv add mlx-lm` "
                    "(Apple Silicon only). Without it the adapter falls back "
                    "to raw content."
                ),
            )
        return StageHealth(
            status="ready",
            message=(
                f"mlx-lm available, model={self._config.model} "
                "(loaded lazily on first enrich call)"
            ),
        )

    def enrich(self, chunk: Chunk, ctx: StageContext) -> str:
        if not self._config.include_summary:
            return chunk.content
        summary = self._summarise(chunk.content)
        if not summary:
            return chunk.content
        return f"# summary: {summary}\n{chunk.content}"

    # ------------------------------------------------------------------ helpers

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        try:
            from mlx_lm import generate, load
            self._mlx_load = load
            self._mlx_generate = generate
            self._model, self._tokenizer = load(self._config.model)
            self._loaded = True
            return True
        except ImportError:
            return False
        except Exception:
            # Model download / load failed — caller falls back gracefully.
            return False

    def _summarise(self, content: str) -> str | None:
        if not self._ensure_loaded():
            return None
        prompt = self._config.prompt_template.format(content=content[:8000])
        try:
            text = self._mlx_generate(  # type: ignore[misc]
                self._model, self._tokenizer,
                prompt=prompt,
                max_tokens=self._config.max_tokens,
                verbose=False,
            )
        except Exception:
            return None
        if not isinstance(text, str):
            return None
        # First non-empty, non-fence line.
        for line in text.strip().splitlines():
            cleaned = line.strip().strip("`").strip()
            if cleaned:
                return cleaned[:250]
        return None

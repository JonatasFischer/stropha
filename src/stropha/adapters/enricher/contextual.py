"""Contextual Prefix Enricher — LLM-generated semantic description prepended
to chunks before embedding.

Based on Anthropic's "Contextual Retrieval" technique (Sept 2024) which showed
+35% recall improvement by prepending context descriptions to chunks.

Unlike the simpler `ollama` enricher (one-line summary), this enricher generates
a richer contextual description that includes:
- What this code does
- Where it fits in the codebase architecture
- Related concepts and components

The description is prepended to the embedding_text, not the stored content,
so the original code is preserved but the embedding captures richer semantics.

Example:
    Original chunk:
        def submit_answer(self, exercise_id: ExerciseId, answer: Answer): ...

    After enrichment:
        [Context: This method belongs to StudyService, the application service
        layer for study sessions. It handles answer submission in the Phase 1
        acquisition flow, updating streaks and triggering FSRS calculation
        when mastery threshold is reached.]

        def submit_answer(self, exercise_id: ExerciseId, answer: Answer): ...

Backend selection (automatic):
- MLX (Apple Silicon): fastest, no daemon needed
- Ollama: cross-platform fallback

Failure semantics: Any error falls back to raw content (same as ollama enricher).
The index keeps building; warnings appear in logs.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from ...models import Chunk
from ...pipeline.base import StageContext, StageHealth
from ...pipeline.registry import register_adapter
from ...stages.enricher import EnricherStage

_DEFAULT_PROMPT = """\
Describe this code chunk's purpose and context in 2-3 sentences.
Include:
- What this code does (functionality)
- Where it fits (layer, module, responsibility)
- Related concepts or components it interacts with

Code:
```{language}
{content}
```

File: {rel_path}
Symbol: {symbol}

Write ONLY the description, no code, no preamble. Start with "This..."
"""

_CONTEXT_PREFIX = "[Context: {description}]\n\n"


class ContextualEnricherConfig(BaseModel):
    prompt_template: str = Field(
        default=_DEFAULT_PROMPT,
        description=(
            "Format string with placeholders: {content}, {language}, "
            "{rel_path}, {symbol}"
        ),
    )
    max_content_chars: int = Field(
        default=4000,
        ge=500,
        le=16000,
        description="Max chars of content to send to LLM (truncates long chunks).",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature. 0 for deterministic output.",
    )
    max_tokens: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Max tokens for the generated description.",
    )
    max_description_chars: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Max chars for the generated description.",
    )


@register_adapter(stage="enricher", name="contextual")
class ContextualEnricher(EnricherStage):
    """LLM-generated contextual descriptions prepended to chunks.

    Implements Anthropic's Contextual Retrieval technique for improved
    embedding quality. The description provides semantic context that
    helps the embedding model understand the chunk's role in the codebase.

    Uses the unified inference backend (MLX on Apple Silicon, Ollama fallback).
    """

    Config = ContextualEnricherConfig

    def __init__(self, config: ContextualEnricherConfig | None = None) -> None:
        self._config = config or ContextualEnricherConfig()

    @property
    def stage_name(self) -> str:
        return "enricher"

    @property
    def adapter_name(self) -> str:
        return "contextual"

    @property
    def adapter_id(self) -> str:
        # Hash the prompt so edits invalidate the enrichment cache
        prompt_hash = hashlib.sha256(
            self._config.prompt_template.encode()
        ).hexdigest()[:8]
        return (
            f"contextual"
            f":t={self._config.temperature}"
            f":p={prompt_hash}"
        )

    @property
    def config_schema(self) -> type[BaseModel]:
        return ContextualEnricherConfig

    def health(self) -> StageHealth:
        """Check inference backend availability."""
        from ...inference import get_backend

        backend = get_backend()
        is_healthy, message = backend.health_check()

        if is_healthy:
            return StageHealth(
                status="ready",
                message=f"contextual enricher using {backend.name} backend",
            )
        return StageHealth(
            status="warning",
            message=f"Inference backend issue: {message}",
        )

    def enrich(self, chunk: Chunk, ctx: StageContext) -> str:
        """Generate contextual description and prepend to content.

        Returns the original content if LLM call fails (graceful fallback).
        """
        description = self._generate_context(chunk)
        if not description:
            return chunk.content

        # Prepend context in a structured format
        prefix = _CONTEXT_PREFIX.format(description=description)
        return f"{prefix}{chunk.content}"

    def _generate_context(self, chunk: Chunk) -> str | None:
        """Generate contextual description using the inference backend.

        Uses MLX on Apple Silicon, falls back to Ollama on other platforms.
        Returns None on any failure (graceful).
        """
        # Build prompt with chunk metadata
        content = chunk.content[: self._config.max_content_chars]
        prompt = self._config.prompt_template.format(
            content=content,
            language=chunk.language or "unknown",
            rel_path=chunk.rel_path,
            symbol=chunk.symbol or "(anonymous)",
        )

        # Use unified inference backend
        from ...inference import generate

        text = generate(
            prompt,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        if not text:
            return None

        # Clean up the response
        # Remove any code fences or markdown artifacts
        cleaned = text.replace("```", "").strip()
        # Take first paragraph if multiple
        if "\n\n" in cleaned:
            cleaned = cleaned.split("\n\n")[0]
        # Normalize whitespace
        cleaned = " ".join(cleaned.split())

        # Truncate to max length
        if len(cleaned) > self._config.max_description_chars:
            cleaned = cleaned[: self._config.max_description_chars - 3] + "..."

        return cleaned

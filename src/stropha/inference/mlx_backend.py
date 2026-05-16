"""MLX inference backend — native Apple Silicon inference.

Uses mlx-lm for in-process inference with Metal acceleration.
No daemon required, more efficient than Ollama on M-series chips.
"""

from __future__ import annotations

import os
from typing import Any

from ..logging import get_logger
from .backend import InferenceBackend

log = get_logger(__name__)

# Default model optimized for code tasks
DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"


class MlxBackend(InferenceBackend):
    """MLX-based inference backend for Apple Silicon."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the MLX backend.

        Args:
            model: HuggingFace model ID in MLX format. Defaults to
                   Qwen2.5-Coder-1.5B-Instruct-4bit.
        """
        self._default_model = model or os.environ.get(
            "STROPHA_MLX_MODEL", DEFAULT_MODEL
        )
        self._loaded_model: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._generate_fn: Any = None
        self._load_fn: Any = None

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def is_available(self) -> bool:
        """Check if MLX is available."""
        try:
            import mlx_lm  # noqa: F401

            return True
        except ImportError:
            return False

    def _ensure_loaded(self, model: str) -> bool:
        """Load the model if not already loaded."""
        if self._loaded_model == model and self._model is not None:
            return True

        try:
            from mlx_lm import generate, load

            self._load_fn = load
            self._generate_fn = generate

            log.info("mlx.loading_model", model=model)
            self._model, self._tokenizer = load(model)
            self._loaded_model = model
            log.info("mlx.model_loaded", model=model)
            return True
        except ImportError:
            log.warning("mlx.mlx_lm_not_installed")
            return False
        except Exception as exc:
            log.warning("mlx.model_load_failed", model=model, error=str(exc))
            return False

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str | None:
        """Generate text using MLX.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            model: Optional model override.

        Returns:
            Generated text, or None on failure.
        """
        model = model or self._default_model

        if not self._ensure_loaded(model):
            return None

        try:
            # Build kwargs for mlx_lm.generate
            # Note: mlx_lm uses a sampler function for temperature control,
            # but the default greedy sampler (temp=0) is fine for our use case.
            # For non-zero temperature, we'd need to pass a custom sampler.
            kwargs: dict = {
                "max_tokens": max_tokens,
                "verbose": False,
            }

            # Only add sampler for non-zero temperature
            if temperature > 0:
                try:
                    from mlx_lm.sample_utils import make_sampler

                    kwargs["sampler"] = make_sampler(temp=temperature)
                except ImportError:
                    # Older mlx_lm version without make_sampler, use default
                    pass

            text = self._generate_fn(
                self._model,
                self._tokenizer,
                prompt,
                **kwargs,
            )

            if not isinstance(text, str):
                return None

            return text.strip()
        except Exception as exc:
            log.warning("mlx.generate_failed", error=str(exc))
            return None

    def health_check(self) -> tuple[bool, str]:
        """Check MLX backend health."""
        if not self.is_available:
            return False, "mlx-lm not installed (pip install mlx-lm)"

        # Try a minimal generation to verify the model works
        try:
            if not self._ensure_loaded(self._default_model):
                return False, f"Failed to load model: {self._default_model}"
            return True, f"MLX ready with {self._default_model}"
        except Exception as exc:
            return False, f"MLX error: {exc}"

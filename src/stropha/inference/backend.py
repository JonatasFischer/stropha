"""Base inference backend and auto-detection logic.

The inference system supports two backends:
1. MLX (Apple Silicon) - faster, more efficient, no daemon needed
2. Ollama - cross-platform, easier model management

Auto-detection prefers MLX when available, falling back to Ollama.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Cached backend instance
_BACKEND: "InferenceBackend | None" = None


class InferenceBackend(ABC):
    """Abstract base class for local LLM inference backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'mlx', 'ollama')."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and ready."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str | None:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            model: Optional model override.

        Returns:
            Generated text, or None on failure.
        """
        ...

    def health_check(self) -> tuple[bool, str]:
        """Check backend health.

        Returns:
            (is_healthy, message) tuple.
        """
        if self.is_available:
            return True, f"{self.name} backend ready"
        return False, f"{self.name} backend not available"


def is_mlx_available() -> bool:
    """Check if MLX is available (Apple Silicon only)."""
    try:
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def is_ollama_available() -> bool:
    """Check if Ollama daemon is reachable."""
    import json
    from urllib import error as urllib_error
    from urllib import request as urllib_request

    ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    try:
        req = urllib_request.Request(
            f"{ollama_url.rstrip('/')}/api/tags",
            method="GET",
        )
        with urllib_request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return "models" in data
    except (urllib_error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return False


def get_backend(prefer: str | None = None) -> InferenceBackend:
    """Get the best available inference backend.

    Args:
        prefer: Preferred backend ('mlx' or 'ollama'). If not available,
                falls back to the other. If None, auto-detects.

    Returns:
        An InferenceBackend instance.

    The backend is cached globally for reuse.
    """
    global _BACKEND

    # Check for preference override via env var
    env_prefer = os.environ.get("STROPHA_INFERENCE_BACKEND")
    if env_prefer:
        prefer = env_prefer.lower()

    if _BACKEND is not None:
        # Return cached if preference matches or no preference
        if prefer is None or _BACKEND.name == prefer:
            return _BACKEND

    from .mlx_backend import MlxBackend
    from .ollama_backend import OllamaBackend

    # Try preferred backend first
    if prefer == "mlx":
        backend = MlxBackend()
        if backend.is_available:
            _BACKEND = backend
            log.info("inference.backend_selected", backend="mlx")
            return _BACKEND
        log.info("inference.mlx_unavailable_fallback_ollama")
        _BACKEND = OllamaBackend()
        return _BACKEND

    if prefer == "ollama":
        backend = OllamaBackend()
        if backend.is_available:
            _BACKEND = backend
            log.info("inference.backend_selected", backend="ollama")
            return _BACKEND
        log.info("inference.ollama_unavailable_fallback_mlx")
        _BACKEND = MlxBackend()
        return _BACKEND

    # Auto-detect: prefer MLX (faster, more efficient)
    if is_mlx_available():
        _BACKEND = MlxBackend()
        log.info("inference.backend_auto_selected", backend="mlx")
        return _BACKEND

    # Fall back to Ollama
    _BACKEND = OllamaBackend()
    log.info("inference.backend_auto_selected", backend="ollama")
    return _BACKEND


def generate(
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    model: str | None = None,
    backend: str | None = None,
) -> str | None:
    """Generate text using the best available backend.

    Convenience function that auto-selects and caches the backend.

    Args:
        prompt: The input prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        model: Optional model override.
        backend: Optional backend preference ('mlx' or 'ollama').

    Returns:
        Generated text, or None on failure.
    """
    be = get_backend(prefer=backend)
    return be.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
    )


def reset_backend() -> None:
    """Reset the cached backend. Useful for testing."""
    global _BACKEND
    _BACKEND = None

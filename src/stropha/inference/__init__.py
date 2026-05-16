"""Local LLM inference backends.

This module provides a unified interface for local LLM inference,
supporting both MLX (Apple Silicon) and Ollama backends.

Usage:
    from stropha.inference import get_backend, generate

    # Auto-detect best backend
    backend = get_backend()
    response = backend.generate("Your prompt here")

    # Or use directly
    response = generate("Your prompt here")
"""

from .backend import (
    InferenceBackend,
    generate,
    get_backend,
    is_mlx_available,
    is_ollama_available,
)
from .mlx_backend import MlxBackend
from .ollama_backend import OllamaBackend

__all__ = [
    "InferenceBackend",
    "MlxBackend",
    "OllamaBackend",
    "generate",
    "get_backend",
    "is_mlx_available",
    "is_ollama_available",
]

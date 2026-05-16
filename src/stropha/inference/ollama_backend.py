"""Ollama inference backend — HTTP-based local LLM inference.

Uses Ollama's REST API for inference. Requires the Ollama daemon
to be running (`ollama serve`).
"""

from __future__ import annotations

import json
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..logging import get_logger
from .backend import InferenceBackend

log = get_logger(__name__)

# Default model for code tasks
DEFAULT_MODEL = "qwen2.5-coder:1.5b"


class OllamaBackend(InferenceBackend):
    """Ollama HTTP-based inference backend."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the Ollama backend.

        Args:
            model: Default Ollama model name.
            base_url: Ollama API base URL.
            timeout: Request timeout in seconds.
        """
        self._default_model = model or os.environ.get(
            "STROPHA_OLLAMA_MODEL", DEFAULT_MODEL
        )
        self._base_url = (
            base_url
            or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def is_available(self) -> bool:
        """Check if Ollama daemon is reachable."""
        try:
            req = urllib_request.Request(
                f"{self._base_url}/api/tags",
                method="GET",
            )
            with urllib_request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return "models" in data
        except (urllib_error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            return False

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str | None:
        """Generate text using Ollama.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate (mapped to num_predict).
            temperature: Sampling temperature.
            model: Optional model override.

        Returns:
            Generated text, or None on failure.
        """
        model = model or self._default_model

        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
        ).encode("utf-8")

        try:
            req = urllib_request.Request(
                f"{self._base_url}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))

            text = (payload.get("response") or "").strip()
            return text if text else None

        except (urllib_error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
            log.warning("ollama.generate_failed", error=str(exc))
            return None

    def health_check(self) -> tuple[bool, str]:
        """Check Ollama backend health."""
        if not self.is_available:
            return False, f"Ollama not reachable at {self._base_url}"

        # Check if default model is available
        try:
            req = urllib_request.Request(
                f"{self._base_url}/api/tags",
                method="GET",
            )
            with urllib_request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                
                # Check for exact match or prefix match (e.g., "qwen2.5-coder:1.5b" in "qwen2.5-coder:1.5b-fp16")
                model_found = any(
                    self._default_model in m or m.startswith(self._default_model.split(":")[0])
                    for m in models
                )
                
                if model_found:
                    return True, f"Ollama ready with {self._default_model}"
                return False, f"Model {self._default_model} not found. Run: ollama pull {self._default_model}"

        except Exception as exc:
            return False, f"Ollama error: {exc}"

"""Unit tests for ``stropha.adapters.enricher.ollama.OllamaEnricher``.

The enricher MUST never block indexing — every HTTP failure path falls
back to raw content. We verify this by mocking ``urllib.request.urlopen``.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

from stropha.adapters.enricher.ollama import (
    OllamaEnricher,
    OllamaEnricherConfig,
)
from stropha.models import Chunk
from stropha.pipeline.base import StageContext


def _make_chunk(content: str = "def foo(): pass") -> Chunk:
    return Chunk(
        chunk_id="sha-1", rel_path="x.py", language="python",
        kind="function", symbol="foo", parent_chunk_id=None,
        start_line=1, end_line=5,
        content=content, content_hash="hash-1",
    )


def _fake_response(payload: dict) -> io.BytesIO:
    """Return a context-manager-compatible bytes stream for urlopen mock."""
    body = json.dumps(payload).encode("utf-8")
    bio = io.BytesIO(body)
    bio.__enter__ = lambda self=bio: self
    bio.__exit__ = lambda self=bio, *args: False
    return bio


# --------------------------------------------------------------------------- adapter_id


def test_adapter_id_includes_model_and_temperature() -> None:
    e = OllamaEnricher(OllamaEnricherConfig(model="qwen2.5-coder:7b", temperature=0.3))
    assert e.adapter_id.startswith("ollama:qwen2.5-coder:7b:t=0.3")


def test_adapter_id_changes_with_prompt_edit() -> None:
    e1 = OllamaEnricher(OllamaEnricherConfig(prompt_template="A {content}"))
    e2 = OllamaEnricher(OllamaEnricherConfig(prompt_template="B {content}"))
    assert e1.adapter_id != e2.adapter_id


def test_adapter_name_is_ollama() -> None:
    assert OllamaEnricher().adapter_name == "ollama"


# --------------------------------------------------------------------------- registry


def test_registered_in_registry() -> None:
    """`stropha adapters list --stage enricher` must surface it."""
    from stropha.pipeline.registry import lookup_adapter

    cls = lookup_adapter("enricher", "ollama")
    assert cls is OllamaEnricher


# --------------------------------------------------------------------------- enrich (success)


def test_enrich_prepends_summary_on_success() -> None:
    enricher = OllamaEnricher()
    chunk = _make_chunk("def add(a,b): return a+b")
    with patch("stropha.adapters.enricher.ollama.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({"response": "Adds two numbers."})
        out = enricher.enrich(chunk, StageContext())
    assert out.startswith("# summary: Adds two numbers.")
    assert chunk.content in out


def test_enrich_strips_markdown_fences_and_picks_first_line() -> None:
    enricher = OllamaEnricher()
    chunk = _make_chunk()
    with patch("stropha.adapters.enricher.ollama.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "response": "```\nFirst real line.\nSecond line.\n```",
        })
        out = enricher.enrich(chunk, StageContext())
    assert "First real line." in out
    assert "Second line." not in out


def test_enrich_truncates_long_summaries_to_250_chars() -> None:
    enricher = OllamaEnricher()
    chunk = _make_chunk()
    long = "x" * 1000
    with patch("stropha.adapters.enricher.ollama.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({"response": long})
        out = enricher.enrich(chunk, StageContext())
    # 250 chars + "# summary: " prefix
    summary_line = out.splitlines()[0]
    assert len(summary_line) <= 270


# --------------------------------------------------------------------------- enrich (failure → fallback)


def test_enrich_falls_back_on_http_error() -> None:
    """Network failure MUST NOT block indexing."""
    from urllib.error import URLError

    enricher = OllamaEnricher()
    chunk = _make_chunk("body")
    with patch("stropha.adapters.enricher.ollama.urllib_request.urlopen") as mock_open:
        mock_open.side_effect = URLError("connection refused")
        out = enricher.enrich(chunk, StageContext())
    assert out == "body"


def test_enrich_falls_back_on_invalid_json() -> None:
    enricher = OllamaEnricher()
    chunk = _make_chunk("body")
    with patch("stropha.adapters.enricher.ollama.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({"oops": "no response key"})
        out = enricher.enrich(chunk, StageContext())
    assert out == "body"


def test_enrich_falls_back_on_empty_response() -> None:
    enricher = OllamaEnricher()
    chunk = _make_chunk("body")
    with patch("stropha.adapters.enricher.ollama.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({"response": "   "})
        out = enricher.enrich(chunk, StageContext())
    assert out == "body"


def test_include_summary_disable_short_circuits() -> None:
    enricher = OllamaEnricher(OllamaEnricherConfig(include_summary=False))
    chunk = _make_chunk("body")
    with patch("stropha.adapters.enricher.ollama.urllib_request.urlopen") as mock_open:
        # Should not even be called
        out = enricher.enrich(chunk, StageContext())
    mock_open.assert_not_called()
    assert out == "body"


# --------------------------------------------------------------------------- health


def test_health_warning_when_daemon_unreachable() -> None:
    from urllib.error import URLError

    enricher = OllamaEnricher()
    with patch("stropha.adapters.enricher.ollama.urllib_request.urlopen") as mock_open:
        mock_open.side_effect = URLError("connection refused")
        h = enricher.health()
    assert h.status == "warning"
    assert "unreachable" in h.message.lower() or "ollama" in h.message.lower()


def test_health_warning_when_model_not_pulled() -> None:
    enricher = OllamaEnricher(OllamaEnricherConfig(model="qwen2.5-coder:1.5b"))
    with patch("stropha.adapters.enricher.ollama.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "models": [{"name": "llama3:8b"}],
        })
        h = enricher.health()
    assert h.status == "warning"
    assert "pull" in h.message.lower()


def test_health_ready_when_model_present() -> None:
    enricher = OllamaEnricher(OllamaEnricherConfig(model="qwen2.5-coder:1.5b"))
    with patch("stropha.adapters.enricher.ollama.urllib_request.urlopen") as mock_open:
        mock_open.return_value = _fake_response({
            "models": [{"name": "qwen2.5-coder:1.5b"}, {"name": "llama3:8b"}],
        })
        h = enricher.health()
    assert h.status == "ready"
    assert "qwen2.5-coder:1.5b" in h.message

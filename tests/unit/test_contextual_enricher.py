"""Unit tests for ``stropha.adapters.enricher.contextual.ContextualEnricher``.

The contextual enricher generates semantic descriptions for chunks to improve
embedding quality. Like the ollama enricher, it MUST never block indexing —
every HTTP failure path falls back to raw content.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from stropha.adapters.enricher.contextual import (
    ContextualEnricher,
    ContextualEnricherConfig,
)
from stropha.models import Chunk
from stropha.pipeline.base import StageContext


def _make_chunk(
    content: str = "def submit_answer(self, exercise_id, answer): pass",
    symbol: str = "submit_answer",
    rel_path: str = "src/services/study.py",
    language: str = "python",
) -> Chunk:
    return Chunk(
        chunk_id="sha-1",
        rel_path=rel_path,
        language=language,
        kind="function",
        symbol=symbol,
        parent_chunk_id=None,
        start_line=1,
        end_line=5,
        content=content,
        content_hash="hash-1",
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
    e = ContextualEnricher(
        ContextualEnricherConfig(model="qwen2.5-coder:7b", temperature=0.3)
    )
    assert e.adapter_id.startswith("contextual:qwen2.5-coder:7b:t=0.3")


def test_adapter_id_changes_with_prompt_edit() -> None:
    e1 = ContextualEnricher(
        ContextualEnricherConfig(prompt_template="A {content} {language} {rel_path} {symbol}")
    )
    e2 = ContextualEnricher(
        ContextualEnricherConfig(prompt_template="B {content} {language} {rel_path} {symbol}")
    )
    assert e1.adapter_id != e2.adapter_id


def test_adapter_name_is_contextual() -> None:
    assert ContextualEnricher().adapter_name == "contextual"


# --------------------------------------------------------------------------- registry


def test_registered_in_registry() -> None:
    """`stropha adapters list --stage enricher` must surface it."""
    from stropha.pipeline.registry import lookup_adapter

    cls = lookup_adapter("enricher", "contextual")
    assert cls is ContextualEnricher


# --------------------------------------------------------------------------- enrich (success)


def test_enrich_prepends_context_on_success() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.return_value = _fake_response({
            "response": (
                "This method handles answer submission in the study service, "
                "updating user progress and triggering FSRS calculations."
            )
        })
        out = enricher.enrich(chunk, StageContext())

    assert out.startswith("[Context: This method handles answer submission")
    assert chunk.content in out


def test_enrich_includes_context_prefix_format() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.return_value = _fake_response({
            "response": "This is a test description."
        })
        out = enricher.enrich(chunk, StageContext())

    assert "[Context:" in out
    assert "]\n\n" in out  # Context ends with ] and double newline


# --------------------------------------------------------------------------- enrich (failures)


def test_enrich_returns_raw_on_timeout() -> None:
    enricher = ContextualEnricher(ContextualEnricherConfig(timeout_s=1))
    chunk = _make_chunk()
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.side_effect = TimeoutError("timed out")
        out = enricher.enrich(chunk, StageContext())

    assert out == chunk.content


def test_enrich_returns_raw_on_connection_error() -> None:
    from urllib import error as urllib_error

    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.side_effect = urllib_error.URLError("Connection refused")
        out = enricher.enrich(chunk, StageContext())

    assert out == chunk.content


def test_enrich_returns_raw_on_empty_response() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.return_value = _fake_response({"response": ""})
        out = enricher.enrich(chunk, StageContext())

    assert out == chunk.content


def test_enrich_returns_raw_on_invalid_json() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        # Return invalid JSON
        bio = io.BytesIO(b"not json")
        bio.__enter__ = lambda self=bio: self
        bio.__exit__ = lambda self=bio, *args: False
        mock_open.return_value = bio
        out = enricher.enrich(chunk, StageContext())

    assert out == chunk.content


# --------------------------------------------------------------------------- response cleaning


def test_strips_code_fences_from_response() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.return_value = _fake_response({
            "response": "```\nThis is the description.\n```"
        })
        out = enricher.enrich(chunk, StageContext())

    assert "```" not in out
    assert "[Context: This is the description.]" in out


def test_takes_first_paragraph_if_multiple() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.return_value = _fake_response({
            "response": "First paragraph.\n\nSecond paragraph with more detail."
        })
        out = enricher.enrich(chunk, StageContext())

    assert "First paragraph." in out
    assert "Second paragraph" not in out


def test_truncates_long_descriptions() -> None:
    enricher = ContextualEnricher(
        ContextualEnricherConfig(max_description_chars=100)  # Minimum allowed
    )
    chunk = _make_chunk()
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        long_response = "This is a very long description that exceeds the limit and keeps going on and on. " * 10
        mock_open.return_value = _fake_response({"response": long_response})
        out = enricher.enrich(chunk, StageContext())

    # Check it's truncated with ellipsis
    context_part = out.split("]\n\n")[0]
    # "[Context: " is 10 chars, so description should be ~90 chars + "..."
    assert len(context_part) < 120
    assert "..." in context_part


# --------------------------------------------------------------------------- health


def test_health_ready_when_ollama_available() -> None:
    enricher = ContextualEnricher()
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.return_value = _fake_response({
            "models": [{"name": "qwen2.5-coder:1.5b"}]
        })
        health = enricher.health()

    assert health.status == "ready"
    assert "contextual enricher" in health.message


def test_health_warning_when_ollama_unreachable() -> None:
    from urllib import error as urllib_error

    enricher = ContextualEnricher()
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.side_effect = urllib_error.URLError("Connection refused")
        health = enricher.health()

    assert health.status == "warning"
    assert "unreachable" in health.message.lower()


def test_health_warning_when_model_not_pulled() -> None:
    enricher = ContextualEnricher(
        ContextualEnricherConfig(model="nonexistent-model:1b")
    )
    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.return_value = _fake_response({
            "models": [{"name": "qwen2.5-coder:1.5b"}]
        })
        health = enricher.health()

    assert health.status == "warning"
    assert "not pulled" in health.message


# --------------------------------------------------------------------------- config


def test_env_var_overrides_base_url(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://custom:12345")
    enricher = ContextualEnricher()
    assert enricher._config.base_url == "http://custom:12345"


def test_prompt_includes_chunk_metadata() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk(
        content="class UserService: ...",
        symbol="UserService",
        rel_path="src/services/user.py",
        language="python",
    )

    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.return_value = _fake_response({"response": "Test description."})
        enricher.enrich(chunk, StageContext())

        # Check the prompt was built with metadata
        call_args = mock_open.call_args
        request = call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))
        prompt = body["prompt"]

        assert "UserService" in prompt
        assert "src/services/user.py" in prompt
        assert "python" in prompt


def test_content_truncation() -> None:
    enricher = ContextualEnricher(
        ContextualEnricherConfig(max_content_chars=500)  # Minimum allowed
    )
    long_content = "x" * 2000  # Much longer than 500
    chunk = _make_chunk(content=long_content)

    with patch(
        "stropha.adapters.enricher.contextual.urllib_request.urlopen"
    ) as mock_open:
        mock_open.return_value = _fake_response({"response": "Description."})
        enricher.enrich(chunk, StageContext())

        # Check content was truncated in prompt
        call_args = mock_open.call_args
        request = call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))
        prompt = body["prompt"]

        # Content should be truncated to 500 chars (plus prompt template overhead)
        assert "x" * 500 in prompt
        assert "x" * 600 not in prompt  # Truncated

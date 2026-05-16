"""Unit tests for ``stropha.adapters.enricher.contextual.ContextualEnricher``.

The contextual enricher generates semantic descriptions for chunks to improve
embedding quality. Like the ollama enricher, it MUST never block indexing —
every inference failure path falls back to raw content.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

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


# --------------------------------------------------------------------------- adapter_id


def test_adapter_id_includes_temperature_and_prompt_hash() -> None:
    e = ContextualEnricher(
        ContextualEnricherConfig(temperature=0.3)
    )
    # adapter_id format: contextual:t=<temp>:p=<prompt_hash>
    assert e.adapter_id.startswith("contextual:t=0.3:p=")


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
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "This method handles answer submission in the study service, "
            "updating user progress and triggering FSRS calculations."
        )
        out = enricher.enrich(chunk, StageContext())

    assert out.startswith("[Context: This method handles answer submission")
    assert chunk.content in out


def test_enrich_includes_context_prefix_format() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "This is a test description."
        out = enricher.enrich(chunk, StageContext())

    assert "[Context:" in out
    assert "]\n\n" in out  # Context ends with ] and double newline


# --------------------------------------------------------------------------- enrich (failures)


def test_enrich_returns_raw_on_failure() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = None  # Simulate failure
        out = enricher.enrich(chunk, StageContext())

    assert out == chunk.content


def test_enrich_returns_raw_on_empty_response() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = ""  # Empty response
        out = enricher.enrich(chunk, StageContext())

    assert out == chunk.content


# --------------------------------------------------------------------------- response cleaning


def test_strips_code_fences_from_response() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "```\nThis is the description.\n```"
        out = enricher.enrich(chunk, StageContext())

    assert "```" not in out
    assert "[Context: This is the description.]" in out


def test_takes_first_paragraph_if_multiple() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "First paragraph.\n\nSecond paragraph with more detail."
        out = enricher.enrich(chunk, StageContext())

    assert "First paragraph." in out
    assert "Second paragraph" not in out


def test_truncates_long_descriptions() -> None:
    enricher = ContextualEnricher(
        ContextualEnricherConfig(max_description_chars=100)  # Minimum allowed
    )
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        long_response = "This is a very long description that exceeds the limit and keeps going on and on. " * 10
        mock_gen.return_value = long_response
        out = enricher.enrich(chunk, StageContext())

    # Check it's truncated with ellipsis
    context_part = out.split("]\n\n")[0]
    # "[Context: " is 10 chars, so description should be ~90 chars + "..."
    assert len(context_part) < 120
    assert "..." in context_part


# --------------------------------------------------------------------------- health


def test_health_ready_when_backend_available() -> None:
    enricher = ContextualEnricher()
    mock_backend = MagicMock()
    mock_backend.name = "mlx"
    mock_backend.health_check.return_value = (True, "Model loaded")
    with patch("stropha.inference.get_backend", return_value=mock_backend):
        health = enricher.health()

    assert health.status == "ready"
    assert "contextual enricher" in health.message
    assert "mlx" in health.message


def test_health_warning_when_backend_unhealthy() -> None:
    enricher = ContextualEnricher()
    mock_backend = MagicMock()
    mock_backend.name = "ollama"
    mock_backend.health_check.return_value = (False, "Ollama unreachable")
    with patch("stropha.inference.get_backend", return_value=mock_backend):
        health = enricher.health()

    assert health.status == "warning"
    assert "unreachable" in health.message.lower()


# --------------------------------------------------------------------------- config


def test_prompt_includes_chunk_metadata() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk(
        content="class UserService: ...",
        symbol="UserService",
        rel_path="src/services/user.py",
        language="python",
    )

    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "Test description."
        enricher.enrich(chunk, StageContext())

        # Check the prompt was built with metadata
        call_args = mock_gen.call_args
        prompt = call_args[0][0]  # First positional argument

        assert "UserService" in prompt
        assert "src/services/user.py" in prompt
        assert "python" in prompt


def test_content_truncation() -> None:
    enricher = ContextualEnricher(
        ContextualEnricherConfig(max_content_chars=500)  # Minimum allowed
    )
    long_content = "x" * 2000  # Much longer than 500
    chunk = _make_chunk(content=long_content)

    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "Description."
        enricher.enrich(chunk, StageContext())

        # Check content was truncated in prompt
        call_args = mock_gen.call_args
        prompt = call_args[0][0]  # First positional argument

        # Content should be truncated to 500 chars (plus prompt template overhead)
        assert "x" * 500 in prompt
        assert "x" * 600 not in prompt  # Truncated

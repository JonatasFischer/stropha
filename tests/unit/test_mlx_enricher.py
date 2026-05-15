"""Unit tests for ``stropha.adapters.enricher.mlx.MlxEnricher``.

We never import the real ``mlx_lm`` package here — every test patches
``mlx_lm.load`` / ``mlx_lm.generate`` so the suite stays fast and runs on
non-Apple CI too.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from stropha.adapters.enricher.mlx import MlxEnricher, MlxEnricherConfig
from stropha.models import Chunk
from stropha.pipeline.base import StageContext


def _make_chunk(content: str = "def foo(): pass") -> Chunk:
    return Chunk(
        chunk_id="sha-1", rel_path="x.py", language="python",
        kind="function", symbol="foo", parent_chunk_id=None,
        start_line=1, end_line=5,
        content=content, content_hash="hash-1",
    )


def _install_fake_mlx(monkeypatch, *, generate_return: str = "Sample summary."):
    """Inject a stub ``mlx_lm`` module into ``sys.modules``."""
    mod = types.ModuleType("mlx_lm")
    mod.load = MagicMock(return_value=("model_obj", "tokenizer_obj"))  # type: ignore[attr-defined]
    mod.generate = MagicMock(return_value=generate_return)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", mod)
    return mod


def _uninstall_fake_mlx(monkeypatch) -> None:
    """Force ``mlx_lm`` to be missing for tests that need ImportError."""
    monkeypatch.setitem(sys.modules, "mlx_lm", None)


# --------------------------------------------------------------------------- adapter_id


def test_adapter_id_includes_model_temperature_prompt_hash() -> None:
    e = MlxEnricher(MlxEnricherConfig(
        model="mlx-community/foo-1.5b-4bit", temperature=0.2,
    ))
    aid = e.adapter_id
    assert aid.startswith("mlx:mlx-community/foo-1.5b-4bit:t=0.2:p=")


def test_adapter_id_changes_with_prompt_edit() -> None:
    e1 = MlxEnricher(MlxEnricherConfig(prompt_template="A {content}"))
    e2 = MlxEnricher(MlxEnricherConfig(prompt_template="B {content}"))
    assert e1.adapter_id != e2.adapter_id


def test_adapter_name_is_mlx() -> None:
    assert MlxEnricher().adapter_name == "mlx"


# --------------------------------------------------------------------------- registry


def test_registered_in_registry() -> None:
    from stropha.pipeline.registry import lookup_adapter

    cls = lookup_adapter("enricher", "mlx")
    assert cls is MlxEnricher


# --------------------------------------------------------------------------- health


def test_health_warning_when_mlx_lm_missing(monkeypatch) -> None:
    _uninstall_fake_mlx(monkeypatch)
    e = MlxEnricher()
    h = e.health()
    assert h.status == "warning"
    assert "mlx-lm" in h.message.lower()


def test_health_ready_when_mlx_lm_importable(monkeypatch) -> None:
    _install_fake_mlx(monkeypatch)
    e = MlxEnricher()
    h = e.health()
    assert h.status == "ready"
    assert "mlx-lm available" in h.message


# --------------------------------------------------------------------------- enrich (success)


def test_enrich_prepends_summary_on_success(monkeypatch) -> None:
    _install_fake_mlx(monkeypatch, generate_return="Computes the sum of two numbers.")
    enricher = MlxEnricher()
    chunk = _make_chunk("def add(a,b): return a+b")
    out = enricher.enrich(chunk, StageContext())
    assert out.startswith("# summary: Computes the sum of two numbers.")
    assert chunk.content in out


def test_enrich_strips_markdown_fences(monkeypatch) -> None:
    _install_fake_mlx(monkeypatch, generate_return="```\nFirst line.\nSecond line.\n```")
    enricher = MlxEnricher()
    out = enricher.enrich(_make_chunk(), StageContext())
    assert "First line." in out
    assert "Second line." not in out


def test_enrich_truncates_long_summaries(monkeypatch) -> None:
    _install_fake_mlx(monkeypatch, generate_return="x" * 1000)
    enricher = MlxEnricher()
    out = enricher.enrich(_make_chunk(), StageContext())
    summary_line = out.splitlines()[0]
    assert len(summary_line) <= 270  # 250 + "# summary: " prefix


# --------------------------------------------------------------------------- enrich (failure → fallback)


def test_enrich_falls_back_when_mlx_lm_missing(monkeypatch) -> None:
    _uninstall_fake_mlx(monkeypatch)
    enricher = MlxEnricher()
    chunk = _make_chunk("body")
    out = enricher.enrich(chunk, StageContext())
    assert out == "body"


def test_enrich_falls_back_when_load_fails(monkeypatch) -> None:
    """``load()`` raising any exception → graceful fallback."""
    mod = _install_fake_mlx(monkeypatch)
    mod.load.side_effect = RuntimeError("model not found")  # type: ignore[attr-defined]
    enricher = MlxEnricher()
    out = enricher.enrich(_make_chunk("body"), StageContext())
    assert out == "body"


def test_enrich_falls_back_when_generate_raises(monkeypatch) -> None:
    mod = _install_fake_mlx(monkeypatch)
    mod.generate.side_effect = RuntimeError("oom")  # type: ignore[attr-defined]
    enricher = MlxEnricher()
    out = enricher.enrich(_make_chunk("body"), StageContext())
    assert out == "body"


def test_enrich_falls_back_on_empty_response(monkeypatch) -> None:
    _install_fake_mlx(monkeypatch, generate_return="   ")
    enricher = MlxEnricher()
    out = enricher.enrich(_make_chunk("body"), StageContext())
    assert out == "body"


def test_include_summary_disabled_short_circuits(monkeypatch) -> None:
    mod = _install_fake_mlx(monkeypatch)
    enricher = MlxEnricher(MlxEnricherConfig(include_summary=False))
    out = enricher.enrich(_make_chunk("body"), StageContext())
    assert out == "body"
    mod.load.assert_not_called()  # type: ignore[attr-defined]
    mod.generate.assert_not_called()  # type: ignore[attr-defined]


def test_load_only_called_once_across_enrich_calls(monkeypatch) -> None:
    """The expensive `load()` must be cached after the first successful call."""
    mod = _install_fake_mlx(monkeypatch)
    enricher = MlxEnricher()
    enricher.enrich(_make_chunk("a"), StageContext())
    enricher.enrich(_make_chunk("b"), StageContext())
    enricher.enrich(_make_chunk("c"), StageContext())
    assert mod.load.call_count == 1  # type: ignore[attr-defined]
    assert mod.generate.call_count == 3  # type: ignore[attr-defined]

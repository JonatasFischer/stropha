"""Tests for the bundled enricher adapters (noop + hierarchical)."""

from __future__ import annotations

from stropha.adapters.enricher.hierarchical import (
    HierarchicalEnricher,
    HierarchicalEnricherConfig,
)
from stropha.adapters.enricher.noop import NoopEnricher
from stropha.models import Chunk
from stropha.pipeline.base import StageContext


def _make_chunk(
    chunk_id: str = "child",
    content: str = "method body",
    parent_id: str | None = None,
    kind: str = "method",
    symbol: str | None = "doStuff",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        rel_path="src/Example.java",
        language="java",
        kind=kind,
        symbol=symbol,
        parent_chunk_id=parent_id,
        start_line=10,
        end_line=20,
        content=content,
        content_hash="h" + chunk_id,
    )


# ---------------------------------------------------------------------------
# Noop
# ---------------------------------------------------------------------------


def test_noop_enricher_returns_content_unchanged() -> None:
    enr = NoopEnricher()
    chunk = _make_chunk(content="print('hi')")
    assert enr.enrich(chunk, StageContext()) == "print('hi')"


def test_noop_enricher_id_is_stable() -> None:
    assert NoopEnricher().adapter_id == "noop"


# ---------------------------------------------------------------------------
# Hierarchical
# ---------------------------------------------------------------------------


def test_hierarchical_with_no_parent_falls_back_to_content() -> None:
    enr = HierarchicalEnricher()
    chunk = _make_chunk(parent_id=None)
    assert enr.enrich(chunk, StageContext()) == "method body"


def test_hierarchical_prepends_parent_skeleton_when_available() -> None:
    parent = _make_chunk(
        chunk_id="parent",
        content="class Foo { ... }",
        kind="class",
        symbol="Foo",
    )
    child = _make_chunk(parent_id="parent", content="method body")
    enr = HierarchicalEnricher()
    out = enr.enrich(child, StageContext(parent_chunk=parent))
    assert "in class Foo" in out
    assert out.endswith("method body")


def test_hierarchical_includes_repo_url_when_configured() -> None:
    enr = HierarchicalEnricher(
        HierarchicalEnricherConfig(include_repo_url=True, include_parent_skeleton=False)
    )
    chunk = _make_chunk()
    out = enr.enrich(chunk, StageContext(repo_key="github.com/foo/bar"))
    assert out.startswith("# repo: github.com/foo/bar")


def test_hierarchical_adapter_id_reflects_toggles() -> None:
    """Switching a toggle MUST change adapter_id so cache + drift invalidate."""
    default = HierarchicalEnricher().adapter_id
    with_repo = HierarchicalEnricher(
        HierarchicalEnricherConfig(include_repo_url=True)
    ).adapter_id
    bare = HierarchicalEnricher(
        HierarchicalEnricherConfig(
            include_parent_skeleton=False, include_repo_url=False
        )
    ).adapter_id
    assert len({default, with_repo, bare}) == 3

"""Tests for reranker adapters."""

from __future__ import annotations

import pytest

from stropha.adapters.retrieval.reranker.noop import NoopReranker
from stropha.models import SearchHit
from stropha.pipeline.registry import lookup_adapter


def _hit(chunk_id: str, score: float = 1.0, rank: int = 1) -> SearchHit:
    """Create a minimal SearchHit for testing."""
    return SearchHit(
        chunk_id=chunk_id,
        rel_path="test.py",
        start_line=1,
        end_line=10,
        language="python",
        kind="function",
        symbol="test_func",
        snippet="def test_func(): pass",
        score=score,
        rank=rank,
    )


class TestNoopReranker:
    """Tests for the noop reranker (pass-through)."""

    def test_noop_reranker_is_registered(self) -> None:
        cls = lookup_adapter("reranker", "noop")
        assert cls is NoopReranker

    def test_noop_reranker_returns_hits_unchanged(self) -> None:
        reranker = NoopReranker()
        hits = [_hit("a"), _hit("b"), _hit("c")]
        result = reranker.rerank("test query", hits, top_k=10)
        assert result == hits

    def test_noop_reranker_truncates_to_top_k(self) -> None:
        reranker = NoopReranker()
        hits = [_hit("a", rank=1), _hit("b", rank=2), _hit("c", rank=3), _hit("d", rank=4), _hit("e", rank=5)]
        result = reranker.rerank("test query", hits, top_k=3)
        assert len(result) == 3
        assert [h.chunk_id for h in result] == ["a", "b", "c"]

    def test_noop_reranker_handles_empty_hits(self) -> None:
        reranker = NoopReranker()
        result = reranker.rerank("test query", [], top_k=10)
        assert result == []

    def test_noop_reranker_health(self) -> None:
        reranker = NoopReranker()
        health = reranker.health()
        assert health.status == "ready"
        assert "pass-through" in health.message

    def test_noop_reranker_adapter_id(self) -> None:
        reranker = NoopReranker()
        assert reranker.adapter_id == "reranker:noop"
        assert reranker.adapter_name == "noop"
        assert reranker.stage_name == "reranker"


class TestCrossEncoderReranker:
    """Tests for the cross-encoder reranker."""

    def test_cross_encoder_is_registered(self) -> None:
        cls = lookup_adapter("reranker", "cross-encoder")
        assert cls.__name__ == "CrossEncoderReranker"

    def test_cross_encoder_lazy_loads_model(self) -> None:
        from stropha.adapters.retrieval.reranker.cross_encoder import (
            CrossEncoderReranker,
        )
        reranker = CrossEncoderReranker()
        # Model should not be loaded until first rerank call
        assert not reranker._model_loaded
        health = reranker.health()
        assert health.detail["loaded"] is False

    def test_cross_encoder_adapter_id_includes_model(self) -> None:
        from stropha.adapters.retrieval.reranker.cross_encoder import (
            CrossEncoderReranker,
            CrossEncoderRerankerConfig,
        )
        config = CrossEncoderRerankerConfig(model="test-model")
        reranker = CrossEncoderReranker(config)
        assert "test-model" in reranker.adapter_id

    @pytest.mark.slow
    def test_cross_encoder_reranks_hits(self) -> None:
        """Integration test: actually loads model and reranks.
        
        Marked as slow because it downloads the model on first run.
        """
        from stropha.adapters.retrieval.reranker.cross_encoder import (
            CrossEncoderReranker,
            CrossEncoderRerankerConfig,
        )
        # Use smallest model for speed
        config = CrossEncoderRerankerConfig(
            model="Xenova/ms-marco-MiniLM-L-6-v2"
        )
        reranker = CrossEncoderReranker(config)
        
        # Create hits with different snippets
        hit1 = SearchHit(
            chunk_id="irrelevant",
            rel_path="test.py",
            start_line=1,
            end_line=10,
            language="python",
            kind="function",
            symbol="unrelated",
            snippet="def unrelated_function(): pass",
            score=1.0,
            rank=1,
        )
        hit2 = SearchHit(
            chunk_id="relevant",
            rel_path="test.py",
            start_line=1,
            end_line=10,
            language="python",
            kind="function",
            symbol="test_func",
            snippet="def test_function(): pass",
            score=0.5,
            rank=2,
        )
        hits = [hit1, hit2]
        
        result = reranker.rerank("test function", hits, top_k=2)
        
        # Model should now be loaded
        assert reranker._model_loaded
        assert len(result) == 2


class TestHybridRrfWithReranker:
    """Tests for hybrid-rrf with reranker integration."""

    def test_hybrid_rrf_config_accepts_reranker(self) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig
        
        config = HybridRrfConfig(
            top_k=10,
            candidate_k=50,
            reranker={"adapter": "noop", "config": {}},
        )
        assert config.reranker is not None
        assert config.reranker["adapter"] == "noop"

    def test_hybrid_rrf_config_reranker_optional(self) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig
        
        config = HybridRrfConfig()
        assert config.reranker is None

"""Tests for query intelligence features (HyDE, Query Rewriting)."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from stropha.retrieval.hyde import maybe_hyde_rewrite, _enabled as hyde_enabled
from stropha.retrieval.query_rewrite import maybe_rewrite_query, _enabled as rewrite_enabled


class TestHyDE:
    """Tests for HyDE (Hypothetical Document Embeddings)."""

    def test_hyde_disabled_by_default(self) -> None:
        """HyDE should be disabled when env var is not set."""
        with patch.dict(os.environ, {"STROPHA_HYDE_ENABLED": "0"}, clear=False):
            assert not hyde_enabled()
            result = maybe_hyde_rewrite("test query")
            assert result is None

    def test_hyde_returns_none_for_empty_query(self) -> None:
        """HyDE should return None for empty queries."""
        with patch.dict(os.environ, {"STROPHA_HYDE_ENABLED": "1"}, clear=False):
            assert maybe_hyde_rewrite("") is None
            assert maybe_hyde_rewrite("   ") is None

    @patch("stropha.inference.generate")
    def test_hyde_calls_inference_backend(self, mock_gen: MagicMock) -> None:
        """HyDE should call inference backend when enabled."""
        mock_gen.return_value = "def calculate_sum(a, b): return a + b"

        with patch.dict(os.environ, {"STROPHA_HYDE_ENABLED": "1"}, clear=False):
            result = maybe_hyde_rewrite("how to add two numbers")
            assert result is not None
            assert "calculate_sum" in result or "return" in result

    @patch("stropha.inference.generate")
    def test_hyde_handles_inference_failure(self, mock_gen: MagicMock) -> None:
        """HyDE should return None when inference fails."""
        mock_gen.return_value = None  # Simulate failure

        with patch.dict(os.environ, {"STROPHA_HYDE_ENABLED": "1"}, clear=False):
            result = maybe_hyde_rewrite("test query")
            assert result is None

    @patch("stropha.inference.generate")
    def test_hyde_handles_empty_response(self, mock_gen: MagicMock) -> None:
        """HyDE should return None when inference returns empty."""
        mock_gen.return_value = ""

        with patch.dict(os.environ, {"STROPHA_HYDE_ENABLED": "1"}, clear=False):
            result = maybe_hyde_rewrite("test query")
            assert result is None


class TestQueryRewrite:
    """Tests for Query Rewriting."""

    def test_rewrite_disabled_by_default(self) -> None:
        """Query rewriting should be disabled when env var is not set."""
        with patch.dict(os.environ, {"STROPHA_QUERY_REWRITE_ENABLED": "0"}, clear=False):
            assert not rewrite_enabled()
            result = maybe_rewrite_query("test query")
            assert result is None

    def test_rewrite_returns_none_for_empty_query(self) -> None:
        """Query rewriting should return None for empty queries."""
        with patch.dict(os.environ, {"STROPHA_QUERY_REWRITE_ENABLED": "1"}, clear=False):
            assert maybe_rewrite_query("") is None
            assert maybe_rewrite_query("   ") is None

    @patch("stropha.inference.generate")
    def test_rewrite_calls_inference_backend(self, mock_gen: MagicMock) -> None:
        """Query rewriting should call inference backend when enabled."""
        mock_gen.return_value = "FSRS calculator test FsrsCalculatorTest"

        with patch.dict(os.environ, {"STROPHA_QUERY_REWRITE_ENABLED": "1"}, clear=False):
            result = maybe_rewrite_query("onde tem teste pra fsrs")
            assert result is not None
            # Should contain original query + rewritten terms
            assert "fsrs" in result.lower()

    @patch("stropha.inference.generate")
    def test_rewrite_combines_original_and_expanded(self, mock_gen: MagicMock) -> None:
        """Query rewriting should combine original query with expanded terms."""
        mock_gen.return_value = "authentication login security"

        with patch.dict(os.environ, {"STROPHA_QUERY_REWRITE_ENABLED": "1"}, clear=False):
            result = maybe_rewrite_query("how does auth work")
            assert result is not None
            # Should contain both original and expanded
            assert "how does auth work" in result
            assert "authentication" in result

    @patch("stropha.inference.generate")
    def test_rewrite_handles_inference_failure(self, mock_gen: MagicMock) -> None:
        """Query rewriting should return None when inference fails."""
        mock_gen.return_value = None  # Simulate failure

        with patch.dict(os.environ, {"STROPHA_QUERY_REWRITE_ENABLED": "1"}, clear=False):
            result = maybe_rewrite_query("test query")
            assert result is None

    @patch("stropha.inference.generate")
    def test_rewrite_truncates_long_output(self, mock_gen: MagicMock) -> None:
        """Query rewriting should truncate very long outputs."""
        long_response = "term " * 200  # 1000 chars
        mock_gen.return_value = long_response

        with patch.dict(os.environ, {"STROPHA_QUERY_REWRITE_ENABLED": "1"}, clear=False):
            result = maybe_rewrite_query("short query")
            assert result is not None
            assert len(result) <= 500


class TestHybridRrfWithQueryIntelligence:
    """Tests for hybrid-rrf integration with HyDE and Query Rewriting."""

    def test_config_accepts_hyde_flag(self) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig
        
        config = HybridRrfConfig(hyde_enabled=True)
        assert config.hyde_enabled is True
        
        config = HybridRrfConfig(hyde_enabled=False)
        assert config.hyde_enabled is False

    def test_config_accepts_query_rewrite_flag(self) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig
        
        config = HybridRrfConfig(query_rewrite_enabled=True)
        assert config.query_rewrite_enabled is True
        
        config = HybridRrfConfig(query_rewrite_enabled=False)
        assert config.query_rewrite_enabled is False

    def test_config_defaults_disabled(self) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig
        
        config = HybridRrfConfig()
        assert config.hyde_enabled is False
        assert config.query_rewrite_enabled is False


class TestHybridRrfSearchIntegration:
    """Integration tests for HyDE and query rewrite in hybrid-rrf.search()."""

    @pytest.fixture
    def recording_storage(self):
        """Storage stub that records search calls and queries."""
        from stropha.models import SearchHit
        
        class _RecordingStorage:
            def __init__(self) -> None:
                self.dense_calls: list[tuple[list[float], int]] = []
                self.bm25_calls: list[tuple[str, int]] = []
                self.symbol_calls: list[tuple[str, int]] = []

            @staticmethod
            def _hit(rank: int, chunk_id: str, score: float = 1.0) -> SearchHit:
                return SearchHit(
                    rank=rank,
                    score=score,
                    rel_path=f"file_{chunk_id}.py",
                    language="python",
                    kind="file",
                    symbol=None,
                    start_line=1,
                    end_line=10,
                    snippet=f"snippet {chunk_id}",
                    chunk_id=chunk_id,
                )

            def search_dense(self, vec: list[float], k: int) -> list[SearchHit]:
                self.dense_calls.append((vec, k))
                return [self._hit(i + 1, f"d{i}") for i in range(min(k, 3))]

            def search_bm25(self, query: str, k: int) -> list[SearchHit]:
                self.bm25_calls.append((query, k))
                return [self._hit(i + 1, f"b{i}") for i in range(min(k, 3))]

            def search_symbol_tokens(self, query: str, k: int) -> list[SearchHit]:
                self.symbol_calls.append((query, k))
                return [self._hit(i + 1, f"s{i}") for i in range(min(k, 3))]

        return _RecordingStorage()

    @pytest.fixture
    def recording_embedder(self):
        """Embedder stub that records which text was embedded."""
        from stropha.pipeline.base import StageHealth
        from pydantic import BaseModel
        
        class _Config(BaseModel):
            pass
        
        class _RecordingEmbedder:
            def __init__(self) -> None:
                self.query_texts: list[str] = []
            
            @property
            def stage_name(self) -> str: return "embedder"
            @property
            def adapter_name(self) -> str: return "stub"
            @property
            def adapter_id(self) -> str: return "stub"
            @property
            def model_name(self) -> str: return "stub"
            @property
            def dim(self) -> int: return 4
            @property
            def batch_size(self) -> int: return 16
            @property
            def config_schema(self): return _Config
            def health(self): return StageHealth(status="ready", message="stub")
            def embed_documents(self, texts): return [[0.0] * 4 for _ in texts]
            def embed_query(self, text: str) -> list[float]:
                self.query_texts.append(text)
                return [1.0, 0.0, 0.0, 0.0]

        return _RecordingEmbedder()

    @patch("stropha.inference.generate")
    def test_search_uses_hyde_when_config_enabled(
        self, mock_gen: MagicMock, recording_storage, recording_embedder
    ) -> None:
        """When hyde_enabled=True, search should use HyDE-rewritten query for embedding."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
        
        mock_gen.return_value = "def hypothetical_code(): pass"

        config = HybridRrfConfig(hyde_enabled=True)
        retrieval = HybridRrfRetrieval(config, storage=recording_storage, embedder=recording_embedder)
        
        retrieval.search("how does login work")
        
        # Embedder should receive the HyDE-rewritten query (hypothetical code)
        assert len(recording_embedder.query_texts) == 1
        assert "hypothetical_code" in recording_embedder.query_texts[0]

    @patch("stropha.inference.generate")
    def test_search_uses_hyde_when_env_enabled(
        self, mock_gen: MagicMock, recording_storage, recording_embedder
    ) -> None:
        """When STROPHA_HYDE_ENABLED=1, search should use HyDE even without config flag."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
        
        mock_gen.return_value = "class AuthService: pass"

        with patch.dict(os.environ, {"STROPHA_HYDE_ENABLED": "1"}, clear=False):
            config = HybridRrfConfig()  # hyde_enabled defaults to False
            retrieval = HybridRrfRetrieval(config, storage=recording_storage, embedder=recording_embedder)
            
            retrieval.search("authentication flow")
            
            assert len(recording_embedder.query_texts) == 1
            assert "AuthService" in recording_embedder.query_texts[0]

    @patch("stropha.inference.generate")
    def test_search_uses_query_rewrite_when_config_enabled(
        self, mock_gen: MagicMock, recording_storage, recording_embedder
    ) -> None:
        """When query_rewrite_enabled=True, search should use rewritten query for all streams."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
        
        mock_gen.return_value = "FSRS calculator FsrsCalculator test"

        config = HybridRrfConfig(query_rewrite_enabled=True)
        retrieval = HybridRrfRetrieval(config, storage=recording_storage, embedder=recording_embedder)
        
        retrieval.search("onde tem teste pra fsrs")
        
        # BM25 and symbol streams should receive the rewritten query
        assert len(recording_storage.bm25_calls) == 1
        bm25_query = recording_storage.bm25_calls[0][0]
        # Rewritten query combines original + expanded terms
        assert "fsrs" in bm25_query.lower()
        assert "FSRS" in bm25_query or "FsrsCalculator" in bm25_query

    @patch("stropha.inference.generate")
    def test_search_uses_query_rewrite_when_env_enabled(
        self, mock_gen: MagicMock, recording_storage, recording_embedder
    ) -> None:
        """When STROPHA_QUERY_REWRITE_ENABLED=1, search should use rewriting."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
        
        mock_gen.return_value = "mastery streak transition"

        with patch.dict(os.environ, {"STROPHA_QUERY_REWRITE_ENABLED": "1"}, clear=False):
            config = HybridRrfConfig()  # query_rewrite_enabled defaults to False
            retrieval = HybridRrfRetrieval(config, storage=recording_storage, embedder=recording_embedder)
            
            retrieval.search("como funciona mastery")
            
            assert len(recording_storage.bm25_calls) == 1
            bm25_query = recording_storage.bm25_calls[0][0]
            assert "mastery" in bm25_query.lower()
            assert "streak" in bm25_query or "transition" in bm25_query

    def test_search_uses_both_hyde_and_query_rewrite(
        self, recording_storage, recording_embedder
    ) -> None:
        """When both enabled, HyDE affects dense embedding, query rewrite affects all streams."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
        
        # Mock both functions directly since they both use the inference backend
        with patch("stropha.retrieval.query_rewrite.maybe_rewrite_query") as mock_rewrite, \
             patch("stropha.retrieval.hyde.maybe_hyde_rewrite") as mock_hyde:
            
            # Query rewrite returns expanded terms
            mock_rewrite.return_value = "how does auth work authentication login AuthService"
            
            # HyDE returns hypothetical code
            mock_hyde.return_value = "class LoginHandler { authenticate() {} }"

            config = HybridRrfConfig(hyde_enabled=True, query_rewrite_enabled=True)
            retrieval = HybridRrfRetrieval(config, storage=recording_storage, embedder=recording_embedder)
            
            retrieval.search("how does auth work")
            
            # Both should have been called with force_enabled=True
            mock_rewrite.assert_called_once()
            mock_hyde.assert_called_once()
            
            # Verify force_enabled was passed
            _, kwargs = mock_rewrite.call_args
            assert kwargs.get("force_enabled") is True
            
            _, kwargs = mock_hyde.call_args
            assert kwargs.get("force_enabled") is True
            
            # Dense embedding should use HyDE output
            assert len(recording_embedder.query_texts) == 1
            assert "LoginHandler" in recording_embedder.query_texts[0]
            
            # BM25/symbol should use query rewrite output
            assert len(recording_storage.bm25_calls) == 1
            bm25_query = recording_storage.bm25_calls[0][0]
            assert "authentication" in bm25_query or "AuthService" in bm25_query

    def test_search_fallback_when_hyde_fails(
        self, recording_storage, recording_embedder
    ) -> None:
        """When HyDE fails, search should fall back to original query."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
        
        with patch("stropha.inference.generate") as mock_gen:
            mock_gen.return_value = None  # Simulate failure
            
            config = HybridRrfConfig(hyde_enabled=True)
            retrieval = HybridRrfRetrieval(config, storage=recording_storage, embedder=recording_embedder)
            
            retrieval.search("test query")
            
            # Should still embed something (the original query)
            assert len(recording_embedder.query_texts) == 1
            assert recording_embedder.query_texts[0] == "test query"

    def test_search_fallback_when_query_rewrite_fails(
        self, recording_storage, recording_embedder
    ) -> None:
        """When query rewrite fails, search should fall back to original query."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
        
        with patch("stropha.inference.generate") as mock_gen:
            mock_gen.return_value = None  # Simulate failure
            
            config = HybridRrfConfig(query_rewrite_enabled=True)
            retrieval = HybridRrfRetrieval(config, storage=recording_storage, embedder=recording_embedder)
            
            retrieval.search("original query")
            
            # BM25 should receive the original query
            assert len(recording_storage.bm25_calls) == 1
            assert recording_storage.bm25_calls[0][0] == "original query"

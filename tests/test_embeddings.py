"""Tests for embeddings module."""

import pytest
from unittest.mock import MagicMock, patch

from config.settings import EmbeddingConfig


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        assert config.provider == "sentence-transformers"
        assert config.sentence_transformers_model == "all-MiniLM-L6-v2"
        assert config.openai_model == "text-embedding-3-small"
        assert config.cache_size == 1000
        assert config.rule_weight == 0.4
        assert config.semantic_weight == 0.6
        assert config.enabled is True

    def test_weights_sum_to_one(self):
        """Test that weights must sum to 1."""
        # Default should be valid
        config = EmbeddingConfig()
        assert abs(config.rule_weight + config.semantic_weight - 1.0) < 0.01

    def test_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(ValueError, match="EMBEDDING_PROVIDER must be one of"):
            with patch.dict("os.environ", {"EMBEDDING_PROVIDER": "invalid"}):
                from importlib import reload
                import config.settings
                reload(config.settings)
                EmbeddingConfig()

    def test_invalid_weights(self):
        """Test invalid weight raises error."""
        with pytest.raises(ValueError):
            config = EmbeddingConfig()
            config.rule_weight = 0.5
            config.semantic_weight = 0.6  # Sum > 1
            config.__post_init__()


class TestEmbeddingUtilities:
    """Tests for embedding utility functions."""

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy required"
    )
    def test_compute_hybrid_similarity(self):
        """Test hybrid similarity computation."""
        from src.processors.embeddings import compute_hybrid_similarity

        # Equal weights
        result = compute_hybrid_similarity(0.5, 0.5, 0.5, 0.5)
        assert result == 0.5

        # Custom weights
        result = compute_hybrid_similarity(1.0, 0.0, 0.4, 0.6)
        assert result == 0.4

        result = compute_hybrid_similarity(0.0, 1.0, 0.4, 0.6)
        assert result == 0.6

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy required"
    )
    def test_embedding_to_bytes_and_back(self):
        """Test embedding serialization round-trip."""
        import numpy as np
        from src.processors.embeddings import embedding_to_bytes, bytes_to_embedding

        original = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        as_bytes = embedding_to_bytes(original)
        recovered = bytes_to_embedding(as_bytes, 4)

        assert np.allclose(original, recovered)

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy required"
    )
    def test_compute_centroid(self):
        """Test centroid computation."""
        import numpy as np
        from src.processors.embeddings import compute_centroid

        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        centroid = compute_centroid(embeddings)

        # Centroid should be normalized
        assert abs(np.linalg.norm(centroid) - 1.0) < 0.01

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy required"
    )
    def test_compute_centroid_empty_raises(self):
        """Test that empty list raises error."""
        from src.processors.embeddings import compute_centroid

        with pytest.raises(ValueError, match="Cannot compute centroid of empty list"):
            compute_centroid([])


class TestEmbeddingManager:
    """Tests for EmbeddingManager."""

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy required"
    )
    def test_singleton(self):
        """Test singleton pattern."""
        from src.processors.embeddings import EmbeddingManager

        # Reset singleton for test
        EmbeddingManager._instance = None

        manager1 = EmbeddingManager.get_instance()
        manager2 = EmbeddingManager.get_instance()
        assert manager1 is manager2

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy required"
    )
    def test_cache_key_generation(self):
        """Test cache key is deterministic."""
        from src.processors.embeddings import EmbeddingManager

        manager = EmbeddingManager()
        key1 = manager._cache_key("test text")
        key2 = manager._cache_key("test text")
        key3 = manager._cache_key("different text")

        assert key1 == key2
        assert key1 != key3

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy required"
    )
    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        import numpy as np
        from src.processors.embeddings import EmbeddingManager

        manager = EmbeddingManager()

        # Same vector = similarity 1
        vec = np.array([1.0, 0.0, 0.0])
        assert manager.cosine_similarity(vec, vec) == 1.0

        # Orthogonal vectors = similarity 0
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        assert abs(manager.cosine_similarity(vec1, vec2)) < 0.01

        # Opposite vectors = similarity -1
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        assert manager.cosine_similarity(vec1, vec2) == -1.0

    @pytest.mark.skipif(
        not pytest.importorskip("numpy", reason="numpy not installed"),
        reason="numpy required"
    )
    def test_clear_cache(self):
        """Test cache clearing."""
        from src.processors.embeddings import EmbeddingManager

        manager = EmbeddingManager()
        manager._cache["test"] = "value"
        manager.clear_cache()
        assert len(manager._cache) == 0


class TestEventStreamIntegration:
    """Tests for embedding integration in event stream."""

    def test_cluster_candidate_fields(self):
        """Test ClusterCandidate has new fields."""
        from src.processors.event_stream import ClusterCandidate

        candidate = ClusterCandidate(
            cluster_id=1,
            cluster_title="Test Cluster",
            score=0.75,
            method="hybrid",
            rule_score=0.5,
            semantic_score=0.9,
        )

        assert candidate.cluster_id == 1
        assert candidate.score == 0.75
        assert candidate.method == "hybrid"
        assert candidate.rule_score == 0.5
        assert candidate.semantic_score == 0.9

    def test_event_stream_processor_init(self):
        """Test processor initializes with embedding support."""
        from src.processors.event_stream import EventStreamProcessor

        mock_db = MagicMock()
        processor = EventStreamProcessor(db=mock_db, use_mock=True)

        assert processor._embedding_manager is None  # Lazy loaded
        assert processor._centroid_cache == {}

    def test_embedding_manager_disabled(self):
        """Test embedding manager returns None when disabled."""
        from src.processors.event_stream import EventStreamProcessor

        mock_db = MagicMock()
        processor = EventStreamProcessor(db=mock_db, use_mock=True)

        # Disable embeddings in settings
        with patch.object(processor.settings.embedding, 'enabled', False):
            manager = processor._get_embedding_manager()
            assert manager is None

"""Tests for the cache optimizer module."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.optimization.cache_optimizer import CacheOptimizer, CacheMetrics, CacheEfficiency
from src.optimization.dedup_optimizer import (
    DedupOptimizer,
    compute_title_hash,
    compute_content_hash,
)
from src.storage.database import SyncDatabase


class TestCacheMetrics:
    """Tests for CacheMetrics dataclass."""

    def test_utilization_rate_zero(self):
        """Test utilization rate with zero entries."""
        metrics = CacheMetrics(total_entries=0, valid_entries=0)
        assert metrics.utilization_rate == 0.0

    def test_utilization_rate_calculation(self):
        """Test utilization rate calculation."""
        metrics = CacheMetrics(total_entries=100, valid_entries=75)
        assert metrics.utilization_rate == 75.0

    def test_utilization_rate_full(self):
        """Test utilization rate when all entries are valid."""
        metrics = CacheMetrics(total_entries=50, valid_entries=50)
        assert metrics.utilization_rate == 100.0


class TestCacheEfficiency:
    """Tests for CacheEfficiency dataclass."""

    def test_hit_rate_zero(self):
        """Test hit rate with zero counts."""
        efficiency = CacheEfficiency()
        assert efficiency.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        efficiency = CacheEfficiency(hit_count=30, miss_count=70)
        assert efficiency.hit_rate == 30.0

    def test_recommendations_default(self):
        """Test that recommendations defaults to empty list."""
        efficiency = CacheEfficiency()
        assert efficiency.recommendations == []


class TestCacheOptimizerIntegration:
    """Integration tests for CacheOptimizer with real database."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SyncDatabase(db_path)
            db.connect()
            yield db
            db.close()

    def test_get_cache_metrics_empty(self, temp_db):
        """Test getting metrics from empty cache."""
        optimizer = CacheOptimizer(temp_db)
        metrics = optimizer.get_cache_metrics()

        assert metrics.total_entries == 0
        assert metrics.valid_entries == 0
        assert metrics.expired_entries == 0

    def test_get_cache_metrics_with_entries(self, temp_db):
        """Test getting metrics with cache entries."""
        optimizer = CacheOptimizer(temp_db)

        # Add some cache entries
        temp_db.set_ai_cache("hash1", '{"test": 1}', ttl_seconds=3600)
        temp_db.set_ai_cache("hash2", '{"test": 2}', ttl_seconds=3600)

        metrics = optimizer.get_cache_metrics()

        assert metrics.total_entries == 2
        assert metrics.valid_entries == 2
        assert metrics.estimated_size_kb > 0

    def test_cleanup_and_optimize(self, temp_db):
        """Test cleanup removes expired entries."""
        optimizer = CacheOptimizer(temp_db)

        # Add entries with short TTL
        temp_db.set_ai_cache("expire1", '{"test": 1}', ttl_seconds=1)
        temp_db.set_ai_cache("valid1", '{"test": 2}', ttl_seconds=3600)

        # Wait for expiry
        import time

        time.sleep(1.1)

        result = optimizer.cleanup_and_optimize()

        assert result["optimized"] is True
        assert result["expired_removed"] >= 1

    def test_analyze_cache_efficiency_empty(self, temp_db):
        """Test efficiency analysis on empty cache."""
        optimizer = CacheOptimizer(temp_db)
        efficiency = optimizer.analyze_cache_efficiency()

        assert len(efficiency.recommendations) > 0
        # Should suggest cache is underutilized
        assert any("underutilized" in r.lower() for r in efficiency.recommendations)

    def test_get_cache_summary(self, temp_db):
        """Test formatted cache summary."""
        optimizer = CacheOptimizer(temp_db)

        # Add an entry
        temp_db.set_ai_cache("test1", '{"data": "value"}', ttl_seconds=3600)

        summary = optimizer.get_cache_summary()

        assert "AI Cache Status" in summary
        assert "Total Entries" in summary
        assert "Recommendations" in summary


class TestComputeTitleHash:
    """Tests for the compute_title_hash function."""

    def test_empty_title(self):
        """Test with empty title."""
        assert compute_title_hash("") == ""

    def test_short_title(self):
        """Test with very short title."""
        assert compute_title_hash("Hi") == ""

    def test_normalized_hash(self):
        """Test that normalization produces consistent hashes."""
        # These should produce the same hash
        hash1 = compute_title_hash("OpenAI announces GPT-5")
        hash2 = compute_title_hash("OPENAI ANNOUNCES GPT-5")
        assert hash1 == hash2

    def test_rt_prefix_removed(self):
        """Test that RT prefix is normalized."""
        hash1 = compute_title_hash("RT OpenAI announces GPT-5")
        hash2 = compute_title_hash("OpenAI announces GPT-5")
        assert hash1 == hash2

    def test_mentions_removed(self):
        """Test that @mentions are normalized."""
        hash1 = compute_title_hash("@openai: OpenAI announces GPT-5")
        hash2 = compute_title_hash("OpenAI announces GPT-5")
        assert hash1 == hash2

    def test_urls_removed(self):
        """Test that URLs are normalized."""
        hash1 = compute_title_hash("Check this out https://example.com")
        hash2 = compute_title_hash("Check this out")
        assert hash1 == hash2

    def test_different_titles_different_hashes(self):
        """Test that different titles produce different hashes."""
        hash1 = compute_title_hash("OpenAI announces GPT-5")
        hash2 = compute_title_hash("Google releases Gemini 2")
        assert hash1 != hash2

    def test_caching_works(self):
        """Test that LRU cache is working."""
        # Call twice to hit cache
        hash1 = compute_title_hash("Test title for caching")
        hash2 = compute_title_hash("Test title for caching")
        assert hash1 == hash2

        # Check cache info
        info = compute_title_hash.cache_info()
        assert info.hits >= 1


class TestComputeContentHash:
    """Tests for the compute_content_hash function."""

    def test_empty_content(self):
        """Test with empty content."""
        result = compute_content_hash("", "")
        assert len(result) == 16

    def test_same_content_same_hash(self):
        """Test that same content produces same hash."""
        hash1 = compute_content_hash("Title", "Content body here")
        hash2 = compute_content_hash("Title", "Content body here")
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Test that different content produces different hash."""
        hash1 = compute_content_hash("Title", "Content A")
        hash2 = compute_content_hash("Title", "Content B")
        assert hash1 != hash2

    def test_long_content_truncated(self):
        """Test that only first 500 chars are used."""
        long_content = "x" * 1000
        hash1 = compute_content_hash("Title", long_content)
        hash2 = compute_content_hash("Title", long_content[:500])
        # Should be equal since only first 500 chars matter
        assert hash1 == hash2


class TestDedupOptimizer:
    """Tests for the DedupOptimizer class."""

    def test_empty_index(self):
        """Test operations on empty index."""
        optimizer = DedupOptimizer()
        assert optimizer.size == 0
        assert optimizer.find_duplicate_by_title("Test") is None

    def test_add_and_find(self):
        """Test adding and finding items."""
        optimizer = DedupOptimizer()
        optimizer.add_item(1, "OpenAI announces GPT-5", "Content here")

        assert optimizer.size == 1
        result = optimizer.find_duplicate_by_title("OpenAI announces GPT-5")
        assert result == 1

    def test_normalized_matching(self):
        """Test that normalized titles match."""
        optimizer = DedupOptimizer()
        optimizer.add_item(1, "OpenAI announces GPT-5")

        # Should match with different case
        result = optimizer.find_duplicate_by_title("OPENAI ANNOUNCES GPT-5")
        assert result == 1

    def test_content_dedup(self):
        """Test content-based deduplication."""
        optimizer = DedupOptimizer()
        optimizer.add_item(1, "Title", "Some content body")

        result = optimizer.find_duplicate_by_content("Title", "Some content body")
        assert result == 1

    def test_remove_item(self):
        """Test removing items from index."""
        optimizer = DedupOptimizer()
        optimizer.add_item(1, "Test Title", "Content")

        optimizer.remove_item("Test Title", "Content")
        assert optimizer.find_duplicate_by_title("Test Title") is None

    def test_clear(self):
        """Test clearing the index."""
        optimizer = DedupOptimizer()
        optimizer.add_item(1, "Title 1")
        optimizer.add_item(2, "Title 2")

        optimizer.clear()
        assert optimizer.size == 0

    def test_get_stats(self):
        """Test getting statistics."""
        optimizer = DedupOptimizer()
        optimizer.add_item(1, "Title 1", "Content 1")

        stats = optimizer.get_stats()
        assert stats["title_hashes"] == 1
        assert stats["content_hashes"] == 1
        assert "cache_size" in stats

    def test_build_index(self):
        """Test building index from list of items."""
        optimizer = DedupOptimizer()

        # Create mock items with longer titles (short titles get filtered)
        items = []
        for i in range(5):
            item = MagicMock()
            item.id = i + 1  # Start from 1 to avoid id=0 issues
            item.title = f"This is a longer title for item number {i}"
            item.content = f"Content body for item {i}"
            items.append(item)

        count = optimizer.build_index(items)
        assert count == 5
        assert optimizer.size == 5

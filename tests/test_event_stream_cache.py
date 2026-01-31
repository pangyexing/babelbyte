"""Tests for caching strategies in EventStreamProcessor."""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.processors.event_stream import (
    ClusterCandidate,
    EventStreamProcessor,
    _extract_entities_cached,
    _extract_keywords_cached,
)
from src.storage.models import ContentItem, EventCluster, SourceType


# ============================================
# Fixtures
# ============================================


@pytest.fixture(autouse=True)
def clear_all_caches():
    """Clear all caches before and after each test."""
    EventStreamProcessor.clear_cache()
    yield
    EventStreamProcessor.clear_cache()


@pytest.fixture
def mock_db():
    """Create a mock database."""
    db = MagicMock()
    db.get_recent_event_clusters = MagicMock(return_value=[])
    return db


@pytest.fixture
def processor(mock_db):
    """Create an EventStreamProcessor with mock database."""
    return EventStreamProcessor(db=mock_db, use_mock=True)


@pytest.fixture
def sample_content_item():
    """Create a sample content item."""
    now = datetime.now()
    return ContentItem(
        id=1,
        subscription_id=1,
        source_type=SourceType.REDDIT,
        external_id="reddit_1",
        title="OpenAI announces GPT-5 release",
        content="OpenAI has officially announced GPT-5, their latest AI model.",
        url="https://example.com/1",
        author="user1",
        published_at=now,
        fetched_at=now,
        summary="OpenAI announced GPT-5 with improved capabilities.",
        category="AI",
        importance_score=9,
        processed_at=now,
    )


@pytest.fixture
def sample_clusters():
    """Create sample event clusters."""
    now = datetime.now()
    return [
        EventCluster(
            id=1,
            event_title="OpenAI GPT-5 Release",
            category="AI",
            first_seen_at=now,
            last_updated_at=now,
            article_count=3,
        ),
        EventCluster(
            id=2,
            event_title="Google Gemini Update",
            category="AI",
            first_seen_at=now,
            last_updated_at=now,
            article_count=2,
        ),
    ]


# ============================================
# Tests: LRU Cache for Entity Extraction
# ============================================


class TestEntityExtractionCache:
    """Tests for LRU cache on entity extraction."""

    def test_extract_entities_returns_frozenset(self):
        """Test that _extract_entities_cached returns a frozenset."""
        result = _extract_entities_cached("OpenAI announces new model")
        assert isinstance(result, frozenset)

    def test_extract_entities_finds_companies(self):
        """Test that entities are correctly extracted."""
        result = _extract_entities_cached("OpenAI and Google compete in AI")
        assert "openai" in result
        assert "google" in result

    def test_extract_entities_finds_products(self):
        """Test that product names are extracted."""
        result = _extract_entities_cached("GPT-5 vs Claude vs Gemini")
        assert "claude" in result
        assert "gemini" in result

    def test_extract_entities_cache_hit(self):
        """Test that repeated calls hit the cache."""
        text = "Microsoft and Apple partnership"

        # First call
        result1 = _extract_entities_cached(text)

        # Get cache info before second call
        cache_info_before = _extract_entities_cached.cache_info()

        # Second call with same text
        result2 = _extract_entities_cached(text)

        # Get cache info after second call
        cache_info_after = _extract_entities_cached.cache_info()

        # Results should be identical
        assert result1 == result2

        # Cache hits should increase by 1
        assert cache_info_after.hits == cache_info_before.hits + 1

    def test_extract_entities_cache_miss_different_text(self):
        """Test that different text causes cache miss."""
        text1 = "Tesla stock rises"
        text2 = "Nvidia earnings report"

        # First calls
        _extract_entities_cached(text1)
        cache_info_before = _extract_entities_cached.cache_info()

        _extract_entities_cached(text2)
        cache_info_after = _extract_entities_cached.cache_info()

        # Cache misses should increase
        assert cache_info_after.misses == cache_info_before.misses + 1


# ============================================
# Tests: LRU Cache for Keyword Extraction
# ============================================


class TestKeywordExtractionCache:
    """Tests for LRU cache on keyword extraction."""

    def test_extract_keywords_returns_frozenset(self):
        """Test that _extract_keywords_cached returns a frozenset."""
        result = _extract_keywords_cached("AI breakthrough announced")
        assert isinstance(result, frozenset)

    def test_extract_keywords_filters_stopwords(self):
        """Test that stopwords are filtered out."""
        result = _extract_keywords_cached("the quick brown fox is in the forest")
        assert "the" not in result
        assert "is" not in result
        assert "in" not in result
        assert "quick" in result
        assert "brown" in result

    def test_extract_keywords_handles_chinese(self):
        """Test that Chinese keywords are extracted."""
        result = _extract_keywords_cached("人工智能发展迅速")
        assert "人工智能发展迅速" in result or len(result) > 0

    def test_extract_keywords_cache_hit(self):
        """Test that repeated calls hit the cache."""
        text = "Machine learning breakthrough"

        # First call
        result1 = _extract_keywords_cached(text)

        # Get cache info before second call
        cache_info_before = _extract_keywords_cached.cache_info()

        # Second call with same text
        result2 = _extract_keywords_cached(text)

        # Get cache info after second call
        cache_info_after = _extract_keywords_cached.cache_info()

        # Results should be identical
        assert result1 == result2

        # Cache hits should increase by 1
        assert cache_info_after.hits == cache_info_before.hits + 1


# ============================================
# Tests: TTL Cache for Recent Clusters
# ============================================


class TestClusterQueryCache:
    """Tests for TTL cache on recent clusters query."""

    def test_clusters_cache_avoids_repeated_queries(self, processor, mock_db, sample_clusters):
        """Test that cached clusters avoid repeated database queries."""
        mock_db.get_recent_event_clusters.return_value = sample_clusters

        # First call - should query database
        result1 = processor._get_recent_clusters_cached(time_window_days=3, category="AI")

        # Second call - should use cache
        result2 = processor._get_recent_clusters_cached(time_window_days=3, category="AI")

        # Results should be identical
        assert result1 == result2
        assert len(result1) == 2

        # Database should only be called once
        assert mock_db.get_recent_event_clusters.call_count == 1

    def test_clusters_cache_different_keys(self, processor, mock_db, sample_clusters):
        """Test that different cache keys query database separately."""
        mock_db.get_recent_event_clusters.return_value = sample_clusters

        # Call with category "AI"
        processor._get_recent_clusters_cached(time_window_days=3, category="AI")

        # Call with category "Tech" - different key
        processor._get_recent_clusters_cached(time_window_days=3, category="Tech")

        # Database should be called twice (different keys)
        assert mock_db.get_recent_event_clusters.call_count == 2

    def test_clusters_cache_different_time_windows(self, processor, mock_db, sample_clusters):
        """Test that different time windows have separate cache entries."""
        mock_db.get_recent_event_clusters.return_value = sample_clusters

        # Call with 3 days
        processor._get_recent_clusters_cached(time_window_days=3, category="AI")

        # Call with 7 days - different key
        processor._get_recent_clusters_cached(time_window_days=7, category="AI")

        # Database should be called twice
        assert mock_db.get_recent_event_clusters.call_count == 2

    def test_clusters_cache_expires_after_ttl(self, processor, mock_db, sample_clusters):
        """Test that cache expires after TTL."""
        mock_db.get_recent_event_clusters.return_value = sample_clusters

        # Set a very short TTL for testing
        original_ttl = EventStreamProcessor._clusters_cache_ttl
        EventStreamProcessor._clusters_cache_ttl = 0.1  # 100ms

        try:
            # First call
            processor._get_recent_clusters_cached(time_window_days=3, category="AI")
            assert mock_db.get_recent_event_clusters.call_count == 1

            # Wait for cache to expire
            time.sleep(0.15)

            # Second call - cache expired, should query again
            processor._get_recent_clusters_cached(time_window_days=3, category="AI")
            assert mock_db.get_recent_event_clusters.call_count == 2
        finally:
            # Restore original TTL
            EventStreamProcessor._clusters_cache_ttl = original_ttl

    def test_clusters_cache_none_category(self, processor, mock_db, sample_clusters):
        """Test cache works with None category."""
        mock_db.get_recent_event_clusters.return_value = sample_clusters

        # Call with None category
        result1 = processor._get_recent_clusters_cached(time_window_days=3, category=None)
        result2 = processor._get_recent_clusters_cached(time_window_days=3, category=None)

        assert result1 == result2
        assert mock_db.get_recent_event_clusters.call_count == 1


# ============================================
# Tests: Session Cache for AI Confirmation
# ============================================


class TestAIConfirmationCache:
    """Tests for session cache on AI confirmation."""

    def test_ai_confirm_cache_hit(self, processor, sample_content_item):
        """Test that repeated AI confirmation calls use cache."""
        candidates = [
            ClusterCandidate(cluster_id=1, cluster_title="GPT-5 Release", score=0.8, method="entity"),
            ClusterCandidate(cluster_id=2, cluster_title="AI News", score=0.5, method="keyword"),
        ]

        # First call (mock mode returns top candidate if score > 0.5)
        result1 = processor.confirm_same_event_with_ai(sample_content_item, candidates)

        # Second call - should use cache
        result2 = processor.confirm_same_event_with_ai(sample_content_item, candidates)

        assert result1 == result2
        assert result1.cluster_id == 1

    def test_ai_confirm_cache_different_items(self, processor):
        """Test that different items have separate cache entries."""
        now = datetime.now()
        item1 = ContentItem(
            id=1,
            subscription_id=1,
            source_type=SourceType.REDDIT,
            external_id="1",
            title="Item 1",
            content="Content 1",
            url="https://example.com/1",
            author="user",
            published_at=now,
            fetched_at=now,
        )
        item2 = ContentItem(
            id=2,
            subscription_id=1,
            source_type=SourceType.REDDIT,
            external_id="2",
            title="Item 2",
            content="Content 2",
            url="https://example.com/2",
            author="user",
            published_at=now,
            fetched_at=now,
        )

        candidates = [
            ClusterCandidate(cluster_id=1, cluster_title="Event 1", score=0.8, method="entity"),
        ]

        # Calls for different items
        processor.confirm_same_event_with_ai(item1, candidates)
        processor.confirm_same_event_with_ai(item2, candidates)

        # Cache should have 2 entries
        assert len(processor._ai_confirm_cache) == 2

    def test_ai_confirm_cache_different_candidates(self, processor, sample_content_item):
        """Test that different candidate lists have separate cache entries."""
        candidates1 = [
            ClusterCandidate(cluster_id=1, cluster_title="Event 1", score=0.8, method="entity"),
        ]
        candidates2 = [
            ClusterCandidate(cluster_id=2, cluster_title="Event 2", score=0.7, method="keyword"),
        ]

        # Calls with different candidates
        processor.confirm_same_event_with_ai(sample_content_item, candidates1)
        processor.confirm_same_event_with_ai(sample_content_item, candidates2)

        # Cache should have 2 entries (different candidate tuples)
        assert len(processor._ai_confirm_cache) == 2

    def test_ai_confirm_cache_stores_none_result(self, processor, sample_content_item):
        """Test that None results are also cached."""
        # Candidates with low scores (< 0.5 in mock mode returns None)
        candidates = [
            ClusterCandidate(cluster_id=1, cluster_title="Event 1", score=0.3, method="keyword"),
        ]

        result1 = processor.confirm_same_event_with_ai(sample_content_item, candidates)
        result2 = processor.confirm_same_event_with_ai(sample_content_item, candidates)

        assert result1 is None
        assert result2 is None

        # Cache should have the entry
        cache_key = (sample_content_item.id, (1,))
        assert cache_key in processor._ai_confirm_cache

    def test_ai_confirm_empty_candidates_not_cached(self, processor, sample_content_item):
        """Test that empty candidates list returns None without caching."""
        result = processor.confirm_same_event_with_ai(sample_content_item, [])

        assert result is None
        assert len(processor._ai_confirm_cache) == 0


# ============================================
# Tests: Clear Cache
# ============================================


class TestClearCache:
    """Tests for the clear_cache class method."""

    def test_clear_cache_clears_clusters_cache(self, processor, mock_db, sample_clusters):
        """Test that clear_cache clears the clusters cache."""
        mock_db.get_recent_event_clusters.return_value = sample_clusters

        # Populate cache
        processor._get_recent_clusters_cached(time_window_days=3, category="AI")
        assert len(EventStreamProcessor._clusters_cache) > 0

        # Clear cache
        EventStreamProcessor.clear_cache()

        assert len(EventStreamProcessor._clusters_cache) == 0

    def test_clear_cache_clears_lru_caches(self):
        """Test that clear_cache clears LRU caches."""
        # Populate LRU caches
        _extract_entities_cached("OpenAI test")
        _extract_keywords_cached("keyword test")

        info_before_entities = _extract_entities_cached.cache_info()
        info_before_keywords = _extract_keywords_cached.cache_info()

        assert info_before_entities.currsize > 0
        assert info_before_keywords.currsize > 0

        # Clear cache
        EventStreamProcessor.clear_cache()

        info_after_entities = _extract_entities_cached.cache_info()
        info_after_keywords = _extract_keywords_cached.cache_info()

        assert info_after_entities.currsize == 0
        assert info_after_keywords.currsize == 0

    def test_clear_cache_does_not_affect_instance_cache(self, processor, sample_content_item):
        """Test that clear_cache does not clear instance-level AI cache."""
        candidates = [
            ClusterCandidate(cluster_id=1, cluster_title="Event", score=0.8, method="entity"),
        ]

        # Populate instance cache
        processor.confirm_same_event_with_ai(sample_content_item, candidates)
        assert len(processor._ai_confirm_cache) > 0

        # Class method clear_cache only clears class-level caches
        EventStreamProcessor.clear_cache()

        # Instance cache is NOT cleared by class method
        # (This is expected behavior - instance cache is per-session)
        assert len(processor._ai_confirm_cache) > 0


# ============================================
# Tests: Integration with find_cluster_candidates
# ============================================


class TestFindClusterCandidatesWithCache:
    """Tests for find_cluster_candidates using cached functions."""

    def test_find_candidates_uses_cached_extraction(self, processor, mock_db, sample_clusters):
        """Test that find_cluster_candidates uses cached extraction functions."""
        mock_db.get_recent_event_clusters.return_value = sample_clusters

        now = datetime.now()
        item = ContentItem(
            id=1,
            subscription_id=1,
            source_type=SourceType.REDDIT,
            external_id="1",
            title="OpenAI releases GPT-5",
            content="OpenAI has released GPT-5 today.",
            url="https://example.com/1",
            author="user",
            published_at=now,
            fetched_at=now,
            category="AI",
        )

        # Clear caches and get initial state
        EventStreamProcessor.clear_cache()
        entities_before = _extract_entities_cached.cache_info()
        keywords_before = _extract_keywords_cached.cache_info()

        # First call
        processor.find_cluster_candidates(item)

        entities_after = _extract_entities_cached.cache_info()
        keywords_after = _extract_keywords_cached.cache_info()

        # Cache misses should have increased (new extractions)
        assert entities_after.misses > entities_before.misses
        assert keywords_after.misses > keywords_before.misses

        # Second call with same item
        entities_before2 = _extract_entities_cached.cache_info()
        processor.find_cluster_candidates(item)
        entities_after2 = _extract_entities_cached.cache_info()

        # Cache hits should increase on second call
        assert entities_after2.hits > entities_before2.hits

    def test_find_candidates_uses_cached_clusters(self, processor, mock_db, sample_clusters):
        """Test that find_cluster_candidates uses cached cluster query."""
        mock_db.get_recent_event_clusters.return_value = sample_clusters

        now = datetime.now()
        item1 = ContentItem(
            id=1,
            subscription_id=1,
            source_type=SourceType.REDDIT,
            external_id="1",
            title="AI News",
            content="Content",
            url="https://example.com/1",
            author="user",
            published_at=now,
            fetched_at=now,
            category="AI",
        )
        item2 = ContentItem(
            id=2,
            subscription_id=1,
            source_type=SourceType.REDDIT,
            external_id="2",
            title="More AI News",
            content="Content",
            url="https://example.com/2",
            author="user",
            published_at=now,
            fetched_at=now,
            category="AI",
        )

        # Process two items with same category
        processor.find_cluster_candidates(item1)
        processor.find_cluster_candidates(item2)

        # Database should only be called once (cached for same category)
        assert mock_db.get_recent_event_clusters.call_count == 1


# ============================================
# Tests: Progress Callback
# ============================================


class TestProgressCallback:
    """Tests for progress callback in cluster_unprocessed_items."""

    def test_progress_callback_is_called(self):
        """Test that progress callback is called for each item."""
        from src.processors.event_stream import cluster_unprocessed_items

        mock_db = MagicMock()
        now = datetime.now()

        # Create test items with LOW importance (won't create clusters)
        items = [
            ContentItem(
                id=i,
                subscription_id=1,
                source_type=SourceType.REDDIT,
                external_id=f"item_{i}",
                title=f"Test item {i}",
                content="Content",
                url=f"https://example.com/{i}",
                author="user",
                published_at=now,
                fetched_at=now,
                importance_score=4,  # Low importance, won't create clusters
                category="Tech",
            )
            for i in range(1, 4)
        ]

        mock_db.get_undelivered_items.return_value = items
        mock_db.get_recent_event_clusters.return_value = []

        # Track progress calls
        progress_calls = []

        def track_progress(current, total, clustered):
            progress_calls.append((current, total, clustered))

        cluster_unprocessed_items(
            db=mock_db, use_mock=True, limit=10, progress_callback=track_progress
        )

        # Should have 3 progress calls (one per item)
        assert len(progress_calls) == 3
        # Items don't cluster (low importance, no existing clusters)
        assert progress_calls[0] == (1, 3, 0)
        assert progress_calls[1] == (2, 3, 0)
        assert progress_calls[2] == (3, 3, 0)

    def test_progress_callback_tracks_clustered_count(self):
        """Test that progress callback correctly tracks clustered count."""
        from src.processors.event_stream import cluster_unprocessed_items

        mock_db = MagicMock()
        now = datetime.now()

        # Create high-importance items that will create new clusters
        items = [
            ContentItem(
                id=i,
                subscription_id=1,
                source_type=SourceType.REDDIT,
                external_id=f"item_{i}",
                title=f"OpenAI announces model {i}",
                content="OpenAI content",
                url=f"https://example.com/{i}",
                author="user",
                published_at=now,
                fetched_at=now,
                importance_score=8,  # High importance to create clusters
                category="AI",
            )
            for i in range(1, 4)
        ]

        mock_db.get_undelivered_items.return_value = items
        mock_db.get_recent_event_clusters.return_value = []

        # Mock cluster creation
        cluster_counter = [0]

        def mock_create_cluster(cluster):
            cluster_counter[0] += 1
            cluster.id = cluster_counter[0]
            return cluster

        mock_db.create_event_cluster.side_effect = mock_create_cluster
        mock_db.add_event_member.return_value = None

        # Track progress calls
        progress_calls = []

        def track_progress(current, total, clustered):
            progress_calls.append((current, total, clustered))

        result = cluster_unprocessed_items(
            db=mock_db, use_mock=True, limit=10, progress_callback=track_progress
        )

        # All items should create clusters (high importance, no existing clusters)
        assert result == 3
        assert len(progress_calls) == 3

        # Clustered count should increase
        assert progress_calls[0][2] == 1  # First item clustered
        assert progress_calls[1][2] == 2  # Second item clustered
        assert progress_calls[2][2] == 3  # Third item clustered

    def test_no_callback_works(self):
        """Test that clustering works without progress callback."""
        from src.processors.event_stream import cluster_unprocessed_items

        mock_db = MagicMock()
        mock_db.get_undelivered_items.return_value = []

        # Should not raise error
        result = cluster_unprocessed_items(db=mock_db, use_mock=True, limit=10)
        assert result == 0


# ============================================
# Tests: Cache Usage During Batch Clustering
# ============================================


class TestBatchClusteringCache:
    """Tests for cache usage during batch clustering."""

    def test_batch_clustering_uses_entity_cache(self):
        """Test that batch clustering uses entity extraction cache."""
        from src.processors.event_stream import cluster_unprocessed_items

        mock_db = MagicMock()
        now = datetime.now()

        # Create items with same entity (OpenAI)
        items = [
            ContentItem(
                id=i,
                subscription_id=1,
                source_type=SourceType.REDDIT,
                external_id=f"item_{i}",
                title=f"OpenAI news item {i}",
                content="OpenAI content here",
                url=f"https://example.com/{i}",
                author="user",
                published_at=now,
                fetched_at=now,
                importance_score=5,
                category="AI",
            )
            for i in range(1, 6)
        ]

        # Add an existing cluster so extraction is triggered
        existing_cluster = EventCluster(
            id=1,
            event_title="OpenAI GPT Release",
            category="AI",
            first_seen_at=now,
            last_updated_at=now,
            article_count=2,
        )

        mock_db.get_undelivered_items.return_value = items
        mock_db.get_recent_event_clusters.return_value = [existing_cluster]

        # Clear cache and get initial state
        EventStreamProcessor.clear_cache()
        cache_info_before = _extract_entities_cached.cache_info()

        cluster_unprocessed_items(db=mock_db, use_mock=True, limit=10)

        cache_info_after = _extract_entities_cached.cache_info()

        # Should have cache entries (extraction was called)
        assert cache_info_after.currsize > 0
        # The cluster title is extracted once and cached
        # Items each have unique text, but cluster title hits cache on subsequent items
        assert cache_info_after.hits > 0

    def test_batch_clustering_uses_cluster_query_cache(self):
        """Test that batch clustering uses cluster query cache."""
        from src.processors.event_stream import cluster_unprocessed_items

        mock_db = MagicMock()
        now = datetime.now()

        # Create items with same category
        items = [
            ContentItem(
                id=i,
                subscription_id=1,
                source_type=SourceType.REDDIT,
                external_id=f"item_{i}",
                title=f"Tech news {i}",
                content="Content",
                url=f"https://example.com/{i}",
                author="user",
                published_at=now,
                fetched_at=now,
                importance_score=5,  # Low importance, won't create clusters
                category="Tech",
            )
            for i in range(1, 6)
        ]

        mock_db.get_undelivered_items.return_value = items
        mock_db.get_recent_event_clusters.return_value = []

        cluster_unprocessed_items(db=mock_db, use_mock=True, limit=10)

        # All items have same category, so cluster query should be cached
        # Should only call database once for "Tech" category
        assert mock_db.get_recent_event_clusters.call_count == 1

    def test_batch_clustering_different_categories_queries_each(self):
        """Test that different categories trigger separate queries."""
        from src.processors.event_stream import cluster_unprocessed_items

        mock_db = MagicMock()
        now = datetime.now()

        # Create items with different categories
        items = [
            ContentItem(
                id=1,
                subscription_id=1,
                source_type=SourceType.REDDIT,
                external_id="item_1",
                title="AI news",
                content="Content",
                url="https://example.com/1",
                author="user",
                published_at=now,
                fetched_at=now,
                importance_score=5,
                category="AI",
            ),
            ContentItem(
                id=2,
                subscription_id=1,
                source_type=SourceType.REDDIT,
                external_id="item_2",
                title="Tech news",
                content="Content",
                url="https://example.com/2",
                author="user",
                published_at=now,
                fetched_at=now,
                importance_score=5,
                category="Tech",
            ),
            ContentItem(
                id=3,
                subscription_id=1,
                source_type=SourceType.REDDIT,
                external_id="item_3",
                title="More AI news",
                content="Content",
                url="https://example.com/3",
                author="user",
                published_at=now,
                fetched_at=now,
                importance_score=5,
                category="AI",
            ),
        ]

        mock_db.get_undelivered_items.return_value = items
        mock_db.get_recent_event_clusters.return_value = []

        cluster_unprocessed_items(db=mock_db, use_mock=True, limit=10)

        # Should call database twice: once for AI, once for Tech
        # Third item (AI) should use cached result
        assert mock_db.get_recent_event_clusters.call_count == 2


# ============================================
# Run tests if executed directly
# ============================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

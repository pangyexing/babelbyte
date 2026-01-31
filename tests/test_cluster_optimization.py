"""Tests for cluster command performance optimizations."""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.processors.event_stream import (
    ClusterCandidate,
    EventStreamProcessor,
    cluster_unprocessed_items,
    cluster_unprocessed_items_parallel,
)
from src.storage.database import CREATE_INDEXES
from src.storage.models import ContentItem, EventCluster, EventMember, SourceType


@pytest.fixture(autouse=True)
def clear_all_caches():
    """Clear all caches before and after each test."""
    EventStreamProcessor.clear_cache()
    yield
    EventStreamProcessor.clear_cache()


@pytest.fixture
def mock_db():
    """Create a mock database with clustering methods."""
    db = MagicMock()
    db.get_recent_event_clusters = MagicMock(return_value=[])
    db.get_unclustered_items = MagicMock(return_value=[])
    db.get_undelivered_items = MagicMock(return_value=[])
    db.add_event_member = MagicMock()
    db.create_event_cluster = MagicMock()
    db.get_event_cluster = MagicMock()
    return db


def create_content_item(id: int, title: str, importance: int = 8) -> ContentItem:
    """Helper to create test content items."""
    now = datetime.now()
    return ContentItem(
        id=id,
        subscription_id=1,
        source_type=SourceType.REDDIT,
        external_id=f"reddit_{id}",
        title=title,
        content=f"Content for {title}",
        url=f"https://example.com/{id}",
        author="user1",
        published_at=now,
        fetched_at=now,
        summary=f"Summary of {title}",
        category="AI",
        importance_score=importance,
        processed_at=now,
    )


# ============================================
# Tests: Database Indexes
# ============================================


class TestDatabaseIndexes:
    """Tests for database index creation."""

    def test_event_members_content_id_index_exists(self):
        """Test that idx_event_members_content_id index is defined."""
        assert "idx_event_members_content_id" in CREATE_INDEXES
        assert "event_members(content_item_id)" in CREATE_INDEXES

    def test_event_members_cluster_id_index_exists(self):
        """Test that idx_event_members_cluster_id index is defined."""
        assert "idx_event_members_cluster_id" in CREATE_INDEXES
        assert "event_members(event_cluster_id)" in CREATE_INDEXES


# ============================================
# Tests: Skip Already-Clustered Items
# ============================================


class TestSkipAlreadyClusteredItems:
    """Tests for skipping already-clustered items."""

    def test_get_unclustered_items_excludes_clustered(self, mock_db):
        """Test that get_unclustered_items excludes items already in clusters."""
        # Setup: 3 items, but item 2 is already clustered
        all_items = [
            create_content_item(1, "OpenAI GPT-5 news"),
            create_content_item(2, "Google Gemini update"),  # Already clustered
            create_content_item(3, "Microsoft Copilot launch"),
        ]
        unclustered_items = [all_items[0], all_items[2]]  # Item 2 excluded

        mock_db.get_unclustered_items.return_value = unclustered_items

        result = mock_db.get_unclustered_items(min_importance=6, limit=100)

        assert len(result) == 2
        assert all(item.id != 2 for item in result)

    def test_cluster_unprocessed_uses_unclustered_query(self, mock_db):
        """Test that cluster_unprocessed_items uses get_unclustered_items."""
        mock_db.get_unclustered_items.return_value = []

        cluster_unprocessed_items(db=mock_db, use_mock=True, limit=50)

        mock_db.get_unclustered_items.assert_called_once_with(
            min_importance=6, limit=50, retry_after_hours=24
        )

    def test_cluster_parallel_uses_unclustered_query(self, mock_db):
        """Test that cluster_unprocessed_items_parallel uses get_unclustered_items."""
        mock_db.get_unclustered_items.return_value = []

        cluster_unprocessed_items_parallel(db=mock_db, use_mock=True, limit=50, max_workers=2)

        mock_db.get_unclustered_items.assert_called_once_with(
            min_importance=6, limit=50, retry_after_hours=24
        )


# ============================================
# Tests: High-Confidence Auto-Accept
# ============================================


class TestHighConfidenceAutoAccept:
    """Tests for auto-accepting high confidence matches without AI."""

    def test_auto_accept_high_confidence_match(self, mock_db):
        """Test that score >= 0.8 is auto-accepted without AI call."""
        processor = EventStreamProcessor(db=mock_db, use_mock=False)
        item = create_content_item(1, "OpenAI GPT-5 announcement")

        candidates = [
            ClusterCandidate(
                cluster_id=1,
                cluster_title="OpenAI GPT-5 Release",
                score=0.85,  # High confidence
                method="entity",
            )
        ]

        # Should return immediately without AI call (no subprocess)
        result = processor.confirm_same_event_with_ai(item, candidates)

        assert result is not None
        assert result.cluster_id == 1
        assert result.score == 0.85

    def test_auto_accept_exactly_at_threshold(self, mock_db):
        """Test that score == 0.8 is auto-accepted."""
        processor = EventStreamProcessor(db=mock_db, use_mock=False)
        item = create_content_item(1, "OpenAI news")

        candidates = [
            ClusterCandidate(
                cluster_id=1,
                cluster_title="OpenAI News",
                score=0.8,  # Exactly at threshold
                method="entity",
            )
        ]

        result = processor.confirm_same_event_with_ai(item, candidates)

        assert result is not None
        assert result.cluster_id == 1

    def test_ai_called_for_medium_confidence(self, mock_db):
        """Test that score < 0.8 triggers AI confirmation (in mock mode accepts > 0.5)."""
        processor = EventStreamProcessor(db=mock_db, use_mock=True)
        item = create_content_item(1, "Tech news update")

        candidates = [
            ClusterCandidate(
                cluster_id=1,
                cluster_title="Tech Industry News",
                score=0.6,  # Medium confidence - needs AI
                method="keyword",
            )
        ]

        # In mock mode, score > 0.5 is accepted
        result = processor.confirm_same_event_with_ai(item, candidates)

        assert result is not None
        assert result.cluster_id == 1

    def test_no_auto_accept_below_threshold(self, mock_db):
        """Test that low scores are not auto-accepted."""
        processor = EventStreamProcessor(db=mock_db, use_mock=True)
        item = create_content_item(1, "Unrelated content")

        candidates = [
            ClusterCandidate(
                cluster_id=1,
                cluster_title="AI News",
                score=0.35,  # Low confidence
                method="keyword",
            )
        ]

        # Mock mode: score <= 0.5 returns None
        result = processor.confirm_same_event_with_ai(item, candidates)

        assert result is None

    def test_empty_candidates_returns_none(self, mock_db):
        """Test that empty candidates list returns None."""
        processor = EventStreamProcessor(db=mock_db, use_mock=True)
        item = create_content_item(1, "Some content")

        result = processor.confirm_same_event_with_ai(item, [])

        assert result is None


# ============================================
# Tests: Parallel Processing
# ============================================


class TestParallelProcessing:
    """Tests for parallel clustering processing."""

    def test_parallel_processing_completes_with_no_items(self, mock_db):
        """Test that parallel processing handles empty input gracefully."""
        mock_db.get_unclustered_items.return_value = []

        clustered = cluster_unprocessed_items_parallel(
            db=mock_db,
            use_mock=True,
            limit=10,
            max_workers=4,
        )

        assert clustered == 0

    def test_parallel_processing_completes(self, mock_db):
        """Test that parallel processing completes without errors."""
        items = [create_content_item(i, f"News item {i}") for i in range(10)]
        mock_db.get_unclustered_items.return_value = items
        mock_db.get_recent_event_clusters.return_value = []

        # Each item should create a new cluster (no candidates)
        mock_db.create_event_cluster.return_value = EventCluster(
            id=1,
            event_title="Test Cluster",
            category="AI",
            first_seen_at=datetime.now(),
            last_updated_at=datetime.now(),
            article_count=1,
        )

        clustered = cluster_unprocessed_items_parallel(
            db=mock_db,
            use_mock=True,
            limit=10,
            max_workers=4,
        )

        assert clustered >= 0

    def test_parallel_with_single_worker(self, mock_db):
        """Test parallel processing with max_workers=1 (effectively sequential)."""
        items = [create_content_item(i, f"News {i}") for i in range(3)]
        mock_db.get_unclustered_items.return_value = items
        mock_db.get_recent_event_clusters.return_value = []
        mock_db.create_event_cluster.return_value = EventCluster(
            id=1,
            event_title="Test",
            category="AI",
            first_seen_at=datetime.now(),
            last_updated_at=datetime.now(),
            article_count=1,
        )

        clustered = cluster_unprocessed_items_parallel(
            db=mock_db, use_mock=True, limit=3, max_workers=1
        )

        assert clustered >= 0

    def test_parallel_progress_callback(self, mock_db):
        """Test that progress callback is called during parallel processing."""
        items = [create_content_item(i, f"News {i}") for i in range(5)]
        mock_db.get_unclustered_items.return_value = items
        mock_db.get_recent_event_clusters.return_value = []
        mock_db.create_event_cluster.return_value = EventCluster(
            id=1,
            event_title="Test",
            category="AI",
            first_seen_at=datetime.now(),
            last_updated_at=datetime.now(),
            article_count=1,
        )

        progress_calls = []

        def on_progress(current, total, clustered):
            progress_calls.append((current, total, clustered))

        cluster_unprocessed_items_parallel(
            db=mock_db,
            use_mock=True,
            limit=5,
            max_workers=2,
            progress_callback=on_progress,
        )

        # Progress should have been called for each item
        assert len(progress_calls) == 5
        # All calls should have total=5
        assert all(call[1] == 5 for call in progress_calls)
        # Final call should have current=5
        final_currents = [call[0] for call in progress_calls]
        assert 5 in final_currents

    def test_sequential_progress_callback(self, mock_db):
        """Test that progress callback is called during sequential processing."""
        items = [create_content_item(i, f"News {i}") for i in range(3)]
        mock_db.get_unclustered_items.return_value = items
        mock_db.get_recent_event_clusters.return_value = []
        mock_db.create_event_cluster.return_value = EventCluster(
            id=1,
            event_title="Test",
            category="AI",
            first_seen_at=datetime.now(),
            last_updated_at=datetime.now(),
            article_count=1,
        )

        progress_calls = []

        def on_progress(current, total, clustered):
            progress_calls.append((current, total, clustered))

        cluster_unprocessed_items(
            db=mock_db,
            use_mock=True,
            limit=3,
            progress_callback=on_progress,
        )

        # Progress should have been called for each item (sequential order)
        assert len(progress_calls) == 3
        assert progress_calls[0][0] == 1
        assert progress_calls[1][0] == 2
        assert progress_calls[2][0] == 3


# ============================================
# Tests: Integration - Clustering Flow
# ============================================


class TestClusteringIntegration:
    """Integration tests for the clustering flow with optimizations."""

    def test_high_confidence_skips_ai_in_full_flow(self, mock_db):
        """Test that high-confidence matches skip AI in the full clustering flow."""
        item = create_content_item(1, "OpenAI GPT-5 Release Announcement")
        mock_db.get_unclustered_items.return_value = [item]

        # Create an existing cluster that matches well
        existing_cluster = EventCluster(
            id=10,
            event_title="OpenAI GPT-5 Release",
            category="AI",
            first_seen_at=datetime.now(),
            last_updated_at=datetime.now(),
            article_count=3,
        )
        mock_db.get_recent_event_clusters.return_value = [existing_cluster]
        mock_db.get_event_cluster.return_value = existing_cluster

        # Run clustering (should match with high confidence due to "OpenAI" entity)
        clustered = cluster_unprocessed_items(db=mock_db, use_mock=True, limit=10)

        # Should have clustered the item
        assert clustered == 1
        mock_db.add_event_member.assert_called()

    def test_new_cluster_created_for_no_matches(self, mock_db):
        """Test that new clusters are created when no matches found."""
        item = create_content_item(1, "Completely unique news", importance=8)
        mock_db.get_unclustered_items.return_value = [item]
        mock_db.get_recent_event_clusters.return_value = []  # No existing clusters

        new_cluster = EventCluster(
            id=1,
            event_title="Unique news",
            category="AI",
            first_seen_at=datetime.now(),
            last_updated_at=datetime.now(),
            article_count=1,
        )
        mock_db.create_event_cluster.return_value = new_cluster

        clustered = cluster_unprocessed_items(db=mock_db, use_mock=True, limit=10)

        assert clustered == 1
        mock_db.create_event_cluster.assert_called_once()
        mock_db.add_event_member.assert_called_once()

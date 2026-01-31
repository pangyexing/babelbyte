"""Tests for digest idempotency - ensuring multiple runs don't cause duplicate processing."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.analytics.token_tracker import TokenTracker, AICallType, get_tracker
from src.storage.database import SyncDatabase
from src.storage.models import (
    ContentItem,
    EventCluster,
    EventMember,
    SourceType,
    Subscription,
    SubscriptionType,
)


class TestDigestIdempotency:
    """Test that multiple runs of digest don't cause redundant AI calls."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SyncDatabase(db_path)
            db.connect()
            yield db
            db.close()

    @pytest.fixture
    def sample_subscription(self, temp_db):
        """Create a sample subscription."""
        sub = Subscription(
            source_type=SourceType.REDDIT,
            subscription_type=SubscriptionType.SUBREDDIT,
            name="test",
            enabled=True,
            created_at=datetime.now(),
        )
        return temp_db.add_subscription(sub)

    @pytest.fixture
    def tracker(self):
        """Get and reset the token tracker."""
        tracker = get_tracker()
        tracker.reset()
        return tracker

    def test_processed_items_not_reprocessed(self, temp_db, sample_subscription, tracker):
        """Already processed items should not trigger AI calls on subsequent runs."""
        # Add a processed item
        item = ContentItem(
            subscription_id=sample_subscription.id,
            source_type=SourceType.REDDIT,
            external_id="processed1",
            title="Already processed item",
            content="This item was processed before",
            url="https://example.com/1",
            author="test",
            published_at=datetime.now(),
            fetched_at=datetime.now(),
            processed_at=datetime.now(),  # Mark as processed
            summary="This is a summary",
            category="技术",
            importance_score=7,
        )
        temp_db.add_content_item(item)

        # Get unprocessed items - should be empty
        unprocessed = temp_db.get_unprocessed_items(limit=100)
        assert (
            len(unprocessed) == 0
        ), "Processed items should not be returned by get_unprocessed_items"

    def test_clustered_items_not_reclustered(self, temp_db, sample_subscription):
        """Items already in a cluster should not be considered for clustering again."""
        # Add a processed item
        item = ContentItem(
            subscription_id=sample_subscription.id,
            source_type=SourceType.REDDIT,
            external_id="clustered1",
            title="Clustered item",
            content="This item is already clustered",
            url="https://example.com/2",
            author="test",
            published_at=datetime.now(),
            fetched_at=datetime.now(),
            processed_at=datetime.now(),
            summary="Summary",
            category="AI",
            importance_score=8,
        )
        item = temp_db.add_content_item(item)

        # Create a cluster and add the item
        cluster = EventCluster(
            event_title="Test Event",
            category="AI",
            first_seen_at=datetime.now(),
            last_updated_at=datetime.now(),
            article_count=1,
        )
        cluster = temp_db.create_event_cluster(cluster)

        member = EventMember(
            event_cluster_id=cluster.id,
            content_item_id=item.id,
            similarity_score=0.9,
            detection_method="test",
        )
        temp_db.add_event_member(member)

        # Check that item is recognized as clustered
        assert temp_db.is_item_in_cluster(item.id), "Item should be in cluster"

        # Get unclustered items - should not include this item
        unclustered = temp_db.get_unclustered_items(min_importance=5, limit=100)
        item_ids = [i.id for i in unclustered]
        assert item.id not in item_ids, "Clustered items should not be in unclustered list"

    def test_delivered_items_excluded(self, temp_db, sample_subscription):
        """Items already delivered should not appear in new digest."""
        # Add a delivered item
        item = ContentItem(
            subscription_id=sample_subscription.id,
            source_type=SourceType.REDDIT,
            external_id="delivered1",
            title="Delivered item",
            content="This item was already delivered",
            url="https://example.com/3",
            author="test",
            published_at=datetime.now(),
            fetched_at=datetime.now(),
            processed_at=datetime.now(),
            summary="Summary",
            category="技术",
            importance_score=7,
            delivered=True,
            delivered_at=datetime.now(),
        )
        temp_db.add_content_item(item)

        # Get undelivered items
        undelivered = temp_db.get_undelivered_items(min_importance=5, limit=50)
        item_ids = [i.id for i in undelivered]
        assert item.id not in item_ids, "Delivered items should not be in undelivered list"

    def test_cluster_attempted_items_skipped(self, temp_db, sample_subscription):
        """Items with recent cluster_attempted_at should be skipped."""
        # Add an item with recent cluster attempt
        item = ContentItem(
            subscription_id=sample_subscription.id,
            source_type=SourceType.REDDIT,
            external_id="attempted1",
            title="Recently attempted item",
            content="Clustering was attempted recently",
            url="https://example.com/4",
            author="test",
            published_at=datetime.now(),
            fetched_at=datetime.now(),
            processed_at=datetime.now(),
            summary="Summary",
            category="技术",
            importance_score=6,
        )
        item = temp_db.add_content_item(item)

        # Mark as recently attempted
        temp_db.mark_cluster_attempted(item.id)

        # Get unclustered items with retry filter
        unclustered = temp_db.get_unclustered_items(
            min_importance=5, limit=100, retry_after_hours=24
        )
        item_ids = [i.id for i in unclustered]
        assert item.id not in item_ids, "Recently attempted items should be skipped"

    def test_old_cluster_attempts_retried(self, temp_db, sample_subscription):
        """Items with old cluster_attempted_at should be retried."""
        # Add an item with old cluster attempt
        item = ContentItem(
            subscription_id=sample_subscription.id,
            source_type=SourceType.REDDIT,
            external_id="old_attempt1",
            title="Old attempted item",
            content="Clustering was attempted long ago",
            url="https://example.com/5",
            author="test",
            published_at=datetime.now(),
            fetched_at=datetime.now(),
            processed_at=datetime.now(),
            summary="Summary",
            category="技术",
            importance_score=6,
        )
        item = temp_db.add_content_item(item)

        # Simulate old cluster attempt (manually set in database)
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        conn = temp_db._async_db._connection
        temp_db._run(
            conn.execute(
                "UPDATE content_items SET cluster_attempted_at = ? WHERE id = ?",
                (old_time, item.id),
            )
        )
        temp_db._run(conn.commit())

        # Get unclustered items with 24-hour retry filter
        unclustered = temp_db.get_unclustered_items(
            min_importance=5, limit=100, retry_after_hours=24
        )
        item_ids = [i.id for i in unclustered]
        assert item.id in item_ids, "Items with old cluster attempts should be retried"


class TestEventConfirmationCache:
    """Tests for event confirmation cache behavior."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SyncDatabase(db_path)
            db.connect()
            yield db
            db.close()

    def test_cache_stores_confirmation_result(self, temp_db):
        """Test that event confirmation results are cached."""
        from src.processors.event_stream import EventStreamProcessor, ClusterCandidate

        processor = EventStreamProcessor(db=temp_db, use_mock=True)

        # Create a mock item
        item = MagicMock()
        item.id = 1
        item.title = "Test Title"
        item.summary = "Test Summary"
        item.content = "Test Content"

        # Create candidates
        candidates = [
            ClusterCandidate(cluster_id=1, cluster_title="Event 1", score=0.4, method="keyword"),
            ClusterCandidate(cluster_id=2, cluster_title="Event 2", score=0.3, method="keyword"),
        ]

        # Call confirm (mock mode accepts if score > 0.35)
        result = processor.confirm_same_event_with_ai(item, candidates)

        # Should return the first candidate (score 0.4 > 0.35)
        assert result is not None
        assert result.cluster_id == 1

    def test_cache_prevents_duplicate_ai_calls(self, temp_db):
        """Test that cached results prevent redundant AI calls."""
        from src.processors.event_stream import EventStreamProcessor, ClusterCandidate

        processor = EventStreamProcessor(db=temp_db, use_mock=True)

        item = MagicMock()
        item.id = 2
        item.title = "Another Test"
        item.summary = "Summary"
        item.content = "Content"

        candidates = [
            ClusterCandidate(cluster_id=3, cluster_title="Event 3", score=0.5, method="title"),
        ]

        # First call
        result1 = processor.confirm_same_event_with_ai(item, candidates)

        # Second call should hit cache
        result2 = processor.confirm_same_event_with_ai(item, candidates)

        # Both should return same result
        assert result1 == result2


class TestAICacheIntegration:
    """Tests for AI cache preventing duplicate content processing."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SyncDatabase(db_path)
            db.connect()
            yield db
            db.close()

    def test_ai_cache_hit_tracking(self, temp_db):
        """Test that AI cache hits are tracked correctly."""
        from config.settings import get_settings

        # Store a cached result
        content_hash = "test_hash_12345"
        cached_result = '{"summary": "Test", "category": "AI", "importance_score": 7}'
        temp_db.set_ai_cache(content_hash, cached_result, ttl_seconds=3600)

        # Verify it can be retrieved
        retrieved = temp_db.get_ai_cache(content_hash)
        assert retrieved == cached_result

    def test_expired_cache_not_returned(self, temp_db):
        """Test that expired cache entries are not returned."""
        content_hash = "expired_hash_123"
        cached_result = '{"summary": "Old", "category": "AI", "importance_score": 5}'

        # Store with very short TTL
        temp_db.set_ai_cache(content_hash, cached_result, ttl_seconds=1)

        # Wait for expiry (in real test, we'd mock datetime)
        import time

        time.sleep(1.1)

        # Should not be returned
        retrieved = temp_db.get_ai_cache(content_hash)
        assert retrieved is None


class TestDuplicateClusterMembership:
    """Tests for preventing duplicate cluster memberships."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SyncDatabase(db_path)
            db.connect()
            yield db
            db.close()

    @pytest.fixture
    def sample_subscription(self, temp_db):
        """Create a sample subscription."""
        sub = Subscription(
            source_type=SourceType.REDDIT,
            subscription_type=SubscriptionType.SUBREDDIT,
            name="test",
            enabled=True,
            created_at=datetime.now(),
        )
        return temp_db.add_subscription(sub)

    def test_item_can_only_be_in_one_cluster(self, temp_db, sample_subscription):
        """Test that an item can only belong to one cluster."""
        # Add an item
        item = ContentItem(
            subscription_id=sample_subscription.id,
            source_type=SourceType.REDDIT,
            external_id="unique1",
            title="Test item",
            content="Content",
            url="https://example.com/unique",
            author="test",
            published_at=datetime.now(),
            fetched_at=datetime.now(),
            processed_at=datetime.now(),
            summary="Summary",
            category="AI",
            importance_score=8,
        )
        item = temp_db.add_content_item(item)

        # Create two clusters
        cluster1 = EventCluster(
            event_title="Event 1",
            category="AI",
            first_seen_at=datetime.now(),
            last_updated_at=datetime.now(),
            article_count=0,
        )
        cluster1 = temp_db.create_event_cluster(cluster1)

        cluster2 = EventCluster(
            event_title="Event 2",
            category="AI",
            first_seen_at=datetime.now(),
            last_updated_at=datetime.now(),
            article_count=0,
        )
        cluster2 = temp_db.create_event_cluster(cluster2)

        # Add item to first cluster
        member1 = EventMember(
            event_cluster_id=cluster1.id,
            content_item_id=item.id,
            similarity_score=0.8,
            detection_method="test",
        )
        result1 = temp_db.add_event_member(member1)
        assert result1 is True, "First membership should succeed"

        # Try to add same item to second cluster - should fail (INSERT OR IGNORE)
        member2 = EventMember(
            event_cluster_id=cluster2.id,
            content_item_id=item.id,
            similarity_score=0.9,
            detection_method="test",
        )
        result2 = temp_db.add_event_member(member2)
        assert result2 is False, "Duplicate membership should be ignored"

        # Verify item is only in first cluster
        members1 = temp_db.get_event_members(cluster1.id)
        members2 = temp_db.get_event_members(cluster2.id)
        assert len(members1) == 1
        assert len(members2) == 0

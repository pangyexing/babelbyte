"""Tests for event clustering integration in digest."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from src.storage.database import Database, SyncDatabase
from src.storage.models import (
    ContentItem,
    DigestItem,
    EventCluster,
    EventDigestItem,
    EventMember,
    SourceType,
)
from src.processors.digest_processor import (
    DigestGenerator,
    DigestResult,
    create_digest_preview,
)


# ============================================
# Fixtures
# ============================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def sync_db(temp_db_path):
    """Create a sync database for testing."""
    db = SyncDatabase(db_path=temp_db_path)
    db.connect()
    yield db
    db.close()


@pytest.fixture
def sample_content_items():
    """Create sample content items for testing."""
    now = datetime.now()
    return [
        ContentItem(
            id=1,
            subscription_id=1,
            source_type=SourceType.REDDIT,
            external_id="reddit_1",
            title="OpenAI announces GPT-5",
            content="OpenAI has announced the release of GPT-5...",
            url="https://example.com/1",
            author="user1",
            published_at=now,
            fetched_at=now,
            summary="OpenAI announced GPT-5 with improved capabilities.",
            category="AI",
            importance_score=9,
            processed_at=now,
            one_liner="OpenAI releases GPT-5, a major leap in AI.",
            delivered=False,
        ),
        ContentItem(
            id=2,
            subscription_id=1,
            source_type=SourceType.TWITTER,
            external_id="twitter_1",
            title="GPT-5 reaction from Sam Altman",
            content="Sam Altman tweets about GPT-5 launch...",
            url="https://example.com/2",
            author="samaltman",
            published_at=now,
            fetched_at=now,
            summary="Sam Altman shares excitement about GPT-5.",
            category="AI",
            importance_score=8,
            processed_at=now,
            one_liner="Sam Altman celebrates GPT-5 launch.",
            delivered=False,
        ),
        ContentItem(
            id=3,
            subscription_id=2,
            source_type=SourceType.REDDIT,
            external_id="reddit_2",
            title="Google releases Gemini 2.0",
            content="Google has released Gemini 2.0...",
            url="https://example.com/3",
            author="user2",
            published_at=now,
            fetched_at=now,
            summary="Google releases Gemini 2.0 to compete with GPT-5.",
            category="AI",
            importance_score=7,
            processed_at=now,
            one_liner="Google counters with Gemini 2.0 release.",
            delivered=False,
        ),
        ContentItem(
            id=4,
            subscription_id=1,
            source_type=SourceType.REDDIT,
            external_id="reddit_3",
            title="Unrelated tech news",
            content="Some other tech news...",
            url="https://example.com/4",
            author="user3",
            published_at=now,
            fetched_at=now,
            summary="Some unrelated tech news article.",
            category="Tech",
            importance_score=5,
            processed_at=now,
            delivered=False,
        ),
    ]


@pytest.fixture
def sample_event_cluster():
    """Create a sample event cluster."""
    now = datetime.now()
    return EventCluster(
        id=1,
        event_title="OpenAI GPT-5 Release",
        category="AI",
        first_seen_at=now,
        last_updated_at=now,
        article_count=2,
    )


# ============================================
# Tests: EventDigestItem Model
# ============================================


class TestEventDigestItem:
    """Tests for the EventDigestItem dataclass."""

    def test_event_title_includes_count(self, sample_content_items, sample_event_cluster):
        """Test that event_title includes article count."""
        members = sample_content_items[:2]
        event_item = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=members,
            representative_item=members[0],
        )
        assert "(2篇报道)" in event_item.event_title
        assert "OpenAI GPT-5 Release" in event_item.event_title

    def test_importance_score_is_max(self, sample_content_items, sample_event_cluster):
        """Test that importance_score is max of members."""
        members = sample_content_items[:2]  # scores 9 and 8
        event_item = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=members,
            representative_item=members[0],
        )
        assert event_item.importance_score == 9

    def test_category_from_cluster(self, sample_content_items, sample_event_cluster):
        """Test that category comes from cluster."""
        members = sample_content_items[:2]
        event_item = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=members,
            representative_item=members[0],
        )
        assert event_item.category == "AI"

    def test_summary_from_representative(self, sample_content_items, sample_event_cluster):
        """Test that summary comes from representative item."""
        members = sample_content_items[:2]
        event_item = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=members,
            representative_item=members[0],
        )
        assert event_item.summary == members[0].summary

    def test_one_liner_from_representative(self, sample_content_items, sample_event_cluster):
        """Test that one_liner comes from representative item."""
        members = sample_content_items[:2]
        event_item = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=members,
            representative_item=members[0],
        )
        assert event_item.one_liner == members[0].one_liner

    def test_content_item_returns_representative(self, sample_content_items, sample_event_cluster):
        """Test that content_item property returns representative."""
        members = sample_content_items[:2]
        event_item = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=members,
            representative_item=members[0],
        )
        assert event_item.content_item == members[0]

    def test_source_display_multiple_sources(self, sample_content_items, sample_event_cluster):
        """Test source_display with multiple source types."""
        members = sample_content_items[:2]  # Reddit and Twitter
        event_item = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=members,
            representative_item=members[0],
        )
        assert "Reddit" in event_item.source_display
        assert "Twitter" in event_item.source_display

    def test_is_event_returns_true(self, sample_content_items, sample_event_cluster):
        """Test that is_event returns True for EventDigestItem."""
        members = sample_content_items[:2]
        event_item = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=members,
            representative_item=members[0],
        )
        assert event_item.is_event is True


class TestDigestItemIsEvent:
    """Test that DigestItem.is_event returns False."""

    def test_digest_item_is_event_false(self, sample_content_items):
        """Test that is_event returns False for DigestItem."""
        item = sample_content_items[0]
        digest_item = DigestItem(
            content_item=item,
            summary=item.summary,
            category=item.category,
            importance_score=item.importance_score,
        )
        assert digest_item.is_event is False


# ============================================
# Tests: DigestResult with Events
# ============================================


class TestDigestResult:
    """Tests for the updated DigestResult dataclass."""

    def test_total_items_includes_event_members(self, sample_content_items, sample_event_cluster):
        """Test that total_items includes all event members."""
        # Create an event with 2 members
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        # Create a regular item
        regular_item = DigestItem(
            content_item=sample_content_items[3],
            summary=sample_content_items[3].summary,
            category=sample_content_items[3].category,
            importance_score=sample_content_items[3].importance_score,
        )

        result = DigestResult(
            items=[regular_item],
            events=[event],
        )

        # total_items = 1 regular + 2 event members = 3
        assert result.total_items == 3

    def test_by_category_includes_both_events_and_items(
        self, sample_content_items, sample_event_cluster
    ):
        """Test that by_category includes both events and items."""
        # Event in AI category
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        # Regular item in AI category
        ai_item = DigestItem(
            content_item=sample_content_items[2],
            summary=sample_content_items[2].summary,
            category="AI",
            importance_score=sample_content_items[2].importance_score,
        )

        # Regular item in Tech category
        tech_item = DigestItem(
            content_item=sample_content_items[3],
            summary=sample_content_items[3].summary,
            category="Tech",
            importance_score=sample_content_items[3].importance_score,
        )

        result = DigestResult(
            items=[ai_item, tech_item],
            events=[event],
        )

        by_cat = result.by_category
        assert "AI" in by_cat
        assert "Tech" in by_cat
        assert len(by_cat["AI"]) == 2  # event + ai_item
        assert len(by_cat["Tech"]) == 1  # tech_item

    def test_by_category_sorted_by_importance(self, sample_content_items, sample_event_cluster):
        """Test that by_category items are sorted by importance."""
        # Event with importance 9
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        # Regular item with importance 7
        regular_item = DigestItem(
            content_item=sample_content_items[2],
            summary=sample_content_items[2].summary,
            category="AI",
            importance_score=7,
        )

        result = DigestResult(
            items=[regular_item],
            events=[event],
        )

        by_cat = result.by_category
        ai_items = by_cat["AI"]

        # Event (importance 9) should come first
        assert ai_items[0].is_event is True
        assert ai_items[0].importance_score == 9
        assert ai_items[1].is_event is False
        assert ai_items[1].importance_score == 7

    def test_all_content_ids(self, sample_content_items, sample_event_cluster):
        """Test that all_content_ids includes all item IDs."""
        # Event with 2 members (IDs 1, 2)
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        # Regular item (ID 4)
        regular_item = DigestItem(
            content_item=sample_content_items[3],
            summary=sample_content_items[3].summary,
            category=sample_content_items[3].category,
            importance_score=sample_content_items[3].importance_score,
        )

        result = DigestResult(
            items=[regular_item],
            events=[event],
        )

        ids = result.all_content_ids
        assert set(ids) == {1, 2, 4}

    def test_empty_digest_result(self):
        """Test an empty DigestResult."""
        result = DigestResult()
        assert result.total_items == 0
        assert result.all_content_ids == []
        assert result.by_category == {}


# ============================================
# Tests: create_digest_preview
# ============================================


class TestCreateDigestPreview:
    """Tests for the create_digest_preview function."""

    def test_preview_shows_event_count(self, sample_content_items, sample_event_cluster):
        """Test that preview shows event count."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        regular_item = DigestItem(
            content_item=sample_content_items[3],
            summary=sample_content_items[3].summary,
            category=sample_content_items[3].category,
            importance_score=sample_content_items[3].importance_score,
        )

        result = DigestResult(items=[regular_item], events=[event])
        preview = create_digest_preview(result)

        assert "1 个事件" in preview
        assert "1 条独立内容" in preview

    def test_preview_shows_event_marker(self, sample_content_items, sample_event_cluster):
        """Test that preview shows [EVENT] marker for events."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        result = DigestResult(items=[], events=[event])
        preview = create_digest_preview(result)

        assert "[EVENT]" in preview

    def test_preview_shows_related_sources(self, sample_content_items, sample_event_cluster):
        """Test that preview shows related sources for events."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        result = DigestResult(items=[], events=[event])
        preview = create_digest_preview(result)

        assert "相关报道:" in preview

    def test_preview_empty_digest(self):
        """Test preview for empty digest."""
        result = DigestResult()
        preview = create_digest_preview(result)

        assert "暂无新内容" in preview

    def test_preview_regular_items_only(self, sample_content_items):
        """Test preview with only regular items (no events)."""
        regular_item = DigestItem(
            content_item=sample_content_items[0],
            summary=sample_content_items[0].summary,
            category=sample_content_items[0].category,
            importance_score=sample_content_items[0].importance_score,
        )

        result = DigestResult(items=[regular_item], events=[])
        preview = create_digest_preview(result)

        # Should not show event-specific text
        assert "[EVENT]" not in preview
        assert "个事件" not in preview


# ============================================
# Tests: Database Query Methods
# ============================================


@pytest.fixture
def populated_db(sync_db, sample_content_items, sample_event_cluster):
    """Populate database with test data."""
    # We need to use the database to add subscriptions first
    from src.storage.models import Subscription, SubscriptionType

    # Add subscriptions
    sub1 = Subscription(
        source_type=SourceType.REDDIT,
        subscription_type=SubscriptionType.SUBREDDIT,
        name="MachineLearning",
        enabled=True,
        created_at=datetime.now(),
    )
    sub1 = sync_db.add_subscription(sub1)

    sub2 = Subscription(
        source_type=SourceType.TWITTER,
        subscription_type=SubscriptionType.TWITTER_USER,
        name="samaltman",
        enabled=True,
        created_at=datetime.now(),
    )
    sub2 = sync_db.add_subscription(sub2)

    # Add content items
    now = datetime.now()
    items = []
    for i, data in enumerate(
        [
            ("OpenAI GPT-5", "AI", 9, SourceType.REDDIT),
            ("GPT-5 reaction", "AI", 8, SourceType.TWITTER),
            ("Gemini 2.0", "AI", 7, SourceType.REDDIT),
            ("Unrelated news", "Tech", 5, SourceType.REDDIT),
        ]
    ):
        title, category, importance, source = data
        item = ContentItem(
            subscription_id=sub1.id if source == SourceType.REDDIT else sub2.id,
            source_type=source,
            external_id=f"ext_{i}",
            title=title,
            content=f"Content for {title}",
            url=f"https://example.com/{i}",
            author=f"author_{i}",
            published_at=now,
            fetched_at=now,
            summary=f"Summary of {title}",
            category=category,
            importance_score=importance,
            processed_at=now,
            delivered=False,
        )
        item = sync_db.add_content_item(item)
        items.append(item)

    # Create event cluster
    cluster = EventCluster(
        event_title="GPT-5 Release",
        category="AI",
        first_seen_at=now,
        last_updated_at=now,
        article_count=2,
    )
    cluster = sync_db.create_event_cluster(cluster)

    # Add first two items to the cluster
    for item in items[:2]:
        member = EventMember(
            event_cluster_id=cluster.id,
            content_item_id=item.id,
            similarity_score=0.9,
            detection_method="test",
        )
        sync_db.add_event_member(member)

    return {
        "db": sync_db,
        "items": items,
        "cluster": cluster,
    }


class TestDatabaseQueryMethods:
    """Tests for the new database query methods."""

    def test_get_undelivered_clustered_items(self, populated_db):
        """Test getting clustered items grouped by cluster ID."""
        db = populated_db["db"]
        cluster = populated_db["cluster"]

        result = db.get_undelivered_clustered_items(min_importance=5, limit=50)

        assert cluster.id in result
        assert len(result[cluster.id]) == 2

    def test_get_undelivered_clustered_items_importance_filter(self, populated_db):
        """Test that importance filter works for clustered items."""
        db = populated_db["db"]

        # With high importance filter, should get fewer items
        result = db.get_undelivered_clustered_items(min_importance=9, limit=50)

        # Only the item with importance 9 should be returned
        total_items = sum(len(members) for members in result.values())
        assert total_items == 1

    def test_get_undelivered_unclustered_items(self, populated_db):
        """Test getting unclustered items."""
        db = populated_db["db"]

        result = db.get_undelivered_unclustered_items(min_importance=5, limit=50)

        # Items 3 and 4 are not in any cluster
        assert len(result) == 2
        titles = [item.title for item in result]
        assert "Gemini 2.0" in titles
        assert "Unrelated news" in titles

    def test_get_undelivered_unclustered_items_importance_filter(self, populated_db):
        """Test that importance filter works for unclustered items."""
        db = populated_db["db"]

        # With importance >= 6, should exclude "Unrelated news" (importance 5)
        result = db.get_undelivered_unclustered_items(min_importance=6, limit=50)

        assert len(result) == 1
        assert result[0].title == "Gemini 2.0"

    def test_clustered_and_unclustered_are_disjoint(self, populated_db):
        """Test that clustered and unclustered items don't overlap."""
        db = populated_db["db"]

        clustered = db.get_undelivered_clustered_items(min_importance=1, limit=100)
        unclustered = db.get_undelivered_unclustered_items(min_importance=1, limit=100)

        clustered_ids = set()
        for members in clustered.values():
            for item in members:
                clustered_ids.add(item.id)

        unclustered_ids = {item.id for item in unclustered}

        # No overlap
        assert clustered_ids.isdisjoint(unclustered_ids)


# ============================================
# Tests: DigestGenerator with Clustering
# ============================================


class TestDigestGeneratorWithClustering:
    """Tests for DigestGenerator with event clustering."""

    def test_generate_digest_includes_events(self, populated_db):
        """Test that generate_digest includes events."""
        db = populated_db["db"]

        generator = DigestGenerator(db=db, use_mock=True)
        result = generator.generate_digest(
            min_importance=5,
            max_items=30,
            run_clustering=False,  # Don't run clustering, use existing
        )

        assert len(result.events) == 1
        assert result.events[0].event_cluster.event_title == "GPT-5 Release"

    def test_generate_digest_includes_unclustered(self, populated_db):
        """Test that generate_digest includes unclustered items."""
        db = populated_db["db"]

        generator = DigestGenerator(db=db, use_mock=True)
        result = generator.generate_digest(
            min_importance=5,
            max_items=30,
            run_clustering=False,
        )

        # Items 3 and 4 are not clustered
        assert len(result.items) == 2

    def test_generate_digest_no_cluster_option(self, populated_db):
        """Test that run_clustering=False skips clustering."""
        db = populated_db["db"]

        generator = DigestGenerator(db=db, use_mock=True)

        # This should not fail even without running clustering
        result = generator.generate_digest(
            min_importance=5,
            max_items=30,
            run_clustering=False,
        )

        assert result is not None

    def test_generate_digest_empty_when_no_content(self, sync_db):
        """Test that generate_digest returns empty result when no content."""
        generator = DigestGenerator(db=sync_db, use_mock=True)
        result = generator.generate_digest(
            min_importance=5,
            max_items=30,
            run_clustering=False,
        )

        assert len(result.items) == 0
        assert len(result.events) == 0

    def test_mark_digest_delivered_marks_event_members(self, populated_db):
        """Test that mark_digest_delivered marks all event members."""
        db = populated_db["db"]

        generator = DigestGenerator(db=db, use_mock=True)
        result = generator.generate_digest(
            min_importance=5,
            max_items=30,
            run_clustering=False,
        )

        # Mark as delivered
        generator.mark_digest_delivered(result)

        # Verify all items are marked
        for item_id in result.all_content_ids:
            item = db.get_content_item(item_id)
            assert item.delivered is True


# ============================================
# Run tests if executed directly
# ============================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

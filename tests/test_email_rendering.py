"""Tests for email rendering with event support."""

import pytest
from datetime import datetime

from src.storage.models import (
    ContentItem,
    DigestItem,
    EventCluster,
    EventDigestItem,
    SourceType,
)
from src.processors.digest_processor import DigestResult
from src.delivery.email_sender import EmailSender


# ============================================
# Fixtures
# ============================================


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
            title="OpenAI announces GPT-5 with breakthrough capabilities",
            content="OpenAI has announced the release of GPT-5...",
            url="https://example.com/1",
            author="user1",
            published_at=now,
            fetched_at=now,
            summary="OpenAI announced GPT-5 with improved reasoning and longer context.",
            category="AI",
            importance_score=9,
            processed_at=now,
            one_liner="OpenAI releases GPT-5, a major leap in AI capabilities.",
            delivered=False,
        ),
        ContentItem(
            id=2,
            subscription_id=2,
            source_type=SourceType.TWITTER,
            external_id="twitter_1",
            title="Sam Altman announces GPT-5 release",
            content="Sam Altman tweets about GPT-5 launch...",
            url="https://twitter.com/samaltman/status/123",
            author="samaltman",
            published_at=now,
            fetched_at=now,
            summary="Sam Altman shares excitement about GPT-5 release.",
            category="AI",
            importance_score=8,
            processed_at=now,
            one_liner="Sam Altman celebrates GPT-5 launch.",
            delivered=False,
        ),
        ContentItem(
            id=3,
            subscription_id=1,
            source_type=SourceType.REDDIT,
            external_id="reddit_2",
            title="Unrelated tech news about cloud computing",
            content="Some other tech news...",
            url="https://example.com/3",
            author="user3",
            published_at=now,
            fetched_at=now,
            summary="Article about cloud computing developments.",
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


@pytest.fixture
def email_sender():
    """Create an email sender instance."""
    return EmailSender()


# ============================================
# Tests: HTML Rendering
# ============================================


class TestEmailHtmlRendering:
    """Tests for email HTML rendering with events."""

    def test_render_html_with_events(
        self, email_sender, sample_content_items, sample_event_cluster
    ):
        """Test that HTML rendering works with events."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        regular_item = DigestItem(
            content_item=sample_content_items[2],
            summary=sample_content_items[2].summary,
            category=sample_content_items[2].category,
            importance_score=sample_content_items[2].importance_score,
        )

        result = DigestResult(items=[regular_item], events=[event])
        html = email_sender._render_digest_html(result)

        # Check basic structure
        assert "BabelByte" in html
        assert "<!DOCTYPE html>" in html

    def test_html_shows_event_count(
        self, email_sender, sample_content_items, sample_event_cluster
    ):
        """Test that HTML shows event count in stats."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        result = DigestResult(items=[], events=[event])
        html = email_sender._render_digest_html(result)

        # Should show 1 event
        assert "1" in html
        assert "个事件" in html

    def test_html_shows_event_badge(
        self, email_sender, sample_content_items, sample_event_cluster
    ):
        """Test that HTML shows event badge for events."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        result = DigestResult(items=[], events=[event])
        html = email_sender._render_digest_html(result)

        assert "event-badge" in html
        assert "事件" in html

    def test_html_shows_event_sources(
        self, email_sender, sample_content_items, sample_event_cluster
    ):
        """Test that HTML shows related sources for events."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        result = DigestResult(items=[], events=[event])
        html = email_sender._render_digest_html(result)

        assert "event-sources" in html
        assert "相关报道" in html
        # Check that member URLs are included
        assert "example.com/1" in html
        assert "twitter.com/samaltman" in html

    def test_html_shows_total_items_including_event_members(
        self, email_sender, sample_content_items, sample_event_cluster
    ):
        """Test that total_items count includes event members."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        regular_item = DigestItem(
            content_item=sample_content_items[2],
            summary=sample_content_items[2].summary,
            category=sample_content_items[2].category,
            importance_score=sample_content_items[2].importance_score,
        )

        result = DigestResult(items=[regular_item], events=[event])

        # total_items = 2 event members + 1 regular = 3
        assert result.total_items == 3

        html = email_sender._render_digest_html(result)
        # Should show "3" in the stats
        assert ">3<" in html or ">3 " in html or " 3<" in html

    def test_html_without_events(self, email_sender, sample_content_items):
        """Test HTML rendering without events (backward compatibility)."""
        regular_item = DigestItem(
            content_item=sample_content_items[0],
            summary=sample_content_items[0].summary,
            category=sample_content_items[0].category,
            importance_score=sample_content_items[0].importance_score,
        )

        result = DigestResult(items=[regular_item], events=[])
        html = email_sender._render_digest_html(result)

        # Should not show event count section
        assert "个事件" not in html
        # Should show regular item
        assert sample_content_items[0].title[:30] in html


# ============================================
# Tests: Plain Text Rendering
# ============================================


class TestEmailTextRendering:
    """Tests for email plain text rendering with events."""

    def test_render_text_with_events(
        self, email_sender, sample_content_items, sample_event_cluster
    ):
        """Test that text rendering works with events."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        result = DigestResult(items=[], events=[event])
        text = email_sender._render_digest_text(result)

        # Check basic structure
        assert "BabelByte" in text
        assert "每日摘要" in text

    def test_text_shows_event_count(
        self, email_sender, sample_content_items, sample_event_cluster
    ):
        """Test that text shows event count."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        result = DigestResult(items=[], events=[event])
        text = email_sender._render_digest_text(result)

        assert "1 个事件" in text

    def test_text_shows_event_marker(
        self, email_sender, sample_content_items, sample_event_cluster
    ):
        """Test that text shows [事件] marker."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        result = DigestResult(items=[], events=[event])
        text = email_sender._render_digest_text(result)

        assert "[事件]" in text

    def test_text_shows_related_sources(
        self, email_sender, sample_content_items, sample_event_cluster
    ):
        """Test that text shows related sources."""
        event_members = sample_content_items[:2]
        event = EventDigestItem(
            event_cluster=sample_event_cluster,
            members=event_members,
            representative_item=event_members[0],
        )

        result = DigestResult(items=[], events=[event])
        text = email_sender._render_digest_text(result)

        assert "相关报道:" in text

    def test_text_without_events(self, email_sender, sample_content_items):
        """Test text rendering without events."""
        regular_item = DigestItem(
            content_item=sample_content_items[0],
            summary=sample_content_items[0].summary,
            category=sample_content_items[0].category,
            importance_score=sample_content_items[0].importance_score,
        )

        result = DigestResult(items=[regular_item], events=[])
        text = email_sender._render_digest_text(result)

        # Should not show event marker
        assert "[事件]" not in text
        assert "个事件" not in text
        # Should show regular item
        assert sample_content_items[0].title[:30] in text

    def test_text_empty_digest(self, email_sender):
        """Test text rendering for empty digest."""
        result = DigestResult()
        text = email_sender._render_digest_text(result)

        assert "今日暂无新内容" in text


# ============================================
# Run tests if executed directly
# ============================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

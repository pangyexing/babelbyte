"""Tests for Hacker News and RSS fetchers."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.fetchers.hackernews import HackerNewsFetcher
from src.fetchers.rss import GenericRSSFetcher
from src.storage.models import (
    SourceType,
    Subscription,
    SubscriptionType,
)


class TestHackerNewsFetcher:
    """Tests for HackerNewsFetcher."""

    def test_source_type(self):
        """Test fetcher has correct source type."""
        fetcher = HackerNewsFetcher()
        assert fetcher.source_type == SourceType.HACKERNEWS

    def test_get_feed_url_front(self):
        """Test getting feed URL for frontpage."""
        fetcher = HackerNewsFetcher()
        sub = Subscription(
            source_type=SourceType.HACKERNEWS,
            subscription_type=SubscriptionType.HN_FRONT,
            name="tech",
        )
        url = fetcher._get_feed_url(sub)
        assert url == "https://hnrss.org/frontpage"

    def test_get_feed_url_best(self):
        """Test getting feed URL for best."""
        fetcher = HackerNewsFetcher()
        sub = Subscription(
            source_type=SourceType.HACKERNEWS,
            subscription_type=SubscriptionType.HN_BEST,
            name="tech",
        )
        url = fetcher._get_feed_url(sub)
        assert url == "https://hnrss.org/best"

    def test_get_feed_url_custom_override(self):
        """Test custom feed URL override."""
        fetcher = HackerNewsFetcher()
        sub = Subscription(
            source_type=SourceType.HACKERNEWS,
            subscription_type=SubscriptionType.HN_FRONT,
            name="custom",
            feed_url_override="https://hnrss.org/newest?points=100",
        )
        url = fetcher._get_feed_url(sub)
        assert url == "https://hnrss.org/newest?points=100"

    def test_clean_html(self):
        """Test HTML cleaning."""
        fetcher = HackerNewsFetcher()
        html = "<p>Hello <b>World</b></p>&amp; Test"
        cleaned = fetcher._clean_html(html)
        assert cleaned == "Hello World & Test"

    def test_should_fetch_content_short(self):
        """Test that short content triggers external fetch."""
        fetcher = HackerNewsFetcher()
        assert fetcher._should_fetch_content("short") is True
        assert fetcher._should_fetch_content("") is True

    def test_should_fetch_content_long(self):
        """Test that long content doesn't trigger external fetch."""
        fetcher = HackerNewsFetcher()
        long_content = "x" * 300
        assert fetcher._should_fetch_content(long_content) is False

    @pytest.mark.asyncio
    async def test_fetch_success(self):
        """Test successful fetch with mocked HTTP."""
        fetcher = HackerNewsFetcher(fetch_link_content=False)
        sub = Subscription(
            id=1,
            source_type=SourceType.HACKERNEWS,
            subscription_type=SubscriptionType.HN_FRONT,
            name="tech",
        )

        mock_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Hacker News</title>
                <item>
                    <title>Test Article</title>
                    <link>https://example.com/article</link>
                    <comments>https://news.ycombinator.com/item?id=123</comments>
                    <pubDate>Sat, 01 Jan 2025 12:00:00 GMT</pubDate>
                    <guid>https://news.ycombinator.com/item?id=123</guid>
                </item>
            </channel>
        </rss>"""

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.text = mock_rss
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await fetcher.fetch(sub)

            assert result.success is True
            assert len(result.items) == 1
            assert result.items[0].title == "Test Article"

    @pytest.mark.asyncio
    async def test_validate_subscription(self):
        """Test subscription validation."""
        fetcher = HackerNewsFetcher()
        sub = Subscription(
            source_type=SourceType.HACKERNEWS,
            subscription_type=SubscriptionType.HN_FRONT,
            name="tech",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200

            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            is_valid = await fetcher.validate_subscription(sub)
            assert is_valid is True


class TestGenericRSSFetcher:
    """Tests for GenericRSSFetcher."""

    def test_source_type(self):
        """Test fetcher has correct source type."""
        fetcher = GenericRSSFetcher()
        assert fetcher.source_type == SourceType.RSS

    def test_get_entry_id_from_id(self):
        """Test entry ID extraction from id field."""
        fetcher = GenericRSSFetcher()
        entry = {"id": "unique-id-123", "link": "https://example.com"}
        assert fetcher._get_entry_id(entry) == "unique-id-123"

    def test_get_entry_id_from_link(self):
        """Test entry ID extraction from link field."""
        fetcher = GenericRSSFetcher()
        entry = {"link": "https://example.com/article"}
        assert fetcher._get_entry_id(entry) == "https://example.com/article"

    def test_get_entry_id_fallback(self):
        """Test entry ID fallback to title+date."""
        fetcher = GenericRSSFetcher()
        entry = {"title": "Test", "published": "2025-01-01"}
        assert fetcher._get_entry_id(entry) == "Test:2025-01-01"

    def test_clean_html(self):
        """Test HTML cleaning."""
        fetcher = GenericRSSFetcher()
        html = "<div><p>Hello &amp; World</p></div>"
        cleaned = fetcher._clean_html(html)
        assert cleaned == "Hello & World"

    def test_should_fetch_full_content(self):
        """Test full content fetch determination."""
        fetcher = GenericRSSFetcher()
        assert fetcher._should_fetch_full_content("") is True
        assert fetcher._should_fetch_full_content("short") is True
        assert fetcher._should_fetch_full_content("x" * 600) is False

    @pytest.mark.asyncio
    async def test_fetch_no_url(self):
        """Test fetch fails without URL."""
        fetcher = GenericRSSFetcher()
        sub = Subscription(
            id=1,
            source_type=SourceType.RSS,
            subscription_type=SubscriptionType.RSS_FEED,
            name="Test Feed",
        )

        result = await fetcher.fetch(sub)
        assert result.success is False
        assert "No feed URL" in result.error_message

    @pytest.mark.asyncio
    async def test_fetch_success(self):
        """Test successful RSS fetch."""
        fetcher = GenericRSSFetcher(fetch_link_content=False)
        sub = Subscription(
            id=1,
            source_type=SourceType.RSS,
            subscription_type=SubscriptionType.RSS_FEED,
            name="Test Feed",
            feed_url_override="https://example.com/feed.xml",
        )

        mock_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <item>
                    <title>Test Article</title>
                    <link>https://example.com/article</link>
                    <description>Article description</description>
                    <pubDate>Sat, 01 Jan 2025 12:00:00 GMT</pubDate>
                    <guid>article-123</guid>
                </item>
            </channel>
        </rss>"""

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.text = mock_rss
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await fetcher.fetch(sub)

            assert result.success is True
            assert len(result.items) == 1
            assert result.items[0].title == "Test Article"


class TestSubscriptionModels:
    """Tests for subscription model enhancements."""

    def test_source_types(self):
        """Test all source types exist."""
        assert SourceType.REDDIT.value == "reddit"
        assert SourceType.TWITTER.value == "twitter"
        assert SourceType.HACKERNEWS.value == "hackernews"
        assert SourceType.RSS.value == "rss"

    def test_subscription_types(self):
        """Test all subscription types exist."""
        assert SubscriptionType.SUBREDDIT.value == "subreddit"
        assert SubscriptionType.HN_FRONT.value == "hn_front"
        assert SubscriptionType.HN_NEW.value == "hn_new"
        assert SubscriptionType.HN_BEST.value == "hn_best"
        assert SubscriptionType.HN_ASK.value == "hn_ask"
        assert SubscriptionType.HN_SHOW.value == "hn_show"
        assert SubscriptionType.RSS_FEED.value == "rss_feed"

    def test_subscription_display_name_hackernews(self):
        """Test display name for HN subscription."""
        sub = Subscription(
            source_type=SourceType.HACKERNEWS,
            subscription_type=SubscriptionType.HN_FRONT,
            name="tech",
        )
        assert sub.display_name == "HN Front: tech"

    def test_subscription_display_name_rss(self):
        """Test display name for RSS subscription."""
        sub = Subscription(
            source_type=SourceType.RSS,
            subscription_type=SubscriptionType.RSS_FEED,
            name="TechCrunch",
        )
        assert sub.display_name == "RSS: TechCrunch"

    def test_subscription_feed_url_hackernews(self):
        """Test feed URL property for HN."""
        sub = Subscription(
            source_type=SourceType.HACKERNEWS,
            subscription_type=SubscriptionType.HN_BEST,
            name="tech",
        )
        assert sub.feed_url == "https://hnrss.org/best"

    def test_subscription_feed_url_override(self):
        """Test feed URL override takes precedence."""
        sub = Subscription(
            source_type=SourceType.HACKERNEWS,
            subscription_type=SubscriptionType.HN_FRONT,
            name="tech",
            feed_url_override="https://custom.url/feed",
        )
        assert sub.feed_url == "https://custom.url/feed"

    def test_subscription_new_fields(self):
        """Test new subscription fields."""
        sub = Subscription(
            source_type=SourceType.RSS,
            subscription_type=SubscriptionType.RSS_FEED,
            name="test",
            feed_url_override="https://example.com/feed",
            last_entry_id="entry-123",
        )
        assert sub.feed_url_override == "https://example.com/feed"
        assert sub.last_entry_id == "entry-123"

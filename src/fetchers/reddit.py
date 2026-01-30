"""Reddit content fetcher using RSS feeds."""

import html
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional

import feedparser
import httpx

from src.fetchers.base import BaseFetcher, FetchResult
from src.storage.models import ContentItem, SourceType, Subscription, SubscriptionType


class RedditFetcher(BaseFetcher):
    """Fetcher for Reddit content using RSS feeds."""

    source_type = SourceType.REDDIT

    # Reddit RSS URLs
    SUBREDDIT_RSS_URL = "https://www.reddit.com/r/{name}/.rss"
    USER_RSS_URL = "https://www.reddit.com/user/{name}/.rss"

    # Custom User-Agent to avoid being blocked
    USER_AGENT = "BabelByte/1.0 (Content Aggregator)"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def _get_feed_url(self, subscription: Subscription) -> str:
        """Get the RSS feed URL for a subscription."""
        if subscription.subscription_type == SubscriptionType.SUBREDDIT:
            return self.SUBREDDIT_RSS_URL.format(name=subscription.name)
        return self.USER_RSS_URL.format(name=subscription.name)

    async def fetch(self, subscription: Subscription) -> FetchResult:
        """Fetch content from Reddit RSS feed."""
        feed_url = self._get_feed_url(subscription)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    feed_url,
                    headers={"User-Agent": self.USER_AGENT},
                    follow_redirects=True,
                )
                response.raise_for_status()

            # Parse the RSS feed
            feed = feedparser.parse(response.text)

            if feed.bozo and not feed.entries:
                return FetchResult(
                    subscription=subscription,
                    success=False,
                    error_message=f"Failed to parse RSS feed: {feed.bozo_exception}",
                )

            items = []
            for entry in feed.entries:
                item = self._parse_entry(subscription, entry)
                if item:
                    items.append(item)

            return FetchResult(
                subscription=subscription,
                items=items,
                success=True,
            )

        except httpx.HTTPStatusError as e:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"HTTP error {e.response.status_code}: {e.response.text[:200]}",
            )
        except httpx.RequestError as e:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"Request error: {str(e)}",
            )
        except Exception as e:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    async def validate_subscription(self, subscription: Subscription) -> bool:
        """Validate that a Reddit subscription exists and is accessible."""
        feed_url = self._get_feed_url(subscription)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.head(
                    feed_url,
                    headers={"User-Agent": self.USER_AGENT},
                    follow_redirects=True,
                )
                return response.status_code == 200
        except Exception:
            return False

    def _parse_entry(self, subscription: Subscription, entry: dict) -> Optional[ContentItem]:
        """Parse a single RSS entry into a ContentItem."""
        try:
            # Extract external ID from the entry
            external_id = entry.get("id", entry.get("link", ""))
            if not external_id:
                return None

            # Extract title
            title = entry.get("title", "")

            # Extract content from various possible fields
            content = ""
            if "content" in entry and entry["content"]:
                content = entry["content"][0].get("value", "")
            elif "summary" in entry:
                content = entry.get("summary", "")

            # Clean HTML from content
            content = self._clean_html(content)

            # Extract URL
            url = entry.get("link", "")

            # Extract author
            author = entry.get("author", "")
            if not author and "authors" in entry and entry["authors"]:
                author = entry["authors"][0].get("name", "")

            # Parse published date
            published_at = datetime.now()
            if "published_parsed" in entry and entry["published_parsed"]:
                try:
                    published_at = datetime(*entry["published_parsed"][:6])
                except (TypeError, ValueError):
                    pass
            elif "published" in entry:
                try:
                    published_at = parsedate_to_datetime(entry["published"])
                except (TypeError, ValueError):
                    pass

            return self.create_content_item(
                subscription=subscription,
                external_id=external_id,
                title=title,
                content=content[:5000],  # Limit content length
                url=url,
                author=author,
                published_at=published_at,
            )

        except Exception:
            return None

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and decode entities."""
        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

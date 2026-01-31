"""Hacker News content fetcher using hnrss.org RSS feeds."""

import html
import logging
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional

import feedparser
import httpx

from src.fetchers.base import BaseFetcher, FetchResult
from src.fetchers.link_extractor import LinkExtractor
from src.storage.models import ContentItem, SourceType, Subscription, SubscriptionType

logger = logging.getLogger(__name__)


class HackerNewsFetcher(BaseFetcher):
    """Fetcher for Hacker News content using hnrss.org RSS feeds."""

    source_type = SourceType.HACKERNEWS

    # Hacker News RSS feed URLs via hnrss.org
    HN_FEEDS = {
        SubscriptionType.HN_FRONT: "https://hnrss.org/frontpage",
        SubscriptionType.HN_NEW: "https://hnrss.org/newest",
        SubscriptionType.HN_BEST: "https://hnrss.org/best",
        SubscriptionType.HN_ASK: "https://hnrss.org/ask",
        SubscriptionType.HN_SHOW: "https://hnrss.org/show",
    }

    USER_AGENT = "BabelByte/1.0 (Content Aggregator)"

    def __init__(self, timeout: float = 30.0, fetch_link_content: bool = True):
        self.timeout = timeout
        self.fetch_link_content = fetch_link_content
        self.link_extractor = LinkExtractor(timeout=15.0)

    def _get_feed_url(self, subscription: Subscription) -> Optional[str]:
        """Get the RSS feed URL for a subscription."""
        if subscription.feed_url_override:
            return subscription.feed_url_override
        return self.HN_FEEDS.get(subscription.subscription_type)

    async def fetch(self, subscription: Subscription) -> FetchResult:
        """Fetch content from Hacker News RSS feed."""
        feed_url = self._get_feed_url(subscription)

        if not feed_url:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"Unknown HN feed type: {subscription.subscription_type}",
            )

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
            newest_entry_id = subscription.last_entry_id

            for entry in feed.entries:
                external_id = entry.get("id") or entry.get("link", "")
                if not external_id:
                    continue

                # Skip entries we've already seen (incremental fetch)
                if subscription.last_entry_id and external_id <= subscription.last_entry_id:
                    continue

                item = self._parse_entry(subscription, entry)
                if item:
                    items.append((item, entry))
                    # Track newest external_id for next fetch
                    if newest_entry_id is None or external_id > newest_entry_id:
                        newest_entry_id = external_id

            # Update subscription with newest entry_id
            subscription.last_entry_id = newest_entry_id

            # Enhance link-posts with external content
            if self.fetch_link_content:
                items = await self._enhance_link_posts(items)
            else:
                items = [item for item, _ in items]

            if items:
                logger.info(f"Fetched {len(items)} new posts from {subscription.display_name}")
            else:
                logger.debug(f"No new posts from {subscription.display_name}")

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
        """Validate that a HN subscription is accessible."""
        feed_url = self._get_feed_url(subscription)
        if not feed_url:
            return False

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

    async def _enhance_link_posts(
        self, items: list[tuple[ContentItem, dict]]
    ) -> list[ContentItem]:
        """Enhance HN posts by fetching external link content."""
        enhanced_items = []

        for item, entry in items:
            # HN posts typically link to external content
            external_url = self._extract_external_url(entry)

            if external_url and self._should_fetch_content(item.content):
                logger.debug(f"Fetching external link: {external_url}")
                result = await self.link_extractor.extract(external_url)

                if result.success and result.content:
                    item.content = f"[External Link] {external_url}\n\n{result.content}"
                    logger.info(
                        f"Enhanced HN post: {item.title[:50]}... (+{len(result.content)} chars)"
                    )
                else:
                    item.content = f"[External Link] {external_url}\n\n{item.content or ''}"

            enhanced_items.append(item)

        return enhanced_items

    def _should_fetch_content(self, content: str) -> bool:
        """Check if we should fetch external content."""
        if not content:
            return True
        # If content is short (HN RSS typically only includes comments link), fetch external
        return len(content.strip()) < 200

    def _extract_external_url(self, entry: dict) -> Optional[str]:
        """Extract external URL from RSS entry."""
        # hnrss.org includes the external URL in the 'link' field
        # while the HN discussion link is in 'comments'
        link = entry.get("link", "")
        comments = entry.get("comments", "")

        # If link is different from comments URL (HN discussion), it's external
        if link and comments and link != comments and "news.ycombinator.com" not in link:
            return link

        # Fallback: check for link in content
        content_html = ""
        if "content" in entry and entry["content"]:
            content_html = entry["content"][0].get("value", "")
        elif "summary" in entry:
            content_html = entry.get("summary", "")

        # Look for article link in the HTML
        link_match = re.search(r'<a\s+href="([^"]+)"[^>]*>Article</a>', content_html, re.I)
        if link_match:
            url = link_match.group(1)
            if "news.ycombinator.com" not in url:
                return url

        return None

    def _parse_entry(self, subscription: Subscription, entry: dict) -> Optional[ContentItem]:
        """Parse a single RSS entry into a ContentItem."""
        try:
            external_id = entry.get("id") or entry.get("link", "")
            if not external_id:
                return None

            title = entry.get("title", "")

            # Extract content from various possible fields
            content = ""
            if "content" in entry and entry["content"]:
                content = entry["content"][0].get("value", "")
            elif "summary" in entry:
                content = entry.get("summary", "")

            # Clean HTML from content
            content = self._clean_html(content)

            # Use comments URL as the canonical URL (HN discussion)
            url = entry.get("comments") or entry.get("link", "")

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
                content=content[:5000],
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

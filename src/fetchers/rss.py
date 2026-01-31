"""Generic RSS content fetcher for any RSS/Atom feed."""

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
from src.storage.models import ContentItem, SourceType, Subscription

logger = logging.getLogger(__name__)


class GenericRSSFetcher(BaseFetcher):
    """Fetcher for any RSS/Atom feed."""

    source_type = SourceType.RSS

    USER_AGENT = "BabelByte/1.0 (Content Aggregator)"

    def __init__(self, timeout: float = 30.0, fetch_link_content: bool = True):
        self.timeout = timeout
        self.fetch_link_content = fetch_link_content
        self.link_extractor = LinkExtractor(timeout=15.0)

    async def fetch(self, subscription: Subscription) -> FetchResult:
        """Fetch content from RSS feed."""
        feed_url = subscription.feed_url_override

        if not feed_url:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message="No feed URL configured for RSS subscription",
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
                external_id = self._get_entry_id(entry)
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

            # Enhance posts with full content if configured
            if self.fetch_link_content:
                items = await self._enhance_posts(items)
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
        """Validate that an RSS subscription is accessible."""
        feed_url = subscription.feed_url_override
        if not feed_url:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    feed_url,
                    headers={"User-Agent": self.USER_AGENT},
                    follow_redirects=True,
                )
                if response.status_code != 200:
                    return False

                # Try to parse to validate it's actually a feed
                feed = feedparser.parse(response.text)
                return bool(feed.entries) or bool(feed.feed.get("title"))
        except Exception:
            return False

    async def _enhance_posts(
        self, items: list[tuple[ContentItem, dict]]
    ) -> list[ContentItem]:
        """Enhance RSS posts by fetching full article content if needed."""
        enhanced_items = []

        for item, entry in items:
            # Check if content is short and we should fetch full article
            if self._should_fetch_full_content(item.content):
                article_url = entry.get("link", "")

                if article_url:
                    logger.debug(f"Fetching full article: {article_url}")
                    result = await self.link_extractor.extract(article_url)

                    if result.success and result.content:
                        item.content = result.content
                        logger.info(
                            f"Enhanced RSS post: {item.title[:50]}... (+{len(result.content)} chars)"
                        )

            enhanced_items.append(item)

        return enhanced_items

    def _should_fetch_full_content(self, content: str) -> bool:
        """Check if we should fetch full article content."""
        if not content:
            return True
        # If content is short (typical for RSS summaries), fetch full article
        return len(content.strip()) < 500

    def _get_entry_id(self, entry: dict) -> str:
        """Get a unique ID for an entry."""
        # Try various ID fields
        entry_id = entry.get("id") or entry.get("guid", {})
        if isinstance(entry_id, dict):
            entry_id = entry_id.get("value", "")

        if not entry_id:
            entry_id = entry.get("link", "")

        if not entry_id:
            # Fallback: create ID from title + date
            title = entry.get("title", "")
            published = entry.get("published", "")
            entry_id = f"{title}:{published}"

        return entry_id

    def _parse_entry(self, subscription: Subscription, entry: dict) -> Optional[ContentItem]:
        """Parse a single RSS entry into a ContentItem."""
        try:
            external_id = self._get_entry_id(entry)
            if not external_id:
                return None

            title = entry.get("title", "")

            # Extract content from various possible fields
            content = ""

            # Try content:encoded first (full content)
            if "content" in entry and entry["content"]:
                for c in entry["content"]:
                    if c.get("type") == "text/html" or not content:
                        content = c.get("value", "")

            # Fallback to summary
            if not content and "summary" in entry:
                content = entry.get("summary", "")

            # Fallback to description
            if not content and "description" in entry:
                content = entry.get("description", "")

            # Clean HTML from content
            content = self._clean_html(content)

            # Extract URL
            url = entry.get("link", "")
            if isinstance(url, dict):
                url = url.get("href", "")

            # Extract author
            author = entry.get("author", "")
            if not author and "authors" in entry and entry["authors"]:
                author = entry["authors"][0].get("name", "")
            if not author and "author_detail" in entry:
                author = entry["author_detail"].get("name", "")

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
            elif "updated_parsed" in entry and entry["updated_parsed"]:
                try:
                    published_at = datetime(*entry["updated_parsed"][:6])
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

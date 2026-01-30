"""Base classes and utilities for content fetchers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.storage.models import ContentItem, SourceType, Subscription


@dataclass
class FetchResult:
    """Result of a fetch operation."""

    subscription: Subscription
    items: list[ContentItem] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    fetched_at: datetime = field(default_factory=datetime.now)

    @property
    def count(self) -> int:
        return len(self.items)


class BaseFetcher(ABC):
    """Abstract base class for content fetchers."""

    source_type: SourceType

    @abstractmethod
    async def fetch(self, subscription: Subscription) -> FetchResult:
        """
        Fetch content from the source.

        Args:
            subscription: The subscription to fetch content for.

        Returns:
            FetchResult containing the fetched items.
        """
        pass

    @abstractmethod
    async def validate_subscription(self, subscription: Subscription) -> bool:
        """
        Validate that a subscription is valid and accessible.

        Args:
            subscription: The subscription to validate.

        Returns:
            True if valid, False otherwise.
        """
        pass

    def create_content_item(
        self,
        subscription: Subscription,
        external_id: str,
        title: str,
        content: str,
        url: str,
        author: str,
        published_at: datetime,
    ) -> ContentItem:
        """Helper to create a ContentItem with common fields filled."""
        return ContentItem(
            subscription_id=subscription.id,
            source_type=self.source_type,
            external_id=external_id,
            title=title,
            content=content,
            url=url,
            author=author,
            published_at=published_at,
            fetched_at=datetime.now(),
        )

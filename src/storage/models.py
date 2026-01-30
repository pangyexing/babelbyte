"""Data models for BabelByte."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SourceType(Enum):
    """Content source type."""

    REDDIT = "reddit"
    TWITTER = "twitter"


class SubscriptionType(Enum):
    """Subscription type within a source."""

    SUBREDDIT = "subreddit"
    REDDIT_USER = "reddit_user"
    TWITTER_USER = "twitter_user"


@dataclass
class Subscription:
    """A subscription to a content source."""

    id: Optional[int] = None
    source_type: SourceType = SourceType.REDDIT
    subscription_type: SubscriptionType = SubscriptionType.SUBREDDIT
    name: str = ""  # e.g., "MachineLearning" for subreddit, "elonmusk" for twitter
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_fetched_at: Optional[datetime] = None

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        if self.source_type == SourceType.REDDIT:
            if self.subscription_type == SubscriptionType.SUBREDDIT:
                return f"r/{self.name}"
            return f"u/{self.name}"
        return f"@{self.name}"

    @property
    def feed_url(self) -> Optional[str]:
        """RSS feed URL for Reddit sources."""
        if self.source_type == SourceType.REDDIT:
            if self.subscription_type == SubscriptionType.SUBREDDIT:
                return f"https://www.reddit.com/r/{self.name}/.rss"
            return f"https://www.reddit.com/user/{self.name}/.rss"
        return None


@dataclass
class ContentItem:
    """A piece of content fetched from a source."""

    id: Optional[int] = None
    subscription_id: int = 0
    source_type: SourceType = SourceType.REDDIT
    external_id: str = ""  # Platform-specific ID
    title: str = ""
    content: str = ""
    url: str = ""
    author: str = ""
    published_at: datetime = field(default_factory=datetime.now)
    fetched_at: datetime = field(default_factory=datetime.now)

    # AI-processed fields
    summary: Optional[str] = None
    category: Optional[str] = None
    importance_score: Optional[int] = None  # 1-10
    processed_at: Optional[datetime] = None

    # Delivery tracking
    delivered: bool = False
    delivered_at: Optional[datetime] = None


@dataclass
class UserProfile:
    """User configuration and preferences."""

    id: Optional[int] = None
    email: str = ""
    digest_enabled: bool = True
    digest_time: str = "08:00"  # HH:MM format
    categories_of_interest: list[str] = field(default_factory=list)
    min_importance_score: int = 5  # Only include items with score >= this
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class DigestItem:
    """A processed item ready for the digest."""

    content_item: ContentItem
    summary: str
    category: str
    importance_score: int

    @property
    def source_display(self) -> str:
        """Display name for the source."""
        if self.content_item.source_type == SourceType.REDDIT:
            return "Reddit"
        return "Twitter"

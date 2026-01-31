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


class ItemState(Enum):
    """Content item state for knowledge base."""

    UNREAD = "unread"
    READ = "read"
    SAVED = "saved"
    ARCHIVED = "archived"
    FLAGGED = "flagged"


@dataclass
class KeyPoint:
    """A key point extracted from content."""

    type: str  # 数字/时间/实体/事实
    value: str
    impact: str = ""


@dataclass
class ImpactAssessment:
    """Impact assessment for a content item."""

    short_term: str = ""
    long_term: str = ""
    certainty: str = "uncertain"  # certain/uncertain


@dataclass
class ActionableItem:
    """An actionable item extracted from content."""

    type: str  # 跟进/验证/决策/触发器
    description: str
    priority: str = "中"  # 高/中/低


@dataclass
class EnhancedProcessingData:
    """Enhanced AI processing data for Phase 1."""

    one_liner: str = ""
    key_points: list[KeyPoint] = field(default_factory=list)
    impact_assessment: Optional[ImpactAssessment] = None
    actionable_items: list[ActionableItem] = field(default_factory=list)


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
    # Twitter-specific: cache user_id and last_tweet_id to reduce API calls
    twitter_user_id: Optional[str] = None
    last_tweet_id: Optional[str] = None
    # Reddit-specific: last external_id for incremental fetching
    last_reddit_id: Optional[str] = None

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

    # AI-processed fields (basic)
    summary: Optional[str] = None
    category: Optional[str] = None
    importance_score: Optional[int] = None  # 1-10
    processed_at: Optional[datetime] = None

    # AI-processed fields (enhanced - Phase 1)
    one_liner: Optional[str] = None  # One sentence conclusion
    key_points: Optional[str] = None  # JSON: list of KeyPoint
    impact_assessment: Optional[str] = None  # JSON: ImpactAssessment
    actionable_items: Optional[str] = None  # JSON: list of ActionableItem

    # State management (Phase 4)
    state: ItemState = ItemState.UNREAD

    # Delivery tracking
    delivered: bool = False
    delivered_at: Optional[datetime] = None

    def get_enhanced_data(self) -> Optional[EnhancedProcessingData]:
        """Parse enhanced processing data from JSON fields."""
        import json

        try:
            key_points = []
            if self.key_points:
                for kp in json.loads(self.key_points):
                    key_points.append(KeyPoint(**kp))

            impact = None
            if self.impact_assessment:
                impact = ImpactAssessment(**json.loads(self.impact_assessment))

            actionables = []
            if self.actionable_items:
                for ai in json.loads(self.actionable_items):
                    actionables.append(ActionableItem(**ai))

            return EnhancedProcessingData(
                one_liner=self.one_liner or "",
                key_points=key_points,
                impact_assessment=impact,
                actionable_items=actionables,
            )
        except (json.JSONDecodeError, TypeError):
            return None


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

    @property
    def enhanced_data(self) -> Optional[EnhancedProcessingData]:
        """Get enhanced processing data."""
        return self.content_item.get_enhanced_data()

    @property
    def is_event(self) -> bool:
        """Check if this is an event item (for template rendering)."""
        return False


@dataclass
class EventDigestItem:
    """An event cluster ready for the digest."""

    event_cluster: "EventCluster"
    members: list[ContentItem]  # Sorted by importance
    representative_item: ContentItem  # Highest importance member

    @property
    def event_title(self) -> str:
        """Display title with article count."""
        return f"{self.event_cluster.event_title} ({len(self.members)}篇报道)"

    @property
    def category(self) -> str:
        """Get event category."""
        return self.event_cluster.category

    @property
    def importance_score(self) -> int:
        """Get max importance score from members."""
        return max((m.importance_score or 0) for m in self.members)

    @property
    def summary(self) -> str:
        """Get summary from representative item."""
        return self.representative_item.summary or ""

    @property
    def one_liner(self) -> Optional[str]:
        """Get one-liner from representative item."""
        return self.representative_item.one_liner

    @property
    def enhanced_data(self) -> Optional[EnhancedProcessingData]:
        """Get enhanced data from representative item."""
        return self.representative_item.get_enhanced_data()

    @property
    def content_item(self) -> ContentItem:
        """For template compatibility - returns representative item."""
        return self.representative_item

    @property
    def source_display(self) -> str:
        """Display sources from all members."""
        sources = set()
        for m in self.members:
            if m.source_type == SourceType.REDDIT:
                sources.add("Reddit")
            else:
                sources.add("Twitter")
        return "/".join(sorted(sources))

    @property
    def is_event(self) -> bool:
        """Check if this is an event item (for template rendering)."""
        return True


# ============================================
# Phase 2: Event Stream Models
# ============================================


@dataclass
class EventCluster:
    """A cluster of related content items about the same event."""

    id: Optional[int] = None
    event_title: str = ""
    category: str = ""
    first_seen_at: datetime = field(default_factory=datetime.now)
    last_updated_at: datetime = field(default_factory=datetime.now)
    article_count: int = 0


@dataclass
class EventMember:
    """Association between an event cluster and a content item."""

    event_cluster_id: int = 0
    content_item_id: int = 0
    similarity_score: float = 0.0
    detection_method: str = "rule"  # 'rule' or 'ai'


@dataclass
class EventTimeline:
    """Timeline entry for an event cluster."""

    event_cluster_id: int = 0
    entry_date: str = ""  # YYYY-MM-DD
    summary: str = ""
    consensus_level: str = "high"  # 'high' or 'conflicted'


# ============================================
# Phase 3: Topic Radar Models
# ============================================


@dataclass
class Topic:
    """A topic for content categorization."""

    id: Optional[int] = None
    name: str = ""
    description: str = ""
    keywords: Optional[str] = None  # JSON: list of keywords
    created_at: datetime = field(default_factory=datetime.now)

    def get_keywords(self) -> list[str]:
        """Parse keywords from JSON."""
        import json

        if self.keywords:
            try:
                return json.loads(self.keywords)
            except json.JSONDecodeError:
                return []
        return []


@dataclass
class ContentTopic:
    """Association between content and topic."""

    content_id: int = 0
    topic_id: int = 0
    relevance: float = 0.0  # 0-1


@dataclass
class TopicSnapshot:
    """Snapshot of a topic at a point in time."""

    id: Optional[int] = None
    topic_id: int = 0
    snapshot_date: str = ""  # YYYY-MM-DD
    summary: str = ""
    key_entities: Optional[str] = None  # JSON
    metrics: Optional[str] = None  # JSON
    trend: str = "stable"  # 'up' / 'down' / 'stable'


# ============================================
# Phase 5: Action List Models
# ============================================


class ActionStatus(Enum):
    """Status of an action item."""

    PENDING = "pending"
    DONE = "done"
    DISMISSED = "dismissed"


@dataclass
class ActionItem:
    """An action item extracted from content."""

    id: Optional[int] = None
    content_item_id: int = 0
    type: str = ""  # 跟进/验证/决策/触发器
    description: str = ""
    priority: str = "中"  # 高/中/低
    status: ActionStatus = ActionStatus.PENDING
    due_date: Optional[str] = None  # YYYY-MM-DD
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class Trigger:
    """User-defined trigger for automatic actions."""

    id: Optional[int] = None
    name: str = ""
    condition: str = ""  # e.g., "公司名=OpenAI AND 事件类型=融资"
    action: str = "notify"  # 'notify' / 'add_action'
    enabled: bool = True

"""SQLite database operations for BabelByte."""

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from config.settings import get_settings
from src.storage.models import (
    ActionItem,
    ActionStatus,
    ContentItem,
    ContentTopic,
    EventCluster,
    EventMember,
    EventTimeline,
    ItemState,
    SourceType,
    Subscription,
    SubscriptionType,
    Topic,
    TopicSnapshot,
    Trigger,
    UserProfile,
)

# SQL statements for table creation
CREATE_SUBSCRIPTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    subscription_type TEXT NOT NULL,
    name TEXT NOT NULL,
    enabled INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    last_fetched_at TEXT,
    twitter_user_id TEXT,
    last_tweet_id TEXT,
    last_reddit_id TEXT,
    UNIQUE(source_type, subscription_type, name)
)
"""

CREATE_CONTENT_ITEMS_TABLE = """
CREATE TABLE IF NOT EXISTS content_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id INTEGER NOT NULL,
    source_type TEXT NOT NULL,
    external_id TEXT NOT NULL,
    title TEXT,
    content TEXT,
    url TEXT,
    author TEXT,
    published_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    summary TEXT,
    category TEXT,
    importance_score INTEGER,
    processed_at TEXT,
    one_liner TEXT,
    key_points TEXT,
    impact_assessment TEXT,
    actionable_items TEXT,
    state TEXT DEFAULT 'unread',
    delivered INTEGER DEFAULT 0,
    delivered_at TEXT,
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id),
    UNIQUE(source_type, external_id)
)
"""

# Phase 2: Event Stream tables
CREATE_EVENT_CLUSTERS_TABLE = """
CREATE TABLE IF NOT EXISTS event_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_title TEXT NOT NULL,
    category TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_updated_at TEXT NOT NULL,
    article_count INTEGER DEFAULT 0
)
"""

CREATE_EVENT_MEMBERS_TABLE = """
CREATE TABLE IF NOT EXISTS event_members (
    event_cluster_id INTEGER NOT NULL,
    content_item_id INTEGER NOT NULL,
    similarity_score REAL DEFAULT 0.0,
    detection_method TEXT DEFAULT 'rule',
    FOREIGN KEY (event_cluster_id) REFERENCES event_clusters(id),
    FOREIGN KEY (content_item_id) REFERENCES content_items(id),
    UNIQUE(event_cluster_id, content_item_id)
)
"""

CREATE_EVENT_TIMELINE_TABLE = """
CREATE TABLE IF NOT EXISTS event_timeline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_cluster_id INTEGER NOT NULL,
    entry_date TEXT NOT NULL,
    summary TEXT,
    consensus_level TEXT DEFAULT 'high',
    FOREIGN KEY (event_cluster_id) REFERENCES event_clusters(id)
)
"""

# Phase 3: Topic Radar tables
CREATE_TOPICS_TABLE = """
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    keywords TEXT,
    created_at TEXT NOT NULL
)
"""

CREATE_CONTENT_TOPICS_TABLE = """
CREATE TABLE IF NOT EXISTS content_topics (
    content_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    relevance REAL DEFAULT 0.0,
    FOREIGN KEY (content_id) REFERENCES content_items(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    PRIMARY KEY(content_id, topic_id)
)
"""

CREATE_TOPIC_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS topic_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL,
    summary TEXT,
    key_entities TEXT,
    metrics TEXT,
    trend TEXT DEFAULT 'stable',
    FOREIGN KEY (topic_id) REFERENCES topics(id)
)
"""

# Phase 5: Action List tables
CREATE_ACTION_ITEMS_TABLE = """
CREATE TABLE IF NOT EXISTS action_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_item_id INTEGER,
    type TEXT NOT NULL,
    description TEXT NOT NULL,
    priority TEXT DEFAULT '中',
    status TEXT DEFAULT 'pending',
    due_date TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (content_item_id) REFERENCES content_items(id)
)
"""

CREATE_TRIGGERS_TABLE = """
CREATE TABLE IF NOT EXISTS triggers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    condition TEXT NOT NULL,
    action TEXT DEFAULT 'notify',
    enabled INTEGER DEFAULT 1
)
"""

CREATE_USER_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS user_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    digest_enabled INTEGER DEFAULT 1,
    digest_time TEXT DEFAULT '08:00',
    categories_of_interest TEXT,
    min_importance_score INTEGER DEFAULT 5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

# Performance indexes for common queries
CREATE_INDEXES = """
-- Index for fetching unprocessed items (processed_at IS NULL)
CREATE INDEX IF NOT EXISTS idx_content_unprocessed
    ON content_items(processed_at) WHERE processed_at IS NULL;

-- Index for fetching undelivered items with importance filtering
CREATE INDEX IF NOT EXISTS idx_content_undelivered
    ON content_items(delivered, importance_score) WHERE delivered = 0;

-- Index for URL deduplication lookups
CREATE INDEX IF NOT EXISTS idx_content_url ON content_items(url);

-- Index for source/external_id uniqueness checks
CREATE INDEX IF NOT EXISTS idx_content_source_external
    ON content_items(source_type, external_id);

-- Phase 4: Knowledge Base indexes
CREATE INDEX IF NOT EXISTS idx_content_category ON content_items(category);
CREATE INDEX IF NOT EXISTS idx_content_importance ON content_items(importance_score);
CREATE INDEX IF NOT EXISTS idx_content_published_at ON content_items(published_at);
CREATE INDEX IF NOT EXISTS idx_content_state ON content_items(state);

-- Event clusters indexes
CREATE INDEX IF NOT EXISTS idx_event_clusters_category ON event_clusters(category);
CREATE INDEX IF NOT EXISTS idx_event_clusters_updated ON event_clusters(last_updated_at);

-- Topics indexes
CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name);

-- Action items indexes
CREATE INDEX IF NOT EXISTS idx_action_items_status ON action_items(status);
CREATE INDEX IF NOT EXISTS idx_action_items_priority ON action_items(priority);
"""

# Full-text search virtual table for Phase 4: Knowledge Base
CREATE_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
    title,
    content,
    summary,
    category,
    author,
    content_id UNINDEXED
);
"""


class Database:
    """Async SQLite database manager."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_settings().database.path
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Connect to the database and initialize tables."""
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._create_tables()

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        async with self._connection.cursor() as cursor:
            # Core tables
            await cursor.execute(CREATE_SUBSCRIPTIONS_TABLE)
            await cursor.execute(CREATE_CONTENT_ITEMS_TABLE)
            await cursor.execute(CREATE_USER_PROFILES_TABLE)

            # Phase 2: Event Stream tables
            await cursor.execute(CREATE_EVENT_CLUSTERS_TABLE)
            await cursor.execute(CREATE_EVENT_MEMBERS_TABLE)
            await cursor.execute(CREATE_EVENT_TIMELINE_TABLE)

            # Phase 3: Topic Radar tables
            await cursor.execute(CREATE_TOPICS_TABLE)
            await cursor.execute(CREATE_CONTENT_TOPICS_TABLE)
            await cursor.execute(CREATE_TOPIC_SNAPSHOTS_TABLE)

            # Phase 5: Action List tables
            await cursor.execute(CREATE_ACTION_ITEMS_TABLE)
            await cursor.execute(CREATE_TRIGGERS_TABLE)

            # Create performance indexes
            for statement in CREATE_INDEXES.strip().split(";"):
                statement = statement.strip()
                if statement and not statement.startswith("--"):
                    await cursor.execute(statement)

            # Create FTS table (Phase 4)
            await cursor.execute(CREATE_FTS_TABLE)

            await self._connection.commit()
        await self._migrate_tables()

    async def _migrate_tables(self) -> None:
        """Run database migrations for schema updates."""
        async with self._connection.cursor() as cursor:
            # Check if twitter_user_id column exists in subscriptions
            await cursor.execute("PRAGMA table_info(subscriptions)")
            sub_columns = {row[1] for row in await cursor.fetchall()}

            if "twitter_user_id" not in sub_columns:
                await cursor.execute(
                    "ALTER TABLE subscriptions ADD COLUMN twitter_user_id TEXT"
                )
            if "last_tweet_id" not in sub_columns:
                await cursor.execute(
                    "ALTER TABLE subscriptions ADD COLUMN last_tweet_id TEXT"
                )
            if "last_reddit_id" not in sub_columns:
                await cursor.execute(
                    "ALTER TABLE subscriptions ADD COLUMN last_reddit_id TEXT"
                )

            # Check content_items columns for Phase 1 & 4 enhancements
            await cursor.execute("PRAGMA table_info(content_items)")
            content_columns = {row[1] for row in await cursor.fetchall()}

            # Phase 1: Enhanced digest fields
            if "one_liner" not in content_columns:
                await cursor.execute(
                    "ALTER TABLE content_items ADD COLUMN one_liner TEXT"
                )
            if "key_points" not in content_columns:
                await cursor.execute(
                    "ALTER TABLE content_items ADD COLUMN key_points TEXT"
                )
            if "impact_assessment" not in content_columns:
                await cursor.execute(
                    "ALTER TABLE content_items ADD COLUMN impact_assessment TEXT"
                )
            if "actionable_items" not in content_columns:
                await cursor.execute(
                    "ALTER TABLE content_items ADD COLUMN actionable_items TEXT"
                )

            # Phase 4: State management
            if "state" not in content_columns:
                await cursor.execute(
                    "ALTER TABLE content_items ADD COLUMN state TEXT DEFAULT 'unread'"
                )

            await self._connection.commit()

    # Subscription operations

    async def add_subscription(self, subscription: Subscription) -> Subscription:
        """Add a new subscription."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO subscriptions (source_type, subscription_type, name, enabled, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    subscription.source_type.value,
                    subscription.subscription_type.value,
                    subscription.name,
                    1 if subscription.enabled else 0,
                    subscription.created_at.isoformat(),
                ),
            )
            await self._connection.commit()
            subscription.id = cursor.lastrowid
            return subscription

    async def get_subscription(self, sub_id: int) -> Optional[Subscription]:
        """Get a subscription by ID."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("SELECT * FROM subscriptions WHERE id = ?", (sub_id,))
            row = await cursor.fetchone()
            if row:
                return self._row_to_subscription(row)
            return None

    async def get_subscription_by_name(
        self, source_type: SourceType, subscription_type: SubscriptionType, name: str
    ) -> Optional[Subscription]:
        """Get a subscription by source type and name."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT * FROM subscriptions
                WHERE source_type = ? AND subscription_type = ? AND name = ?
                """,
                (source_type.value, subscription_type.value, name),
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_subscription(row)
            return None

    async def list_subscriptions(self, enabled_only: bool = False) -> list[Subscription]:
        """List all subscriptions."""
        async with self._connection.cursor() as cursor:
            if enabled_only:
                await cursor.execute("SELECT * FROM subscriptions WHERE enabled = 1")
            else:
                await cursor.execute("SELECT * FROM subscriptions")
            rows = await cursor.fetchall()
            return [self._row_to_subscription(row) for row in rows]

    async def update_subscription(self, subscription: Subscription) -> None:
        """Update a subscription."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                UPDATE subscriptions
                SET enabled = ?, last_fetched_at = ?, twitter_user_id = ?, last_tweet_id = ?,
                    last_reddit_id = ?
                WHERE id = ?
                """,
                (
                    1 if subscription.enabled else 0,
                    subscription.last_fetched_at.isoformat() if subscription.last_fetched_at else None,
                    subscription.twitter_user_id,
                    subscription.last_tweet_id,
                    subscription.last_reddit_id,
                    subscription.id,
                ),
            )
            await self._connection.commit()

    async def delete_subscription(self, sub_id: int) -> None:
        """Delete a subscription."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("DELETE FROM subscriptions WHERE id = ?", (sub_id,))
            await self._connection.commit()

    def _row_to_subscription(self, row: aiosqlite.Row) -> Subscription:
        """Convert a database row to a Subscription object."""
        return Subscription(
            id=row["id"],
            source_type=SourceType(row["source_type"]),
            subscription_type=SubscriptionType(row["subscription_type"]),
            name=row["name"],
            enabled=bool(row["enabled"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_fetched_at=(
                datetime.fromisoformat(row["last_fetched_at"])
                if row["last_fetched_at"]
                else None
            ),
            twitter_user_id=row["twitter_user_id"] if "twitter_user_id" in row.keys() else None,
            last_tweet_id=row["last_tweet_id"] if "last_tweet_id" in row.keys() else None,
            last_reddit_id=row["last_reddit_id"] if "last_reddit_id" in row.keys() else None,
        )

    # Content item operations

    async def add_content_item(self, item: ContentItem) -> ContentItem:
        """Add a new content item."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT OR IGNORE INTO content_items
                (subscription_id, source_type, external_id, title, content, url, author,
                 published_at, fetched_at, summary, category, importance_score, processed_at,
                 delivered, delivered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.subscription_id,
                    item.source_type.value,
                    item.external_id,
                    item.title,
                    item.content,
                    item.url,
                    item.author,
                    item.published_at.isoformat(),
                    item.fetched_at.isoformat(),
                    item.summary,
                    item.category,
                    item.importance_score,
                    item.processed_at.isoformat() if item.processed_at else None,
                    1 if item.delivered else 0,
                    item.delivered_at.isoformat() if item.delivered_at else None,
                ),
            )
            await self._connection.commit()
            item.id = cursor.lastrowid
            return item

    async def get_content_item(self, item_id: int) -> Optional[ContentItem]:
        """Get a content item by ID."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("SELECT * FROM content_items WHERE id = ?", (item_id,))
            row = await cursor.fetchone()
            if row:
                return self._row_to_content_item(row)
            return None

    async def get_unprocessed_items(self, limit: int = 100) -> list[ContentItem]:
        """Get content items that haven't been processed by AI."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT * FROM content_items
                WHERE processed_at IS NULL
                ORDER BY published_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_content_item(row) for row in rows]

    async def get_undelivered_items(
        self, min_importance: int = 1, limit: int = 50
    ) -> list[ContentItem]:
        """Get processed items that haven't been delivered."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT * FROM content_items
                WHERE processed_at IS NOT NULL
                AND delivered = 0
                AND importance_score >= ?
                ORDER BY importance_score DESC, published_at DESC
                LIMIT ?
                """,
                (min_importance, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_content_item(row) for row in rows]

    async def update_content_item(self, item: ContentItem) -> None:
        """Update a content item."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                UPDATE content_items
                SET summary = ?, category = ?, importance_score = ?, processed_at = ?,
                    one_liner = ?, key_points = ?, impact_assessment = ?, actionable_items = ?,
                    state = ?, delivered = ?, delivered_at = ?
                WHERE id = ?
                """,
                (
                    item.summary,
                    item.category,
                    item.importance_score,
                    item.processed_at.isoformat() if item.processed_at else None,
                    item.one_liner,
                    item.key_points,
                    item.impact_assessment,
                    item.actionable_items,
                    item.state.value,
                    1 if item.delivered else 0,
                    item.delivered_at.isoformat() if item.delivered_at else None,
                    item.id,
                ),
            )
            await self._connection.commit()

            # Update FTS index if processed
            if item.processed_at and item.id:
                await self._update_fts_index(item)

    async def _update_fts_index(self, item: ContentItem) -> None:
        """Update FTS index for a content item."""
        async with self._connection.cursor() as cursor:
            # Delete existing entry
            await cursor.execute(
                "DELETE FROM content_fts WHERE content_id = ?", (item.id,)
            )
            # Insert new entry
            await cursor.execute(
                """
                INSERT INTO content_fts (title, content, summary, category, author, content_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item.title or "",
                    item.content or "",
                    item.summary or "",
                    item.category or "",
                    item.author or "",
                    item.id,
                ),
            )
            await self._connection.commit()

    async def mark_items_delivered(self, item_ids: list[int]) -> None:
        """Mark multiple items as delivered."""
        now = datetime.now().isoformat()
        async with self._connection.cursor() as cursor:
            await cursor.executemany(
                "UPDATE content_items SET delivered = 1, delivered_at = ? WHERE id = ?",
                [(now, item_id) for item_id in item_ids],
            )
            await self._connection.commit()

    async def content_exists(self, source_type: SourceType, external_id: str) -> bool:
        """Check if a content item already exists."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                "SELECT 1 FROM content_items WHERE source_type = ? AND external_id = ?",
                (source_type.value, external_id),
            )
            return await cursor.fetchone() is not None

    async def url_exists(self, url: str) -> bool:
        """Check if content with this URL already exists (for cross-source deduplication)."""
        if not url:
            return False
        # Normalize URL: strip tracking params, trailing slashes
        normalized_url = self._normalize_url(url)
        if not normalized_url:
            return False
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                "SELECT 1 FROM content_items WHERE url = ? OR url = ? LIMIT 1",
                (url, normalized_url),
            )
            return await cursor.fetchone() is not None

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        if not url:
            return ""
        # Remove common tracking parameters
        from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
        try:
            parsed = urlparse(url)
            # Skip normalization for reddit/twitter URLs (use external_id instead)
            if any(domain in parsed.netloc for domain in ["reddit.com", "twitter.com", "x.com"]):
                return ""
            # Remove tracking params
            tracking_params = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "ref", "source"}
            query_params = parse_qs(parsed.query)
            filtered_params = {k: v for k, v in query_params.items() if k.lower() not in tracking_params}
            clean_query = urlencode(filtered_params, doseq=True)
            # Rebuild URL without tracking params and trailing slash
            clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), parsed.params, clean_query, ""))
            return clean_url
        except Exception:
            return url

    def _row_to_content_item(self, row: aiosqlite.Row) -> ContentItem:
        """Convert a database row to a ContentItem object."""
        row_keys = row.keys()

        # Handle state field (Phase 4)
        state = ItemState.UNREAD
        if "state" in row_keys and row["state"]:
            try:
                state = ItemState(row["state"])
            except ValueError:
                state = ItemState.UNREAD

        return ContentItem(
            id=row["id"],
            subscription_id=row["subscription_id"],
            source_type=SourceType(row["source_type"]),
            external_id=row["external_id"],
            title=row["title"],
            content=row["content"],
            url=row["url"],
            author=row["author"],
            published_at=datetime.fromisoformat(row["published_at"]),
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
            summary=row["summary"],
            category=row["category"],
            importance_score=row["importance_score"],
            processed_at=(
                datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else None
            ),
            # Enhanced fields (Phase 1)
            one_liner=row["one_liner"] if "one_liner" in row_keys else None,
            key_points=row["key_points"] if "key_points" in row_keys else None,
            impact_assessment=row["impact_assessment"] if "impact_assessment" in row_keys else None,
            actionable_items=row["actionable_items"] if "actionable_items" in row_keys else None,
            # State (Phase 4)
            state=state,
            delivered=bool(row["delivered"]),
            delivered_at=(
                datetime.fromisoformat(row["delivered_at"]) if row["delivered_at"] else None
            ),
        )

    # User profile operations

    async def get_or_create_profile(self, email: str) -> UserProfile:
        """Get or create a user profile."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("SELECT * FROM user_profiles WHERE email = ?", (email,))
            row = await cursor.fetchone()
            if row:
                return self._row_to_user_profile(row)

            # Create new profile
            now = datetime.now().isoformat()
            await cursor.execute(
                """
                INSERT INTO user_profiles (email, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (email, now, now),
            )
            await self._connection.commit()

            profile = UserProfile(
                id=cursor.lastrowid,
                email=email,
                created_at=datetime.fromisoformat(now),
                updated_at=datetime.fromisoformat(now),
            )
            return profile

    async def update_profile(self, profile: UserProfile) -> None:
        """Update a user profile."""
        profile.updated_at = datetime.now()
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                UPDATE user_profiles
                SET digest_enabled = ?, digest_time = ?, categories_of_interest = ?,
                    min_importance_score = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    1 if profile.digest_enabled else 0,
                    profile.digest_time,
                    ",".join(profile.categories_of_interest),
                    profile.min_importance_score,
                    profile.updated_at.isoformat(),
                    profile.id,
                ),
            )
            await self._connection.commit()

    def _row_to_user_profile(self, row: aiosqlite.Row) -> UserProfile:
        """Convert a database row to a UserProfile object."""
        categories = row["categories_of_interest"]
        return UserProfile(
            id=row["id"],
            email=row["email"],
            digest_enabled=bool(row["digest_enabled"]),
            digest_time=row["digest_time"],
            categories_of_interest=categories.split(",") if categories else [],
            min_importance_score=row["min_importance_score"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # ============================================
    # Phase 4: Knowledge Base Operations
    # ============================================

    async def search_content(
        self,
        query: str,
        category: Optional[str] = None,
        min_importance: Optional[int] = None,
        state: Optional[ItemState] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ContentItem]:
        """
        Full-text search for content items.

        Args:
            query: Search query string
            category: Filter by category
            min_importance: Filter by minimum importance score
            state: Filter by item state
            from_date: Filter by published_at >= from_date (YYYY-MM-DD)
            to_date: Filter by published_at <= to_date (YYYY-MM-DD)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of matching ContentItem objects
        """
        async with self._connection.cursor() as cursor:
            # Build the query with FTS
            sql = """
                SELECT c.* FROM content_items c
                INNER JOIN content_fts fts ON c.id = fts.content_id
                WHERE content_fts MATCH ?
            """
            params: list = [query]

            if category:
                sql += " AND c.category = ?"
                params.append(category)

            if min_importance:
                sql += " AND c.importance_score >= ?"
                params.append(min_importance)

            if state:
                sql += " AND c.state = ?"
                params.append(state.value)

            if from_date:
                sql += " AND c.published_at >= ?"
                params.append(from_date)

            if to_date:
                sql += " AND c.published_at <= ?"
                params.append(to_date + "T23:59:59")

            sql += " ORDER BY c.importance_score DESC, c.published_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
            return [self._row_to_content_item(row) for row in rows]

    async def browse_by_date(
        self,
        date: str,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> list[ContentItem]:
        """
        Browse content items by date.

        Args:
            date: Date string (YYYY-MM-DD)
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of ContentItem objects from that date
        """
        async with self._connection.cursor() as cursor:
            sql = """
                SELECT * FROM content_items
                WHERE published_at >= ? AND published_at < ?
            """
            params: list = [f"{date}T00:00:00", f"{date}T23:59:59"]

            if category:
                sql += " AND category = ?"
                params.append(category)

            sql += " ORDER BY importance_score DESC, published_at DESC LIMIT ?"
            params.append(limit)

            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
            return [self._row_to_content_item(row) for row in rows]

    async def update_item_state(self, item_id: int, state: ItemState) -> None:
        """Update the state of a content item."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE content_items SET state = ? WHERE id = ?",
                (state.value, item_id),
            )
            await self._connection.commit()

    async def get_category_stats(self) -> dict[str, int]:
        """Get count of items per category."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT category, COUNT(*) as count
                FROM content_items
                WHERE category IS NOT NULL
                GROUP BY category
                ORDER BY count DESC
                """
            )
            rows = await cursor.fetchall()
            return {row["category"]: row["count"] for row in rows}

    async def get_state_stats(self) -> dict[str, int]:
        """Get count of items per state."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT state, COUNT(*) as count
                FROM content_items
                GROUP BY state
                ORDER BY count DESC
                """
            )
            rows = await cursor.fetchall()
            return {row["state"]: row["count"] for row in rows}

    async def rebuild_fts_index(self) -> int:
        """Rebuild the full-text search index from all processed items."""
        async with self._connection.cursor() as cursor:
            # Clear existing FTS data
            await cursor.execute("DELETE FROM content_fts")

            # Rebuild from processed items
            await cursor.execute(
                """
                INSERT INTO content_fts (title, content, summary, category, author, content_id)
                SELECT title, content, summary, category, author, id
                FROM content_items
                WHERE processed_at IS NOT NULL
                """
            )
            await self._connection.commit()

            # Get count
            await cursor.execute("SELECT COUNT(*) FROM content_fts")
            row = await cursor.fetchone()
            return row[0] if row else 0

    # ============================================
    # Phase 2: Event Stream Operations
    # ============================================

    async def create_event_cluster(self, cluster: EventCluster) -> EventCluster:
        """Create a new event cluster."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO event_clusters (event_title, category, first_seen_at, last_updated_at, article_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cluster.event_title,
                    cluster.category,
                    cluster.first_seen_at.isoformat(),
                    cluster.last_updated_at.isoformat(),
                    cluster.article_count,
                ),
            )
            await self._connection.commit()
            cluster.id = cursor.lastrowid
            return cluster

    async def get_event_cluster(self, cluster_id: int) -> Optional[EventCluster]:
        """Get an event cluster by ID."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("SELECT * FROM event_clusters WHERE id = ?", (cluster_id,))
            row = await cursor.fetchone()
            if row:
                return self._row_to_event_cluster(row)
            return None

    async def get_recent_event_clusters(
        self, days: int = 7, category: Optional[str] = None, limit: int = 50
    ) -> list[EventCluster]:
        """Get recent event clusters."""
        async with self._connection.cursor() as cursor:
            from_date = (datetime.now() - __import__("datetime").timedelta(days=days)).isoformat()
            sql = "SELECT * FROM event_clusters WHERE last_updated_at >= ?"
            params: list = [from_date]

            if category:
                sql += " AND category = ?"
                params.append(category)

            sql += " ORDER BY last_updated_at DESC LIMIT ?"
            params.append(limit)

            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
            return [self._row_to_event_cluster(row) for row in rows]

    async def add_event_member(self, member: EventMember) -> None:
        """Add a content item to an event cluster."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT OR REPLACE INTO event_members (event_cluster_id, content_item_id, similarity_score, detection_method)
                VALUES (?, ?, ?, ?)
                """,
                (
                    member.event_cluster_id,
                    member.content_item_id,
                    member.similarity_score,
                    member.detection_method,
                ),
            )
            # Update article count
            await cursor.execute(
                """
                UPDATE event_clusters
                SET article_count = (SELECT COUNT(*) FROM event_members WHERE event_cluster_id = ?),
                    last_updated_at = ?
                WHERE id = ?
                """,
                (member.event_cluster_id, datetime.now().isoformat(), member.event_cluster_id),
            )
            await self._connection.commit()

    async def get_event_members(self, cluster_id: int) -> list[ContentItem]:
        """Get all content items in an event cluster."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT c.* FROM content_items c
                INNER JOIN event_members em ON c.id = em.content_item_id
                WHERE em.event_cluster_id = ?
                ORDER BY c.published_at DESC
                """,
                (cluster_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_content_item(row) for row in rows]

    async def add_event_timeline(self, timeline: EventTimeline) -> None:
        """Add a timeline entry for an event cluster."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO event_timeline (event_cluster_id, entry_date, summary, consensus_level)
                VALUES (?, ?, ?, ?)
                """,
                (
                    timeline.event_cluster_id,
                    timeline.entry_date,
                    timeline.summary,
                    timeline.consensus_level,
                ),
            )
            await self._connection.commit()

    def _row_to_event_cluster(self, row: aiosqlite.Row) -> EventCluster:
        """Convert a database row to an EventCluster object."""
        return EventCluster(
            id=row["id"],
            event_title=row["event_title"],
            category=row["category"],
            first_seen_at=datetime.fromisoformat(row["first_seen_at"]),
            last_updated_at=datetime.fromisoformat(row["last_updated_at"]),
            article_count=row["article_count"],
        )

    # ============================================
    # Phase 3: Topic Radar Operations
    # ============================================

    async def create_topic(self, topic: Topic) -> Topic:
        """Create a new topic."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO topics (name, description, keywords, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    topic.name,
                    topic.description,
                    topic.keywords,
                    topic.created_at.isoformat(),
                ),
            )
            await self._connection.commit()
            topic.id = cursor.lastrowid
            return topic

    async def get_topic(self, topic_id: int) -> Optional[Topic]:
        """Get a topic by ID."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("SELECT * FROM topics WHERE id = ?", (topic_id,))
            row = await cursor.fetchone()
            if row:
                return self._row_to_topic(row)
            return None

    async def get_topic_by_name(self, name: str) -> Optional[Topic]:
        """Get a topic by name."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("SELECT * FROM topics WHERE name = ?", (name,))
            row = await cursor.fetchone()
            if row:
                return self._row_to_topic(row)
            return None

    async def list_topics(self) -> list[Topic]:
        """List all topics."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("SELECT * FROM topics ORDER BY name")
            rows = await cursor.fetchall()
            return [self._row_to_topic(row) for row in rows]

    async def delete_topic(self, topic_id: int) -> None:
        """Delete a topic."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("DELETE FROM content_topics WHERE topic_id = ?", (topic_id,))
            await cursor.execute("DELETE FROM topic_snapshots WHERE topic_id = ?", (topic_id,))
            await cursor.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
            await self._connection.commit()

    async def add_content_topic(self, content_id: int, topic_id: int, relevance: float) -> None:
        """Associate content with a topic."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT OR REPLACE INTO content_topics (content_id, topic_id, relevance)
                VALUES (?, ?, ?)
                """,
                (content_id, topic_id, relevance),
            )
            await self._connection.commit()

    async def get_topic_content(
        self, topic_id: int, min_relevance: float = 0.5, limit: int = 50
    ) -> list[ContentItem]:
        """Get content items related to a topic."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT c.* FROM content_items c
                INNER JOIN content_topics ct ON c.id = ct.content_id
                WHERE ct.topic_id = ? AND ct.relevance >= ?
                ORDER BY ct.relevance DESC, c.published_at DESC
                LIMIT ?
                """,
                (topic_id, min_relevance, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_content_item(row) for row in rows]

    async def add_topic_snapshot(self, snapshot: TopicSnapshot) -> TopicSnapshot:
        """Add a topic snapshot."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO topic_snapshots (topic_id, snapshot_date, summary, key_entities, metrics, trend)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.topic_id,
                    snapshot.snapshot_date,
                    snapshot.summary,
                    snapshot.key_entities,
                    snapshot.metrics,
                    snapshot.trend,
                ),
            )
            await self._connection.commit()
            snapshot.id = cursor.lastrowid
            return snapshot

    async def get_topic_snapshots(
        self, topic_id: int, limit: int = 10
    ) -> list[TopicSnapshot]:
        """Get recent snapshots for a topic."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT * FROM topic_snapshots
                WHERE topic_id = ?
                ORDER BY snapshot_date DESC
                LIMIT ?
                """,
                (topic_id, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_topic_snapshot(row) for row in rows]

    def _row_to_topic(self, row: aiosqlite.Row) -> Topic:
        """Convert a database row to a Topic object."""
        return Topic(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            keywords=row["keywords"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_topic_snapshot(self, row: aiosqlite.Row) -> TopicSnapshot:
        """Convert a database row to a TopicSnapshot object."""
        return TopicSnapshot(
            id=row["id"],
            topic_id=row["topic_id"],
            snapshot_date=row["snapshot_date"],
            summary=row["summary"],
            key_entities=row["key_entities"],
            metrics=row["metrics"],
            trend=row["trend"],
        )

    # ============================================
    # Phase 5: Action List Operations
    # ============================================

    async def create_action_item(self, action: ActionItem) -> ActionItem:
        """Create a new action item."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO action_items (content_item_id, type, description, priority, status, due_date, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action.content_item_id,
                    action.type,
                    action.description,
                    action.priority,
                    action.status.value,
                    action.due_date,
                    action.created_at.isoformat(),
                ),
            )
            await self._connection.commit()
            action.id = cursor.lastrowid
            return action

    async def get_action_items(
        self,
        status: Optional[ActionStatus] = None,
        priority: Optional[str] = None,
        limit: int = 50,
    ) -> list[ActionItem]:
        """Get action items with optional filters."""
        async with self._connection.cursor() as cursor:
            sql = "SELECT * FROM action_items WHERE 1=1"
            params: list = []

            if status:
                sql += " AND status = ?"
                params.append(status.value)

            if priority:
                sql += " AND priority = ?"
                params.append(priority)

            sql += " ORDER BY CASE priority WHEN '高' THEN 1 WHEN '中' THEN 2 ELSE 3 END, created_at DESC LIMIT ?"
            params.append(limit)

            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
            return [self._row_to_action_item(row) for row in rows]

    async def update_action_status(
        self, action_id: int, status: ActionStatus
    ) -> None:
        """Update action item status."""
        async with self._connection.cursor() as cursor:
            completed_at = datetime.now().isoformat() if status == ActionStatus.DONE else None
            await cursor.execute(
                "UPDATE action_items SET status = ?, completed_at = ? WHERE id = ?",
                (status.value, completed_at, action_id),
            )
            await self._connection.commit()

    async def delete_action_item(self, action_id: int) -> None:
        """Delete an action item."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("DELETE FROM action_items WHERE id = ?", (action_id,))
            await self._connection.commit()

    def _row_to_action_item(self, row: aiosqlite.Row) -> ActionItem:
        """Convert a database row to an ActionItem object."""
        return ActionItem(
            id=row["id"],
            content_item_id=row["content_item_id"],
            type=row["type"],
            description=row["description"],
            priority=row["priority"],
            status=ActionStatus(row["status"]),
            due_date=row["due_date"],
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )

    # Trigger operations

    async def create_trigger(self, trigger: Trigger) -> Trigger:
        """Create a new trigger."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO triggers (name, condition, action, enabled)
                VALUES (?, ?, ?, ?)
                """,
                (trigger.name, trigger.condition, trigger.action, 1 if trigger.enabled else 0),
            )
            await self._connection.commit()
            trigger.id = cursor.lastrowid
            return trigger

    async def get_triggers(self, enabled_only: bool = True) -> list[Trigger]:
        """Get all triggers."""
        async with self._connection.cursor() as cursor:
            if enabled_only:
                await cursor.execute("SELECT * FROM triggers WHERE enabled = 1")
            else:
                await cursor.execute("SELECT * FROM triggers")
            rows = await cursor.fetchall()
            return [self._row_to_trigger(row) for row in rows]

    async def delete_trigger(self, trigger_id: int) -> None:
        """Delete a trigger."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("DELETE FROM triggers WHERE id = ?", (trigger_id,))
            await self._connection.commit()

    def _row_to_trigger(self, row: aiosqlite.Row) -> Trigger:
        """Convert a database row to a Trigger object."""
        return Trigger(
            id=row["id"],
            name=row["name"],
            condition=row["condition"],
            action=row["action"],
            enabled=bool(row["enabled"]),
        )


# Synchronous wrapper for CLI usage
class SyncDatabase:
    """Synchronous wrapper for Database class."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_settings().database.path
        self._async_db = Database(db_path)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def _run(self, coro):
        """Run an async coroutine synchronously."""
        return self._get_loop().run_until_complete(coro)

    def connect(self) -> None:
        self._run(self._async_db.connect())

    def close(self) -> None:
        self._run(self._async_db.close())

    def add_subscription(self, subscription: Subscription) -> Subscription:
        return self._run(self._async_db.add_subscription(subscription))

    def get_subscription(self, sub_id: int) -> Optional[Subscription]:
        return self._run(self._async_db.get_subscription(sub_id))

    def get_subscription_by_name(
        self, source_type: SourceType, subscription_type: SubscriptionType, name: str
    ) -> Optional[Subscription]:
        return self._run(
            self._async_db.get_subscription_by_name(source_type, subscription_type, name)
        )

    def list_subscriptions(self, enabled_only: bool = False) -> list[Subscription]:
        return self._run(self._async_db.list_subscriptions(enabled_only))

    def update_subscription(self, subscription: Subscription) -> None:
        self._run(self._async_db.update_subscription(subscription))

    def delete_subscription(self, sub_id: int) -> None:
        self._run(self._async_db.delete_subscription(sub_id))

    def add_content_item(self, item: ContentItem) -> ContentItem:
        return self._run(self._async_db.add_content_item(item))

    def get_content_item(self, item_id: int) -> Optional[ContentItem]:
        return self._run(self._async_db.get_content_item(item_id))

    def get_unprocessed_items(self, limit: int = 100) -> list[ContentItem]:
        return self._run(self._async_db.get_unprocessed_items(limit))

    def get_undelivered_items(
        self, min_importance: int = 1, limit: int = 50
    ) -> list[ContentItem]:
        return self._run(self._async_db.get_undelivered_items(min_importance, limit))

    def update_content_item(self, item: ContentItem) -> None:
        self._run(self._async_db.update_content_item(item))

    def mark_items_delivered(self, item_ids: list[int]) -> None:
        self._run(self._async_db.mark_items_delivered(item_ids))

    def content_exists(self, source_type: SourceType, external_id: str) -> bool:
        return self._run(self._async_db.content_exists(source_type, external_id))

    def url_exists(self, url: str) -> bool:
        return self._run(self._async_db.url_exists(url))

    def get_or_create_profile(self, email: str) -> UserProfile:
        return self._run(self._async_db.get_or_create_profile(email))

    def update_profile(self, profile: UserProfile) -> None:
        self._run(self._async_db.update_profile(profile))

    # Phase 4: Knowledge Base sync wrappers

    def search_content(
        self,
        query: str,
        category: Optional[str] = None,
        min_importance: Optional[int] = None,
        state: Optional[ItemState] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ContentItem]:
        return self._run(self._async_db.search_content(
            query, category, min_importance, state, from_date, to_date, limit, offset
        ))

    def browse_by_date(
        self, date: str, category: Optional[str] = None, limit: int = 100
    ) -> list[ContentItem]:
        return self._run(self._async_db.browse_by_date(date, category, limit))

    def update_item_state(self, item_id: int, state: ItemState) -> None:
        self._run(self._async_db.update_item_state(item_id, state))

    def get_category_stats(self) -> dict[str, int]:
        return self._run(self._async_db.get_category_stats())

    def get_state_stats(self) -> dict[str, int]:
        return self._run(self._async_db.get_state_stats())

    def rebuild_fts_index(self) -> int:
        return self._run(self._async_db.rebuild_fts_index())

    # Phase 2: Event Stream sync wrappers

    def create_event_cluster(self, cluster: EventCluster) -> EventCluster:
        return self._run(self._async_db.create_event_cluster(cluster))

    def get_event_cluster(self, cluster_id: int) -> Optional[EventCluster]:
        return self._run(self._async_db.get_event_cluster(cluster_id))

    def get_recent_event_clusters(
        self, days: int = 7, category: Optional[str] = None, limit: int = 50
    ) -> list[EventCluster]:
        return self._run(self._async_db.get_recent_event_clusters(days, category, limit))

    def add_event_member(self, member: EventMember) -> None:
        self._run(self._async_db.add_event_member(member))

    def get_event_members(self, cluster_id: int) -> list[ContentItem]:
        return self._run(self._async_db.get_event_members(cluster_id))

    def add_event_timeline(self, timeline: EventTimeline) -> None:
        self._run(self._async_db.add_event_timeline(timeline))

    # Phase 3: Topic Radar sync wrappers

    def create_topic(self, topic: Topic) -> Topic:
        return self._run(self._async_db.create_topic(topic))

    def get_topic(self, topic_id: int) -> Optional[Topic]:
        return self._run(self._async_db.get_topic(topic_id))

    def get_topic_by_name(self, name: str) -> Optional[Topic]:
        return self._run(self._async_db.get_topic_by_name(name))

    def list_topics(self) -> list[Topic]:
        return self._run(self._async_db.list_topics())

    def delete_topic(self, topic_id: int) -> None:
        self._run(self._async_db.delete_topic(topic_id))

    def add_content_topic(self, content_id: int, topic_id: int, relevance: float) -> None:
        self._run(self._async_db.add_content_topic(content_id, topic_id, relevance))

    def get_topic_content(
        self, topic_id: int, min_relevance: float = 0.5, limit: int = 50
    ) -> list[ContentItem]:
        return self._run(self._async_db.get_topic_content(topic_id, min_relevance, limit))

    def add_topic_snapshot(self, snapshot: TopicSnapshot) -> TopicSnapshot:
        return self._run(self._async_db.add_topic_snapshot(snapshot))

    def get_topic_snapshots(self, topic_id: int, limit: int = 10) -> list[TopicSnapshot]:
        return self._run(self._async_db.get_topic_snapshots(topic_id, limit))

    # Phase 5: Action List sync wrappers

    def create_action_item(self, action: ActionItem) -> ActionItem:
        return self._run(self._async_db.create_action_item(action))

    def get_action_items(
        self,
        status: Optional[ActionStatus] = None,
        priority: Optional[str] = None,
        limit: int = 50,
    ) -> list[ActionItem]:
        return self._run(self._async_db.get_action_items(status, priority, limit))

    def update_action_status(self, action_id: int, status: ActionStatus) -> None:
        self._run(self._async_db.update_action_status(action_id, status))

    def delete_action_item(self, action_id: int) -> None:
        self._run(self._async_db.delete_action_item(action_id))

    def create_trigger(self, trigger: Trigger) -> Trigger:
        return self._run(self._async_db.create_trigger(trigger))

    def get_triggers(self, enabled_only: bool = True) -> list[Trigger]:
        return self._run(self._async_db.get_triggers(enabled_only))

    def delete_trigger(self, trigger_id: int) -> None:
        self._run(self._async_db.delete_trigger(trigger_id))

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

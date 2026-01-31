"""SQLite database operations for BabelByte."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from config.settings import get_settings
from src.storage.models import (
    ActionItem,
    ActionStatus,
    ContentItem,
    EventCluster,
    EventMember,
    EventTimeline,
    ItemState,
    SourceType,
    Subscription,
    SubscriptionType,
    TokenUsage,
    Topic,
    TopicSnapshot,
    TopicSuggestion,
    Trigger,
    UserProfile,
)

logger = logging.getLogger(__name__)

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
    cluster_attempted_at TEXT,
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
    content_item_id INTEGER NOT NULL UNIQUE,
    similarity_score REAL DEFAULT 0.0,
    detection_method TEXT DEFAULT 'rule',
    FOREIGN KEY (event_cluster_id) REFERENCES event_clusters(id),
    FOREIGN KEY (content_item_id) REFERENCES content_items(id)
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

# AI Cache table for storing processed results
CREATE_AI_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS ai_cache (
    content_hash TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
)
"""

# Token Usage table for tracking AI calls
CREATE_TOKEN_USAGE_TABLE = """
CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    cached INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0,
    success INTEGER DEFAULT 1,
    error TEXT
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

# Embedding tables for semantic similarity
CREATE_CONTENT_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS content_embeddings (
    content_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (content_id) REFERENCES content_items(id)
)
"""

CREATE_CLUSTER_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS cluster_embeddings (
    cluster_id INTEGER PRIMARY KEY,
    centroid_embedding BLOB NOT NULL,
    member_count INTEGER NOT NULL,
    last_updated_at TEXT NOT NULL,
    FOREIGN KEY (cluster_id) REFERENCES event_clusters(id)
)
"""

# Topic suggestions table for automatic topic discovery
CREATE_TOPIC_SUGGESTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS topic_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    keywords TEXT NOT NULL,
    frequency INTEGER NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    sample_titles TEXT,
    status TEXT DEFAULT 'pending',
    suggested_at TEXT NOT NULL,
    reviewed_at TEXT,
    merged_with_topic_id INTEGER,
    FOREIGN KEY (merged_with_topic_id) REFERENCES topics(id)
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

-- Event members indexes (for cluster performance optimization)
CREATE INDEX IF NOT EXISTS idx_event_members_content_id ON event_members(content_item_id);
CREATE INDEX IF NOT EXISTS idx_event_members_cluster_id ON event_members(event_cluster_id);

-- AI cache indexes
CREATE INDEX IF NOT EXISTS idx_ai_cache_expires ON ai_cache(expires_at);

-- Token usage indexes
CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp ON token_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_token_usage_call_type ON token_usage(call_type);
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

            # AI Cache table
            await cursor.execute(CREATE_AI_CACHE_TABLE)

            # Token Usage table
            await cursor.execute(CREATE_TOKEN_USAGE_TABLE)

            # Embedding tables
            await cursor.execute(CREATE_CONTENT_EMBEDDINGS_TABLE)
            await cursor.execute(CREATE_CLUSTER_EMBEDDINGS_TABLE)

            # Topic suggestions table
            await cursor.execute(CREATE_TOPIC_SUGGESTIONS_TABLE)

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
                await cursor.execute("ALTER TABLE subscriptions ADD COLUMN twitter_user_id TEXT")
            if "last_tweet_id" not in sub_columns:
                await cursor.execute("ALTER TABLE subscriptions ADD COLUMN last_tweet_id TEXT")
            if "last_reddit_id" not in sub_columns:
                await cursor.execute("ALTER TABLE subscriptions ADD COLUMN last_reddit_id TEXT")
            if "feed_url_override" not in sub_columns:
                await cursor.execute("ALTER TABLE subscriptions ADD COLUMN feed_url_override TEXT")
            if "last_entry_id" not in sub_columns:
                await cursor.execute("ALTER TABLE subscriptions ADD COLUMN last_entry_id TEXT")

            # Check content_items columns for Phase 1 & 4 enhancements
            await cursor.execute("PRAGMA table_info(content_items)")
            content_columns = {row[1] for row in await cursor.fetchall()}

            # Phase 1: Enhanced digest fields
            if "one_liner" not in content_columns:
                await cursor.execute("ALTER TABLE content_items ADD COLUMN one_liner TEXT")
            if "key_points" not in content_columns:
                await cursor.execute("ALTER TABLE content_items ADD COLUMN key_points TEXT")
            if "impact_assessment" not in content_columns:
                await cursor.execute("ALTER TABLE content_items ADD COLUMN impact_assessment TEXT")
            if "actionable_items" not in content_columns:
                await cursor.execute("ALTER TABLE content_items ADD COLUMN actionable_items TEXT")

            # Phase 4: State management
            if "state" not in content_columns:
                await cursor.execute(
                    "ALTER TABLE content_items ADD COLUMN state TEXT DEFAULT 'unread'"
                )
            if "cluster_attempted_at" not in content_columns:
                await cursor.execute(
                    "ALTER TABLE content_items ADD COLUMN cluster_attempted_at TEXT"
                )

            # Migrate event_members table to enforce UNIQUE content_item_id
            # This prevents the same content item from being in multiple clusters
            await self._migrate_event_members_unique_constraint(cursor)

            await self._connection.commit()

    async def _migrate_event_members_unique_constraint(self, cursor) -> None:
        """Migrate event_members table to enforce UNIQUE on content_item_id.

        SQLite doesn't support adding constraints to existing tables, so we need to:
        1. Check if migration is needed (old schema has composite unique constraint)
        2. Create new table with correct schema
        3. Copy data (keeping only one row per content_item_id with highest similarity_score)
        4. Drop old table and rename new one
        """
        # Check current table schema
        await cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='event_members'"
        )
        row = await cursor.fetchone()
        if not row:
            return  # Table doesn't exist, will be created with correct schema

        table_sql = row[0]
        # Check if migration is needed: old schema has UNIQUE(event_cluster_id, content_item_id)
        # New schema has content_item_id INTEGER NOT NULL UNIQUE
        if "UNIQUE(event_cluster_id, content_item_id)" not in table_sql:
            return  # Already migrated or using new schema

        logger.info("Migrating event_members table to enforce UNIQUE content_item_id...")

        # Create new table with correct schema
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_members_new (
                event_cluster_id INTEGER NOT NULL,
                content_item_id INTEGER NOT NULL UNIQUE,
                similarity_score REAL DEFAULT 0.0,
                detection_method TEXT DEFAULT 'rule',
                FOREIGN KEY (event_cluster_id) REFERENCES event_clusters(id),
                FOREIGN KEY (content_item_id) REFERENCES content_items(id)
            )
        """)

        # Copy data, keeping only the row with highest similarity_score for each content_item_id
        await cursor.execute("""
            INSERT OR IGNORE INTO event_members_new
                (event_cluster_id, content_item_id, similarity_score, detection_method)
            SELECT event_cluster_id, content_item_id, similarity_score, detection_method
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY content_item_id ORDER BY similarity_score DESC
                ) as rn
                FROM event_members
            ) WHERE rn = 1
        """)

        # Drop old table and rename new one
        await cursor.execute("DROP TABLE event_members")
        await cursor.execute("ALTER TABLE event_members_new RENAME TO event_members")

        # Recreate indexes
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_members_content_id
            ON event_members(content_item_id)
        """)
        await cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_members_cluster_id
            ON event_members(event_cluster_id)
        """)

        # Update all cluster article counts
        await cursor.execute("""
            UPDATE event_clusters SET article_count = (
                SELECT COUNT(*) FROM event_members WHERE event_cluster_id = event_clusters.id
            )
        """)

        logger.info("Migration completed: event_members table now enforces UNIQUE content_item_id")

    # Subscription operations

    async def add_subscription(self, subscription: Subscription) -> Subscription:
        """Add a new subscription."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO subscriptions (source_type, subscription_type, name, enabled, created_at,
                    feed_url_override, last_entry_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subscription.source_type.value,
                    subscription.subscription_type.value,
                    subscription.name,
                    1 if subscription.enabled else 0,
                    subscription.created_at.isoformat(),
                    subscription.feed_url_override,
                    subscription.last_entry_id,
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
                    last_reddit_id = ?, feed_url_override = ?, last_entry_id = ?
                WHERE id = ?
                """,
                (
                    1 if subscription.enabled else 0,
                    (
                        subscription.last_fetched_at.isoformat()
                        if subscription.last_fetched_at
                        else None
                    ),
                    subscription.twitter_user_id,
                    subscription.last_tweet_id,
                    subscription.last_reddit_id,
                    subscription.feed_url_override,
                    subscription.last_entry_id,
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
        row_keys = row.keys()
        return Subscription(
            id=row["id"],
            source_type=SourceType(row["source_type"]),
            subscription_type=SubscriptionType(row["subscription_type"]),
            name=row["name"],
            enabled=bool(row["enabled"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_fetched_at=(
                datetime.fromisoformat(row["last_fetched_at"]) if row["last_fetched_at"] else None
            ),
            twitter_user_id=row["twitter_user_id"] if "twitter_user_id" in row_keys else None,
            last_tweet_id=row["last_tweet_id"] if "last_tweet_id" in row_keys else None,
            last_reddit_id=row["last_reddit_id"] if "last_reddit_id" in row_keys else None,
            feed_url_override=row["feed_url_override"] if "feed_url_override" in row_keys else None,
            last_entry_id=row["last_entry_id"] if "last_entry_id" in row_keys else None,
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

    async def get_unclustered_items(
        self,
        min_importance: int = 5,
        limit: int = 100,
        retry_after_hours: Optional[int] = None,
    ) -> list[ContentItem]:
        """Get processed items not yet assigned to any cluster.

        This is an optimization for clustering - it skips items that are already
        in a cluster, avoiding redundant processing.

        Args:
            min_importance: Minimum importance score to include
            limit: Maximum items to return

        Returns:
            List of ContentItem not in any cluster
        """
        async with self._connection.cursor() as cursor:
            cutoff = None
            if retry_after_hours is not None and retry_after_hours > 0:
                cutoff = (
                    datetime.now() - __import__("datetime").timedelta(hours=retry_after_hours)
                ).isoformat()
            await cursor.execute(
                """
                SELECT c.* FROM content_items c
                LEFT JOIN event_members em ON c.id = em.content_item_id
                WHERE c.processed_at IS NOT NULL
                  AND c.delivered = 0
                  AND c.importance_score >= ?
                  AND em.content_item_id IS NULL
                  AND (
                        ? IS NULL
                        OR c.cluster_attempted_at IS NULL
                        OR c.cluster_attempted_at <= ?
                  )
                ORDER BY c.importance_score DESC, c.published_at DESC
                LIMIT ?
                """,
                (min_importance, cutoff, cutoff, limit),
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

    async def mark_cluster_attempted(self, item_id: int) -> None:
        """Mark a content item as having been attempted for clustering."""
        async with self._connection.cursor() as cursor:
            now = datetime.now().isoformat()
            await cursor.execute(
                "UPDATE content_items SET cluster_attempted_at = ? WHERE id = ?",
                (now, item_id),
            )
            await self._connection.commit()

    async def _update_fts_index(self, item: ContentItem) -> None:
        """Update FTS index for a content item."""
        async with self._connection.cursor() as cursor:
            # Delete existing entry
            await cursor.execute("DELETE FROM content_fts WHERE content_id = ?", (item.id,))
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

    async def title_exists(self, title: str) -> bool:
        """Check if content with exact same title already exists.

        This helps detect duplicate content from different sources.
        """
        if not title or len(title) < 10:
            return False
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                "SELECT 1 FROM content_items WHERE title = ? LIMIT 1",
                (title,),
            )
            return await cursor.fetchone() is not None

    async def find_similar_title(self, title: str, days: int = 7) -> Optional[int]:
        """Find content item with very similar title (for deduplication).

        Uses normalized title comparison to detect near-duplicates.

        Args:
            title: Title to search for
            days: Look back this many days

        Returns:
            Content item ID if similar title found, None otherwise
        """
        if not title or len(title) < 10:
            return None

        # Normalize: remove RT prefix, @mentions, URLs, lowercase
        import re
        normalized = title.strip().lower()
        normalized = re.sub(r"^rt\s+", "", normalized)
        normalized = re.sub(r"@\w+[:\s]*", "", normalized)
        normalized = re.sub(r"https?://\S+", "", normalized)
        normalized = " ".join(normalized.split())

        if len(normalized) < 10:
            return None

        async with self._connection.cursor() as cursor:
            # Search for exact normalized match or very similar titles
            from_date = (
                datetime.now() - __import__("datetime").timedelta(days=days)
            ).isoformat()

            await cursor.execute(
                """
                SELECT id, title FROM content_items
                WHERE published_at >= ? AND title IS NOT NULL
                ORDER BY published_at DESC
                LIMIT 500
                """,
                (from_date,),
            )
            rows = await cursor.fetchall()

            for row in rows:
                existing_title = row["title"]
                if not existing_title:
                    continue

                # Normalize existing title
                existing_norm = existing_title.strip().lower()
                existing_norm = re.sub(r"^rt\s+", "", existing_norm)
                existing_norm = re.sub(r"@\w+[:\s]*", "", existing_norm)
                existing_norm = re.sub(r"https?://\S+", "", existing_norm)
                existing_norm = " ".join(existing_norm.split())

                # Exact match after normalization
                if normalized == existing_norm:
                    return row["id"]

                # Check if one is prefix of other (common in truncated tweets)
                if len(normalized) > 20 and len(existing_norm) > 20:
                    if normalized.startswith(existing_norm[:50]):
                        return row["id"]
                    if existing_norm.startswith(normalized[:50]):
                        return row["id"]

            return None

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
            tracking_params = {
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_term",
                "utm_content",
                "ref",
                "source",
            }
            query_params = parse_qs(parsed.query)
            filtered_params = {
                k: v for k, v in query_params.items() if k.lower() not in tracking_params
            }
            clean_query = urlencode(filtered_params, doseq=True)
            # Rebuild URL without tracking params and trailing slash
            clean_url = urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path.rstrip("/"),
                    parsed.params,
                    clean_query,
                    "",
                )
            )
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
            cluster_attempted_at=(
                datetime.fromisoformat(row["cluster_attempted_at"])
                if "cluster_attempted_at" in row_keys and row["cluster_attempted_at"]
                else None
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
            await cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM content_items
                WHERE category IS NOT NULL
                GROUP BY category
                ORDER BY count DESC
                """)
            rows = await cursor.fetchall()
            return {row["category"]: row["count"] for row in rows}

    async def get_state_stats(self) -> dict[str, int]:
        """Get count of items per state."""
        async with self._connection.cursor() as cursor:
            await cursor.execute("""
                SELECT state, COUNT(*) as count
                FROM content_items
                GROUP BY state
                ORDER BY count DESC
                """)
            rows = await cursor.fetchall()
            return {row["state"]: row["count"] for row in rows}

    async def rebuild_fts_index(self) -> int:
        """Rebuild the full-text search index from all processed items."""
        async with self._connection.cursor() as cursor:
            # Clear existing FTS data
            await cursor.execute("DELETE FROM content_fts")

            # Rebuild from processed items
            await cursor.execute("""
                INSERT INTO content_fts (title, content, summary, category, author, content_id)
                SELECT title, content, summary, category, author, id
                FROM content_items
                WHERE processed_at IS NOT NULL
                """)
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

    async def add_event_member(self, member: EventMember) -> bool:
        """Add a content item to an event cluster.

        Uses INSERT OR IGNORE to prevent duplicate cluster memberships.
        Each content item can only belong to one cluster.

        Returns:
            True if the member was added, False if already in a cluster.
        """
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT OR IGNORE INTO event_members
                    (event_cluster_id, content_item_id, similarity_score, detection_method)
                VALUES (?, ?, ?, ?)
                """,
                (
                    member.event_cluster_id,
                    member.content_item_id,
                    member.similarity_score,
                    member.detection_method,
                ),
            )
            # Only update article count if insertion succeeded
            if cursor.rowcount > 0:
                await cursor.execute(
                    """
                    UPDATE event_clusters
                    SET article_count = (
                        SELECT COUNT(*) FROM event_members WHERE event_cluster_id = ?
                    ),
                        last_updated_at = ?
                    WHERE id = ?
                    """,
                    (member.event_cluster_id, datetime.now().isoformat(), member.event_cluster_id),
                )
                await self._connection.commit()
                return True
            return False

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

    async def get_undelivered_clustered_items(
        self, min_importance: int = 5, limit: int = 50
    ) -> dict[int, list[ContentItem]]:
        """
        Get undelivered content items that belong to event clusters.

        Returns items grouped by cluster_id, with members sorted by importance_score desc.

        Args:
            min_importance: Minimum importance score to include
            limit: Maximum total items to return

        Returns:
            Dict mapping cluster_id to list of ContentItem
        """
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT c.*, em.event_cluster_id
                FROM content_items c
                INNER JOIN event_members em ON c.id = em.content_item_id
                WHERE c.processed_at IS NOT NULL
                  AND c.delivered = 0
                  AND c.importance_score >= ?
                ORDER BY c.importance_score DESC, c.published_at DESC
                LIMIT ?
                """,
                (min_importance, limit),
            )
            rows = await cursor.fetchall()

            # Group by cluster_id
            result: dict[int, list[ContentItem]] = {}
            for row in rows:
                cluster_id = row["event_cluster_id"]
                item = self._row_to_content_item(row)
                if cluster_id not in result:
                    result[cluster_id] = []
                result[cluster_id].append(item)

            return result

    async def get_undelivered_unclustered_items(
        self, min_importance: int = 5, limit: int = 50
    ) -> list[ContentItem]:
        """
        Get undelivered content items that do not belong to any event cluster.

        Args:
            min_importance: Minimum importance score to include
            limit: Maximum items to return

        Returns:
            List of ContentItem not in any cluster
        """
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT c.* FROM content_items c
                LEFT JOIN event_members em ON c.id = em.content_item_id
                WHERE c.processed_at IS NOT NULL
                  AND c.delivered = 0
                  AND c.importance_score >= ?
                  AND em.content_item_id IS NULL
                ORDER BY c.importance_score DESC, c.published_at DESC
                LIMIT ?
                """,
                (min_importance, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_content_item(row) for row in rows]

    async def is_item_in_cluster(self, item_id: int) -> bool:
        """Check if a content item is already in any event cluster.

        Args:
            item_id: The content item ID to check.

        Returns:
            True if the item is in a cluster, False otherwise.
        """
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                "SELECT 1 FROM event_members WHERE content_item_id = ? LIMIT 1",
                (item_id,),
            )
            return await cursor.fetchone() is not None

    async def cleanup_duplicate_cluster_memberships(self) -> int:
        """Remove duplicate cluster memberships, keeping the one with highest similarity score.

        This is a cleanup utility for fixing data where items were added to multiple clusters.

        Returns:
            Number of duplicate memberships removed.
        """
        async with self._connection.cursor() as cursor:
            # Delete duplicates, keeping the row with highest similarity_score
            await cursor.execute("""
                DELETE FROM event_members
                WHERE rowid NOT IN (
                    SELECT rowid FROM (
                        SELECT rowid, content_item_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY content_item_id
                                   ORDER BY similarity_score DESC
                               ) as rn
                        FROM event_members
                    ) WHERE rn = 1
                )
            """)
            removed = cursor.rowcount

            # Update all cluster article counts
            await cursor.execute("""
                UPDATE event_clusters SET article_count = (
                    SELECT COUNT(*) FROM event_members
                    WHERE event_cluster_id = event_clusters.id
                )
            """)
            await self._connection.commit()
            return removed

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

    async def get_topic_snapshots(self, topic_id: int, limit: int = 10) -> list[TopicSnapshot]:
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

    # Topic Suggestion operations

    async def create_topic_suggestion(self, suggestion: TopicSuggestion) -> TopicSuggestion:
        """Create a new topic suggestion."""
        import json

        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO topic_suggestions
                    (name, keywords, frequency, confidence, source, sample_titles, status, suggested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    suggestion.name,
                    json.dumps(suggestion.get_keywords(), ensure_ascii=False)
                    if isinstance(suggestion.keywords, str) else
                    json.dumps(suggestion.keywords or [], ensure_ascii=False),
                    suggestion.frequency,
                    suggestion.confidence,
                    suggestion.source,
                    json.dumps(suggestion.get_sample_titles(), ensure_ascii=False)
                    if isinstance(suggestion.sample_titles, str) else
                    json.dumps(suggestion.sample_titles or [], ensure_ascii=False),
                    suggestion.status,
                    suggestion.suggested_at.isoformat(),
                ),
            )
            await self._connection.commit()
            suggestion.id = cursor.lastrowid
            return suggestion

    async def get_topic_suggestion(self, suggestion_id: int) -> Optional[TopicSuggestion]:
        """Get a topic suggestion by ID."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                "SELECT * FROM topic_suggestions WHERE id = ?",
                (suggestion_id,),
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_topic_suggestion(row)
            return None

    async def get_topic_suggestions(
        self, status: Optional[str] = None, limit: int = 50
    ) -> list[TopicSuggestion]:
        """Get topic suggestions with optional status filter."""
        async with self._connection.cursor() as cursor:
            if status:
                await cursor.execute(
                    """
                    SELECT * FROM topic_suggestions
                    WHERE status = ?
                    ORDER BY confidence * frequency DESC
                    LIMIT ?
                    """,
                    (status, limit),
                )
            else:
                await cursor.execute(
                    """
                    SELECT * FROM topic_suggestions
                    ORDER BY suggested_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = await cursor.fetchall()
            return [self._row_to_topic_suggestion(row) for row in rows]

    async def update_topic_suggestion_status(
        self,
        suggestion_id: int,
        status: str,
        merged_with_topic_id: Optional[int] = None,
    ) -> bool:
        """Update the status of a topic suggestion."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                UPDATE topic_suggestions
                SET status = ?, reviewed_at = ?, merged_with_topic_id = ?
                WHERE id = ?
                """,
                (status, datetime.now().isoformat(), merged_with_topic_id, suggestion_id),
            )
            await self._connection.commit()
            return cursor.rowcount > 0

    async def update_topic(self, topic: Topic) -> None:
        """Update a topic's keywords."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                UPDATE topics SET keywords = ?, description = ?
                WHERE id = ?
                """,
                (topic.keywords, topic.description, topic.id),
            )
            await self._connection.commit()

    def _row_to_topic_suggestion(self, row: aiosqlite.Row) -> TopicSuggestion:
        """Convert a database row to a TopicSuggestion object."""
        return TopicSuggestion(
            id=row["id"],
            name=row["name"],
            keywords=row["keywords"],
            frequency=row["frequency"],
            confidence=row["confidence"],
            source=row["source"],
            sample_titles=row["sample_titles"],
            status=row["status"],
            suggested_at=datetime.fromisoformat(row["suggested_at"]),
            reviewed_at=datetime.fromisoformat(row["reviewed_at"]) if row["reviewed_at"] else None,
            merged_with_topic_id=row["merged_with_topic_id"],
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

    async def update_action_status(self, action_id: int, status: ActionStatus) -> None:
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
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
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

    # ============================================
    # AI Cache Operations
    # ============================================

    async def get_ai_cache(self, content_hash: str) -> Optional[str]:
        """
        Get cached AI processing result.

        Args:
            content_hash: SHA256 hash of content (16 chars).

        Returns:
            Cached result JSON string, or None if not found or expired.
        """
        async with self._connection.cursor() as cursor:
            now = datetime.now().isoformat()
            await cursor.execute(
                """
                SELECT result_json FROM ai_cache
                WHERE content_hash = ? AND expires_at > ?
                """,
                (content_hash, now),
            )
            row = await cursor.fetchone()
            return row["result_json"] if row else None

    async def set_ai_cache(
        self, content_hash: str, result_json: str, ttl_seconds: int = 86400
    ) -> None:
        """
        Store AI processing result in cache.

        Args:
            content_hash: SHA256 hash of content (16 chars).
            result_json: Serialized ProcessingResult JSON.
            ttl_seconds: Time-to-live in seconds (default 24 hours).
        """
        from datetime import timedelta

        now = datetime.now()
        expires_at = now + timedelta(seconds=ttl_seconds)

        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT OR REPLACE INTO ai_cache (content_hash, result_json, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (content_hash, result_json, now.isoformat(), expires_at.isoformat()),
            )
            await self._connection.commit()

    async def clear_ai_cache(self) -> int:
        """
        Clear all AI cache entries.

        Returns:
            Number of entries deleted.
        """
        async with self._connection.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) FROM ai_cache")
            row = await cursor.fetchone()
            count = row[0] if row else 0
            await cursor.execute("DELETE FROM ai_cache")
            await self._connection.commit()
            return count

    async def cleanup_expired_cache(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries deleted.
        """
        async with self._connection.cursor() as cursor:
            now = datetime.now().isoformat()
            await cursor.execute("SELECT COUNT(*) FROM ai_cache WHERE expires_at <= ?", (now,))
            row = await cursor.fetchone()
            count = row[0] if row else 0
            await cursor.execute("DELETE FROM ai_cache WHERE expires_at <= ?", (now,))
            await self._connection.commit()
            return count

    async def get_ai_cache_stats(self) -> dict:
        """
        Get AI cache statistics.

        Returns:
            Dict with total_entries, valid_entries, expired_entries, oldest_entry, newest_entry.
        """
        async with self._connection.cursor() as cursor:
            now = datetime.now().isoformat()

            # Total entries
            await cursor.execute("SELECT COUNT(*) FROM ai_cache")
            row = await cursor.fetchone()
            total = row[0] if row else 0

            # Valid (not expired) entries
            await cursor.execute("SELECT COUNT(*) FROM ai_cache WHERE expires_at > ?", (now,))
            row = await cursor.fetchone()
            valid = row[0] if row else 0

            # Oldest entry
            await cursor.execute(
                "SELECT MIN(created_at) FROM ai_cache WHERE expires_at > ?", (now,)
            )
            row = await cursor.fetchone()
            oldest = row[0] if row and row[0] else None

            # Newest entry
            await cursor.execute(
                "SELECT MAX(created_at) FROM ai_cache WHERE expires_at > ?", (now,)
            )
            row = await cursor.fetchone()
            newest = row[0] if row and row[0] else None

            return {
                "total_entries": total,
                "valid_entries": valid,
                "expired_entries": total - valid,
                "oldest_entry": oldest,
                "newest_entry": newest,
            }

    # ============================================
    # Token Usage Operations
    # ============================================

    async def record_token_usage(self, usage: TokenUsage) -> TokenUsage:
        """Record a token usage entry."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO token_usage
                    (call_type, timestamp, cached, input_tokens, output_tokens,
                     duration_ms, success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    usage.call_type,
                    usage.timestamp.isoformat(),
                    1 if usage.cached else 0,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.duration_ms,
                    1 if usage.success else 0,
                    usage.error,
                ),
            )
            await self._connection.commit()
            usage.id = cursor.lastrowid
            return usage

    async def get_token_usage_stats(
        self, since: Optional[datetime] = None
    ) -> dict:
        """
        Get token usage statistics.

        Args:
            since: Only count usage since this time. If None, count all.

        Returns:
            Dict with aggregated statistics.
        """
        async with self._connection.cursor() as cursor:
            where_clause = ""
            params: list = []
            if since:
                where_clause = "WHERE timestamp >= ?"
                params.append(since.isoformat())

            # Total calls
            await cursor.execute(
                f"SELECT COUNT(*) FROM token_usage {where_clause}", params
            )
            row = await cursor.fetchone()
            total_calls = row[0] if row else 0

            # Actual AI calls (not cached)
            await cursor.execute(
                f"SELECT COUNT(*) FROM token_usage {where_clause} {'AND' if where_clause else 'WHERE'} cached = 0",
                params,
            )
            row = await cursor.fetchone()
            actual_ai_calls = row[0] if row else 0

            # Cache hits
            cache_hits = total_calls - actual_ai_calls

            # Token counts (only for non-cached calls)
            await cursor.execute(
                f"""
                SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),
                       COALESCE(SUM(duration_ms), 0)
                FROM token_usage
                {where_clause} {'AND' if where_clause else 'WHERE'} cached = 0
                """,
                params,
            )
            row = await cursor.fetchone()
            input_tokens = row[0] if row else 0
            output_tokens = row[1] if row else 0
            total_duration_ms = row[2] if row else 0

            # Errors
            await cursor.execute(
                f"SELECT COUNT(*) FROM token_usage {where_clause} {'AND' if where_clause else 'WHERE'} success = 0",
                params,
            )
            row = await cursor.fetchone()
            errors = row[0] if row else 0

            # By call type
            await cursor.execute(
                f"""
                SELECT call_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN cached = 1 THEN 1 ELSE 0 END) as cached,
                       SUM(CASE WHEN cached = 0 THEN input_tokens + output_tokens ELSE 0 END) as tokens
                FROM token_usage
                {where_clause}
                GROUP BY call_type
                """,
                params,
            )
            rows = await cursor.fetchall()
            calls_by_type = {
                row["call_type"]: {
                    "total": row["total"],
                    "cached": row["cached"],
                    "tokens": row["tokens"],
                }
                for row in rows
            }

            return {
                "total_calls": total_calls,
                "actual_ai_calls": actual_ai_calls,
                "cache_hits": cache_hits,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_duration_ms": total_duration_ms,
                "errors": errors,
                "calls_by_type": calls_by_type,
            }

    async def clear_token_usage(self, before: Optional[datetime] = None) -> int:
        """
        Clear token usage records.

        Args:
            before: If provided, only clear records before this time.
                    If None, clear all records.

        Returns:
            Number of records deleted.
        """
        async with self._connection.cursor() as cursor:
            if before:
                await cursor.execute(
                    "SELECT COUNT(*) FROM token_usage WHERE timestamp < ?",
                    (before.isoformat(),),
                )
                row = await cursor.fetchone()
                count = row[0] if row else 0
                await cursor.execute(
                    "DELETE FROM token_usage WHERE timestamp < ?",
                    (before.isoformat(),),
                )
            else:
                await cursor.execute("SELECT COUNT(*) FROM token_usage")
                row = await cursor.fetchone()
                count = row[0] if row else 0
                await cursor.execute("DELETE FROM token_usage")
            await self._connection.commit()
            return count

    def _row_to_token_usage(self, row: aiosqlite.Row) -> TokenUsage:
        """Convert a database row to a TokenUsage object."""
        return TokenUsage(
            id=row["id"],
            call_type=row["call_type"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            cached=bool(row["cached"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            duration_ms=row["duration_ms"],
            success=bool(row["success"]),
            error=row["error"],
        )

    # Embedding operations

    async def save_content_embedding(
        self, content_id: int, embedding: bytes, model: str, dimension: int
    ) -> None:
        """Save or update embedding for a content item."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT OR REPLACE INTO content_embeddings
                    (content_id, embedding, model, dimension, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (content_id, embedding, model, dimension, datetime.now().isoformat()),
            )
            await self._connection.commit()

    async def get_content_embedding(self, content_id: int) -> Optional[tuple[bytes, str, int]]:
        """Get embedding for a content item.

        Returns:
            Tuple of (embedding_bytes, model, dimension) or None if not found.
        """
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                "SELECT embedding, model, dimension FROM content_embeddings WHERE content_id = ?",
                (content_id,),
            )
            row = await cursor.fetchone()
            if row:
                return (row["embedding"], row["model"], row["dimension"])
            return None

    async def get_content_embeddings_batch(
        self, content_ids: list[int]
    ) -> dict[int, tuple[bytes, str, int]]:
        """Get embeddings for multiple content items."""
        if not content_ids:
            return {}

        async with self._connection.cursor() as cursor:
            placeholders = ",".join("?" * len(content_ids))
            await cursor.execute(
                f"""
                SELECT content_id, embedding, model, dimension
                FROM content_embeddings
                WHERE content_id IN ({placeholders})
                """,
                content_ids,
            )
            rows = await cursor.fetchall()
            return {
                row["content_id"]: (row["embedding"], row["model"], row["dimension"])
                for row in rows
            }

    async def get_content_ids_without_embeddings(self, limit: int = 100) -> list[int]:
        """Get content item IDs that don't have embeddings yet.

        Note: No longer requires processed_at since embeddings are computed before AI processing.
        """
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT c.id FROM content_items c
                LEFT JOIN content_embeddings ce ON c.id = ce.content_id
                WHERE ce.content_id IS NULL
                ORDER BY c.published_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
            return [row["id"] for row in rows]

    async def get_processed_items_with_embeddings(
        self, since: datetime, limit: int = 200
    ) -> list[tuple["ContentItem", bytes, int]]:
        """Get processed content items that have embeddings.

        Used for finding similar items to reuse AI results.

        Args:
            since: Only return items published after this date.
            limit: Maximum number of items to return.

        Returns:
            List of tuples: (ContentItem, embedding_bytes, dimension)
        """
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT c.*, ce.embedding, ce.dimension
                FROM content_items c
                JOIN content_embeddings ce ON c.id = ce.content_id
                WHERE c.processed_at IS NOT NULL
                    AND c.summary IS NOT NULL
                    AND c.published_at >= ?
                ORDER BY c.published_at DESC
                LIMIT ?
                """,
                (since.isoformat(), limit),
            )
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                item = ContentItem(
                    id=row["id"],
                    subscription_id=row["subscription_id"],
                    external_id=row["external_id"],
                    title=row["title"],
                    content=row["content"],
                    url=row["url"],
                    author=row["author"],
                    published_at=datetime.fromisoformat(row["published_at"])
                    if row["published_at"]
                    else None,
                    created_at=datetime.fromisoformat(row["created_at"])
                    if row["created_at"]
                    else None,
                    processed_at=datetime.fromisoformat(row["processed_at"])
                    if row["processed_at"]
                    else None,
                    summary=row["summary"],
                    category=row["category"],
                    importance_score=row["importance_score"],
                    one_liner=row["one_liner"],
                    key_points=row["key_points"],
                    impact_assessment=row["impact_assessment"],
                    actionable_items=row["actionable_items"],
                )
                results.append((item, row["embedding"], row["dimension"]))
            return results

    async def save_cluster_centroid(
        self, cluster_id: int, centroid: bytes, member_count: int
    ) -> None:
        """Save or update centroid embedding for a cluster."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT OR REPLACE INTO cluster_embeddings
                    (cluster_id, centroid_embedding, member_count, last_updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (cluster_id, centroid, member_count, datetime.now().isoformat()),
            )
            await self._connection.commit()

    async def get_cluster_centroid(self, cluster_id: int) -> Optional[tuple[bytes, int]]:
        """Get centroid embedding for a cluster.

        Returns:
            Tuple of (centroid_bytes, member_count) or None if not found.
        """
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                "SELECT centroid_embedding, member_count FROM cluster_embeddings WHERE cluster_id = ?",
                (cluster_id,),
            )
            row = await cursor.fetchone()
            if row:
                return (row["centroid_embedding"], row["member_count"])
            return None

    async def get_all_cluster_centroids(self) -> dict[int, tuple[bytes, int]]:
        """Get all cluster centroids."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                "SELECT cluster_id, centroid_embedding, member_count FROM cluster_embeddings"
            )
            rows = await cursor.fetchall()
            return {
                row["cluster_id"]: (row["centroid_embedding"], row["member_count"])
                for row in rows
            }

    async def get_embedding_stats(self) -> dict:
        """Get embedding statistics."""
        async with self._connection.cursor() as cursor:
            # Content embeddings
            await cursor.execute("SELECT COUNT(*) FROM content_embeddings")
            row = await cursor.fetchone()
            content_count = row[0] if row else 0

            # Cluster centroids
            await cursor.execute("SELECT COUNT(*) FROM cluster_embeddings")
            row = await cursor.fetchone()
            cluster_count = row[0] if row else 0

            # Content items total
            await cursor.execute(
                "SELECT COUNT(*) FROM content_items WHERE processed_at IS NOT NULL"
            )
            row = await cursor.fetchone()
            processed_items = row[0] if row else 0

            return {
                "content_embeddings": content_count,
                "cluster_centroids": cluster_count,
                "processed_items": processed_items,
                "coverage_percent": round(content_count / processed_items * 100, 1)
                if processed_items > 0 else 0,
            }


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

    def get_undelivered_items(self, min_importance: int = 1, limit: int = 50) -> list[ContentItem]:
        return self._run(self._async_db.get_undelivered_items(min_importance, limit))

    def get_unclustered_items(
        self, min_importance: int = 5, limit: int = 100, retry_after_hours: Optional[int] = None
    ) -> list[ContentItem]:
        return self._run(
            self._async_db.get_unclustered_items(min_importance, limit, retry_after_hours)
        )

    def update_content_item(self, item: ContentItem) -> None:
        self._run(self._async_db.update_content_item(item))

    def mark_cluster_attempted(self, item_id: int) -> None:
        self._run(self._async_db.mark_cluster_attempted(item_id))

    def mark_items_delivered(self, item_ids: list[int]) -> None:
        self._run(self._async_db.mark_items_delivered(item_ids))

    def content_exists(self, source_type: SourceType, external_id: str) -> bool:
        return self._run(self._async_db.content_exists(source_type, external_id))

    def url_exists(self, url: str) -> bool:
        return self._run(self._async_db.url_exists(url))

    def title_exists(self, title: str) -> bool:
        return self._run(self._async_db.title_exists(title))

    def find_similar_title(self, title: str, days: int = 7) -> Optional[int]:
        return self._run(self._async_db.find_similar_title(title, days))

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
        return self._run(
            self._async_db.search_content(
                query, category, min_importance, state, from_date, to_date, limit, offset
            )
        )

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

    def add_event_member(self, member: EventMember) -> bool:
        return self._run(self._async_db.add_event_member(member))

    def is_item_in_cluster(self, item_id: int) -> bool:
        return self._run(self._async_db.is_item_in_cluster(item_id))

    def cleanup_duplicate_cluster_memberships(self) -> int:
        return self._run(self._async_db.cleanup_duplicate_cluster_memberships())

    def get_event_members(self, cluster_id: int) -> list[ContentItem]:
        return self._run(self._async_db.get_event_members(cluster_id))

    def get_undelivered_clustered_items(
        self, min_importance: int = 5, limit: int = 50
    ) -> dict[int, list[ContentItem]]:
        return self._run(self._async_db.get_undelivered_clustered_items(min_importance, limit))

    def get_undelivered_unclustered_items(
        self, min_importance: int = 5, limit: int = 50
    ) -> list[ContentItem]:
        return self._run(self._async_db.get_undelivered_unclustered_items(min_importance, limit))

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

    # Topic Suggestion sync wrappers

    def create_topic_suggestion(self, suggestion: TopicSuggestion) -> TopicSuggestion:
        return self._run(self._async_db.create_topic_suggestion(suggestion))

    def get_topic_suggestion(self, suggestion_id: int) -> Optional[TopicSuggestion]:
        return self._run(self._async_db.get_topic_suggestion(suggestion_id))

    def get_topic_suggestions(
        self, status: Optional[str] = None, limit: int = 50
    ) -> list[TopicSuggestion]:
        return self._run(self._async_db.get_topic_suggestions(status, limit))

    def update_topic_suggestion_status(
        self, suggestion_id: int, status: str, merged_with_topic_id: Optional[int] = None
    ) -> bool:
        return self._run(
            self._async_db.update_topic_suggestion_status(suggestion_id, status, merged_with_topic_id)
        )

    def update_topic(self, topic: Topic) -> None:
        self._run(self._async_db.update_topic(topic))

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

    # AI Cache sync wrappers

    def get_ai_cache(self, content_hash: str) -> Optional[str]:
        return self._run(self._async_db.get_ai_cache(content_hash))

    def set_ai_cache(self, content_hash: str, result_json: str, ttl_seconds: int = 86400) -> None:
        self._run(self._async_db.set_ai_cache(content_hash, result_json, ttl_seconds))

    def clear_ai_cache(self) -> int:
        return self._run(self._async_db.clear_ai_cache())

    def cleanup_expired_cache(self) -> int:
        return self._run(self._async_db.cleanup_expired_cache())

    def get_ai_cache_stats(self) -> dict:
        return self._run(self._async_db.get_ai_cache_stats())

    # Token Usage sync wrappers

    def record_token_usage(self, usage: TokenUsage) -> TokenUsage:
        return self._run(self._async_db.record_token_usage(usage))

    def get_token_usage_stats(self, since: Optional[datetime] = None) -> dict:
        return self._run(self._async_db.get_token_usage_stats(since))

    def clear_token_usage(self, before: Optional[datetime] = None) -> int:
        return self._run(self._async_db.clear_token_usage(before))

    # Embedding sync wrappers

    def save_content_embedding(
        self, content_id: int, embedding: bytes, model: str, dimension: int
    ) -> None:
        self._run(self._async_db.save_content_embedding(content_id, embedding, model, dimension))

    def get_content_embedding(self, content_id: int) -> Optional[tuple[bytes, str, int]]:
        return self._run(self._async_db.get_content_embedding(content_id))

    def get_content_embeddings_batch(
        self, content_ids: list[int]
    ) -> dict[int, tuple[bytes, str, int]]:
        return self._run(self._async_db.get_content_embeddings_batch(content_ids))

    def get_content_ids_without_embeddings(self, limit: int = 100) -> list[int]:
        return self._run(self._async_db.get_content_ids_without_embeddings(limit))

    def get_processed_items_with_embeddings(
        self, since: datetime, limit: int = 200
    ) -> list[tuple["ContentItem", bytes, int]]:
        return self._run(self._async_db.get_processed_items_with_embeddings(since, limit))

    def save_cluster_centroid(self, cluster_id: int, centroid: bytes, member_count: int) -> None:
        self._run(self._async_db.save_cluster_centroid(cluster_id, centroid, member_count))

    def get_cluster_centroid(self, cluster_id: int) -> Optional[tuple[bytes, int]]:
        return self._run(self._async_db.get_cluster_centroid(cluster_id))

    def get_all_cluster_centroids(self) -> dict[int, tuple[bytes, int]]:
        return self._run(self._async_db.get_all_cluster_centroids())

    def get_embedding_stats(self) -> dict:
        return self._run(self._async_db.get_embedding_stats())

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

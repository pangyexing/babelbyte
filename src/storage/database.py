"""SQLite database operations for BabelByte."""

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from config.settings import get_settings
from src.storage.models import (
    ContentItem,
    SourceType,
    Subscription,
    SubscriptionType,
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
    delivered INTEGER DEFAULT 0,
    delivered_at TEXT,
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id),
    UNIQUE(source_type, external_id)
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
            await cursor.execute(CREATE_SUBSCRIPTIONS_TABLE)
            await cursor.execute(CREATE_CONTENT_ITEMS_TABLE)
            await cursor.execute(CREATE_USER_PROFILES_TABLE)
            await self._connection.commit()
        await self._migrate_tables()

    async def _migrate_tables(self) -> None:
        """Run database migrations for schema updates."""
        async with self._connection.cursor() as cursor:
            # Check if twitter_user_id column exists in subscriptions
            await cursor.execute("PRAGMA table_info(subscriptions)")
            columns = {row[1] for row in await cursor.fetchall()}

            if "twitter_user_id" not in columns:
                await cursor.execute(
                    "ALTER TABLE subscriptions ADD COLUMN twitter_user_id TEXT"
                )
            if "last_tweet_id" not in columns:
                await cursor.execute(
                    "ALTER TABLE subscriptions ADD COLUMN last_tweet_id TEXT"
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
                SET enabled = ?, last_fetched_at = ?, twitter_user_id = ?, last_tweet_id = ?
                WHERE id = ?
                """,
                (
                    1 if subscription.enabled else 0,
                    subscription.last_fetched_at.isoformat() if subscription.last_fetched_at else None,
                    subscription.twitter_user_id,
                    subscription.last_tweet_id,
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
                    delivered = ?, delivered_at = ?
                WHERE id = ?
                """,
                (
                    item.summary,
                    item.category,
                    item.importance_score,
                    item.processed_at.isoformat() if item.processed_at else None,
                    1 if item.delivered else 0,
                    item.delivered_at.isoformat() if item.delivered_at else None,
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

    def _row_to_content_item(self, row: aiosqlite.Row) -> ContentItem:
        """Convert a database row to a ContentItem object."""
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

    def get_or_create_profile(self, email: str) -> UserProfile:
        return self._run(self._async_db.get_or_create_profile(email))

    def update_profile(self, profile: UserProfile) -> None:
        self._run(self._async_db.update_profile(profile))

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

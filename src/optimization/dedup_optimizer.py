"""Deduplication optimization utilities for BabelByte."""

import hashlib
import logging
import re
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=10000)
def compute_title_hash(title: str) -> str:
    """
    Compute a normalized title hash for fast duplicate detection.

    Normalization:
    - Lowercase
    - Remove RT prefix
    - Remove @mentions
    - Remove URLs
    - Remove extra whitespace
    - Remove common punctuation

    Args:
        title: The title to hash.

    Returns:
        16-character hex hash of normalized title.
    """
    if not title:
        return ""

    normalized = title.strip().lower()

    # Remove RT prefix (retweets)
    normalized = re.sub(r"^rt\s+", "", normalized)

    # Remove @mentions at start
    normalized = re.sub(r"^@\w+[:\s]*", "", normalized)

    # Remove URLs
    normalized = re.sub(r"https?://\S+", "", normalized)

    # Remove hashtags
    normalized = re.sub(r"#\w+", "", normalized)

    # Remove common punctuation and normalize whitespace
    normalized = re.sub(r"[.,!?;:\"'()\[\]{}]", " ", normalized)
    normalized = " ".join(normalized.split())

    if len(normalized) < 5:
        return ""

    # Create hash
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def compute_content_hash(title: str, content: str) -> str:
    """
    Compute a hash for content deduplication.

    Uses title + first 500 chars of content for comparison.

    Args:
        title: The content title.
        content: The content body.

    Returns:
        16-character hex hash.
    """
    combined = f"{title or ''}\n{(content or '')[:500]}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


class DedupOptimizer:
    """
    Optimizes duplicate detection using pre-computed hashes.

    Maintains an in-memory index of title hashes for fast lookups.
    """

    def __init__(self):
        self._title_hashes: dict[str, int] = {}  # hash -> content_id
        self._content_hashes: dict[str, int] = {}  # hash -> content_id

    def build_index(self, items: list) -> int:
        """
        Build hash index from existing items.

        Args:
            items: List of ContentItem objects.

        Returns:
            Number of items indexed.
        """
        count = 0
        for item in items:
            if item.id and item.title:
                title_hash = compute_title_hash(item.title)
                if title_hash:
                    self._title_hashes[title_hash] = item.id
                    count += 1

                content_hash = compute_content_hash(item.title, item.content)
                if content_hash:
                    self._content_hashes[content_hash] = item.id

        logger.debug(f"Built dedup index with {count} title hashes")
        return count

    def find_duplicate_by_title(self, title: str) -> Optional[int]:
        """
        Find existing item with same normalized title.

        Args:
            title: Title to search for.

        Returns:
            Content item ID if duplicate found, None otherwise.
        """
        title_hash = compute_title_hash(title)
        if not title_hash:
            return None
        return self._title_hashes.get(title_hash)

    def find_duplicate_by_content(self, title: str, content: str) -> Optional[int]:
        """
        Find existing item with same content.

        Args:
            title: Title of content.
            content: Content body.

        Returns:
            Content item ID if duplicate found, None otherwise.
        """
        content_hash = compute_content_hash(title, content)
        return self._content_hashes.get(content_hash)

    def add_item(self, item_id: int, title: str, content: Optional[str] = None) -> None:
        """
        Add an item to the dedup index.

        Args:
            item_id: The content item ID.
            title: The item title.
            content: Optional content body.
        """
        title_hash = compute_title_hash(title)
        if title_hash:
            self._title_hashes[title_hash] = item_id

        if content:
            content_hash = compute_content_hash(title, content)
            self._content_hashes[content_hash] = item_id

    def remove_item(self, title: str, content: Optional[str] = None) -> None:
        """
        Remove an item from the dedup index.

        Args:
            title: The item title.
            content: Optional content body.
        """
        title_hash = compute_title_hash(title)
        if title_hash and title_hash in self._title_hashes:
            del self._title_hashes[title_hash]

        if content:
            content_hash = compute_content_hash(title, content)
            if content_hash in self._content_hashes:
                del self._content_hashes[content_hash]

    def clear(self) -> None:
        """Clear the dedup index."""
        self._title_hashes.clear()
        self._content_hashes.clear()
        compute_title_hash.cache_clear()

    @property
    def size(self) -> int:
        """Return the number of indexed items."""
        return len(self._title_hashes)

    def get_stats(self) -> dict:
        """
        Get dedup index statistics.

        Returns:
            Dict with index statistics.
        """
        cache_info = compute_title_hash.cache_info()
        return {
            "title_hashes": len(self._title_hashes),
            "content_hashes": len(self._content_hashes),
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "cache_size": cache_info.currsize,
            "cache_maxsize": cache_info.maxsize,
        }

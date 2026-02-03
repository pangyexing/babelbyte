"""Deduplication optimization utilities for BabelByte."""

import hashlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.storage.models import ContentItem

logger = logging.getLogger(__name__)


# Default threshold for considering two items as semantic duplicates
# Can be overridden via EMBEDDING_DEDUP_THRESHOLD env var
# Lowered from 0.92 to 0.85 for better token savings (81.8% accuracy validated)
DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD = 0.85

# Default search limit for deduplication
# Can be overridden via EMBEDDING_DEDUP_SEARCH_LIMIT env var
DEFAULT_DEDUP_SEARCH_LIMIT = 5000


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


@dataclass
class SimilarItemResult:
    """Result of finding a similar processed item."""

    source_id: int
    similarity: float
    summary: str
    category: str
    importance_score: int
    one_liner: Optional[str] = None
    key_points: Optional[str] = None
    impact_assessment: Optional[str] = None
    actionable_items: Optional[str] = None


def find_similar_processed_item(
    item: "ContentItem",
    db,
    threshold: Optional[float] = None,
) -> Optional[SimilarItemResult]:
    """
    Find a similar already-processed item using embedding similarity.

    This allows reusing AI results for semantically similar content,
    saving LLM tokens.

    Configuration (env vars):
    - EMBEDDING_DEDUP_THRESHOLD: Similarity threshold (default: 0.85)
    - EMBEDDING_DEDUP_SEARCH_LIMIT: Max items to search (default: 5000)

    Args:
        item: The unprocessed item to find similar content for.
        db: Database instance.
        threshold: Minimum similarity threshold (0-1). If None, uses config value.

    Returns:
        SimilarItemResult if a similar processed item is found, None otherwise.
    """
    try:
        from src.processors.embeddings import EmbeddingManager, bytes_to_embedding
        from config.settings import get_settings

        settings = get_settings()
        if not settings.embedding.enabled:
            return None

        # Use configurable threshold and search limit
        if threshold is None:
            threshold = settings.embedding.dedup_threshold
        search_limit = settings.embedding.dedup_search_limit

        # Get embedding for the current item
        item_embedding_data = db.get_content_embedding(item.id)
        if not item_embedding_data:
            return None

        item_embedding_bytes, _, dimension = item_embedding_data
        item_embedding = bytes_to_embedding(item_embedding_bytes, dimension)

        # Get recent processed items with embeddings
        # Limit search to last 7 days for efficiency
        from datetime import datetime, timedelta

        recent_cutoff = datetime.now() - timedelta(days=7)
        processed_items = db.get_processed_items_with_embeddings(
            since=recent_cutoff, limit=search_limit
        )

        if not processed_items:
            return None

        manager = EmbeddingManager.get_instance()
        best_match = None
        best_similarity = 0.0

        for processed_item, emb_bytes, emb_dimension in processed_items:
            # Skip self
            if processed_item.id == item.id:
                continue

            # Skip items without summary (not properly processed)
            if not processed_item.summary:
                continue

            processed_embedding = bytes_to_embedding(emb_bytes, emb_dimension)
            similarity = manager.cosine_similarity(item_embedding, processed_embedding)

            # Convert from [-1, 1] to [0, 1] range
            similarity_normalized = (similarity + 1) / 2

            if similarity_normalized > best_similarity and similarity_normalized >= threshold:
                best_similarity = similarity_normalized
                best_match = processed_item

        if best_match:
            logger.info(
                f"Found similar processed item (similarity={best_similarity:.3f}): "
                f"'{item.title[:40]}...' -> '{best_match.title[:40]}...'"
            )
            return SimilarItemResult(
                source_id=best_match.id,
                similarity=best_similarity,
                summary=best_match.summary,
                category=best_match.category,
                importance_score=best_match.importance_score,
                one_liner=best_match.one_liner,
                key_points=best_match.key_points,
                impact_assessment=best_match.impact_assessment,
                actionable_items=best_match.actionable_items,
            )

        return None

    except ImportError as e:
        logger.debug(f"Embedding dedup not available: {e}")
        return None
    except (ValueError, TypeError, OSError) as e:
        logger.warning(f"Error finding similar item: {e}")
        return None

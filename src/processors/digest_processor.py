"""Digest processor for AI content processing and digest generation."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.processors.claude_cli import ClaudeCLI, MockClaudeCLI, ProcessingResult
from src.storage.database import Database, SyncDatabase
from src.storage.models import ContentItem, DigestItem

logger = logging.getLogger(__name__)


@dataclass
class DigestResult:
    """Result of digest generation."""

    items: list[DigestItem] = field(default_factory=list)
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def by_category(self) -> dict[str, list[DigestItem]]:
        """Group items by category."""
        result: dict[str, list[DigestItem]] = {}
        for item in self.items:
            if item.category not in result:
                result[item.category] = []
            result[item.category].append(item)
        return result


class DigestProcessor:
    """Processor for generating content digests."""

    def __init__(
        self,
        claude_cli: Optional[ClaudeCLI] = None,
        use_mock: bool = False,
    ):
        if use_mock:
            self.claude = MockClaudeCLI()
        else:
            self.claude = claude_cli or ClaudeCLI()

    def process_item(self, item: ContentItem) -> ProcessingResult:
        """
        Process a single content item with AI.

        Args:
            item: The content item to process.

        Returns:
            ProcessingResult with summary, category, and importance score.
        """
        return self.claude.process_content(item.title, item.content)

    def process_items(self, items: list[ContentItem]) -> list[tuple[ContentItem, ProcessingResult]]:
        """
        Process multiple content items.

        Args:
            items: List of content items to process.

        Returns:
            List of tuples (item, result).
        """
        results = []
        for item in items:
            result = self.process_item(item)
            results.append((item, result))
            logger.info(
                f"Processed: {item.title[:50]}... -> {result.category} "
                f"(importance: {result.importance_score})"
            )
        return results


class DigestGenerator:
    """Generator for creating digests from processed content."""

    def __init__(
        self,
        db: Optional[SyncDatabase] = None,
        processor: Optional[DigestProcessor] = None,
        use_mock: bool = False,
    ):
        self.db = db
        self.processor = processor or DigestProcessor(use_mock=use_mock)

    def process_unprocessed_items(self, limit: int = 50) -> int:
        """
        Process all unprocessed items in the database.

        Args:
            limit: Maximum number of items to process.

        Returns:
            Number of items processed.
        """
        if not self.db:
            raise ValueError("Database not configured")

        items = self.db.get_unprocessed_items(limit=limit)
        if not items:
            logger.info("No unprocessed items found")
            return 0

        processed_count = 0
        for item in items:
            result = self.processor.process_item(item)

            if result.success:
                item.summary = result.summary
                item.category = result.category
                item.importance_score = result.importance_score
                item.processed_at = datetime.now()
                self.db.update_content_item(item)
                processed_count += 1
                logger.info(f"Processed: {item.title[:50]}... -> {result.category}")
            else:
                logger.warning(f"Failed to process: {item.title[:50]}... - {result.error_message}")

        return processed_count

    def generate_digest(
        self,
        min_importance: int = 5,
        max_items: int = 30,
        include_delivered: bool = False,
    ) -> DigestResult:
        """
        Generate a digest from processed items.

        Args:
            min_importance: Minimum importance score to include.
            max_items: Maximum number of items in the digest.
            include_delivered: Whether to include already delivered items.

        Returns:
            DigestResult with the digest items.
        """
        if not self.db:
            raise ValueError("Database not configured")

        # Get undelivered items
        items = self.db.get_undelivered_items(
            min_importance=min_importance,
            limit=max_items,
        )

        digest_items = []
        for item in items:
            if item.summary and item.category and item.importance_score:
                digest_items.append(
                    DigestItem(
                        content_item=item,
                        summary=item.summary,
                        category=item.category,
                        importance_score=item.importance_score,
                    )
                )

        # Sort by importance (descending)
        digest_items.sort(key=lambda x: x.importance_score, reverse=True)

        return DigestResult(
            items=digest_items[:max_items],
            total_processed=len(items),
            successful=len(digest_items),
            failed=len(items) - len(digest_items),
        )

    def mark_digest_delivered(self, digest: DigestResult) -> None:
        """Mark all items in a digest as delivered."""
        if not self.db:
            raise ValueError("Database not configured")

        item_ids = [item.content_item.id for item in digest.items if item.content_item.id]
        if item_ids:
            self.db.mark_items_delivered(item_ids)
            logger.info(f"Marked {len(item_ids)} items as delivered")


def create_digest_preview(digest: DigestResult) -> str:
    """
    Create a text preview of the digest.

    Args:
        digest: The digest to preview.

    Returns:
        Formatted text preview.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("📧 BabelByte 每日摘要预览")
    lines.append(f"生成时间: {digest.generated_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"共 {len(digest.items)} 条内容")
    lines.append("=" * 60)

    if not digest.items:
        lines.append("\n暂无新内容")
        return "\n".join(lines)

    # Group by category
    for category, items in sorted(digest.by_category.items()):
        lines.append(f"\n📁 {category} ({len(items)}条)")
        lines.append("-" * 40)

        for item in items:
            importance_stars = "⭐" * min(item.importance_score // 2, 5)
            lines.append(f"\n{importance_stars} [{item.importance_score}/10]")
            lines.append(f"📌 {item.content_item.title[:60]}")
            lines.append(f"📝 {item.summary}")
            lines.append(f"🔗 {item.content_item.url}")
            lines.append(f"   来源: {item.source_display} | 作者: {item.content_item.author}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)

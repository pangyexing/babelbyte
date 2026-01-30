"""Digest processor for AI content processing and digest generation."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from config.settings import get_settings
from src.processors.base import BaseAIProcessor, MockAIProcessor, ProcessingResult
from src.processors.rule_classifier import (
    RuleClassifier,
    create_skip_result,
    should_skip_ai_processing,
)
from src.storage.database import SyncDatabase
from src.storage.models import ContentItem, DigestItem

logger = logging.getLogger(__name__)


def get_ai_processor(provider: Optional[str] = None, use_mock: bool = False) -> BaseAIProcessor:
    """
    Get the appropriate AI processor based on configuration.

    Args:
        provider: Override provider ("claude", "codex", or "auto")
        use_mock: Use mock processor for testing

    Returns:
        An AI processor instance
    """
    if use_mock:
        return MockAIProcessor()

    settings = get_settings()
    provider = provider or settings.ai.get_provider()

    if provider == "codex":
        from src.processors.openai_cli import CodexCLI

        return CodexCLI()

    if provider == "claude" or provider == "auto":
        from src.processors.claude_cli import ClaudeCLI

        return ClaudeCLI()

    # Default fallback
    from src.processors.claude_cli import ClaudeCLI

    return ClaudeCLI()


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
        ai_processor: Optional[BaseAIProcessor] = None,
        provider: Optional[str] = None,
        use_mock: bool = False,
    ):
        self.ai = ai_processor or get_ai_processor(provider=provider, use_mock=use_mock)
        self.rule_classifier = RuleClassifier()

    def process_item(self, item: ContentItem) -> ProcessingResult:
        """
        Process a single content item with AI.

        Args:
            item: The content item to process.

        Returns:
            ProcessingResult with summary, category, and importance score.
        """
        return self.ai.process_content(item.title, item.content)

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
        provider: Optional[str] = None,
        use_mock: bool = False,
    ):
        self.db = db
        self.processor = processor or DigestProcessor(provider=provider, use_mock=use_mock)

    def process_unprocessed_items(
        self,
        limit: int = 50,
        batch_size: int = 3,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> int:
        """
        Process all unprocessed items in the database using batch processing.

        Optimizations applied:
        1. Pre-filter low-value content (skip AI for link-posts, very short content)
        2. Rule-based classification for known domains/patterns
        3. Dynamic batch sizing (larger batches for short content)

        Args:
            limit: Maximum number of items to process.
            batch_size: Base batch size for normal content (default 3).
            progress_callback: Optional callback for progress updates.
                Signature: (phase: str, current: int, total: int) -> None

        Returns:
            Number of items processed.
        """
        import time

        if not self.db:
            raise ValueError("Database not configured")

        items = self.db.get_unprocessed_items(limit=limit)
        if not items:
            logger.info("No unprocessed items found")
            return 0

        total_start = time.time()
        processed_count = 0
        skipped_count = 0
        rule_classified_count = 0
        ai_processed_count = 0

        # Items that need AI processing
        items_for_ai: list[ContentItem] = []

        # Phase 1: Pre-filter and rule-based classification
        logger.info(f"Phase 1: Pre-filtering {len(items)} items...")
        phase1_total = len(items)
        for idx, item in enumerate(items):
            # Update progress for phase 1
            if progress_callback:
                progress_callback("Pre-filtering", idx + 1, phase1_total)

            # Check if content should be skipped entirely
            should_skip, skip_reason = should_skip_ai_processing(item)
            if should_skip:
                summary, category, importance = create_skip_result(item, skip_reason)
                item.summary = summary
                item.category = category
                item.importance_score = importance
                item.processed_at = datetime.now()
                self.db.update_content_item(item)
                skipped_count += 1
                processed_count += 1
                logger.debug(f"Skipped: {item.title[:40]}... ({skip_reason})")
                continue

            # Try rule-based classification
            rule_result = self.processor.rule_classifier.classify(item)
            if rule_result:
                # Generate a simple summary for rule-classified items
                item.summary = f"{item.title[:50]}" if len(item.title) > 50 else item.title
                item.category = rule_result.category
                item.importance_score = rule_result.importance_score
                item.processed_at = datetime.now()
                self.db.update_content_item(item)
                rule_classified_count += 1
                processed_count += 1
                logger.debug(
                    f"Rule classified: {item.title[:40]}... -> {rule_result.category} "
                    f"({rule_result.reason})"
                )
                continue

            # Needs AI processing
            items_for_ai.append(item)

        prefilter_count = skipped_count + rule_classified_count
        logger.info(
            f"Phase 1 done: {prefilter_count} items handled without AI "
            f"(skipped={skipped_count}, rules={rule_classified_count}), "
            f"{len(items_for_ai)} need AI processing"
        )

        # Phase 2: AI processing with dynamic batch sizing
        if items_for_ai:
            logger.info(f"Phase 2: AI processing {len(items_for_ai)} items...")
            ai_processed_count = self._process_items_with_ai(
                items_for_ai, batch_size, progress_callback
            )
            processed_count += ai_processed_count
        else:
            logger.info("Phase 2: No items need AI processing!")

        total_elapsed = time.time() - total_start
        tokens_saved_pct = (prefilter_count / len(items) * 100) if items else 0
        logger.info(
            f"Processing complete in {total_elapsed:.1f}s: {processed_count}/{len(items)} total "
            f"| Token savings: {tokens_saved_pct:.0f}% ({prefilter_count} items skipped AI)"
        )
        return processed_count

    def _process_items_with_ai(
        self,
        items: list[ContentItem],
        base_batch_size: int = 3,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> int:
        """
        Process items with AI using dynamic batch sizing.

        Short content (<500 chars) uses larger batches (configurable, default 12).
        Long content uses smaller batches (configurable, default 6).

        Args:
            items: Items to process with AI.
            base_batch_size: Legacy parameter, overridden by config.
            progress_callback: Optional callback for progress updates.

        Returns:
            Number of items successfully processed.
        """
        settings = get_settings()

        # Get batch sizes from config
        short_batch_size = settings.ai.batch_size_short  # Default 12
        long_batch_size = settings.ai.batch_size_long    # Default 6

        # Separate items by content length
        short_items = [i for i in items if len(i.content or "") < 500]
        long_items = [i for i in items if len(i.content or "") >= 500]

        processed_count = 0
        total_items = len(items)
        items_done = 0

        # Process short items with larger batches
        if short_items:
            short_processed, short_done = self._process_batch_group(
                short_items, short_batch_size, "short", progress_callback, items_done, total_items
            )
            processed_count += short_processed
            items_done += short_done

        # Process long items with configured batch size
        if long_items:
            long_processed, _ = self._process_batch_group(
                long_items, long_batch_size, "long", progress_callback, items_done, total_items
            )
            processed_count += long_processed

        return processed_count

    def _process_batch_group(
        self,
        items: list[ContentItem],
        batch_size: int,
        group_name: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        offset: int = 0,
        total: int = 0,
    ) -> tuple[int, int]:
        """Process a group of items in batches.

        Args:
            items: Items to process.
            batch_size: Number of items per batch.
            group_name: Name for logging (e.g., "short", "long").
            progress_callback: Optional callback for progress updates.
            offset: Starting offset for progress tracking.
            total: Total items for progress tracking.

        Returns:
            Tuple of (processed_count, items_processed_in_group).
        """
        import time

        processed_count = 0
        items_processed = 0
        total_batches = (len(items) + batch_size - 1) // batch_size
        group_start = time.time()
        actual_total = total if total > 0 else len(items)

        for i in range(0, len(items), batch_size):
            batch_items = items[i : i + batch_size]

            # Prepare batch data: (index, title, content)
            batch_data = [(idx, item.title, item.content) for idx, item in enumerate(batch_items)]

            # Process batch with timing
            batch_num = i // batch_size + 1
            batch_start = time.time()
            logger.info(
                f"[{batch_num}/{total_batches}] Processing {group_name} content "
                f"({len(batch_items)} items)..."
            )

            results = self.processor.ai.process_batch(batch_data)
            batch_elapsed = time.time() - batch_start

            # Update items with results
            batch_success = 0
            for item, result in zip(batch_items, results):
                if result.success:
                    item.summary = result.summary
                    item.category = result.category
                    item.importance_score = result.importance_score
                    item.processed_at = datetime.now()
                    self.db.update_content_item(item)
                    processed_count += 1
                    batch_success += 1
                else:
                    logger.warning(f"Failed: {item.title[:40]}...")

            items_processed += len(batch_items)

            # Update progress after each batch
            if progress_callback:
                progress_callback("AI processing", offset + items_processed, actual_total)

            logger.info(
                f"[{batch_num}/{total_batches}] Done: {batch_success}/{len(batch_items)} "
                f"in {batch_elapsed:.1f}s"
            )

        total_elapsed = time.time() - group_start
        if total_batches > 0:
            logger.info(
                f"Finished {group_name} content: {processed_count}/{len(items)} items "
                f"in {total_elapsed:.1f}s"
            )

        return processed_count, items_processed

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

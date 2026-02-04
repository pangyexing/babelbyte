"""Digest processor for AI content processing and digest generation."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Union

from config.settings import get_settings
from src.processors.base import (
    ActionResult,
    BaseAIProcessor,
    ImpactResult,
    KeyPointResult,
    MockAIProcessor,
    ProcessingResult,
    TaskType,
)
from src.processors.rule_classifier import (
    ImportanceEstimate,
    RuleClassifier,
    create_skip_result,
    estimate_importance,
    should_skip_ai_processing,
)
from src.optimization.dedup_optimizer import find_similar_processed_item
from src.storage.database import SyncDatabase
from src.storage.models import ActionItem, ActionStatus, ContentItem, DigestItem, EventDigestItem

logger = logging.getLogger(__name__)

DigestItemType = Union[DigestItem, EventDigestItem]


def get_ai_processor(
    provider: Optional[str] = None,
    use_mock: bool = False,
    db: Optional[SyncDatabase] = None,
) -> BaseAIProcessor:
    """
    Get the appropriate AI processor based on configuration.

    Args:
        provider: Override provider ("claude", "codex", "ollama", or "auto")
        use_mock: Use mock processor for testing
        db: Optional database instance for cache support

    Returns:
        An AI processor instance
    """
    if use_mock:
        return MockAIProcessor()

    settings = get_settings()
    provider = provider or settings.ai.get_provider()

    if provider == "ollama":
        from src.processors.ollama_api import OllamaAPI

        return OllamaAPI(db=db)

    if provider == "codex":
        from src.processors.openai_cli import CodexCLI

        return CodexCLI(db=db)

    if provider == "claude" or provider == "auto":
        from src.processors.claude_cli import ClaudeCLI

        return ClaudeCLI(db=db)

    # Default fallback
    from src.processors.claude_cli import ClaudeCLI

    return ClaudeCLI(db=db)


@dataclass
class DigestResult:
    """Result of digest generation."""

    items: list[DigestItem] = field(default_factory=list)  # Unclustered items
    events: list[EventDigestItem] = field(default_factory=list)  # Event clusters
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def by_category(self) -> dict[str, list[DigestItemType]]:
        """Group items and events by category, mixed together."""
        result: dict[str, list[DigestItemType]] = {}

        # Add events first (they are more important groupings)
        for event in self.events:
            if event.category not in result:
                result[event.category] = []
            result[event.category].append(event)

        # Add regular items
        for item in self.items:
            if item.category not in result:
                result[item.category] = []
            result[item.category].append(item)

        # Sort each category by importance score
        for category in result:
            result[category].sort(key=lambda x: x.importance_score, reverse=True)

        return result

    @property
    def events_by_category(self) -> dict[str, list[EventDigestItem]]:
        """Group events by category (for separate display)."""
        result: dict[str, list[EventDigestItem]] = {}
        for event in self.events:
            if event.category not in result:
                result[event.category] = []
            result[event.category].append(event)
        # Sort by importance
        for category in result:
            result[category].sort(key=lambda x: x.importance_score, reverse=True)
        return result

    @property
    def items_by_category(self) -> dict[str, list[DigestItem]]:
        """Group individual items by category (for separate display)."""
        result: dict[str, list[DigestItem]] = {}
        for item in self.items:
            if item.category not in result:
                result[item.category] = []
            result[item.category].append(item)
        # Sort by importance
        for category in result:
            result[category].sort(key=lambda x: x.importance_score, reverse=True)
        return result

    @property
    def total_items(self) -> int:
        """Total count including event members."""
        event_member_count = sum(len(e.members) for e in self.events)
        return len(self.items) + event_member_count

    @property
    def all_content_ids(self) -> list[int]:
        """Get all content item IDs for marking as delivered."""
        ids = [item.content_item.id for item in self.items if item.content_item.id]
        for event in self.events:
            ids.extend([m.id for m in event.members if m.id])
        return ids


class DigestProcessor:
    """Processor for generating content digests."""

    def __init__(
        self,
        ai_processor: Optional[BaseAIProcessor] = None,
        provider: Optional[str] = None,
        use_mock: bool = False,
        db: Optional[SyncDatabase] = None,
    ):
        self.ai = ai_processor or get_ai_processor(provider=provider, use_mock=use_mock, db=db)
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
        self.processor = processor or DigestProcessor(provider=provider, use_mock=use_mock, db=db)

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

        # Get items pre-sorted by content_ranker (based on historical AI scores)
        # content_ranker considers: source tier, author reputation, content length
        items = self.db.get_unprocessed_items(limit=limit)
        if not items:
            logger.info("No unprocessed items found")
            return 0

        # Log priority distribution for debugging
        from src.processors.content_ranker import calculate_priority_score, extract_source_key

        if len(items) >= 5:
            top_sources = []
            for item in items[:5]:
                src_type = item.source_type.value if item.source_type else None
                score = calculate_priority_score(item.url or "", item.content, item.author, src_type)
                src = extract_source_key(item.url or "")
                top_sources.append(f"{src}:{score}")
            logger.info(f"Processing {len(items)} items (top 5 by priority: {top_sources})")

        total_start = time.time()
        processed_count = 0
        skipped_count = 0
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

            # Try embedding-based dedup: reuse AI results from similar items
            similar_result = find_similar_processed_item(item, self.db)
            if similar_result:
                item.summary = similar_result.summary
                item.category = similar_result.category
                item.importance_score = similar_result.importance_score
                item.one_liner = similar_result.one_liner
                item.key_points = similar_result.key_points
                item.impact_assessment = similar_result.impact_assessment
                item.actionable_items = similar_result.actionable_items
                item.processed_at = datetime.now()
                self.db.update_content_item(item)
                skipped_count += 1
                processed_count += 1
                logger.debug(
                    f"Embedding dedup: {item.title[:40]}... "
                    f"(similar to #{similar_result.source_id}, score={similar_result.similarity:.3f})"
                )
                continue

            # Needs AI processing (removed basic rule classification to ensure complete enhanced fields)
            items_for_ai.append(item)

        logger.info(
            f"Phase 1 done: {skipped_count} items skipped, "
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
        tokens_saved_pct = (skipped_count / len(items) * 100) if items else 0
        logger.info(
            f"Processing complete in {total_elapsed:.1f}s: {processed_count}/{len(items)} total "
            f"| Token savings: {tokens_saved_pct:.0f}% ({skipped_count} items skipped AI)"
        )
        return processed_count

    def _process_items_with_ai(
        self,
        items: list[ContentItem],
        base_batch_size: int = 3,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> int:
        """
        Process items with AI using model tier selection based on importance estimation.

        Model selection logic:
        1. High importance (>= threshold) or low confidence -> Heavy model
        2. Low importance with high confidence -> Light model
        3. Quality guard: reprocess if light model returns unexpectedly high score

        Two-stage processing (Ollama only):
        When OLLAMA_MODEL_SCREEN is configured, uses 8B model for initial screening,
        then refines high-importance/uncertain items with 32B model.

        Args:
            items: Items to process with AI.
            base_batch_size: Legacy parameter, overridden by config.
            progress_callback: Optional callback for progress updates.

        Returns:
            Number of items successfully processed.
        """
        settings = get_settings()
        config = settings.ai.model_tiers
        guard = config.quality_guard

        # Check if two-stage processing is available (Ollama with screen model)
        if settings.ai.get_provider() == "ollama" and settings.ollama.two_stage_enabled:
            return self._process_items_two_stage(items, progress_callback)

        # Get batch sizes from config
        short_batch_size = settings.ai.batch_size_short
        long_batch_size = settings.ai.batch_size_long

        # Categorize items by model tier if enabled
        if config.enabled:
            heavy_items: list[tuple[ContentItem, ImportanceEstimate]] = []
            light_items: list[tuple[ContentItem, ImportanceEstimate]] = []

            for item in items:
                estimate = estimate_importance(item)

                # Decision logic for model selection
                use_heavy = False
                if estimate.score >= config.heavy_threshold:
                    use_heavy = True
                elif estimate.confidence < config.confidence_cutoff:
                    use_heavy = True  # Low confidence -> conservative (heavy)
                elif guard.unknown_domain_use_heavy and estimate.reason == "unknown":
                    use_heavy = True

                if use_heavy:
                    heavy_items.append((item, estimate))
                else:
                    light_items.append((item, estimate))

            logger.info(f"Model tier selection: {len(heavy_items)} heavy, {len(light_items)} light")

            processed_count = 0
            total_items = len(items)
            items_done = 0
            reprocess_items = []

            # Process light items FIRST (avoids model switching: light -> heavy)
            if light_items:
                light_content_items = [item for item, _ in light_items]
                light_processed, light_done, reprocess_items = self._process_batch_group_tiered(
                    light_content_items,
                    short_batch_size,
                    long_batch_size,
                    TaskType.CONTENT_LOW,
                    "light",
                    progress_callback,
                    items_done,
                    total_items,
                    check_reprocess=guard.reprocess_high_score_from_light,
                    reprocess_threshold=guard.reprocess_threshold,
                )
                processed_count += light_processed
                items_done += light_done

            # Combine heavy items with reprocess items (both use heavy model)
            heavy_content_items = [item for item, _ in heavy_items]
            if reprocess_items:
                logger.info(
                    f"Quality guard: {len(reprocess_items)} items need reprocessing, "
                    f"combining with {len(heavy_content_items)} heavy items"
                )
                heavy_content_items.extend(reprocess_items)

            # Process heavy items (including reprocess) AFTER light
            if heavy_content_items:
                heavy_processed, heavy_done, _ = self._process_batch_group_tiered(
                    heavy_content_items,
                    short_batch_size,
                    long_batch_size,
                    TaskType.CONTENT_HIGH,
                    "heavy",
                    progress_callback,
                    items_done,
                    total_items,
                )
                processed_count += heavy_processed
                items_done += heavy_done

            return processed_count

        # Fallback: original logic without model tiers
        short_items = [i for i in items if len(i.content or "") < 500]
        long_items = [i for i in items if len(i.content or "") >= 500]

        processed_count = 0
        total_items = len(items)
        items_done = 0

        if short_items:
            short_processed, short_done = self._process_batch_group(
                short_items, short_batch_size, "short", progress_callback, items_done, total_items
            )
            processed_count += short_processed
            items_done += short_done

        if long_items:
            long_processed, _ = self._process_batch_group(
                long_items, long_batch_size, "long", progress_callback, items_done, total_items
            )
            processed_count += long_processed

        return processed_count

    def _process_items_two_stage(
        self,
        items: list[ContentItem],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> int:
        """
        Process items using Ollama two-stage processing (8B screen + 32B refine).

        This method delegates to OllamaAPI.process_items_two_stage() which handles:
        1. Rule-based filtering (zero cost)
        2. 8B screening for importance/category
        3. 32B refinement for high-importance/uncertain items

        Args:
            items: Items to process with AI.
            progress_callback: Optional callback for progress updates.

        Returns:
            Number of items successfully processed.
        """
        import time
        from datetime import datetime

        from src.processors.ollama_api import OllamaAPI

        if not items:
            return 0

        # Ensure we have an OllamaAPI instance
        if not isinstance(self.processor.ai, OllamaAPI):
            logger.warning("Two-stage processing requires OllamaAPI, falling back to standard")
            # Fall through to standard processing (handled by caller)
            return 0

        ollama_api: OllamaAPI = self.processor.ai
        total_items = len(items)
        start_time = time.time()

        logger.info(f"Starting two-stage processing for {total_items} items...")

        # Track current phase for progress display
        current_phase_name = "Two-stage processing"
        current_phase_total = total_items

        # Update progress at start
        if progress_callback:
            progress_callback(current_phase_name, 0, total_items)

        # Phase callback: called when entering a new processing phase
        def internal_phase(phase_name: str, phase_total: int) -> None:
            """Reset progress display for new phase."""
            nonlocal current_phase_name, current_phase_total
            current_phase_name = phase_name
            current_phase_total = phase_total
            if progress_callback:
                progress_callback(phase_name, 0, phase_total)

        # Progress callback: updates current phase progress
        def internal_progress(current: int, total: int) -> None:
            if progress_callback:
                progress_callback(current_phase_name, current, total)

        # Process all items through two-stage pipeline (with parallel processing)
        results = ollama_api.process_items_two_stage(
            items, progress_callback=internal_progress, phase_callback=internal_phase
        )

        # Update items with results
        processed_count = 0
        for item, result in zip(items, results):
            if result and result.success:
                self._update_item_with_result(item, result)
                processed_count += 1

        elapsed = time.time() - start_time
        logger.info(
            f"Two-stage processing complete: {processed_count}/{total_items} items "
            f"in {elapsed:.1f}s ({elapsed/total_items:.2f}s/item avg)"
        )

        return processed_count

    def _process_batch_group_tiered(
        self,
        items: list[ContentItem],
        short_batch_size: int,
        long_batch_size: int,
        task_type: TaskType,
        group_name: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        offset: int = 0,
        total: int = 0,
        check_reprocess: bool = False,
        reprocess_threshold: int = 7,
    ) -> tuple[int, int, list[ContentItem]]:
        """
        Process a group of items with specified task type (model tier).

        Args:
            items: Items to process.
            short_batch_size: Batch size for short content.
            long_batch_size: Batch size for long content.
            task_type: Task type for model selection.
            group_name: Name for logging.
            progress_callback: Optional callback.
            offset: Starting offset for progress.
            total: Total items for progress.
            check_reprocess: Whether to check for reprocessing.
            reprocess_threshold: Score threshold for reprocessing.

        Returns:
            Tuple of (processed_count, items_done, reprocess_items).
        """
        import time

        # Separate by content length
        short_items = [i for i in items if len(i.content or "") < 500]
        long_items = [i for i in items if len(i.content or "") >= 500]

        processed_count = 0
        items_done = 0
        reprocess_items = []
        actual_total = total if total > 0 else len(items)
        group_start = time.time()

        # Process short items
        for batch_items in self._chunk_list(short_items, short_batch_size):
            batch_start = time.time()
            batch_data = [(idx, item.title, item.content) for idx, item in enumerate(batch_items)]

            # Call process_batch with task_type
            if hasattr(self.processor.ai, "process_batch"):
                results = self.processor.ai.process_batch(batch_data, task_type)
            else:
                results = [
                    self.processor.ai.process_content(item.title, item.content, task_type)
                    for item in batch_items
                ]

            batch_elapsed = time.time() - batch_start

            for item, result in zip(batch_items, results):
                if result.success:
                    self._update_item_with_result(item, result)
                    processed_count += 1

                    # Check for reprocessing
                    if check_reprocess and result.importance_score >= reprocess_threshold:
                        reprocess_items.append(item)

            items_done += len(batch_items)
            if progress_callback:
                progress_callback("AI processing", offset + items_done, actual_total)

            logger.debug(
                f"[{group_name}] Batch done: {len(batch_items)} items in {batch_elapsed:.1f}s"
            )

        # Process long items
        for batch_items in self._chunk_list(long_items, long_batch_size):
            batch_start = time.time()
            batch_data = [(idx, item.title, item.content) for idx, item in enumerate(batch_items)]

            if hasattr(self.processor.ai, "process_batch"):
                results = self.processor.ai.process_batch(batch_data, task_type)
            else:
                results = [
                    self.processor.ai.process_content(item.title, item.content, task_type)
                    for item in batch_items
                ]

            batch_elapsed = time.time() - batch_start

            for item, result in zip(batch_items, results):
                if result.success:
                    self._update_item_with_result(item, result)
                    processed_count += 1

                    if check_reprocess and result.importance_score >= reprocess_threshold:
                        reprocess_items.append(item)

            items_done += len(batch_items)
            if progress_callback:
                progress_callback("AI processing", offset + items_done, actual_total)

            logger.debug(
                f"[{group_name}] Batch done: {len(batch_items)} items in {batch_elapsed:.1f}s"
            )

        total_elapsed = time.time() - group_start
        logger.info(
            f"Finished {group_name} tier: {processed_count}/{len(items)} items "
            f"in {total_elapsed:.1f}s"
        )

        return processed_count, items_done, reprocess_items

    def _chunk_list(self, lst: list, chunk_size: int) -> list[list]:
        """Split list into chunks."""
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def _update_item_with_result(self, item: ContentItem, result: ProcessingResult) -> None:
        """Update content item with processing result and save to database."""
        item.summary = result.summary
        item.category = result.category
        item.importance_score = result.importance_score
        item.processed_at = datetime.now()

        # Save enhanced fields
        item.one_liner = result.one_liner
        item.key_points = self._serialize_key_points(result.key_points)
        item.impact_assessment = self._serialize_impact(result.impact_assessment)
        item.actionable_items = self._serialize_actions(result.actionable_items)

        self.db.update_content_item(item)

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

                    # Save enhanced fields (Phase 1)
                    item.one_liner = result.one_liner
                    item.key_points = self._serialize_key_points(result.key_points)
                    item.impact_assessment = self._serialize_impact(result.impact_assessment)
                    item.actionable_items = self._serialize_actions(result.actionable_items)

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

    def _serialize_key_points(self, key_points: list[KeyPointResult]) -> Optional[str]:
        """Serialize key points to JSON string."""
        import json

        if not key_points:
            return None
        return json.dumps(
            [{"type": kp.type, "value": kp.value, "impact": kp.impact} for kp in key_points],
            ensure_ascii=False,
        )

    def _serialize_impact(self, impact: Optional[ImpactResult]) -> Optional[str]:
        """Serialize impact assessment to JSON string."""
        import json

        if not impact:
            return None
        return json.dumps(
            {
                "short_term": impact.short_term,
                "long_term": impact.long_term,
                "certainty": impact.certainty,
            },
            ensure_ascii=False,
        )

    def _serialize_actions(self, actions: list[ActionResult]) -> Optional[str]:
        """Serialize actionable items to JSON string."""
        import json

        if not actions:
            return None
        return json.dumps(
            [
                {"type": a.type, "description": a.description, "priority": a.priority}
                for a in actions
            ],
            ensure_ascii=False,
        )

    def generate_digest(
        self,
        min_importance: int = 5,
        max_items: int = 30,
        include_delivered: bool = False,
        run_clustering: bool = True,
        clustering_progress_callback: Optional[callable] = None,
        parallel_clustering: bool = True,
        clustering_workers: int = 4,
    ) -> DigestResult:
        """
        Generate a digest from processed items, with event clustering support.

        Args:
            min_importance: Minimum importance score to include.
            max_items: Maximum number of items/events in the digest.
            include_delivered: Whether to include already delivered items.
            run_clustering: Whether to run event clustering before generating digest.
            clustering_progress_callback: Optional callback(current, total, clustered) for
                clustering progress updates.
            parallel_clustering: Use parallel processing for clustering (default: True).
            clustering_workers: Number of parallel workers for clustering (default: 4).

        Returns:
            DigestResult with digest items and event clusters.
        """
        if not self.db:
            raise ValueError("Database not configured")

        # Step 1: Run clustering if enabled
        if run_clustering:
            from src.processors.event_stream import (
                cluster_unprocessed_items,
                cluster_unprocessed_items_parallel,
            )

            use_mock = isinstance(self.processor.ai, MockAIProcessor)

            if parallel_clustering:
                clustered_count = cluster_unprocessed_items_parallel(
                    db=self.db,
                    use_mock=use_mock,
                    limit=100,
                    max_workers=clustering_workers,
                    progress_callback=clustering_progress_callback,
                )
            else:
                clustered_count = cluster_unprocessed_items(
                    db=self.db,
                    use_mock=use_mock,
                    limit=100,
                    progress_callback=clustering_progress_callback,
                )
            if clustered_count > 0:
                logger.info(f"Clustered {clustered_count} items into events")

        # Step 2: Get clustered items grouped by cluster
        clustered_data = self.db.get_undelivered_clustered_items(
            min_importance=min_importance,
            limit=max_items * 2,  # Get more to allow for grouping
        )

        # Step 3: Build EventDigestItem for each cluster
        events: list[EventDigestItem] = []
        for cluster_id, members in clustered_data.items():
            cluster = self.db.get_event_cluster(cluster_id)
            if cluster and members:
                # Members are already sorted by importance desc from the query
                events.append(
                    EventDigestItem(
                        event_cluster=cluster,
                        members=members,
                        representative_item=members[0],  # Highest importance
                    )
                )

        # Sort events by importance
        events.sort(key=lambda e: e.importance_score, reverse=True)

        # Step 4: Get unclustered items
        unclustered = self.db.get_undelivered_unclustered_items(
            min_importance=min_importance,
            limit=max_items,
        )

        digest_items = []
        for item in unclustered:
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

        # Limit total count (events + items)
        # Each event counts as 1 toward the limit
        total_count = len(events) + len(digest_items)
        if total_count > max_items:
            # Prioritize by importance across both lists
            all_items: list[tuple[int, bool, int]] = []
            for i, e in enumerate(events):
                all_items.append((e.importance_score, True, i))
            for i, d in enumerate(digest_items):
                all_items.append((d.importance_score, False, i))

            all_items.sort(key=lambda x: x[0], reverse=True)
            all_items = all_items[:max_items]

            # Rebuild filtered lists
            event_indices = {idx for score, is_event, idx in all_items if is_event}
            item_indices = {idx for score, is_event, idx in all_items if not is_event}

            events = [e for i, e in enumerate(events) if i in event_indices]
            digest_items = [d for i, d in enumerate(digest_items) if i in item_indices]

        total_processed = len(unclustered) + sum(len(e.members) for e in events)
        successful = len(digest_items) + len(events)

        return DigestResult(
            items=digest_items,
            events=events,
            total_processed=total_processed,
            successful=successful,
            failed=total_processed - successful,
        )

    def mark_digest_delivered(self, digest: DigestResult) -> None:
        """Mark all items in a digest as delivered, including event members."""
        if not self.db:
            raise ValueError("Database not configured")

        item_ids = digest.all_content_ids
        if item_ids:
            self.db.mark_items_delivered(item_ids)
            event_member_count = sum(len(e.members) for e in digest.events)
            logger.info(
                f"Marked {len(item_ids)} items as delivered "
                f"({len(digest.items)} individual + {event_member_count} in "
                f"{len(digest.events)} events)"
            )

    def extract_action_items(self, digest: DigestResult) -> int:
        """
        Extract actionable items from digest and save to database.

        Phase 5: Automatically creates action items from high-importance content
        that has actionable_items in its enhanced data.

        Args:
            digest: The digest result containing processed items

        Returns:
            Number of action items created
        """
        if not self.db:
            raise ValueError("Database not configured")

        created = 0
        for digest_item in digest.items:
            content = digest_item.content_item
            enhanced = content.get_enhanced_data()

            if not enhanced or not enhanced.actionable_items:
                continue

            # Only extract actions from high-importance items
            if content.importance_score and content.importance_score < 7:
                continue

            for action_data in enhanced.actionable_items:
                action = ActionItem(
                    content_item_id=content.id,
                    type=action_data.type,
                    description=action_data.description,
                    priority=action_data.priority,
                    status=ActionStatus.PENDING,
                    created_at=datetime.now(),
                )
                self.db.create_action_item(action)
                created += 1
                logger.debug(
                    f"Created action item: [{action.priority}] {action.type}: "
                    f"{action.description[:30]}..."
                )

        if created:
            logger.info(f"Extracted {created} action items from digest")

        return created


def create_digest_preview(digest: DigestResult) -> str:
    """
    Create an enhanced text preview of the digest with rich formatting.

    Shows:
    - Event/article title
    - Importance score
    - One-liner (key insight)
    - Summary
    - Key points
    - Impact assessment (for high-importance items)
    - Related articles (for events)
    - Source info

    Uses box-drawing characters for visual structure.

    Args:
        digest: The digest to preview.

    Returns:
        Formatted text preview.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("BabelByte 每日摘要预览")
    lines.append(f"生成时间: {digest.generated_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"共 {digest.total_items} 条内容")
    if digest.events:
        lines.append(f"其中 {len(digest.events)} 个事件, {len(digest.items)} 条独立内容")
    lines.append("=" * 60)

    if not digest.items and not digest.events:
        lines.append("\n暂无新内容")
        return "\n".join(lines)

    # Group by category (includes both events and items)
    for category, category_items in sorted(digest.by_category.items()):
        lines.append(f"\n┌─ [{category}] ({len(category_items)}条) " + "─" * 30)

        for item in category_items:
            lines.append("│")

            if item.is_event:
                # Event preview with enhanced format
                lines.append(f"│ 事件 | {item.event_cluster.event_title}")
                lines.append(f"│ {item.importance_score}/10")

                # One-liner (key insight)
                one_liner = item.one_liner
                if one_liner:
                    lines.append(f"│ [核心] {one_liner}")

                # Summary
                if item.summary:
                    lines.append(f"│ {item.summary}")

                # Key points from representative item
                enhanced = item.enhanced_data
                if enhanced and enhanced.key_points:
                    kp_str = " | ".join(
                        [f"{kp.type}: {kp.value}" for kp in enhanced.key_points[:3]]
                    )
                    lines.append(f"│ 关键点: {kp_str}")

                # Impact assessment for high importance
                if item.importance_score >= 7 and enhanced and enhanced.impact_assessment:
                    impact = enhanced.impact_assessment
                    if impact.short_term:
                        lines.append(f"│ 短期影响: {impact.short_term}")

                # Related articles
                lines.append("│ 相关报道:")
                for member in item.members[:3]:
                    title_display = member.title[:60]
                    if len(member.title) > 60:
                        title_display += "..."
                    lines.append(f"│   - {title_display}")
                    lines.append(f"│     {member.url}")
                if len(item.members) > 3:
                    lines.append(f"│   ...还有 {len(item.members) - 3} 篇报道")

                lines.append(f"│ 来源: {item.source_display}")

            else:
                # Regular item preview with enhanced format
                title = item.content_item.title
                title_display = title[:60] + "..." if len(title) > 60 else title
                lines.append(f"│ 文章 | {title_display}")
                lines.append(f"│ {item.importance_score}/10")

                # One-liner
                enhanced = item.content_item.get_enhanced_data()
                one_liner = item.content_item.one_liner
                if one_liner:
                    lines.append(f"│ [核心] {one_liner}")

                # Summary
                if item.summary:
                    lines.append(f"│ {item.summary}")

                # Key points
                if enhanced and enhanced.key_points:
                    kp_str = " | ".join(
                        [f"{kp.type}: {kp.value}" for kp in enhanced.key_points[:3]]
                    )
                    lines.append(f"│ 关键点: {kp_str}")

                # Impact assessment for high importance
                if item.importance_score >= 7 and enhanced and enhanced.impact_assessment:
                    impact = enhanced.impact_assessment
                    if impact.short_term:
                        lines.append(f"│ 短期影响: {impact.short_term}")

                lines.append(f"│ {item.content_item.url}")
                author = item.content_item.author or "未知"
                lines.append(f"│ 来源: {item.source_display} | 作者: {author}")

            lines.append("│" + "─" * 50)

        lines.append("└" + "─" * 50)

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)

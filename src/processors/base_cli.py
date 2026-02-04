"""Base class for CLI-based AI processors with shared implementation."""

import json
import logging
import subprocess
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from config.settings import get_settings
from src.analytics.token_tracker import AICallType, record_ai_call
from src.processors.base import BaseAIProcessor, ProcessingResult, TaskType
from src.processors.rule_classifier import (
    create_skip_result,
    estimate_importance,
    is_paper_content,
    should_skip_ai_processing,
)

if TYPE_CHECKING:
    from src.storage.database import SyncDatabase
    from src.storage.models import ContentItem

logger = logging.getLogger(__name__)


class BaseCLIProcessor(BaseAIProcessor):
    """
    Base class for CLI-based AI processors (Claude CLI, Codex CLI, etc.).

    This class provides shared implementation for:
    - Cache handling (check/store)
    - Token tracking
    - Batch processing with fallback
    - JSON response parsing

    Subclasses must implement:
    - cli_path property: Path to the CLI executable
    - _get_model_name(): Get the model name for heavy/light tasks
    - _build_cli_command(): Build the CLI command list
    - _run_subprocess(): Execute the CLI with proper input handling
    """

    def __init__(
        self,
        cli_path: Optional[str] = None,
        timeout: int = 60,
        db: Optional["SyncDatabase"] = None,
    ):
        self._cli_path = cli_path
        self.timeout = timeout
        self.db = db
        self._settings = None

    @property
    def settings(self):
        """Lazy load settings."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    @abstractmethod
    def cli_path(self) -> str:
        """Path to the CLI executable."""
        pass

    @abstractmethod
    def _get_model_name(self, task_type: TaskType, heavy: bool) -> str:
        """
        Get model name for the given task type.

        Args:
            task_type: Type of task being performed.
            heavy: True for heavy model, False for light model.

        Returns:
            Model name to use.
        """
        pass

    @abstractmethod
    def _build_cli_command(self, model: Optional[str] = None) -> list[str]:
        """
        Build the CLI command list.

        Args:
            model: Optional model name to use.

        Returns:
            Command list for subprocess.
        """
        pass

    @abstractmethod
    def _run_subprocess(
        self, cmd: list[str], prompt: str, timeout: int
    ) -> subprocess.CompletedProcess:
        """
        Execute the CLI subprocess.

        Args:
            cmd: Command list.
            prompt: The prompt to send.
            timeout: Timeout in seconds.

        Returns:
            Completed subprocess result.
        """
        pass

    @property
    @abstractmethod
    def cli_name(self) -> str:
        """Name of the CLI for error messages (e.g., 'Claude', 'Codex')."""
        pass

    def get_model_for_task(self, task_type: TaskType) -> str:
        """
        Select model based on task type.

        Args:
            task_type: Type of task being performed.

        Returns:
            Model name to use, or empty string to use default.
        """
        config = self.settings.ai.model_tiers
        if not config.enabled:
            return ""

        # Check task-level override
        task_name = task_type.name.lower()
        if task_name in config.task_overrides:
            return config.task_overrides[task_name]

        # Heavy tasks use heavy model
        heavy_tasks = {
            TaskType.CONTENT_HIGH,
            TaskType.CONTENT_UNCERTAIN,
            TaskType.REPORT,
        }
        is_heavy = task_type in heavy_tasks

        return self._get_model_name(task_type, is_heavy)

    def _run_cli(
        self, prompt: str, task_type: Optional[TaskType] = None, timeout: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        """
        Run CLI with optional model selection.

        Args:
            prompt: The prompt to send.
            task_type: Task type for model selection.
            timeout: Optional timeout override.

        Returns:
            Completed subprocess result.
        """
        model = None
        if task_type:
            model = self.get_model_for_task(task_type)
            if model:
                logger.debug(f"Using model {model} for task {task_type.name}")

        cmd = self._build_cli_command(model)
        return self._run_subprocess(cmd, prompt, timeout or self.timeout)

    def _get_ai_call_type(self, task_type: TaskType) -> AICallType:
        """Map task type to AI call type for tracking."""
        call_type_map = {
            TaskType.CONTENT_HIGH: AICallType.CONTENT_HEAVY,
            TaskType.CONTENT_LOW: AICallType.CONTENT_LIGHT,
            TaskType.CONTENT_MINIMAL: AICallType.CONTENT_LIGHT,
            TaskType.CONTENT_UNCERTAIN: AICallType.CONTENT_HEAVY,
            TaskType.PAPER_FULL: AICallType.CONTENT_HEAVY,
            TaskType.PAPER_SCREEN: AICallType.CONTENT_LIGHT,
        }
        return call_type_map.get(task_type, AICallType.CONTENT_HEAVY)

    def process_content(
        self, title: str, content: str, task_type: Optional[TaskType] = None
    ) -> ProcessingResult:
        """
        Process content using the CLI.

        Args:
            title: The title of the content.
            content: The main content text.
            task_type: Task type for model selection. Defaults to CONTENT_HIGH.

        Returns:
            ProcessingResult with summary, category, and importance score.
        """
        if task_type is None:
            task_type = TaskType.CONTENT_HIGH

        # Check cache if enabled
        content_hash = None
        if self.settings.ai.cache_enabled and self.db:
            content_hash = self._get_content_hash(title, content)
            cached_json = self.db.get_ai_cache(content_hash)
            if cached_json:
                logger.debug(f"AI cache hit: {title[:30]}...")
                record_ai_call(
                    call_type=self._get_ai_call_type(task_type),
                    cached=True,
                    input_chars=len(title) + len(content),
                )
                return self._deserialize_result(cached_json)

        # Select prompt based on task type
        if task_type == TaskType.PAPER_FULL:
            prompt = self._build_paper_prompt(title, content)
        elif task_type == TaskType.CONTENT_MINIMAL:
            prompt = self._build_simple_prompt(title, content)
        elif task_type == TaskType.CONTENT_LOW:
            prompt = self._build_light_prompt(title, content)
        else:
            prompt = self._build_prompt(title, content)

        start_time = time.time()

        try:
            result = self._run_cli(prompt, task_type)
            duration_ms = int((time.time() - start_time) * 1000)
            ai_call_type = self._get_ai_call_type(task_type)

            if result.returncode != 0:
                record_ai_call(
                    call_type=ai_call_type,
                    cached=False,
                    input_chars=len(prompt),
                    output_chars=len(result.stderr),
                    duration_ms=duration_ms,
                    success=False,
                    error=result.stderr[:100],
                )
                return ProcessingResult(
                    summary="",
                    category="其他",
                    importance_score=5,
                    success=False,
                    error_message=f"{self.cli_name} CLI error: {result.stderr}",
                    raw_response=result.stdout,
                )

            processing_result = self._parse_json_response(result.stdout)

            record_ai_call(
                call_type=ai_call_type,
                cached=False,
                input_chars=len(prompt),
                output_chars=len(result.stdout),
                duration_ms=duration_ms,
                success=processing_result.success,
            )

            # Store in cache if successful
            if processing_result.success and self.settings.ai.cache_enabled and self.db:
                if content_hash is None:
                    content_hash = self._get_content_hash(title, content)
                self.db.set_ai_cache(
                    content_hash,
                    self._serialize_result(processing_result),
                    self.settings.ai.cache_ttl,
                )
                logger.debug(f"Cache stored: {title[:30]}...")

            return processing_result

        except subprocess.TimeoutExpired:
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message=f"{self.cli_name} CLI timeout",
            )
        except FileNotFoundError:
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message=f"{self.cli_name} CLI not found at: {self.cli_path}",
            )
        except (OSError, subprocess.SubprocessError) as e:
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message=f"Subprocess error: {str(e)}",
            )

    def process_batch(
        self, items: list[tuple[int, str, str]], task_type: Optional[TaskType] = None
    ) -> list[ProcessingResult]:
        """
        Process multiple items in a single API call with cache support.

        Args:
            items: List of (id, title, content) tuples
            task_type: Task type for model selection. Defaults to CONTENT_HIGH.

        Returns:
            List of ProcessingResult, one per item
        """
        if not items:
            return []

        if task_type is None:
            task_type = TaskType.CONTENT_HIGH

        results: list[Optional[ProcessingResult]] = [None] * len(items)
        items_to_process: list[tuple[int, int, str, str]] = []
        cache_hashes: dict[int, str] = {}

        # 1. Check cache for each item
        if self.settings.ai.cache_enabled and self.db:
            for i, (orig_id, title, content) in enumerate(items):
                content_hash = self._get_content_hash(title, content)
                cached_json = self.db.get_ai_cache(content_hash)
                if cached_json:
                    results[i] = self._deserialize_result(cached_json)
                    logger.debug(f"Batch cache hit: {title[:30]}...")
                    record_ai_call(
                        call_type=AICallType.CONTENT_BATCH,
                        cached=True,
                        input_chars=len(title) + len(content),
                    )
                else:
                    items_to_process.append((i, orig_id, title, content))
                    cache_hashes[i] = content_hash
        else:
            items_to_process = [
                (i, orig_id, title, content) for i, (orig_id, title, content) in enumerate(items)
            ]

        # All items were cached
        if not items_to_process:
            logger.info(f"Batch: all {len(items)} items from cache")
            return results

        cache_hits = len(items) - len(items_to_process)
        if cache_hits > 0:
            logger.info(f"Batch: {cache_hits} cache hits, {len(items_to_process)} to process")

        # 2. Process uncached items in batch
        batch_input = [(orig_id, title, content) for _, orig_id, title, content in items_to_process]
        prompt = self._build_batch_prompt(batch_input)
        batch_timeout = self.timeout + (len(batch_input) * 15)

        start_time = time.time()

        try:
            cli_result = self._run_cli(prompt, task_type, batch_timeout)
            duration_ms = int((time.time() - start_time) * 1000)

            if cli_result.returncode != 0:
                record_ai_call(
                    call_type=AICallType.CONTENT_BATCH,
                    cached=False,
                    input_chars=len(prompt),
                    duration_ms=duration_ms,
                    success=False,
                    error="Batch processing failed",
                )
                # Fall back to individual processing
                for result_idx, orig_id, title, content in items_to_process:
                    results[result_idx] = self.process_content(title, content, task_type)
                return results

            batch_results = self._parse_batch_response(cli_result.stdout, len(batch_input))

            record_ai_call(
                call_type=AICallType.CONTENT_BATCH,
                cached=False,
                input_chars=len(prompt),
                output_chars=len(cli_result.stdout),
                duration_ms=duration_ms,
                success=True,
                input_tokens=len(batch_input) * 215,
            )

            # 3. Fill results and store in cache
            for (result_idx, _, title, _), proc_result in zip(items_to_process, batch_results):
                results[result_idx] = proc_result
                if proc_result.success and self.settings.ai.cache_enabled and self.db:
                    content_hash = cache_hashes.get(result_idx)
                    if content_hash:
                        self.db.set_ai_cache(
                            content_hash,
                            self._serialize_result(proc_result),
                            self.settings.ai.cache_ttl,
                        )
                        logger.debug(f"Batch cache stored: {title[:30]}...")

            return results

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            # Fall back to individual processing
            for result_idx, orig_id, title, content in items_to_process:
                results[result_idx] = self.process_content(title, content, task_type)
            return results

    def _parse_batch_response(self, response: str, expected_count: int) -> list[ProcessingResult]:
        """Parse batch response containing multiple JSON objects."""
        results_by_id: dict[int, ProcessingResult] = {}
        results_in_order: list[ProcessingResult] = []

        # Strategy 1: Try parsing line by line
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue

            try:
                data = json.loads(line)
                result = self._extract_result_from_dict(data)
                if result:
                    idx = data.get("id", -1)
                    if idx >= 0:
                        results_by_id[idx] = result
                    results_in_order.append(result)
            except json.JSONDecodeError:
                continue

        # Strategy 2: Find JSON objects with balanced braces
        if len(results_in_order) < expected_count:
            for match in self._find_json_objects(response):
                try:
                    data = json.loads(match)
                    result = self._extract_result_from_dict(data)
                    if result:
                        idx = data.get("id", -1)
                        if idx >= 0 and idx not in results_by_id:
                            results_by_id[idx] = result
                        if len(results_in_order) < expected_count:
                            results_in_order.append(result)
                except json.JSONDecodeError:
                    continue

        # Build final results
        final_results: list[ProcessingResult] = []
        for i in range(expected_count):
            if i in results_by_id:
                final_results.append(results_by_id[i])
            elif i < len(results_in_order):
                final_results.append(results_in_order[i])
            else:
                logger.debug(f"Batch parse: missing item {i}, response preview: {response[:200]}")
                final_results.append(
                    ProcessingResult(
                        summary="",
                        category="其他",
                        importance_score=5,
                        success=False,
                        error_message=f"Missing result for item {i}",
                    )
                )

        return final_results

    def _extract_result_from_dict(self, data: dict) -> Optional[ProcessingResult]:
        """Extract ProcessingResult from a parsed JSON dict with enhanced fields."""
        if not isinstance(data, dict):
            return None

        summary = data.get("summary", "")
        if not summary:
            return None

        importance = data.get("importance", 5)
        if not isinstance(importance, int):
            try:
                importance = int(importance)
            except (ValueError, TypeError):
                importance = 5

        return self._extract_processing_result(data, json.dumps(data, ensure_ascii=False))

    def _find_json_objects(self, text: str) -> list[str]:
        """Find JSON objects in text by matching balanced braces."""
        objects: list[str] = []
        i = 0
        while i < len(text):
            if text[i] == "{":
                depth = 1
                start = i
                i += 1
                while i < len(text) and depth > 0:
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                    i += 1
                if depth == 0:
                    objects.append(text[start:i])
            else:
                i += 1
        return objects

    def is_available(self) -> bool:
        """Check if CLI is available."""
        try:
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def process_item(self, item: "ContentItem") -> ProcessingResult:
        """
        Process a ContentItem with optimized rule-first approach.

        This method tries multiple optimization strategies before falling back to AI:
        1. Skip processing for low-value content (spam, boilerplate, etc.)
        2. Use rule-only processing for high-confidence matches
        3. Select appropriate prompt complexity based on importance estimate
        4. Fall back to AI processing with appropriate model tier

        Args:
            item: The ContentItem to process.

        Returns:
            ProcessingResult with summary, category, and importance score.
        """
        title = item.title or ""
        content = item.content or ""

        # Strategy 1: Skip low-value content entirely
        should_skip, skip_reason = should_skip_ai_processing(item)
        if should_skip:
            summary, category, importance = create_skip_result(item, skip_reason)
            logger.info(f"Skipped AI processing: {title[:40]}... ({skip_reason})")
            record_ai_call(
                call_type=AICallType.CONTENT_LIGHT,
                cached=True,  # Count as "cached" since no AI call made
                input_chars=len(title) + len(content),
            )
            return ProcessingResult(
                summary=summary,
                category=category,
                importance_score=importance,
                success=True,
            )

        # Strategy 2: Check if this is an academic paper
        rule_settings = self.settings.rule_optimization
        if rule_settings.paper_prompt_enabled and is_paper_content(item):
            logger.info(f"Paper detected: {title[:50]}...")
            return self.process_content(title, content, TaskType.PAPER_FULL)

        # Strategy 3: Estimate importance for model/prompt selection
        importance_est = estimate_importance(item)
        min_conf = rule_settings.minimal_prompt_min_confidence

        # Select task type based on importance estimate
        # High importance → full analysis (heavy model)
        if importance_est.score >= 7:
            task_type = TaskType.CONTENT_HIGH
        # Quality sources (reddit/twitter/hn) → LIGHT_PROMPT (keeps key_points)
        elif importance_est.reason.startswith("source:"):
            task_type = TaskType.CONTENT_LOW
        # Low importance with sufficient confidence → minimal analysis (if enabled)
        elif (
            rule_settings.minimal_prompt_enabled
            and importance_est.confidence >= min_conf
            and importance_est.score <= rule_settings.minimal_prompt_threshold
        ):
            task_type = TaskType.CONTENT_MINIMAL
        # Low importance with sufficient confidence → light analysis
        elif importance_est.confidence >= min_conf and importance_est.score <= 5:
            task_type = TaskType.CONTENT_LOW
        # Low confidence → use heavy model to ensure quality
        elif importance_est.confidence < min_conf:
            task_type = TaskType.CONTENT_UNCERTAIN
        # Medium confidence, medium importance → light analysis
        else:
            task_type = TaskType.CONTENT_LOW

        logger.debug(
            f"Importance estimate: score={importance_est.score}, "
            f"confidence={importance_est.confidence:.2f}, "
            f"reason={importance_est.reason} -> {task_type.name}"
        )

        # Strategy 4: Fall back to AI processing
        return self.process_content(title, content, task_type)

    def process_items_batch(
        self, items: list["ContentItem"], batch_size: int = 6
    ) -> list[ProcessingResult]:
        """
        Process multiple ContentItems with optimized batching.

        Groups items by processing strategy:
        - Skip items: Handled immediately with no AI
        - Rule-only items: Handled immediately with no AI
        - AI items: Batched for efficient processing

        Args:
            items: List of ContentItems to process.
            batch_size: Maximum items per AI batch.

        Returns:
            List of ProcessingResults in same order as input.
        """
        results: list[Optional[ProcessingResult]] = [None] * len(items)
        ai_items: list[tuple[int, "ContentItem", TaskType]] = []

        # First pass: handle skip and rule-only items
        for i, item in enumerate(items):
            title = item.title or ""
            content = item.content or ""

            # Check if should skip
            should_skip, skip_reason = should_skip_ai_processing(item)
            if should_skip:
                summary, category, importance = create_skip_result(item, skip_reason)
                results[i] = ProcessingResult(
                    summary=summary,
                    category=category,
                    importance_score=importance,
                    success=True,
                )
                record_ai_call(
                    call_type=AICallType.CONTENT_LIGHT,
                    cached=True,
                    input_chars=len(title) + len(content),
                )
                continue

            # Estimate importance for task type
            importance_est = estimate_importance(item)
            rule_settings = self.settings.rule_optimization
            min_conf = rule_settings.minimal_prompt_min_confidence
            if importance_est.score >= 7:
                task_type = TaskType.CONTENT_HIGH
            elif importance_est.reason.startswith("source:"):
                task_type = TaskType.CONTENT_LOW
            elif (
                rule_settings.minimal_prompt_enabled
                and importance_est.confidence >= min_conf
                and importance_est.score <= rule_settings.minimal_prompt_threshold
            ):
                task_type = TaskType.CONTENT_MINIMAL
            elif importance_est.confidence >= min_conf and importance_est.score <= 5:
                task_type = TaskType.CONTENT_LOW
            elif importance_est.confidence < min_conf:
                task_type = TaskType.CONTENT_UNCERTAIN
            else:
                task_type = TaskType.CONTENT_LOW

            ai_items.append((i, item, task_type))

        # Log optimization stats
        skipped_count = len(items) - len(ai_items)
        if skipped_count > 0:
            logger.info(
                f"Batch optimization: {skipped_count}/{len(items)} items handled without AI"
            )

        # Second pass: process AI items in batches
        if ai_items:
            # Group by task type for optimal batching
            high_items = [(i, it) for i, it, tt in ai_items
                          if tt in (TaskType.CONTENT_HIGH, TaskType.CONTENT_UNCERTAIN)]
            low_items = [(i, it) for i, it, tt in ai_items if tt == TaskType.CONTENT_LOW]
            minimal_items = [(i, it) for i, it, tt in ai_items if tt == TaskType.CONTENT_MINIMAL]

            # Process each group with appropriate task type
            groups = [
                (high_items, TaskType.CONTENT_HIGH),
                (low_items, TaskType.CONTENT_LOW),
                (minimal_items, TaskType.CONTENT_MINIMAL),
            ]

            for group, task_type in groups:
                if not group:
                    continue

                # Create batch input
                batch_input = [
                    (item.id or idx, item.title or "", item.content or "")
                    for idx, (orig_idx, item) in enumerate(group)
                ]

                # Process in batches
                for batch_start in range(0, len(batch_input), batch_size):
                    batch = batch_input[batch_start:batch_start + batch_size]
                    batch_results = self.process_batch(batch, task_type)

                    # Map results back to original indices
                    for j, result in enumerate(batch_results):
                        orig_idx = group[batch_start + j][0]
                        results[orig_idx] = result

        return results

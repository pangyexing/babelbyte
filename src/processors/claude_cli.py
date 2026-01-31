"""Claude Code CLI wrapper for AI processing."""

import json
import logging
import subprocess
import time
from typing import TYPE_CHECKING, Optional

from config.settings import get_settings
from src.analytics.token_tracker import AICallType, record_ai_call
from src.processors.base import BaseAIProcessor, ProcessingResult, TaskType

if TYPE_CHECKING:
    from src.storage.database import SyncDatabase

logger = logging.getLogger(__name__)


class ClaudeCLI(BaseAIProcessor):
    """Wrapper for Claude Code CLI."""

    def __init__(
        self,
        cli_path: Optional[str] = None,
        timeout: int = 60,
        db: Optional["SyncDatabase"] = None,
    ):
        self.cli_path = cli_path or get_settings().claude.cli_path
        self.timeout = timeout
        self.db = db
        self._settings = None

    @property
    def settings(self):
        """Lazy load settings."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

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
        if task_type in heavy_tasks:
            return config.claude_heavy

        # Light tasks use light model
        return config.claude_light

    def _run_cli(
        self, prompt: str, task_type: Optional[TaskType] = None, timeout: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        """
        Run Claude CLI with optional model selection.

        Args:
            prompt: The prompt to send.
            task_type: Task type for model selection.
            timeout: Optional timeout override.

        Returns:
            Completed subprocess result.
        """
        cmd = [self.cli_path, "-p", prompt, "--output-format", "text"]

        if task_type:
            model = self.get_model_for_task(task_type)
            if model:
                cmd.extend(["--model", model])
                logger.debug(f"Using model {model} for task {task_type.name}")

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout or self.timeout,
        )

    def process_content(
        self, title: str, content: str, task_type: Optional[TaskType] = None
    ) -> ProcessingResult:
        """
        Process content using Claude Code CLI.

        Args:
            title: The title of the content.
            content: The main content text.
            task_type: Task type for model selection. Defaults to CONTENT_HIGH.

        Returns:
            ProcessingResult with summary, category, and importance score.
        """
        # Default to high importance processing if not specified
        if task_type is None:
            task_type = TaskType.CONTENT_HIGH

        # Check cache if enabled
        content_hash = None
        if self.settings.ai.cache_enabled and self.db:
            content_hash = self._get_content_hash(title, content)
            cached_json = self.db.get_ai_cache(content_hash)
            if cached_json:
                logger.debug(f"AI cache hit: {title[:30]}...")
                # Record cache hit for token tracking
                call_type_map = {
                    TaskType.CONTENT_HIGH: AICallType.CONTENT_HEAVY,
                    TaskType.CONTENT_LOW: AICallType.CONTENT_LIGHT,
                    TaskType.CONTENT_UNCERTAIN: AICallType.CONTENT_HEAVY,
                }
                record_ai_call(
                    call_type=call_type_map.get(task_type, AICallType.CONTENT_HEAVY),
                    cached=True,
                    input_chars=len(title) + len(content),
                )
                return self._deserialize_result(cached_json)

        # Use light prompt for low importance content
        if task_type == TaskType.CONTENT_LOW:
            prompt = self._build_light_prompt(title, content)
        else:
            prompt = self._build_prompt(title, content)

        # Track timing for token tracking
        start_time = time.time()

        try:
            result = self._run_cli(prompt, task_type)
            duration_ms = int((time.time() - start_time) * 1000)

            # Map task type to AI call type for tracking
            call_type_map = {
                TaskType.CONTENT_HIGH: AICallType.CONTENT_HEAVY,
                TaskType.CONTENT_LOW: AICallType.CONTENT_LIGHT,
                TaskType.CONTENT_UNCERTAIN: AICallType.CONTENT_HEAVY,
            }
            ai_call_type = call_type_map.get(task_type, AICallType.CONTENT_HEAVY)

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
                    error_message=f"Claude CLI error: {result.stderr}",
                    raw_response=result.stdout,
                )

            processing_result = self._parse_json_response(result.stdout)

            # Record successful AI call
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
                error_message="Claude CLI timeout",
            )
        except FileNotFoundError:
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message=f"Claude CLI not found at: {self.cli_path}",
            )
        except Exception as e:
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
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
        # (result_idx, orig_id, title, content)
        items_to_process: list[tuple[int, int, str, str]] = []
        cache_hashes: dict[int, str] = {}  # result_idx -> content_hash

        # 1. Check cache for each item
        if self.settings.ai.cache_enabled and self.db:
            for i, (orig_id, title, content) in enumerate(items):
                content_hash = self._get_content_hash(title, content)
                cached_json = self.db.get_ai_cache(content_hash)
                if cached_json:
                    results[i] = self._deserialize_result(cached_json)
                    logger.debug(f"Batch cache hit: {title[:30]}...")
                    # Record cache hit
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

        # Dynamic timeout: base + 15s per item (larger batches need more time)
        batch_timeout = self.timeout + (len(batch_input) * 15)

        # Track timing
        start_time = time.time()

        try:
            cli_result = self._run_cli(prompt, task_type, batch_timeout)
            duration_ms = int((time.time() - start_time) * 1000)

            if cli_result.returncode != 0:
                # Record failed batch call
                record_ai_call(
                    call_type=AICallType.CONTENT_BATCH,
                    cached=False,
                    input_chars=len(prompt),
                    duration_ms=duration_ms,
                    success=False,
                    error="Batch processing failed",
                )
                # Fall back to individual processing (which uses cache internally)
                for result_idx, orig_id, title, content in items_to_process:
                    results[result_idx] = self.process_content(title, content, task_type)
                return results

            batch_results = self._parse_batch_response(cli_result.stdout, len(batch_input))

            # Record successful batch call (one call for entire batch)
            record_ai_call(
                call_type=AICallType.CONTENT_BATCH,
                cached=False,
                input_chars=len(prompt),
                output_chars=len(cli_result.stdout),
                duration_ms=duration_ms,
                success=True,
                # Estimate tokens based on batch size
                input_tokens=len(batch_input) * 215,  # Estimated per-item tokens
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

        except Exception:
            # Fall back to individual processing (which uses cache internally)
            for result_idx, orig_id, title, content in items_to_process:
                results[result_idx] = self.process_content(title, content, task_type)
            return results

    def _parse_batch_response(self, response: str, expected_count: int) -> list[ProcessingResult]:
        """Parse batch response containing multiple JSON objects."""
        import logging

        logger = logging.getLogger(__name__)

        # Strategy 1: Try parsing line by line (most common format)
        results_by_id = {}
        results_in_order = []

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

        # Strategy 2: If line-by-line didn't work, try finding JSON objects with balanced braces
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

        # Build final results: prefer by ID, fall back to order
        final_results = []
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
        importance = data.get("importance", 5)

        # Skip if no summary (likely not a valid result)
        if not summary:
            return None

        if not isinstance(importance, int):
            try:
                importance = int(importance)
            except (ValueError, TypeError):
                importance = 5
        _ = max(1, min(10, importance))  # Validation only, actual used in parent method

        # Use parent class method to extract enhanced fields
        return self._extract_processing_result(data, json.dumps(data, ensure_ascii=False))

    def _find_json_objects(self, text: str) -> list[str]:
        """Find JSON objects in text by matching balanced braces."""
        objects = []
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
        """Check if Claude CLI is available."""
        try:
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

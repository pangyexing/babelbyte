"""Ollama API wrapper for local LLM processing.

Uses Ollama's HTTP API (default: http://localhost:11434) for content processing
with locally running models. Ollama uses a single model configuration to avoid
the overhead of switching models during processing.
"""

import json
import logging
import time
import urllib.request
import urllib.error
from typing import TYPE_CHECKING, Optional

from config.settings import get_settings
from src.analytics.token_tracker import AICallType, record_ai_call
from src.processors.base import BaseAIProcessor, ProcessingResult, TaskType
from src.processors.rule_classifier import (
    create_skip_result,
    estimate_importance,
    should_skip_ai_processing,
    try_rule_only_processing,
)

if TYPE_CHECKING:
    from src.storage.database import SyncDatabase
    from src.storage.models import ContentItem

logger = logging.getLogger(__name__)


class OllamaAPI(BaseAIProcessor):
    """Ollama HTTP API wrapper for local LLM processing.

    Unlike CLI-based processors, OllamaAPI uses HTTP requests to communicate
    with the Ollama server. This provides structured JSON responses and
    better error handling.

    Ollama uses a single model (configured via OLLAMA_MODEL) without
    heavy/light tier switching to avoid model loading overhead.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        db: Optional["SyncDatabase"] = None,
    ):
        """Initialize OllamaAPI.

        Args:
            base_url: Ollama API base URL. Defaults to config or http://localhost:11434.
            model: Model name to use. Defaults to config or qwen3:32b.
            timeout: Request timeout in seconds. Defaults to config or 120.
            db: Optional database instance for cache support.
        """
        self._settings = None
        self.db = db

        settings = self.settings
        self._base_url = base_url or settings.ollama.base_url
        self._model = model or settings.ollama.model
        self._timeout = timeout or settings.ollama.timeout

    @property
    def settings(self):
        """Lazy load settings."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def base_url(self) -> str:
        """Ollama API base URL."""
        return self._base_url

    @property
    def model(self) -> str:
        """Model name to use for all requests."""
        return self._model

    @property
    def timeout(self) -> int:
        """Request timeout in seconds."""
        return self._timeout

    def is_available(self) -> bool:
        """Check if Ollama service is running.

        Sends a GET request to /api/tags to verify the service is accessible.
        """
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return False

    def get_model(self, task_type: Optional[TaskType] = None) -> str:
        """Get the model name to use.

        Ollama uses a single model for all tasks to avoid model loading overhead.

        Args:
            task_type: Ignored - Ollama doesn't switch models.

        Returns:
            The configured model name.
        """
        return self._model

    def _call_api(self, prompt: str) -> tuple[bool, str, Optional[str]]:
        """Send a request to Ollama's generate API.

        Args:
            prompt: The prompt to send.

        Returns:
            Tuple of (success, response_text, error_message).
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 2048,
            },
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return True, result.get("response", ""), None

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            return False, "", f"HTTP {e.code}: {error_body}"
        except urllib.error.URLError as e:
            return False, "", f"Connection error: {e.reason}"
        except TimeoutError:
            return False, "", f"Request timeout after {self._timeout}s"
        except json.JSONDecodeError as e:
            return False, "", f"Invalid JSON response: {e}"
        except OSError as e:
            return False, "", f"Network error: {e}"

    def _get_ai_call_type(self, task_type: TaskType) -> AICallType:
        """Map task type to AI call type for tracking.

        Ollama uses CONTENT_HEAVY for all calls since it's a single model.
        """
        # Ollama uses a single model, but we still track by task type
        call_type_map = {
            TaskType.CONTENT_HIGH: AICallType.CONTENT_HEAVY,
            TaskType.CONTENT_LOW: AICallType.CONTENT_LIGHT,
            TaskType.CONTENT_MINIMAL: AICallType.CONTENT_LIGHT,
            TaskType.CONTENT_UNCERTAIN: AICallType.CONTENT_HEAVY,
        }
        return call_type_map.get(task_type, AICallType.CONTENT_HEAVY)

    def process_content(
        self, title: str, content: str, task_type: Optional[TaskType] = None
    ) -> ProcessingResult:
        """Process content using Ollama API.

        Args:
            title: The title of the content.
            content: The main content text.
            task_type: Task type for prompt selection. Defaults to CONTENT_HIGH.

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
        if task_type == TaskType.CONTENT_MINIMAL:
            prompt = self._build_simple_prompt(title, content)
        elif task_type == TaskType.CONTENT_LOW:
            prompt = self._build_light_prompt(title, content)
        else:
            prompt = self._build_prompt(title, content)

        start_time = time.time()
        success, response_text, error = self._call_api(prompt)
        duration_ms = int((time.time() - start_time) * 1000)
        ai_call_type = self._get_ai_call_type(task_type)

        if not success:
            record_ai_call(
                call_type=ai_call_type,
                cached=False,
                input_chars=len(prompt),
                output_chars=0,
                duration_ms=duration_ms,
                success=False,
                error=error[:100] if error else "Unknown error",
            )
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message=f"Ollama API error: {error}",
                raw_response=response_text,
            )

        processing_result = self._parse_json_response(response_text)

        record_ai_call(
            call_type=ai_call_type,
            cached=False,
            input_chars=len(prompt),
            output_chars=len(response_text),
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

    def process_item(self, item: "ContentItem") -> ProcessingResult:
        """Process a ContentItem with optimized rule-first approach.

        This method tries multiple optimization strategies before falling back to AI:
        1. Skip processing for low-value content (spam, boilerplate, etc.)
        2. Use rule-only processing for high-confidence matches
        3. Select appropriate prompt complexity based on importance estimate
        4. Fall back to AI processing

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
                cached=True,
                input_chars=len(title) + len(content),
            )
            return ProcessingResult(
                summary=summary,
                category=category,
                importance_score=importance,
                success=True,
            )

        # Strategy 2: Try rule-only processing for high-confidence matches
        rule_result = try_rule_only_processing(item)
        if rule_result:
            logger.info(f"Rule-only processing: {title[:40]}... -> {rule_result.category}")
            record_ai_call(
                call_type=AICallType.CONTENT_LIGHT,
                cached=True,
                input_chars=len(title) + len(content),
            )
            return rule_result

        # Strategy 3: Estimate importance for prompt selection
        importance_est = estimate_importance(item)
        rule_settings = self.settings.rule_optimization

        # Select task type based on importance estimate
        if importance_est.confidence >= 0.7 and importance_est.score >= 7:
            task_type = TaskType.CONTENT_HIGH
        elif (
            rule_settings.minimal_prompt_enabled
            and importance_est.confidence >= 0.7
            and importance_est.score <= rule_settings.minimal_prompt_threshold
        ):
            task_type = TaskType.CONTENT_MINIMAL
        elif importance_est.confidence >= 0.7 and importance_est.score <= 5:
            task_type = TaskType.CONTENT_LOW
        elif importance_est.confidence < 0.5:
            task_type = TaskType.CONTENT_UNCERTAIN
        else:
            task_type = TaskType.CONTENT_LOW

        logger.debug(
            f"Importance estimate: score={importance_est.score}, "
            f"confidence={importance_est.confidence:.2f}, "
            f"reason={importance_est.reason} -> {task_type.name}"
        )

        # Strategy 4: Fall back to AI processing
        return self.process_content(title, content, task_type)

    def process_batch(
        self, items: list[tuple[int, str, str]], task_type: Optional[TaskType] = None
    ) -> list[ProcessingResult]:
        """Process multiple items with cache support.

        Unlike CLI processors, Ollama processes items individually but still
        benefits from caching. For true batching, consider using /api/chat
        with context, but individual calls work well for most use cases.

        Args:
            items: List of (id, title, content) tuples.
            task_type: Task type for prompt selection. Defaults to CONTENT_HIGH.

        Returns:
            List of ProcessingResult, one per item.
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
                (i, orig_id, title, content)
                for i, (orig_id, title, content) in enumerate(items)
            ]

        # All items were cached
        if not items_to_process:
            logger.info(f"Batch: all {len(items)} items from cache")
            return results

        cache_hits = len(items) - len(items_to_process)
        if cache_hits > 0:
            logger.info(f"Batch: {cache_hits} cache hits, {len(items_to_process)} to process")

        # 2. Process uncached items individually
        for result_idx, orig_id, title, content in items_to_process:
            proc_result = self.process_content(title, content, task_type)
            results[result_idx] = proc_result

            # Cache is handled in process_content, no need to duplicate

        return results

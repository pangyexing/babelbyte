"""Ollama API wrapper for local LLM processing.

Uses Ollama's HTTP API (default: http://localhost:11434) for content processing
with locally running models.

Supports dual-model mode: set OLLAMA_MODEL_LIGHT for simpler tasks (e.g., 14b),
while OLLAMA_MODEL handles complex tasks (e.g., 32b).

Two-stage processing (OLLAMA_MODEL_SCREEN):
When configured, uses a lightweight 8B model for initial screening, then
refines high-importance or uncertain items with the 32B model. Expected to
save 20-40% processing time while maintaining quality for important content.
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

    Supports dual-model mode when OLLAMA_MODEL_LIGHT is configured:
    - Heavy model (OLLAMA_MODEL): complex tasks, high importance content
    - Light model (OLLAMA_MODEL_LIGHT): simple tasks, low importance content
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        model_light: Optional[str] = None,
        model_screen: Optional[str] = None,
        timeout: Optional[int] = None,
        db: Optional["SyncDatabase"] = None,
    ):
        """Initialize OllamaAPI.

        Args:
            base_url: Ollama API base URL. Defaults to config or http://localhost:11434.
            model: Heavy model name. Defaults to config or qwen3:32b.
            model_light: Light model name. Defaults to config (empty = single model mode).
            model_screen: Screen model name for two-stage processing (e.g., qwen3:8b).
            timeout: Request timeout in seconds. Defaults to config or 120.
            db: Optional database instance for cache support.
        """
        self._settings = None
        self.db = db

        settings = self.settings
        self._base_url = base_url or settings.ollama.base_url
        self._model = model or settings.ollama.model
        self._model_light = model_light or settings.ollama.model_light
        self._model_screen = model_screen or settings.ollama.model_screen
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

    @property
    def two_stage_enabled(self) -> bool:
        """Check if two-stage processing is enabled."""
        return bool(self._model_screen and self._model_screen != self._model)

    # Simplified screening prompt - only outputs importance + category
    # fmt: off
    SCREEN_PROMPT = """分析内容，仅输出JSON：
{{"importance": 1-10, "category": "AI/机器学习/编程/技术/创业/研究/设计/其他/不确定"}}

评分:
9-10: 重大发布(GPT-5/Claude-4级)、突破性论文、行业变革
7-8: AI公司官宣、知名人物观点、重要开源、大额融资
5-6: 技术教程、一般研究、产品更新、行业分析
3-4: 普通讨论、转载、个人项目
1-2: 招聘、求助、水帖、广告

标题: {title}
内容: {content}"""
    # fmt: on

    def _build_screen_prompt(self, title: str, content: str) -> str:
        """Build simplified screening prompt for 8B model.

        Only extracts importance score and category for quick triage.
        Content is truncated to 500 chars for faster processing.

        Args:
            title: Content title.
            content: Content body.

        Returns:
            Formatted screening prompt.
        """
        # Shorter truncation for screening
        max_title = 100
        max_content = 500

        if len(title) > max_title:
            title = self._smart_truncate(title, max_title)
        if len(content) > max_content:
            content = self._smart_truncate(content, max_content)

        return self.SCREEN_PROMPT.format(title=title, content=content)

    def _parse_screen_response(self, response: str) -> tuple[int, str]:
        """Parse screening response for importance and category.

        Args:
            response: Raw response from 8B model.

        Returns:
            Tuple of (importance_score, category). Defaults to (5, "不确定") on error.
        """
        try:
            # Clean up response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                start_idx = 1 if lines[0].startswith("```") else 0
                end_idx = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == "```":
                        end_idx = i
                        break
                response = "\n".join(lines[start_idx:end_idx])

            # Find JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                response = response[json_start:json_end]

            data = json.loads(response)
            importance = data.get("importance", 5)
            category = data.get("category", "不确定")

            # Validate importance
            if not isinstance(importance, int):
                try:
                    importance = int(importance)
                except (ValueError, TypeError):
                    importance = 5
            importance = max(1, min(10, importance))

            return importance, category

        except (json.JSONDecodeError, KeyError, TypeError):
            return 5, "不确定"

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
        """Get the model name based on task type.

        If OLLAMA_MODEL_LIGHT is configured, uses light model for simple tasks.
        Otherwise falls back to single model mode.

        Args:
            task_type: Task type for model selection.

        Returns:
            The appropriate model name.
        """
        if not self._model_light:
            return self._model

        if task_type in (TaskType.CONTENT_LOW, TaskType.CONTENT_MINIMAL):
            return self._model_light

        return self._model

    def _call_api(self, prompt: str, model: Optional[str] = None) -> tuple[bool, str, Optional[str]]:
        """Send a request to Ollama's generate API.

        Args:
            prompt: The prompt to send.
            model: Model to use. Defaults to heavy model.

        Returns:
            Tuple of (success, response_text, error_message).
        """
        use_model = model or self._model
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": use_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",  # Keep model loaded for 10 minutes (longer for dual-model)
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
        """Map task type to AI call type for tracking."""
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

        # Select prompt and model based on task type
        if task_type == TaskType.CONTENT_MINIMAL:
            prompt = self._build_simple_prompt(title, content)
        elif task_type == TaskType.CONTENT_LOW:
            prompt = self._build_light_prompt(title, content)
        else:
            prompt = self._build_prompt(title, content)

        model = self.get_model(task_type)
        start_time = time.time()
        success, response_text, error = self._call_api(prompt, model=model)
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
        min_conf = rule_settings.minimal_prompt_min_confidence

        # Select task type based on importance estimate
        # High importance → heavy model
        if importance_est.score >= 7:
            task_type = TaskType.CONTENT_HIGH
        # Quality sources (reddit/twitter/hn) → LIGHT_PROMPT (keeps key_points)
        elif importance_est.reason.startswith("source:"):
            task_type = TaskType.CONTENT_LOW
        # Low importance with sufficient confidence → minimal prompt
        elif (
            rule_settings.minimal_prompt_enabled
            and importance_est.confidence >= min_conf
            and importance_est.score <= rule_settings.minimal_prompt_threshold
        ):
            task_type = TaskType.CONTENT_MINIMAL
        elif importance_est.confidence >= min_conf and importance_est.score <= 5:
            task_type = TaskType.CONTENT_LOW
        # Low confidence → conservative (heavy model)
        elif importance_est.confidence < min_conf:
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

    def process_items_two_stage(
        self, items: list["ContentItem"]
    ) -> list[ProcessingResult]:
        """Two-stage batch processing: 8B screening + 32B refinement.

        This method implements an optimized processing pipeline:
        1. Rule filtering: Skip low-value content (zero cost)
        2. Rule-only processing: High-confidence matches (zero AI cost)
        3. 8B screening: Quick importance/category triage for remaining items
        4. 32B refinement: Full analysis only for important/uncertain items

        Expected to save 20-40% processing time while maintaining quality.

        Args:
            items: List of ContentItems to process.

        Returns:
            List of ProcessingResults in same order as input.
        """
        if not items:
            return []

        if not self.two_stage_enabled:
            # Fall back to standard processing if two-stage not configured
            logger.info("Two-stage disabled, using standard processing")
            return [self.process_item(item) for item in items]

        results: list[Optional[ProcessingResult]] = [None] * len(items)
        to_screen: list[tuple[int, "ContentItem"]] = []

        # ========================================
        # Stage 0: Rule-based filtering (zero cost)
        # ========================================
        skipped_count = 0
        rule_only_count = 0

        for i, item in enumerate(items):
            title = item.title or ""
            content = item.content or ""

            # Strategy 1: Skip low-value content entirely
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
                skipped_count += 1
                continue

            # Strategy 2: Rule-only processing for high-confidence matches
            rule_result = try_rule_only_processing(item)
            if rule_result:
                results[i] = rule_result
                record_ai_call(
                    call_type=AICallType.CONTENT_LIGHT,
                    cached=True,
                    input_chars=len(title) + len(content),
                )
                rule_only_count += 1
                continue

            # Needs AI screening
            to_screen.append((i, item))

        if skipped_count > 0 or rule_only_count > 0:
            logger.info(
                f"Two-stage pre-filter: {skipped_count} skipped, "
                f"{rule_only_count} rule-only, {len(to_screen)} to screen"
            )

        if not to_screen:
            return results

        # ========================================
        # Stage 1: 8B batch screening
        # ========================================
        screen_results: list[tuple[int, str]] = []  # (importance, category)
        start_time = time.time()

        for idx, item in to_screen:
            title = item.title or ""
            content = item.content or ""
            prompt = self._build_screen_prompt(title, content)

            success, response_text, error = self._call_api(prompt, model=self._model_screen)
            if success:
                importance, category = self._parse_screen_response(response_text)
            else:
                logger.warning(f"Screen failed for {title[:30]}...: {error}")
                importance, category = 5, "不确定"

            screen_results.append((importance, category))

        screen_duration = time.time() - start_time
        logger.info(
            f"Two-stage screening: {len(to_screen)} items in {screen_duration:.1f}s "
            f"with {self._model_screen}"
        )

        # ========================================
        # Stage 2: Collect items needing 32B refinement
        # ========================================
        to_heavy: list[tuple[int, "ContentItem"]] = []
        heavy_threshold = self.settings.ai.model_tiers.heavy_threshold  # default 7

        for (idx, item), (importance, category) in zip(to_screen, screen_results):
            title = item.title or ""
            content = item.content or ""

            # Upgrade conditions: high importance OR uncertain category
            needs_upgrade = importance >= heavy_threshold or category == "不确定"

            if needs_upgrade:
                to_heavy.append((idx, item))
            else:
                # Use screening result directly - create lightweight ProcessingResult
                # For non-upgraded items, we use the light prompt to get full result
                # but we know it's low importance so we use light model
                results[idx] = self.process_content(title, content, TaskType.CONTENT_LOW)

        logger.info(
            f"Two-stage upgrade: {len(to_heavy)}/{len(to_screen)} items "
            f"need 32B refinement (threshold: importance >= {heavy_threshold} or category='不确定')"
        )

        # ========================================
        # Stage 3: 32B batch refinement
        # ========================================
        if to_heavy:
            start_time = time.time()
            for idx, item in to_heavy:
                title = item.title or ""
                content = item.content or ""
                # Full analysis with heavy model
                results[idx] = self.process_content(title, content, TaskType.CONTENT_HIGH)

            heavy_duration = time.time() - start_time
            logger.info(
                f"Two-stage refinement: {len(to_heavy)} items in {heavy_duration:.1f}s "
                f"with {self._model}"
            )

        return results

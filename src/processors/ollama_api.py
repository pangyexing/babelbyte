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

    # Simplified screening prompt - outputs importance, category, and confidence
    # fmt: off
    SCREEN_PROMPT = """分析内容，仅输出JSON：
{{"importance": 1-10, "category": "只选一个", "confidence": "high/medium/low"}}

category必须是以下之一：AI、机器学习、编程、技术、创业、创新、金融、研究、设计、其他

评分:
9-10: 重大发布(GPT-5/Claude-4级)、突破性论文、行业变革、重大政策
7-8: AI公司官宣、知名人物观点、重要开源、大额融资、产品发布、商业模式创新
5-6: 技术教程、一般研究、产品更新、行业分析、市场动态、投资趋势
3-4: 普通讨论、转载、个人项目、一般财经新闻
1-2: 招聘、求助、水帖、广告

分类说明:
- 创业: 融资、初创公司、商业模式、创业故事
- 创新: 新产品发布、颠覆性技术、行业变革、专利突破
- 金融: 投资、股市、加密货币、经济政策、金融科技

confidence规则:
- high: 关键词明确匹配分类
- medium: 有一定线索但不完全确定
- low: 内容模糊难以判断

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

    # Valid categories for normalization
    VALID_CATEGORIES = {"AI", "机器学习", "编程", "技术", "创业", "创新", "金融", "研究", "设计", "其他"}

    def _normalize_category(self, category: str) -> str:
        """Normalize category to one of the valid categories."""
        if not category:
            return "其他"
        if category in self.VALID_CATEGORIES:
            return category
        # Handle composite categories
        for sep in ["/", "、", ",", "，"]:
            if sep in category:
                for part in category.split(sep):
                    part = part.strip()
                    if part in self.VALID_CATEGORIES:
                        return part
        # Partial match
        for valid_cat in self.VALID_CATEGORIES:
            if valid_cat in category:
                return valid_cat
        return "其他"

    def _parse_screen_response(self, response: str) -> tuple[int, str, str]:
        """Parse screening response for importance, category, and confidence.

        Args:
            response: Raw response from 8B model.

        Returns:
            Tuple of (importance_score, category, confidence).
            Defaults to (5, "其他", "low") on error.
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
            raw_category = data.get("category", "其他")
            category = self._normalize_category(raw_category)
            confidence = data.get("confidence", "low")

            # Validate confidence
            if confidence not in ("high", "medium", "low"):
                confidence = "low"

            # Handle legacy "不确定" category - convert to "其他" with low confidence
            if category == "不确定":
                category = "其他"
                confidence = "low"

            # Validate importance
            if not isinstance(importance, int):
                try:
                    importance = int(importance)
                except (ValueError, TypeError):
                    importance = 5
            importance = max(1, min(10, importance))

            return importance, category, confidence

        except (json.JSONDecodeError, KeyError, TypeError):
            return 5, "其他", "low"

    def _build_screen_result(self, title: str, category: str, importance: int) -> ProcessingResult:
        """Build a lightweight ProcessingResult from screening data.

        Used for low-importance items that don't need full AI processing.
        Generates summary and one_liner from title, uses screening category/importance.

        Args:
            title: Content title.
            category: Category from 8B screening.
            importance: Importance score from 8B screening.

        Returns:
            ProcessingResult with basic fields populated.
        """
        # Generate summary from title (truncate if needed)
        summary = title[:100] if len(title) > 100 else title

        # Generate one_liner with category prefix
        category_prefixes = {
            "AI": "AI动态",
            "机器学习": "ML资讯",
            "编程": "开发资讯",
            "技术": "技术更新",
            "创业": "创业动态",
            "创新": "创新资讯",
            "金融": "金融快讯",
            "研究": "研究进展",
            "设计": "设计资讯",
        }
        prefix = category_prefixes.get(category, "资讯")
        one_liner = f"{prefix}：{title[:50]}" if len(title) > 50 else f"{prefix}：{title}"

        return ProcessingResult(
            summary=summary,
            category=category,
            importance_score=importance,
            success=True,
            one_liner=one_liner,
            key_points=[],
            impact_assessment=None,
            actionable_items=[],
        )

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

        # Strategy 2: Estimate importance for prompt selection
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
        # Note: Similar content dedup and importance estimation are done
        # externally in digest_processor.py before calling this method.
        # ========================================
        skipped_count = 0

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

            # Needs AI screening
            to_screen.append((i, item))

        if skipped_count > 0:
            logger.info(
                f"Two-stage pre-filter: {skipped_count} skipped, {len(to_screen)} to screen"
            )

        if not to_screen:
            return results

        # ========================================
        # Stage 1: 8B batch screening (with cache)
        # ========================================
        screen_results: list[tuple[int, str, str]] = []  # (importance, category, confidence)
        start_time = time.time()
        screen_cache_hits = 0

        for idx, item in to_screen:
            title = item.title or ""
            content = item.content or ""

            # Check screen cache first
            screen_cache_key = None
            if self.settings.ai.cache_enabled and self.db:
                screen_cache_key = f"screen:{self._get_content_hash(title, content[:500])}"
                cached_screen = self.db.get_ai_cache(screen_cache_key)
                if cached_screen:
                    try:
                        cached_data = json.loads(cached_screen)
                        importance = cached_data.get("importance", 5)
                        category = cached_data.get("category", "其他")
                        confidence = cached_data.get("confidence", "low")
                        screen_results.append((importance, category, confidence))
                        screen_cache_hits += 1
                        continue
                    except (json.JSONDecodeError, KeyError):
                        pass  # Cache miss, proceed with API call

            # Call 8B model for screening
            prompt = self._build_screen_prompt(title, content)
            success, response_text, error = self._call_api(prompt, model=self._model_screen)
            if success:
                importance, category, confidence = self._parse_screen_response(response_text)
                # Store in cache
                if screen_cache_key and self.db:
                    cache_data = json.dumps({
                        "importance": importance,
                        "category": category,
                        "confidence": confidence,
                    })
                    self.db.set_ai_cache(screen_cache_key, cache_data, self.settings.ai.cache_ttl)
            else:
                logger.warning(f"Screen failed for {title[:30]}...: {error}")
                importance, category, confidence = 5, "其他", "low"

            screen_results.append((importance, category, confidence))

        screen_duration = time.time() - start_time
        screen_api_calls = len(to_screen) - screen_cache_hits
        logger.info(
            f"Two-stage screening: {len(to_screen)} items in {screen_duration:.1f}s "
            f"({screen_cache_hits} cache hits, {screen_api_calls} API calls) "
            f"with {self._model_screen}"
        )

        # ========================================
        # Stage 2: Collect items needing 32B refinement
        # ========================================
        to_heavy: list[tuple[int, "ContentItem"]] = []
        heavy_threshold = self.settings.ai.model_tiers.heavy_threshold  # default 8
        low_conf_threshold = self.settings.ai.model_tiers.low_confidence_threshold  # default 5

        for (idx, item), (importance, category, confidence) in zip(to_screen, screen_results):
            title = item.title or ""
            content = item.content or ""

            # Layered upgrade decision logic:
            # | Importance          | Confidence   | Action                    |
            # |---------------------|--------------|---------------------------|
            # | >= heavy_threshold  | any          | 32B refinement            |
            # | >= low_conf_thresh  | low          | 32B refinement            |
            # | otherwise           | any          | use screen result (no AI) |
            #
            # With heavy_threshold=8, low_conf_threshold=6:
            # - 8+ always upgrade
            # - 6-7 with low confidence upgrade
            # - 5 and below use 8B result
            if importance >= heavy_threshold:
                needs_upgrade = True
            elif importance >= low_conf_threshold and confidence == "low":
                needs_upgrade = True
            else:
                needs_upgrade = False

            if needs_upgrade:
                to_heavy.append((idx, item))
            else:
                # Use screening result directly - no additional AI call
                results[idx] = self._build_screen_result(title, category, importance)

        upgrade_pct = len(to_heavy) / len(to_screen) * 100 if to_screen else 0
        logger.info(
            f"Two-stage upgrade: {len(to_heavy)}/{len(to_screen)} items ({upgrade_pct:.1f}%) "
            f"need 32B refinement"
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

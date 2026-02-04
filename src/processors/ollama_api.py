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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

    # num_predict settings per task type
    # Thinking mode is disabled for screen/simple/light (faster, fewer tokens)
    # Heavy: configurable via OLLAMA_THINKING_ENABLED
    # Actual usage: screen ~25, simple ~37, light ~151, heavy ~345 (no-think) / ~928 (think)
    NUM_PREDICT = {
        "screen": 128,   # 8B筛选 (no-think, ~25 tokens)
        "simple": 128,   # 最简分析 (no-think, ~37 tokens)
        "light": 256,    # 轻量分析 (no-think, ~151 tokens)
        "heavy": 512,    # 32B分析 (no-think ~345, think需额外空间见下方)
    }

    # Extra tokens needed when thinking is enabled (heavy tasks only)
    THINKING_EXTRA_TOKENS = 768  # thinking ~600 + buffer

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
        self._session: Optional[requests.Session] = None
        self._session_lock = threading.Lock()
        self._parallel_warning_shown = False

        settings = self.settings
        self._base_url = base_url or settings.ollama.base_url
        self._model = model or settings.ollama.model
        self._model_light = model_light or settings.ollama.model_light
        self._model_screen = model_screen or settings.ollama.model_screen
        self._timeout = timeout or settings.ollama.timeout
        self._keep_alive = settings.ollama.keep_alive

        # Check Ollama parallel config on init if concurrent enabled
        if settings.ollama.concurrent_enabled:
            self._check_ollama_parallel_config()

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
    def session(self) -> requests.Session:
        """Get or create a thread-safe requests session with connection pooling.

        Uses lazy initialization with locking for thread safety.
        Connection pool is sized for concurrent processing.
        """
        if self._session is None:
            with self._session_lock:
                # Double-check locking pattern
                if self._session is None:
                    session = requests.Session()
                    # Configure connection pooling based on max workers
                    max_workers = max(
                        self.settings.ollama.workers_heavy,
                        self.settings.ollama.workers_screen,
                    )
                    adapter = HTTPAdapter(
                        pool_connections=max_workers + 2,
                        pool_maxsize=max_workers + 2,
                        max_retries=Retry(total=2, backoff_factor=0.5),
                    )
                    session.mount("http://", adapter)
                    session.mount("https://", adapter)
                    self._session = session
        return self._session

    @property
    def two_stage_enabled(self) -> bool:
        """Check if two-stage processing is enabled."""
        return bool(self._model_screen and self._model_screen != self._model)

    @property
    def concurrent_enabled(self) -> bool:
        """Check if concurrent processing is enabled."""
        return self.settings.ollama.concurrent_enabled

    def _check_ollama_parallel_config(self) -> None:
        """Check if Ollama server has parallel processing enabled.

        Ollama requires OLLAMA_NUM_PARALLEL environment variable to be set
        on the SERVER side to actually process requests in parallel.
        This method checks and warns if parallel is likely not configured.
        """
        # Check if warning already shown
        if self._parallel_warning_shown:
            return
        self._parallel_warning_shown = True

        # Get configured workers
        workers_heavy = self.settings.ollama.workers_heavy
        workers_screen = self.settings.ollama.workers_screen
        max_workers = max(workers_heavy, workers_screen)

        if max_workers <= 1:
            return  # No parallel needed

        logger.warning(
            f"Ollama concurrent processing enabled (workers: {max_workers}). "
            f"Ensure Ollama server has OLLAMA_NUM_PARALLEL={max_workers} set. "
            f"Without it, requests will be queued and processed serially. "
            f"To enable: export OLLAMA_NUM_PARALLEL={max_workers} && ollama serve"
        )

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
    VALID_CATEGORIES = {
        "AI", "机器学习", "编程", "技术", "创业", "创新", "金融", "研究", "设计", "其他"
    }

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
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except (requests.RequestException, OSError):
            return False

    def unload_model(self, model: Optional[str] = None) -> bool:
        """Unload a model from GPU memory.

        Sends a request with keep_alive=0 to immediately release GPU memory.

        Args:
            model: Model name to unload. If None, unloads the heavy model.

        Returns:
            True if successful, False otherwise.
        """
        use_model = model or self._model
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": use_model,
            "prompt": "",
            "keep_alive": 0,  # Immediately unload
        }

        try:
            response = self.session.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Unloaded model: {use_model}")
                return True
            else:
                logger.warning(f"Failed to unload {use_model}: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.warning(f"Error unloading {use_model}: {e}")
            return False

    def unload_all_models(self) -> None:
        """Unload all models used in two-stage processing.

        Releases GPU memory for heavy, light, and screen models.
        """
        models_to_unload = set()

        # Add all configured models
        if self._model:
            models_to_unload.add(self._model)
        if self._model_light:
            models_to_unload.add(self._model_light)
        if self._model_screen:
            models_to_unload.add(self._model_screen)

        for model in models_to_unload:
            self.unload_model(model)

        logger.info(f"Unloaded {len(models_to_unload)} models from GPU memory")

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

    def _call_api(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        disable_thinking: bool = False,
    ) -> tuple[bool, str, Optional[str]]:
        """Send a request to Ollama's generate API.

        Uses requests.Session with connection pooling for better performance
        in concurrent processing scenarios.

        Args:
            prompt: The prompt to send.
            model: Model to use. Defaults to heavy model.
            max_tokens: Maximum tokens to generate. Defaults to NUM_PREDICT["heavy"].
            disable_thinking: If True, disable qwen3's thinking mode for faster responses.
                Use this for simple tasks like screening that don't need reasoning.

        Returns:
            Tuple of (success, response_text, error_message).
        """
        use_model = model or self._model
        url = f"{self.base_url}/api/generate"
        # Use configurable keep_alive time (in seconds, convert to string format)
        keep_alive_str = f"{self._keep_alive}s"
        # Use provided max_tokens or default to heavy (768)
        num_predict = max_tokens if max_tokens is not None else self.NUM_PREDICT["heavy"]
        payload = {
            "model": use_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive_str,
            "options": {
                "temperature": 0.3,
                "num_predict": num_predict,
            },
        }

        # Disable thinking mode for qwen3 models (faster, fewer tokens)
        if disable_thinking:
            payload["think"] = False

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            result = response.json()
            return True, result.get("response", ""), None

        except requests.HTTPError as e:
            error_body = e.response.text if e.response else str(e)
            return False, "", f"HTTP {e.response.status_code}: {error_body}"
        except requests.ConnectionError as e:
            return False, "", f"Connection error: {e}"
        except requests.Timeout:
            return False, "", f"Request timeout after {self._timeout}s"
        except json.JSONDecodeError as e:
            return False, "", f"Invalid JSON response: {e}"
        except requests.RequestException as e:
            return False, "", f"Request error: {e}"

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
        self,
        title: str,
        content: str,
        task_type: Optional[TaskType] = None,
        _use_db_cache: bool = True,
    ) -> ProcessingResult:
        """Process content using Ollama API.

        Args:
            title: The title of the content.
            content: The main content text.
            task_type: Task type for prompt selection. Defaults to CONTENT_HIGH.
            _use_db_cache: Internal flag to control DB cache access.
                Set to False in parallel processing to avoid asyncio conflicts.

        Returns:
            ProcessingResult with summary, category, and importance score.
        """
        if task_type is None:
            task_type = TaskType.CONTENT_HIGH

        # Check cache if enabled (skip in parallel mode to avoid DB conflicts)
        content_hash = None
        use_cache = _use_db_cache and self.settings.ai.cache_enabled and self.db
        if use_cache:
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

        # Select prompt, model, num_predict, and thinking mode based on task type
        # Simple/Light tasks don't need thinking mode (faster, fewer tokens)
        # Heavy tasks: controlled by OLLAMA_THINKING_ENABLED setting
        if task_type == TaskType.CONTENT_MINIMAL:
            prompt = self._build_simple_prompt(title, content)
            max_tokens = self.NUM_PREDICT["simple"]
            disable_thinking = True  # Simple classification, no reasoning needed
        elif task_type == TaskType.CONTENT_LOW:
            prompt = self._build_light_prompt(title, content)
            max_tokens = self.NUM_PREDICT["light"]
            disable_thinking = True  # Light analysis, key_points don't need deep reasoning
        else:
            prompt = self._build_prompt(title, content)
            # Heavy analysis: use setting to control thinking mode
            # Default: disabled for speed (~18s vs ~43s), same quality output
            disable_thinking = not self.settings.ollama.thinking_enabled
            # Add extra tokens when thinking is enabled (thinking uses ~600+ tokens)
            max_tokens = self.NUM_PREDICT["heavy"]
            if not disable_thinking:
                max_tokens += self.THINKING_EXTRA_TOKENS

        model = self.get_model(task_type)
        start_time = time.time()
        success, response_text, error = self._call_api(
            prompt, model=model, max_tokens=max_tokens, disable_thinking=disable_thinking
        )
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

        # Store in cache if successful (skip in parallel mode)
        if processing_result.success and use_cache:
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

    def process_items_concurrent(
        self,
        items: list["ContentItem"],
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        unload_after: bool = True,
    ) -> list[ProcessingResult]:
        """Process ContentItems concurrently using ThreadPoolExecutor.

        This method provides significant speedup by processing multiple items
        in parallel, better utilizing GPU resources during I/O wait times.

        Args:
            items: List of ContentItems to process.
            max_workers: Number of parallel workers. Defaults to workers_heavy setting.
            progress_callback: Optional callback(current, total) for progress updates.
            unload_after: If True, unload models from GPU after processing (default True).

        Returns:
            List of ProcessingResults in same order as input items.
        """
        if not items:
            return []

        # Fall back to sequential processing if concurrent is disabled
        if not self.concurrent_enabled:
            logger.info("Concurrent processing disabled, using sequential")
            results = []
            for i, item in enumerate(items):
                results.append(self.process_item(item))
                if progress_callback:
                    progress_callback(i + 1, len(items))
            return results

        # Use configured workers or default
        if max_workers is None:
            max_workers = self.settings.ollama.workers_heavy

        total = len(items)
        results: list[Optional[ProcessingResult]] = [None] * total
        completed = 0
        lock = threading.Lock()

        def process_single(idx_item: tuple[int, "ContentItem"]) -> tuple[int, ProcessingResult]:
            """Process a single item and return (index, result)."""
            idx, item = idx_item
            return idx, self.process_item(item)

        logger.info(f"Processing {total} items concurrently with {max_workers} workers")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with their indices
            futures = {
                executor.submit(process_single, (idx, item)): idx
                for idx, item in enumerate(items)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = futures[future]
                    logger.warning(f"Error processing item at index {idx}: {e}")
                    # Create error result
                    results[idx] = ProcessingResult(
                        summary="",
                        category="其他",
                        importance_score=5,
                        success=False,
                        error_message=str(e),
                    )

                with lock:
                    completed += 1
                    current = completed

                if progress_callback:
                    progress_callback(current, total)

        duration = time.time() - start_time
        logger.info(
            f"Concurrent processing completed: {total} items in {duration:.1f}s "
            f"({duration/total:.2f}s/item avg, {max_workers} workers)"
        )

        # Unload models to release GPU memory
        if unload_after:
            self.unload_all_models()

        return results

    def process_items_two_stage(
        self,
        items: list["ContentItem"],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        phase_callback: Optional[Callable[[str, int], None]] = None,
        unload_after: bool = True,
    ) -> list[ProcessingResult]:
        """Two-stage batch processing: 8B screening + 32B refinement.

        This method implements an optimized processing pipeline with optional
        concurrent processing for better GPU utilization:
        1. Rule filtering: Skip low-value content (zero cost)
        2. Rule-only processing: High-confidence matches (zero AI cost)
        3. 8B screening: Quick importance/category triage (parallel if enabled)
        4. 32B refinement: Full analysis for important/uncertain items (parallel if enabled)

        Expected to save 20-40% processing time while maintaining quality.
        With concurrent processing enabled, can achieve 2-2.5x additional speedup.

        Args:
            items: List of ContentItems to process.
            progress_callback: Optional callback(current, total) for progress updates.
            phase_callback: Optional callback(phase_name, phase_total) for phase transitions.
                Called when entering a new processing phase (e.g., "8B screening", "32B refinement").
            unload_after: If True, unload models from GPU after processing (default True).

        Returns:
            List of ProcessingResults in same order as input.
        """
        if not items:
            return []

        if not self.two_stage_enabled:
            # Fall back to concurrent processing if two-stage not configured
            logger.info("Two-stage disabled, using standard processing")
            if self.concurrent_enabled:
                return self.process_items_concurrent(
                    items, progress_callback=progress_callback, unload_after=unload_after
                )
            results = [self.process_item(item) for item in items]
            if unload_after:
                self.unload_all_models()
            return results

        total_items = len(items)
        results: list[Optional[ProcessingResult]] = [None] * total_items
        to_screen: list[tuple[int, "ContentItem"]] = []

        # Thread lock for concurrent progress updates
        lock = threading.Lock()

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
        # Stage 1: 8B batch screening (with cache, optionally parallel)
        # ========================================
        # idx -> (importance, category, confidence)
        screen_results: dict[int, tuple[int, str, str]] = {}
        start_time = time.time()
        screen_cache_hits = 0
        to_screen_api: list[tuple[int, "ContentItem", str]] = []  # items needing API call

        # First pass: check cache
        for idx, item in to_screen:
            title = item.title or ""
            content = item.content or ""

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
                        screen_results[idx] = (importance, category, confidence)
                        screen_cache_hits += 1
                        continue
                    except (json.JSONDecodeError, KeyError):
                        pass  # Cache miss, proceed with API call

            to_screen_api.append((idx, item, screen_cache_key or ""))

        # Second pass: API calls (sequential or parallel based on setting)
        # Cache writes are collected and done in main thread to avoid asyncio issues
        cache_to_write: list[tuple[str, str]] = []  # (cache_key, cache_data)

        # Notify phase transition for 8B screening
        if to_screen_api and phase_callback:
            phase_callback("8B screening", len(to_screen_api))

        if to_screen_api:
            if self.concurrent_enabled and len(to_screen_api) > 1:
                # Parallel 8B screening - cache writes deferred to main thread
                screen_workers = self.settings.ollama.workers_screen

                def screen_single(
                    args: tuple[int, "ContentItem", str]
                ) -> tuple[int, str, tuple[int, str, str]]:
                    """Screen a single item, return (idx, cache_key, result)."""
                    idx, item, cache_key = args
                    title = item.title or ""
                    content = item.content or ""

                    prompt = self._build_screen_prompt(title, content)
                    success, response_text, error = self._call_api(
                        prompt,
                        model=self._model_screen,
                        max_tokens=self.NUM_PREDICT["screen"],
                        disable_thinking=True,  # Fast screening without reasoning
                    )

                    if success:
                        imp, cat, conf = self._parse_screen_response(response_text)
                    else:
                        logger.warning(f"Screen failed for {title[:30]}...: {error}")
                        imp, cat, conf = 5, "其他", "low"

                    return idx, cache_key, (imp, cat, conf)

                logger.info(
                    f"Parallel 8B screening: {len(to_screen_api)} items "
                    f"with {screen_workers} workers"
                )

                screen_done = 0
                screen_total = len(to_screen_api)
                with ThreadPoolExecutor(max_workers=screen_workers) as executor:
                    futures = {
                        executor.submit(screen_single, args): args[0]
                        for args in to_screen_api
                    }
                    for future in as_completed(futures):
                        try:
                            idx, cache_key, (imp, cat, conf) = future.result()
                            screen_results[idx] = (imp, cat, conf)
                            # Collect cache writes for main thread
                            if cache_key and self.db:
                                cache_data = json.dumps({
                                    "importance": imp,
                                    "category": cat,
                                    "confidence": conf,
                                })
                                cache_to_write.append((cache_key, cache_data))
                        except Exception as e:
                            idx = futures[future]
                            logger.warning(f"Screen error for item {idx}: {e}")
                            screen_results[idx] = (5, "其他", "low")

                        # Update progress for 8B screening
                        with lock:
                            screen_done += 1
                            if progress_callback:
                                progress_callback(screen_done, screen_total)
            else:
                # Sequential screening
                screen_done = 0
                screen_total = len(to_screen_api)
                for idx, item, cache_key in to_screen_api:
                    title = item.title or ""
                    content = item.content or ""

                    prompt = self._build_screen_prompt(title, content)
                    success, response_text, error = self._call_api(
                        prompt,
                        model=self._model_screen,
                        max_tokens=self.NUM_PREDICT["screen"],
                        disable_thinking=True,  # Fast screening without reasoning
                    )

                    if success:
                        imp, cat, conf = self._parse_screen_response(response_text)
                        if cache_key and self.db:
                            cache_data = json.dumps({
                                "importance": imp,
                                "category": cat,
                                "confidence": conf,
                            })
                            cache_to_write.append((cache_key, cache_data))
                    else:
                        logger.warning(f"Screen failed for {title[:30]}...: {error}")
                        imp, cat, conf = 5, "其他", "low"

                    screen_results[idx] = (imp, cat, conf)

                    # Update progress for 8B screening
                    screen_done += 1
                    if progress_callback:
                        progress_callback(screen_done, screen_total)

        # Write screen cache in main thread (thread-safe)
        if cache_to_write and self.db:
            ttl = self.settings.ai.cache_ttl
            for cache_key, cache_data in cache_to_write:
                try:
                    self.db.set_ai_cache(cache_key, cache_data, ttl)
                except Exception as e:
                    logger.warning(f"Cache write failed: {e}")

        screen_duration = time.time() - start_time
        screen_api_calls = len(to_screen) - screen_cache_hits
        is_parallel = self.concurrent_enabled and len(to_screen_api) > 1
        parallel_mode = "parallel" if is_parallel else "sequential"
        logger.info(
            f"Two-stage screening ({parallel_mode}): {len(to_screen)} items "
            f"in {screen_duration:.1f}s ({screen_cache_hits} cache hits, "
            f"{screen_api_calls} API calls) with {self._model_screen}"
        )

        # ========================================
        # Stage 2: Collect items needing 32B refinement
        # ========================================
        to_heavy: list[tuple[int, "ContentItem"]] = []
        heavy_threshold = self.settings.ai.model_tiers.heavy_threshold  # default 8
        low_conf_threshold = self.settings.ai.model_tiers.low_confidence_threshold  # default 5

        for idx, item in to_screen:
            importance, category, confidence = screen_results.get(idx, (5, "其他", "low"))
            title = item.title or ""

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
        # Stage 3: 32B batch refinement (sequential or parallel)
        # ========================================
        if to_heavy:
            # Notify phase transition for 32B refinement
            if phase_callback:
                phase_callback("32B refinement", len(to_heavy))

            refine_start = time.time()
            refine_done = 0
            refine_total = len(to_heavy)

            if self.concurrent_enabled and len(to_heavy) > 1:
                # Parallel 32B refinement - disable DB cache to avoid asyncio conflicts
                heavy_workers = self.settings.ollama.workers_heavy

                def refine_single(
                    args: tuple[int, "ContentItem"]
                ) -> tuple[int, str, ProcessingResult]:
                    """Refine a single item with 32B model, return (idx, hash, result)."""
                    idx, item = args
                    title = item.title or ""
                    content = item.content or ""
                    # Disable DB cache in parallel mode
                    result = self.process_content(
                        title, content, TaskType.CONTENT_HIGH, _use_db_cache=False
                    )
                    # Return content hash for main thread cache write
                    content_hash = self._get_content_hash(title, content)
                    return idx, content_hash, result

                logger.info(
                    f"Parallel 32B refinement: {len(to_heavy)} items "
                    f"with {heavy_workers} workers"
                )

                # Collect results for main thread cache write
                heavy_cache_writes: list[tuple[str, ProcessingResult]] = []

                with ThreadPoolExecutor(max_workers=heavy_workers) as executor:
                    futures = {
                        executor.submit(refine_single, args): args[0]
                        for args in to_heavy
                    }
                    for future in as_completed(futures):
                        try:
                            idx, content_hash, result = future.result()
                            results[idx] = result
                            if result.success:
                                heavy_cache_writes.append((content_hash, result))
                        except Exception as e:
                            idx = futures[future]
                            logger.warning(f"Refine error for item {idx}: {e}")
                            results[idx] = ProcessingResult(
                                summary="",
                                category="其他",
                                importance_score=5,
                                success=False,
                                error_message=str(e),
                            )

                        # Update progress for 32B refinement
                        with lock:
                            refine_done += 1
                            if progress_callback:
                                progress_callback(refine_done, refine_total)

                # Write cache in main thread (thread-safe)
                if heavy_cache_writes and self.db and self.settings.ai.cache_enabled:
                    ttl = self.settings.ai.cache_ttl
                    for content_hash, result in heavy_cache_writes:
                        try:
                            self.db.set_ai_cache(
                                content_hash, self._serialize_result(result), ttl
                            )
                        except Exception as e:
                            logger.warning(f"Heavy cache write failed: {e}")
            else:
                # Sequential refinement
                for idx, item in to_heavy:
                    title = item.title or ""
                    content = item.content or ""
                    results[idx] = self.process_content(title, content, TaskType.CONTENT_HIGH)

                    # Update progress for 32B refinement
                    refine_done += 1
                    if progress_callback:
                        progress_callback(refine_done, refine_total)

            heavy_duration = time.time() - refine_start
            is_parallel = self.concurrent_enabled and len(to_heavy) > 1
            parallel_mode = "parallel" if is_parallel else "sequential"
            logger.info(
                f"Two-stage refinement ({parallel_mode}): {len(to_heavy)} items "
                f"in {heavy_duration:.1f}s with {self._model}"
            )

        # Unload models to release GPU memory
        if unload_after:
            self.unload_all_models()

        return results

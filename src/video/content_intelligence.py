"""Content intelligence for video generation.

Provides intelligent template selection and AI-driven script generation
for more natural and engaging video content.
"""

import json
import logging
import re
import time
from typing import Optional

from config.settings import get_settings
from src.analytics.token_tracker import AICallType, record_ai_call
from src.storage.models import ContentItem
from src.video.templates import TemplateType

logger = logging.getLogger(__name__)


def parse_key_points(key_points_json: Optional[str]) -> list[dict]:
    """Parse key_points JSON string into list of dicts.

    Args:
        key_points_json: JSON string like '[{"type": "数字", "value": "100%", "impact": "..."}]'

    Returns:
        List of key point dicts, empty list on error.
    """
    if not key_points_json:
        return []
    try:
        data = json.loads(key_points_json)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def has_numeric_data(key_points: list[dict]) -> bool:
    """Check if key_points contain numeric data suitable for DATA_CARD template.

    Looks for:
    - type == "数字" or "百分比"
    - value containing numbers, percentages, or currency

    Args:
        key_points: List of key point dicts.

    Returns:
        True if numeric data found.
    """
    numeric_patterns = [
        r"\d+%",  # Percentages
        r"\$[\d,]+",  # Dollar amounts
        r"¥[\d,]+",  # Yuan amounts
        r"\d+[亿万千百]",  # Chinese number units
        r"\d+\.\d+[BMK]?",  # Decimal numbers with optional B/M/K suffix
        r"\b\d{2,}\b",  # Numbers with 2+ digits
    ]

    for kp in key_points:
        kp_type = kp.get("type", "")
        kp_value = kp.get("value", "")

        # Check type
        if kp_type in ("数字", "百分比", "金额", "统计"):
            return True

        # Check value for numeric patterns
        for pattern in numeric_patterns:
            if re.search(pattern, kp_value):
                return True

    return False


def select_template(item: ContentItem) -> TemplateType:
    """Select the best video template based on content characteristics.

    Selection logic:
    1. Has numeric data in key_points -> DATA_CARD (data visualization)
    2. Has 3+ key_points -> KEY_POINTS (bullet list format)
    3. Has impact_assessment -> DEEP_ANALYSIS (structured breakdown)
    4. Default -> NEWS_BRIEF (concise news card)

    Args:
        item: ContentItem with AI processing results.

    Returns:
        Recommended TemplateType.
    """
    key_points = parse_key_points(item.key_points)

    # Has numeric/percentage data -> DATA_CARD
    if has_numeric_data(key_points):
        logger.debug("Template selection: DATA_CARD (numeric data found)")
        return TemplateType.DATA_CARD

    # 3+ key points -> KEY_POINTS template
    if len(key_points) >= 3:
        logger.debug(f"Template selection: KEY_POINTS ({len(key_points)} key points)")
        return TemplateType.KEY_POINTS

    # Has impact assessment with meaningful content -> DEEP_ANALYSIS
    if item.impact_assessment:
        try:
            impact = json.loads(item.impact_assessment)
            short_term = impact.get("short_term", "")
            long_term = impact.get("long_term", "")
            # Only use DEEP_ANALYSIS if impact assessment has substantial content
            if len(short_term) > 10 or len(long_term) > 10:
                logger.debug("Template selection: DEEP_ANALYSIS (impact assessment found)")
                return TemplateType.DEEP_ANALYSIS
        except (json.JSONDecodeError, TypeError):
            pass

    # Default to NEWS_BRIEF
    logger.debug("Template selection: NEWS_BRIEF (default)")
    return TemplateType.NEWS_BRIEF


def extract_keywords(item: ContentItem) -> list[str]:
    """Extract keywords from key_points for highlighting.

    Extracts the 'value' field from each key_point.

    Args:
        item: ContentItem with AI processing results.

    Returns:
        List of keyword strings to highlight.
    """
    key_points = parse_key_points(item.key_points)
    keywords = []

    for kp in key_points:
        value = kp.get("value", "")
        if value and len(value) <= 50:  # Reasonable length for highlighting
            keywords.append(value)

    return keywords


class ContentSelector:
    """AI-driven content selection for video generation.

    Evaluates whether content items are suitable for short video broadcast,
    considering factors like newsworthiness, visual appeal, and audience engagement.
    """

    # Prompt for content selection - returns JSON with suitability score and reason
    SELECTION_PROMPT = """判断以下内容是否适合制作成短视频播报（15-60秒），返回JSON：
{{"suitable": true/false, "score": 1-10, "reason": "简短理由"}}

适合短视频的内容特征：
- 新闻性强：产品发布、重大更新、融资消息、行业事件
- 有明确要点：数据、时间、人物、结论
- 话题性强：热点事件、争议话题、突破性进展
- 易于理解：不需要太多背景知识

不适合短视频的内容：
- 纯技术教程、代码讲解
- 招聘信息、广告
- 需要深度阅读的长文分析
- 过于小众或专业的话题
- 旧闻或重复内容

评分标准：
9-10: 非常适合，强新闻性，有明确爆点
7-8: 适合，有新闻价值，要点清晰
5-6: 一般，可以做但不够吸引人
3-4: 不太适合，缺乏新闻性或太专业
1-2: 不适合，招聘/广告/纯技术

标题：{title}
摘要：{summary}
分类：{category}
重要性：{importance}/10
要点：{key_points}

仅返回JSON，无其他文字："""

    def __init__(self, processor=None):
        """Initialize ContentSelector.

        Args:
            processor: Optional AI processor instance. If None, will be created on demand.
        """
        self._processor = processor
        self._settings = None

    @property
    def settings(self):
        """Lazy load settings."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    def _get_processor(self):
        """Get or create AI processor."""
        if self._processor is None:
            from src.processors.digest_processor import get_ai_processor

            self._processor = get_ai_processor()
        return self._processor

    def _format_key_points(self, item: ContentItem) -> str:
        """Format key_points for prompt."""
        key_points = parse_key_points(item.key_points)
        if not key_points:
            return "无"

        formatted = []
        for kp in key_points[:3]:
            value = kp.get("value", "")
            if value:
                formatted.append(f"- {value}")

        return "\n".join(formatted) if formatted else "无"

    def evaluate(self, item: ContentItem) -> tuple[bool, int, str]:
        """Evaluate if content is suitable for video broadcast.

        Args:
            item: ContentItem to evaluate.

        Returns:
            Tuple of (is_suitable, score, reason).
            - is_suitable: True if content should be made into video
            - score: 1-10 suitability score
            - reason: Brief explanation
        """
        # Build prompt
        prompt = self.SELECTION_PROMPT.format(
            title=item.title[:100] if item.title else "无标题",
            summary=item.summary[:150] if item.summary else "无摘要",
            category=item.category or "未分类",
            importance=item.importance_score or 5,
            key_points=self._format_key_points(item),
        )

        processor = self._get_processor()
        start_time = time.time()

        if hasattr(processor, "_call_api"):
            success, response, error = processor._call_api(
                prompt,
                model=processor.model,  # Use 32B model
                max_tokens=128,
                disable_thinking=True,
            )
            duration_ms = int((time.time() - start_time) * 1000)

            record_ai_call(
                call_type=AICallType.CONTENT_HEAVY,
                cached=False,
                input_chars=len(prompt),
                output_chars=len(response) if success else 0,
                duration_ms=duration_ms,
                success=success,
                error=error[:100] if error else None,
            )

            if not success:
                logger.warning(f"Content selection failed: {error}")
                return self._fallback_evaluate(item)

            return self._parse_selection_response(response, item)
        else:
            return self._fallback_evaluate(item)

    def _parse_selection_response(self, response: str, item: ContentItem) -> tuple[bool, int, str]:
        """Parse AI response for selection decision."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(line for line in lines if not line.strip().startswith("```"))

            # Find JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                response = response[json_start:json_end]

            data = json.loads(response)
            suitable = data.get("suitable", False)
            score = data.get("score", 5)
            reason = data.get("reason", "")

            # Validate score
            if not isinstance(score, int):
                score = int(score)
            score = max(1, min(10, score))

            return suitable, score, reason

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse selection response: {e}")
            return self._fallback_evaluate(item)

    def _fallback_evaluate(self, item: ContentItem) -> tuple[bool, int, str]:
        """Rule-based fallback evaluation when AI is unavailable."""
        score = item.importance_score or 5
        category = (item.category or "").lower()

        # Boost for news-worthy categories
        if category in ("ai", "创业", "创新", "金融"):
            score = min(10, score + 1)

        # Check for key points (good for video)
        key_points = parse_key_points(item.key_points)
        if len(key_points) >= 2:
            score = min(10, score + 1)

        # Penalize if no summary
        if not item.summary:
            score = max(1, score - 2)

        suitable = score >= 7
        reason = "基于重要性和分类的规则判断"

        return suitable, score, reason

    def select_batch(
        self,
        items: list[ContentItem],
        min_score: int = 7,
        max_items: int = 10,
    ) -> list[tuple[ContentItem, int, str]]:
        """Select suitable items from a batch for video generation.

        Args:
            items: List of ContentItems to evaluate.
            min_score: Minimum suitability score (default 7).
            max_items: Maximum items to return (default 10).

        Returns:
            List of (item, score, reason) tuples for selected items,
            sorted by score descending.
        """
        results = []

        for item in items:
            suitable, score, reason = self.evaluate(item)
            if suitable and score >= min_score:
                results.append((item, score, reason))
                logger.info(f"Selected: {item.title[:40]}... (score={score}, reason={reason})")
            else:
                logger.debug(f"Skipped: {item.title[:40]}... (score={score}, reason={reason})")

        # Sort by score descending and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_items]


class ScriptGenerator:
    """AI-driven video script generator.

    Generates natural, conversational scripts for TTS narration
    by leveraging the AI processor to polish content.
    """

    # Prompt template for script generation
    # Designed for ~200 tokens output, uses light model for cost efficiency
    SCRIPT_PROMPT = """将以下内容改写为口播脚本（15-30秒朗读时长），要求：
1. 口语化，适合朗读，使用短句
2. 开头吸引注意力，不要用"大家好"、"今天"等老套开场
3. 结尾有行动号召（如：关注了解更多）
4. 总字数不超过150字
5. 不要使用emoji

标题：{title}
摘要：{summary}
要点：{key_points}
结论：{one_liner}

直接返回脚本文本，不需要JSON格式："""

    def __init__(self, processor=None):
        """Initialize ScriptGenerator.

        Args:
            processor: Optional AI processor instance. If None, will be created on demand.
        """
        self._processor = processor
        self._settings = None

    @property
    def settings(self):
        """Lazy load settings."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    def _get_processor(self):
        """Get or create AI processor for script generation."""
        if self._processor is None:
            from src.processors.digest_processor import get_ai_processor

            self._processor = get_ai_processor()
        return self._processor

    def _format_key_points(self, item: ContentItem) -> str:
        """Format key_points for prompt.

        Args:
            item: ContentItem with key_points JSON.

        Returns:
            Formatted string for prompt.
        """
        key_points = parse_key_points(item.key_points)
        if not key_points:
            return "无"

        formatted = []
        for i, kp in enumerate(key_points[:3], 1):  # Max 3 points
            value = kp.get("value", "")
            impact = kp.get("impact", "")
            if value:
                point = f"{i}. {value}"
                if impact:
                    point += f" ({impact})"
                formatted.append(point)

        return "\n".join(formatted) if formatted else "无"

    def generate(self, item: ContentItem) -> str:
        """Generate a polished TTS script from content item.

        Uses AI to transform the content into natural, conversational text
        suitable for voice-over narration.

        Args:
            item: ContentItem with AI processing results.

        Returns:
            Polished script text for TTS.

        Raises:
            Exception: If AI processing fails.
        """
        # Build prompt
        title = item.title or ""
        summary = item.summary or ""
        key_points_str = self._format_key_points(item)
        one_liner = item.one_liner or ""

        prompt = self.SCRIPT_PROMPT.format(
            title=title[:100],  # Truncate long titles
            summary=summary[:200],  # Truncate long summaries
            key_points=key_points_str,
            one_liner=one_liner[:100],
        )

        # Call AI processor
        processor = self._get_processor()
        start_time = time.time()

        # Use processor's _call_api if available (OllamaAPI), otherwise fall back
        if hasattr(processor, "_call_api"):
            # OllamaAPI - use 32B model for better script quality
            success, response, error = processor._call_api(
                prompt,
                model=processor.model,  # Use heavy model (32B)
                max_tokens=256,  # Script is ~150 chars
                disable_thinking=True,
            )
            duration_ms = int((time.time() - start_time) * 1000)

            # Record AI call for tracking
            record_ai_call(
                call_type=AICallType.CONTENT_HEAVY,  # 32B model
                cached=False,
                input_chars=len(prompt),
                output_chars=len(response) if success else 0,
                duration_ms=duration_ms,
                success=success,
                error=error[:100] if error else None,
            )

            if not success:
                logger.warning(f"Script generation failed: {error}")
                return self._fallback_script(item)

            # Clean up response (remove any markdown or extra whitespace)
            script = response.strip()
            if script.startswith("```"):
                # Remove markdown code blocks if present
                lines = script.split("\n")
                script = "\n".join(line for line in lines if not line.strip().startswith("```"))

            return script.strip()
        else:
            # CLI-based processor - fall back to simple script generation
            logger.debug("Using fallback script generation (CLI processor)")
            return self._fallback_script(item)

    def _fallback_script(self, item: ContentItem) -> str:
        """Generate a simple script without AI when AI is unavailable.

        Args:
            item: ContentItem with AI processing results.

        Returns:
            Basic script text.
        """
        parts = []

        # Opening - use title
        title = item.title or "科技快讯"
        parts.append(f"关注！{title}")

        # Summary
        if item.summary:
            parts.append(item.summary[:100])

        # Key points
        key_points = parse_key_points(item.key_points)
        if key_points:
            for i, kp in enumerate(key_points[:2], 1):
                value = kp.get("value", "")
                if value:
                    parts.append(f"第{i}点，{value}")

        # One-liner conclusion
        if item.one_liner:
            parts.append(f"总结：{item.one_liner[:50]}")

        # CTA
        parts.append("关注了解更多。")

        return "。".join(parts)

    def generate_batch(self, items: list[ContentItem]) -> list[str]:
        """Generate scripts for multiple items.

        Args:
            items: List of ContentItems.

        Returns:
            List of script strings, one per item.
        """
        return [self.generate(item) for item in items]

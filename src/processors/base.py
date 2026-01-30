"""Base classes for AI processors."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from config.settings import get_settings


@dataclass
class KeyPointResult:
    """A key point from AI processing."""

    type: str  # 数字/时间/实体/事实
    value: str
    impact: str = ""


@dataclass
class ImpactResult:
    """Impact assessment from AI processing."""

    short_term: str = ""
    long_term: str = ""
    certainty: str = "uncertain"


@dataclass
class ActionResult:
    """Actionable item from AI processing."""

    type: str  # 跟进/验证/决策/触发器
    description: str
    priority: str = "中"


@dataclass
class ProcessingResult:
    """Result of AI processing."""

    summary: str
    category: str
    importance_score: int
    success: bool = True
    error_message: Optional[str] = None
    raw_response: Optional[str] = None

    # Enhanced fields (Phase 1)
    one_liner: str = ""
    key_points: list[KeyPointResult] = field(default_factory=list)
    impact_assessment: Optional[ImpactResult] = None
    actionable_items: list[ActionResult] = field(default_factory=list)


class BaseAIProcessor(ABC):
    """Abstract base class for AI processors."""

    # Enhanced prompt template for content processing (Chinese output)
    # fmt: off
    PROCESS_PROMPT = """分析内容，返回JSON（仅JSON，无其他文字）：
{{
  "summary": "50字中文摘要",
  "category": "AI/编程/产品/技术/创业/科学/商业/其他",
  "importance": 1-10,
  "one_liner": "一句话结论：这条信息对读者意味着什么",
  "key_points": [
    {{"type": "数字/时间/实体/事实", "value": "关键值", "impact": "影响说明"}}
  ],
  "impact_assessment": {{
    "short_term": "短期影响",
    "long_term": "长期影响",
    "certainty": "certain/uncertain"
  }},
  "actionable_items": [
    {{"type": "跟进/验证/决策/触发器", "description": "具体行动", "priority": "高/中/低"}}
  ]
}}

评分标准：9-10重大突破/行业变革，7-8重要更新/值得关注，5-6一般信息，1-4低价值/噪音
key_points：提取最多3个关键点（数字/时间/实体/事实）
actionable_items：仅当importance>=7时提取行动项，否则为空数组

标题：{title}
正文：{content}"""  # noqa: E501
    # fmt: on

    # Simple prompt for low-value or pre-filtered content (uses less tokens)
    SIMPLE_PROMPT = """返回JSON：{{"summary":"中文摘要50字内","category":"AI/编程/产品/技术/创业/科学/商业/其他","importance":1-10}}
评分：9-10重大突破，7-8重要更新，5-6一般，1-4低价值

标题：{title}
正文：{content}"""

    # Batch processing prompt for multiple items
    BATCH_PROMPT = """批量处理以下{count}条内容，每条返回一行JSON（仅JSON，无其他文字）：
{{
  "id": 序号,
  "summary": "50字中文摘要",
  "category": "AI/编程/产品/技术/创业/科学/商业/其他",
  "importance": 1-10,
  "one_liner": "一句话结论",
  "key_points": [{{"type": "类型", "value": "值", "impact": "影响"}}],
  "impact_assessment": {{"short_term": "短期", "long_term": "长期", "certainty": "certain/uncertain"}},
  "actionable_items": [{{"type": "类型", "description": "描述", "priority": "高/中/低"}}]
}}

评分：9-10重大突破，7-8重要，5-6一般，1-4低价值
key_points最多3个，actionable_items仅importance>=7时提取

{items}"""

    @abstractmethod
    def process_content(self, title: str, content: str) -> ProcessingResult:
        """Process content and return summary, category, and importance score."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the AI processor is available."""
        pass

    def _smart_truncate(self, text: str, max_length: int) -> str:
        """
        Truncate text at sentence boundary when possible.

        Args:
            text: Text to truncate.
            max_length: Maximum length.

        Returns:
            Truncated text, preferring sentence boundaries.
        """
        if len(text) <= max_length:
            return text

        # Try to find a sentence boundary within the limit
        truncated = text[:max_length]

        # Look for sentence endings (。！？.!?) in the last 20% of allowed text
        search_start = int(max_length * 0.8)
        sentence_endings = ["。", "！", "？", ".", "!", "?", "\n\n"]

        best_pos = -1
        for ending in sentence_endings:
            pos = truncated.rfind(ending, search_start)
            if pos > best_pos:
                best_pos = pos

        if best_pos > search_start:
            return truncated[: best_pos + 1] + "..."

        # Fall back to word boundary (space or Chinese punctuation)
        word_boundaries = [" ", "，", ",", "；", ";", "：", ":"]
        for boundary in word_boundaries:
            pos = truncated.rfind(boundary, search_start)
            if pos > search_start:
                return truncated[:pos] + "..."

        # Last resort: hard cut
        return truncated + "..."

    def _build_prompt(self, title: str, content: str, max_length: int = -1) -> str:
        """
        Build the prompt with content truncation.

        Args:
            title: Content title.
            content: Content body.
            max_length: Max content length. -1 uses config default, 0 disables truncation.

        Returns:
            Formatted prompt string.
        """
        settings = get_settings()

        # Apply title truncation
        max_title = settings.ai.max_title_length
        if max_title > 0 and len(title) > max_title:
            title = self._smart_truncate(title, max_title)

        # Apply content truncation
        if max_length == -1:
            max_length = settings.ai.max_content_length
        if max_length > 0 and len(content) > max_length:
            content = self._smart_truncate(content, max_length)

        return self.PROCESS_PROMPT.format(title=title, content=content)

    def _build_batch_prompt(self, items: list[tuple[int, str, str]], max_length: int = -1) -> str:
        """
        Build batch prompt for multiple items with truncation.

        Args:
            items: List of (id, title, content) tuples.
            max_length: Max content length per item. -1 uses config default, 0 disables.

        Returns:
            Formatted batch prompt string.
        """
        settings = get_settings()
        max_title = settings.ai.max_title_length

        if max_length == -1:
            max_length = settings.ai.max_content_length

        item_texts = []
        for idx, title, content in items:
            # Truncate title
            if max_title > 0 and len(title) > max_title:
                title = self._smart_truncate(title, max_title)
            # Truncate content
            if max_length > 0 and len(content) > max_length:
                content = self._smart_truncate(content, max_length)
            item_texts.append(f"[{idx}] 标题：{title}\n正文：{content}")

        return self.BATCH_PROMPT.format(count=len(items), items="\n\n".join(item_texts))

    def process_batch(self, items: list[tuple[int, str, str]]) -> list[ProcessingResult]:
        """
        Process multiple items in a single API call (override in subclass).

        Args:
            items: List of (id, title, content) tuples

        Returns:
            List of ProcessingResult, one per item
        """
        # Default: fall back to individual processing
        results = []
        for idx, title, content in items:
            results.append(self.process_content(title, content))
        return results

    def _parse_json_response(self, response: str) -> ProcessingResult:
        """Parse JSON response from AI output."""
        original_response = response
        try:
            # Clean up response - remove markdown code blocks if present
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

            # Try to find JSON object in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                response = response[json_start:json_end]

            data = json.loads(response)
            return self._extract_processing_result(data, response)

        except json.JSONDecodeError:
            # Fallback: try regex extraction for malformed JSON (e.g., unescaped quotes)
            return self._parse_with_regex(original_response)

    def _extract_processing_result(self, data: dict, raw_response: str) -> ProcessingResult:
        """Extract ProcessingResult from parsed JSON dict with enhanced fields."""
        summary = data.get("summary", "")
        category = data.get("category", "其他")
        importance = data.get("importance", 5)

        # Validate importance score
        if not isinstance(importance, int):
            try:
                importance = int(importance)
            except (ValueError, TypeError):
                importance = 5
        importance = max(1, min(10, importance))

        # Extract enhanced fields
        one_liner = data.get("one_liner", "")

        # Parse key_points
        key_points = []
        raw_key_points = data.get("key_points", [])
        if isinstance(raw_key_points, list):
            for kp in raw_key_points:
                if isinstance(kp, dict):
                    key_points.append(KeyPointResult(
                        type=kp.get("type", "事实"),
                        value=kp.get("value", ""),
                        impact=kp.get("impact", ""),
                    ))

        # Parse impact_assessment
        impact_assessment = None
        raw_impact = data.get("impact_assessment")
        if isinstance(raw_impact, dict):
            impact_assessment = ImpactResult(
                short_term=raw_impact.get("short_term", ""),
                long_term=raw_impact.get("long_term", ""),
                certainty=raw_impact.get("certainty", "uncertain"),
            )

        # Parse actionable_items
        actionable_items = []
        raw_actions = data.get("actionable_items", [])
        if isinstance(raw_actions, list):
            for action in raw_actions:
                if isinstance(action, dict):
                    actionable_items.append(ActionResult(
                        type=action.get("type", "跟进"),
                        description=action.get("description", ""),
                        priority=action.get("priority", "中"),
                    ))

        return ProcessingResult(
            summary=summary,
            category=category,
            importance_score=importance,
            success=True,
            raw_response=raw_response,
            one_liner=one_liner,
            key_points=key_points,
            impact_assessment=impact_assessment,
            actionable_items=actionable_items,
        )

    def _parse_with_regex(self, response: str) -> ProcessingResult:
        """Fallback parser using regex for malformed JSON."""
        try:
            # Extract summary - match content between "summary": " and ", "category"
            summary_match = re.search(
                r'"summary"\s*:\s*"(.+?)"\s*,\s*"category"',
                response,
                re.DOTALL
            )
            # Extract category
            category_match = re.search(
                r'"category"\s*:\s*"([^"]+)"',
                response
            )
            # Extract importance
            importance_match = re.search(
                r'"importance"\s*:\s*(\d+)',
                response
            )

            if summary_match and category_match and importance_match:
                summary = summary_match.group(1).strip()
                category = category_match.group(1).strip()
                importance = int(importance_match.group(1))
                importance = max(1, min(10, importance))

                return ProcessingResult(
                    summary=summary,
                    category=category,
                    importance_score=importance,
                    success=True,
                    raw_response=response,
                )

            # If regex also fails, return error
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message="Failed to parse JSON response (regex fallback also failed)",
                raw_response=response,
            )

        except Exception as e:
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message=f"Failed to parse JSON response: {e}",
                raw_response=response,
            )


class MockAIProcessor(BaseAIProcessor):
    """Mock AI processor for testing."""

    def process_content(self, title: str, content: str) -> ProcessingResult:
        """Return mock processing result."""
        summary = f"这是关于「{title[:20]}」的内容摘要。"
        category = "技术"
        importance = 6

        return ProcessingResult(
            summary=summary,
            category=category,
            importance_score=importance,
            success=True,
            one_liner=f"关注：{title[:30]}的最新动态",
            key_points=[
                KeyPointResult(type="事实", value=title[:20], impact="值得了解"),
            ],
            impact_assessment=ImpactResult(
                short_term="短期内可能产生一定影响",
                long_term="长期影响待观察",
                certainty="uncertain",
            ),
            actionable_items=[],
        )

    def is_available(self) -> bool:
        return True

"""Base classes for AI processors."""

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Optional

from config.settings import get_settings


class TaskType(Enum):
    """Task types for model tier selection."""

    CONTENT_HIGH = auto()  # High importance content - full analysis
    CONTENT_LOW = auto()  # Low importance content - light analysis
    CONTENT_MINIMAL = auto()  # Very low importance - minimal analysis (simple prompt)
    CONTENT_UNCERTAIN = auto()  # Low confidence content - use heavy model
    EVENT_CONFIRM = auto()  # Event clustering confirmation
    EVENT_TITLE = auto()  # Event title generation
    TIMELINE = auto()  # Timeline summary
    REPORT = auto()  # Report generation
    PAPER_FULL = auto()  # Academic paper - deep analysis with innovation/practicality
    PAPER_SCREEN = auto()  # Academic paper - quick screening for value judgment


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
class InnovationResult:
    """Paper innovation assessment."""

    core_contribution: str = ""  # Core contribution (1-2 sentences)
    novelty_type: str = ""  # 理论创新/方法创新/应用创新/工程创新
    diff_from_prior: str = ""  # Key difference from prior work
    novelty_score: int = 3  # 1-5 scale


@dataclass
class PracticalityResult:
    """Paper practicality assessment."""

    application_scenarios: list[str] = field(default_factory=list)
    engineering_feasibility: str = "中"  # 高/中/低
    deployment_difficulty: str = "中等"  # 简单/中等/复杂
    practicality_score: int = 3  # 1-5 scale


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

    # Valid categories for classification
    VALID_CATEGORIES = {"AI", "机器学习", "编程", "技术", "创业", "创新", "金融", "研究", "设计", "其他"}

    # Enhanced prompt template for content processing (Chinese output)
    # fmt: off
    PROCESS_PROMPT = """分析内容，返回JSON（仅JSON，无其他文字）：
{{
  "summary": "50字中文摘要",
  "category": "只选一个：AI、机器学习、编程、技术、创业、创新、金融、研究、设计、其他",
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

评分:9-10重大发布/突破性论文,7-8官宣/重要开源/融资,5-6教程/一般研究,3-4普通讨论,1-2招聘/水帖
category：必须是以上10个分类之一，不要组合
key_points：提取最多3个关键点（数字/时间/实体/事实）
actionable_items：仅当importance>=7时提取行动项，否则为空数组

标题：{title}
正文：{content}"""  # noqa: E501
    # fmt: on

    # Simple prompt for low-value or pre-filtered content (uses less tokens)
    # fmt: off
    SIMPLE_PROMPT = """返回JSON：{{"summary":"中文摘要50字内","category":"AI|机器学习|编程|技术|创业|创新|金融|研究|设计|其他(选一个)","importance":1-10}}
评分:9-10重大发布,7-8重要更新,5-6一般,1-4低价值

标题：{title}
正文：{content}"""  # noqa: E501
    # fmt: on

    # Light prompt - retains key_points but skips impact and actions (for light model)
    # fmt: off
    LIGHT_PROMPT = """返回JSON：{{
  "summary": "50字中文摘要",
  "category": "只选一个：AI、机器学习、编程、技术、创业、创新、金融、研究、设计、其他",
  "importance": 1-10,
  "one_liner": "一句话结论",
  "key_points": [{{"type": "数字/时间/实体/事实", "value": "关键值", "impact": "影响"}}]
}}
评分:9-10重大,7-8重要,5-6一般,1-4低价值
key_points：最多3个关键点

标题：{title}
正文：{content}"""
    # fmt: on

    # Academic paper deep analysis prompt
    # fmt: off
    PAPER_PROMPT = """分析学术论文，返回JSON（仅JSON，无其他文字）：
{{
  "summary": "100字中文摘要：研究目标、方法、主要发现",
  "category": "只选一个：AI、机器学习、编程、技术、创业、创新、金融、研究、设计、其他",
  "importance": 1-10,
  "one_liner": "一句话：这篇论文最核心的贡献",
  "innovation": {{
    "core_contribution": "核心贡献（1-2句）",
    "novelty_type": "理论创新/方法创新/应用创新/工程创新",
    "diff_from_prior": "与现有工作的关键区别",
    "novelty_score": 1-5
  }},
  "practicality": {{
    "application_scenarios": ["应用场景1", "应用场景2"],
    "engineering_feasibility": "高/中/低",
    "deployment_difficulty": "简单/中等/复杂",
    "practicality_score": 1-5
  }},
  "key_points": [
    {{"type": "方法/数据/结果/局限", "value": "关键信息", "impact": "影响说明"}}
  ],
  "impact_assessment": {{
    "academic_impact": "学术影响（引用潜力、领域推进）",
    "industry_impact": "工业界影响（实际应用价值）",
    "certainty": "certain/uncertain"
  }},
  "actionable_items": [
    {{"type": "复现/跟进/应用/关注", "description": "具体行动", "priority": "高/中/低"}}
  ]
}}

论文评分（严格标准，与筛选阶段一致）:
10: 极罕见，改变领域的开创性工作(如Transformer/GPT/AlphaFold级别)
9: 顶级突破，新范式或SOTA大幅提升(>10%)，每年仅几篇
7-8: 重要贡献，显著改进或新颖方法，顶会best paper级别
5-6: 普通研究，渐进改进/标准方法/有效但不突出（大部分论文）
3-4: 一般，小幅改进/复现/应用已有方法
1-2: 低价值，无创新/方法存疑

novelty_score: 5=开创性,4=重要创新,3=有意义改进,2=渐进改进,1=无明显创新
practicality_score: 5=即刻可用,4=稍加修改可用,3=需要开发,2=研究阶段,1=纯理论
key_points: 最多4个
actionable_items: 仅importance>=7时提取

标题：{title}
摘要/正文：{content}"""
    # fmt: on

    # Batch processing prompt for multiple items
    # fmt: off
    BATCH_PROMPT = """批量处理以下{count}条内容，每条返回一行JSON（仅JSON，无其他文字）：
{{
  "id": 序号,
  "summary": "50字中文摘要",
  "category": "只选一个：AI、机器学习、编程、技术、创业、创新、金融、研究、设计、其他",
  "importance": 1-10,
  "one_liner": "一句话结论",
  "key_points": [{{"type": "类型", "value": "值", "impact": "影响"}}],
  "impact_assessment": {{"short_term": "短期", "long_term": "长期", "certainty": "certain/uncertain"}},
  "actionable_items": [{{"type": "类型", "description": "描述", "priority": "高/中/低"}}]
}}

评分:9-10重大发布/突破,7-8官宣/开源/融资,5-6教程/研究,3-4讨论,1-2招聘/水帖
category必须是10个分类之一，key_points最多3个，actionable_items仅importance>=7时提取

{items}"""  # noqa: E501
    # fmt: on

    @abstractmethod
    def process_content(self, title: str, content: str) -> ProcessingResult:
        """Process content and return summary, category, and importance score."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the AI processor is available."""
        pass

    # ============================================
    # Cache Helper Methods
    # ============================================

    @staticmethod
    def _get_content_hash(title: str, content: str) -> str:
        """
        Generate content hash as cache key.

        Args:
            title: Content title.
            content: Content body.

        Returns:
            SHA256 hash (16 characters) of combined title and content.
        """
        combined = f"{title}||{content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    @staticmethod
    def _serialize_result(result: ProcessingResult) -> str:
        """
        Serialize ProcessingResult to JSON string.

        Args:
            result: ProcessingResult to serialize.

        Returns:
            JSON string representation.
        """
        impact = asdict(result.impact_assessment) if result.impact_assessment else None
        data = {
            "summary": result.summary,
            "category": result.category,
            "importance_score": result.importance_score,
            "success": result.success,
            "error_message": result.error_message,
            "one_liner": result.one_liner,
            "key_points": [asdict(kp) for kp in result.key_points],
            "impact_assessment": impact,
            "actionable_items": [asdict(ai) for ai in result.actionable_items],
        }
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _deserialize_result(json_str: str) -> ProcessingResult:
        """
        Deserialize JSON string to ProcessingResult.

        Args:
            json_str: JSON string to deserialize.

        Returns:
            ProcessingResult object.
        """
        data = json.loads(json_str)

        # Reconstruct key_points
        key_points = []
        for kp in data.get("key_points", []):
            key_points.append(
                KeyPointResult(
                    type=kp.get("type", "事实"),
                    value=kp.get("value", ""),
                    impact=kp.get("impact", ""),
                )
            )

        # Reconstruct impact_assessment
        impact_assessment = None
        if data.get("impact_assessment"):
            ia = data["impact_assessment"]
            impact_assessment = ImpactResult(
                short_term=ia.get("short_term", ""),
                long_term=ia.get("long_term", ""),
                certainty=ia.get("certainty", "uncertain"),
            )

        # Reconstruct actionable_items
        actionable_items = []
        for ai in data.get("actionable_items", []):
            actionable_items.append(
                ActionResult(
                    type=ai.get("type", "跟进"),
                    description=ai.get("description", ""),
                    priority=ai.get("priority", "中"),
                )
            )

        return ProcessingResult(
            summary=data.get("summary", ""),
            category=data.get("category", "其他"),
            importance_score=data.get("importance_score", 5),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            one_liner=data.get("one_liner", ""),
            key_points=key_points,
            impact_assessment=impact_assessment,
            actionable_items=actionable_items,
        )

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

    def _build_light_prompt(self, title: str, content: str, max_length: int = -1) -> str:
        """
        Build a lightweight prompt for light model processing.

        Args:
            title: Content title.
            content: Content body.
            max_length: Max content length. -1 uses config default, 0 disables truncation.

        Returns:
            Formatted light prompt string.
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

        return self.LIGHT_PROMPT.format(title=title, content=content)

    def _build_simple_prompt(self, title: str, content: str, max_length: int = -1) -> str:
        """
        Build a minimal prompt for very low importance content.

        Uses SIMPLE_PROMPT which only extracts summary, category, and importance.
        No key_points, impact_assessment, or actionable_items.

        Args:
            title: Content title.
            content: Content body.
            max_length: Max content length. -1 uses 500 chars (shorter for simple).

        Returns:
            Formatted simple prompt string.
        """
        settings = get_settings()

        # Apply title truncation (shorter for simple)
        max_title = min(settings.ai.max_title_length, 100)
        if max_title > 0 and len(title) > max_title:
            title = self._smart_truncate(title, max_title)

        # Apply content truncation (much shorter for simple content)
        if max_length == -1:
            max_length = 500  # Shorter default for simple prompt
        if max_length > 0 and len(content) > max_length:
            content = self._smart_truncate(content, max_length)

        return self.SIMPLE_PROMPT.format(title=title, content=content)

    def _build_paper_prompt(self, title: str, content: str, max_length: int = -1) -> str:
        """
        Build prompt for academic paper deep analysis.

        Papers need more content (abstracts are typically 200-500 words).
        Uses PAPER_PROMPT with innovation and practicality assessment.

        Args:
            title: Paper title.
            content: Abstract or full text.
            max_length: Max content length. -1 uses extended default (2500 chars).

        Returns:
            Formatted paper analysis prompt string.
        """
        settings = get_settings()

        # Apply title truncation
        max_title = 200  # Paper titles can be longer
        if max_title > 0 and len(title) > max_title:
            title = self._smart_truncate(title, max_title)

        # Papers need more content (abstracts are structured and longer)
        if max_length == -1:
            max_length = settings.rule_optimization.paper_extended_content_length
        if max_length > 0 and len(content) > max_length:
            content = self._smart_truncate(content, max_length)

        return self.PAPER_PROMPT.format(title=title, content=content)

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

    def process_batch(
        self, items: list[tuple[int, str, str]], task_type: Optional["TaskType"] = None
    ) -> list[ProcessingResult]:
        """
        Process multiple items in a single API call (override in subclass).

        Args:
            items: List of (id, title, content) tuples
            task_type: Optional task type for model selection

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

    # Category aliases for normalization
    CATEGORY_ALIASES = {
        "经济": "金融",
        "投资": "金融",
        "商业": "创业",
        "产品": "创新",
        "开发": "编程",
        "科研": "研究",
        "学术": "研究",
        "ML": "机器学习",
        "深度学习": "机器学习",
    }

    def _normalize_category(self, category: str) -> str:
        """
        Normalize category to one of the valid categories.

        Handles cases where AI returns:
        - Composite categories like "AI/机器学习" -> "AI"
        - Full category list -> "其他"
        - Alias categories like "经济" -> "金融"
        - Unknown categories -> "其他"
        """
        if not category:
            return "其他"

        # If it's already a valid category, return it
        if category in self.VALID_CATEGORIES:
            return category

        # Check aliases
        if category in self.CATEGORY_ALIASES:
            return self.CATEGORY_ALIASES[category]

        # Handle composite categories (e.g., "AI/机器学习", "AI/技术")
        # Take the first valid category
        for sep in ["/", "、", ",", "，"]:
            if sep in category:
                parts = category.split(sep)
                for part in parts:
                    part = part.strip()
                    if part in self.VALID_CATEGORIES:
                        return part
                    if part in self.CATEGORY_ALIASES:
                        return self.CATEGORY_ALIASES[part]

        # Handle partial matches (e.g., "机器学习/编程/技术" contains valid ones)
        for valid_cat in self.VALID_CATEGORIES:
            if valid_cat in category:
                return valid_cat

        return "其他"

    def _extract_processing_result(self, data: dict, raw_response: str) -> ProcessingResult:
        """Extract ProcessingResult from parsed JSON dict with enhanced fields."""
        summary = data.get("summary", "")
        raw_category = data.get("category", "其他")
        category = self._normalize_category(raw_category)
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
                    key_points.append(
                        KeyPointResult(
                            type=kp.get("type", "事实"),
                            value=kp.get("value", ""),
                            impact=kp.get("impact", ""),
                        )
                    )

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
                    actionable_items.append(
                        ActionResult(
                            type=action.get("type", "跟进"),
                            description=action.get("description", ""),
                            priority=action.get("priority", "中"),
                        )
                    )

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
                r'"summary"\s*:\s*"(.+?)"\s*,\s*"category"', response, re.DOTALL
            )
            # Extract category
            category_match = re.search(r'"category"\s*:\s*"([^"]+)"', response)
            # Extract importance
            importance_match = re.search(r'"importance"\s*:\s*(\d+)', response)

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

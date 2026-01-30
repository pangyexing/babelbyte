"""Base classes for AI processors."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProcessingResult:
    """Result of AI processing."""

    summary: str
    category: str
    importance_score: int
    success: bool = True
    error_message: Optional[str] = None
    raw_response: Optional[str] = None


class BaseAIProcessor(ABC):
    """Abstract base class for AI processors."""

    # Shared prompt template for content processing (Chinese output)
    PROCESS_PROMPT = """请用中文处理以下内容，返回纯JSON格式（不要包含markdown代码块标记）：

要求：
1. summary: 简短中文摘要（50字以内）
2. category: 主题分类，从以下选择：AI、编程、产品、技术、创业、科学、商业、其他
3. importance: 重要性评分（1-10分，10分最重要）

评分标准：
- 9-10分：重大突破、行业变革性消息
- 7-8分：重要更新、有价值的见解
- 5-6分：一般有趣的内容
- 3-4分：普通内容
- 1-2分：低价值或重复内容

请只返回JSON，格式如下：
{{"summary": "中文摘要", "category": "分类", "importance": 数字}}

内容标题：{title}

内容正文：
{content}
"""

    @abstractmethod
    def process_content(self, title: str, content: str) -> ProcessingResult:
        """Process content and return summary, category, and importance score."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the AI processor is available."""
        pass

    def _build_prompt(self, title: str, content: str, max_length: int = 3000) -> str:
        """Build the prompt with truncated content."""
        if len(content) > max_length:
            content = content[:max_length] + "..."
        return self.PROCESS_PROMPT.format(title=title, content=content)

    def _parse_json_response(self, response: str) -> ProcessingResult:
        """Parse JSON response from AI output."""
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

            return ProcessingResult(
                summary=summary,
                category=category,
                importance_score=importance,
                success=True,
                raw_response=response,
            )

        except json.JSONDecodeError as e:
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
        )

    def is_available(self) -> bool:
        return True

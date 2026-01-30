"""Base classes for AI processors."""

import json
import re
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
    PROCESS_PROMPT = """返回JSON：{{"summary":"中文摘要50字内","category":"AI/编程/产品/技术/创业/科学/商业/其他","importance":1-10}}
评分：9-10重大突破，7-8重要更新，5-6一般，1-4低价值

标题：{title}
正文：{content}"""

    # Batch processing prompt for multiple items
    BATCH_PROMPT = """批量处理以下{count}条内容，每条返回一行JSON（不要其他内容）：
{{"id":序号,"summary":"中文摘要50字内","category":"AI/编程/产品/技术/创业/科学/商业/其他","importance":1-10}}
评分：9-10重大突破，7-8重要更新，5-6一般，1-4低价值

{items}"""

    @abstractmethod
    def process_content(self, title: str, content: str) -> ProcessingResult:
        """Process content and return summary, category, and importance score."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the AI processor is available."""
        pass

    def _build_prompt(self, title: str, content: str, max_length: int = 0) -> str:
        """Build the prompt with optional content truncation."""
        if max_length > 0 and len(content) > max_length:
            content = content[:max_length] + "..."
        return self.PROCESS_PROMPT.format(title=title, content=content)

    def _build_batch_prompt(self, items: list[tuple[int, str, str]], max_length: int = 0) -> str:
        """Build batch prompt for multiple items (no truncation by default)."""
        item_texts = []
        for idx, title, content in items:
            if max_length > 0 and len(content) > max_length:
                content = content[:max_length] + "..."
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

        except json.JSONDecodeError:
            # Fallback: try regex extraction for malformed JSON (e.g., unescaped quotes)
            return self._parse_with_regex(original_response)

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
        )

    def is_available(self) -> bool:
        return True

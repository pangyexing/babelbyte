"""Claude Code CLI wrapper for AI processing."""

import json
import subprocess
from dataclasses import dataclass
from typing import Optional

from config.settings import get_settings


@dataclass
class ProcessingResult:
    """Result of AI processing."""

    summary: str
    category: str
    importance_score: int
    success: bool = True
    error_message: Optional[str] = None
    raw_response: Optional[str] = None


class ClaudeCLI:
    """Wrapper for Claude Code CLI."""

    # Prompt template for content processing (Chinese output)
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

    def __init__(self, cli_path: Optional[str] = None, timeout: int = 60):
        self.cli_path = cli_path or get_settings().claude.cli_path
        self.timeout = timeout

    def process_content(self, title: str, content: str) -> ProcessingResult:
        """
        Process content using Claude Code CLI.

        Args:
            title: The title of the content.
            content: The main content text.

        Returns:
            ProcessingResult with summary, category, and importance score.
        """
        # Truncate content if too long
        max_content_length = 3000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        prompt = self.PROCESS_PROMPT.format(title=title, content=content)

        try:
            result = subprocess.run(
                [self.cli_path, "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                return ProcessingResult(
                    summary="",
                    category="其他",
                    importance_score=5,
                    success=False,
                    error_message=f"Claude CLI error: {result.stderr}",
                    raw_response=result.stdout,
                )

            return self._parse_response(result.stdout)

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

    def _parse_response(self, response: str) -> ProcessingResult:
        """Parse the JSON response from Claude CLI."""
        try:
            # Clean up response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                # Remove opening code block
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
            # If JSON parsing fails, try to extract information manually
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message=f"Failed to parse JSON response: {e}",
                raw_response=response,
            )

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


class MockClaudeCLI:
    """Mock Claude CLI for testing without actual API access."""

    def process_content(self, title: str, content: str) -> ProcessingResult:
        """Return mock processing result."""
        # Simple mock logic
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

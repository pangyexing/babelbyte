"""Claude Code CLI wrapper for AI processing."""

import subprocess
from typing import Optional

from config.settings import get_settings
from src.processors.base import BaseAIProcessor, ProcessingResult


class ClaudeCLI(BaseAIProcessor):
    """Wrapper for Claude Code CLI."""

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
        prompt = self._build_prompt(title, content)

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

            return self._parse_json_response(result.stdout)

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

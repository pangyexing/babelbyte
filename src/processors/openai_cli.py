"""OpenAI Codex CLI wrapper for AI processing."""

import logging
import subprocess
from typing import Optional

from config.settings import get_settings
from src.processors.base import BaseAIProcessor, ProcessingResult, TaskType

logger = logging.getLogger(__name__)


class CodexCLI(BaseAIProcessor):
    """Wrapper for OpenAI Codex CLI (subscription mode, no API key needed)."""

    def __init__(self, cli_path: Optional[str] = None, timeout: int = 60):
        self.cli_path = cli_path or get_settings().codex.cli_path
        self.timeout = timeout
        self._settings = None

    @property
    def settings(self):
        """Lazy load settings."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    def get_model_for_task(self, task_type: TaskType) -> str:
        """
        Select model based on task type.

        Args:
            task_type: Type of task being performed.

        Returns:
            Model name to use, or empty string to use default.
        """
        config = self.settings.ai.model_tiers
        if not config.enabled:
            return ""

        # Check task-level override
        task_name = task_type.name.lower()
        if task_name in config.task_overrides:
            return config.task_overrides[task_name]

        # Heavy tasks use heavy model
        heavy_tasks = {
            TaskType.CONTENT_HIGH,
            TaskType.CONTENT_UNCERTAIN,
            TaskType.REPORT,
        }
        if task_type in heavy_tasks:
            return config.codex_heavy

        # Light tasks use light model
        return config.codex_light

    def _run_cli(
        self, prompt: str, task_type: Optional[TaskType] = None, timeout: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        """
        Run Codex CLI with optional model selection.

        Args:
            prompt: The prompt to send.
            task_type: Task type for model selection.
            timeout: Optional timeout override.

        Returns:
            Completed subprocess result.
        """
        cmd = [self.cli_path, "exec", "-"]

        if task_type:
            model = self.get_model_for_task(task_type)
            if model:
                cmd = [self.cli_path, "exec", "-m", model, "-"]
                logger.debug(f"Using model {model} for task {task_type.name}")

        return subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout or self.timeout,
        )

    def process_content(
        self, title: str, content: str, task_type: Optional[TaskType] = None
    ) -> ProcessingResult:
        """
        Process content using Codex CLI.

        Args:
            title: The title of the content.
            content: The main content text.
            task_type: Task type for model selection. Defaults to CONTENT_HIGH.

        Returns:
            ProcessingResult with summary, category, and importance score.
        """
        # Default to high importance processing if not specified
        if task_type is None:
            task_type = TaskType.CONTENT_HIGH

        # Use light prompt for low importance content
        if task_type == TaskType.CONTENT_LOW:
            prompt = self._build_light_prompt(title, content)
        else:
            prompt = self._build_prompt(title, content)

        try:
            result = self._run_cli(prompt, task_type)

            if result.returncode != 0:
                return ProcessingResult(
                    summary="",
                    category="其他",
                    importance_score=5,
                    success=False,
                    error_message=f"Codex CLI error: {result.stderr}",
                    raw_response=result.stdout,
                )

            return self._parse_json_response(result.stdout)

        except subprocess.TimeoutExpired:
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message="Codex CLI timeout",
            )
        except FileNotFoundError:
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message=f"Codex CLI not found at: {self.cli_path}",
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
        """Check if Codex CLI is available."""
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


# Alias for backward compatibility
OpenAICLI = CodexCLI

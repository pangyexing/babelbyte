"""Claude Code CLI wrapper for AI processing."""

import subprocess
from typing import TYPE_CHECKING, Optional

from config.settings import get_settings
from src.processors.base import TaskType
from src.processors.base_cli import BaseCLIProcessor

if TYPE_CHECKING:
    from src.storage.database import SyncDatabase


class ClaudeCLI(BaseCLIProcessor):
    """Wrapper for Claude Code CLI."""

    def __init__(
        self,
        cli_path: Optional[str] = None,
        timeout: int = 60,
        db: Optional["SyncDatabase"] = None,
    ):
        super().__init__(cli_path, timeout, db)

    @property
    def cli_path(self) -> str:
        """Path to the Claude CLI executable."""
        if self._cli_path:
            return self._cli_path
        return get_settings().claude.cli_path

    @property
    def cli_name(self) -> str:
        """Name of the CLI for error messages."""
        return "Claude"

    def _get_model_name(self, task_type: TaskType, heavy: bool) -> str:
        """Get Claude model name based on task weight."""
        config = self.settings.ai.model_tiers
        return config.claude_heavy if heavy else config.claude_light

    def _build_cli_command(self, model: Optional[str] = None) -> list[str]:
        """Build the Claude CLI command."""
        cmd = [self.cli_path, "-p", "", "--output-format", "text"]
        if model:
            cmd.extend(["--model", model])
        return cmd

    def _run_subprocess(
        self, cmd: list[str], prompt: str, timeout: int
    ) -> subprocess.CompletedProcess:
        """Execute Claude CLI with prompt as argument."""
        # Claude CLI takes prompt via -p flag, update the command
        cmd[2] = prompt  # Replace empty string placeholder with actual prompt
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

"""OpenAI Codex CLI wrapper for AI processing."""

import subprocess
from typing import TYPE_CHECKING, Optional

from config.settings import get_settings
from src.processors.base import TaskType
from src.processors.base_cli import BaseCLIProcessor

if TYPE_CHECKING:
    from src.storage.database import SyncDatabase


class CodexCLI(BaseCLIProcessor):
    """Wrapper for OpenAI Codex CLI (subscription mode, no API key needed)."""

    def __init__(
        self,
        cli_path: Optional[str] = None,
        timeout: int = 60,
        db: Optional["SyncDatabase"] = None,
    ):
        super().__init__(cli_path, timeout, db)

    @property
    def cli_path(self) -> str:
        """Path to the Codex CLI executable."""
        if self._cli_path:
            return self._cli_path
        return get_settings().codex.cli_path

    @property
    def cli_name(self) -> str:
        """Name of the CLI for error messages."""
        return "Codex"

    def _get_model_name(self, task_type: TaskType, heavy: bool) -> str:
        """Get Codex model name based on task weight."""
        config = self.settings.ai.model_tiers
        return config.codex_heavy if heavy else config.codex_light

    def _build_cli_command(self, model: Optional[str] = None) -> list[str]:
        """Build the Codex CLI command."""
        if model:
            return [self.cli_path, "exec", "-m", model, "-"]
        return [self.cli_path, "exec", "-"]

    def _run_subprocess(
        self, cmd: list[str], prompt: str, timeout: int
    ) -> subprocess.CompletedProcess:
        """Execute Codex CLI with prompt via stdin."""
        return subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


# Alias for backward compatibility
OpenAICLI = CodexCLI

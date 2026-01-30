"""Configuration management for BabelByte."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class TwitterConfig:
    """Twitter API configuration."""

    bearer_token: str = field(default_factory=lambda: os.getenv("TWITTER_BEARER_TOKEN", ""))

    @property
    def is_configured(self) -> bool:
        return bool(self.bearer_token)


@dataclass
class EmailConfig:
    """Email/SMTP configuration."""

    host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", "smtp.139.com"))
    port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "465")))
    user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    from_addr: str = field(default_factory=lambda: os.getenv("EMAIL_FROM", ""))
    to_addr: str = field(default_factory=lambda: os.getenv("EMAIL_TO", ""))

    @property
    def is_configured(self) -> bool:
        return all([self.host, self.port, self.user, self.password, self.from_addr, self.to_addr])


@dataclass
class DatabaseConfig:
    """Database configuration."""

    path: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATABASE_PATH", "data/babelbyte.db")
    )

    def __post_init__(self):
        # Ensure the data directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    path: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("LOG_PATH", "logs/babelbyte.log")
    )

    def __post_init__(self):
        # Ensure the logs directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    digest_send_time: str = field(default_factory=lambda: os.getenv("DIGEST_SEND_TIME", "08:00"))
    fetch_interval_hours: int = field(
        default_factory=lambda: int(os.getenv("FETCH_INTERVAL_HOURS", "6"))
    )

    @property
    def digest_hour(self) -> int:
        return int(self.digest_send_time.split(":")[0])

    @property
    def digest_minute(self) -> int:
        return int(self.digest_send_time.split(":")[1])


@dataclass
class ClaudeConfig:
    """Claude CLI configuration."""

    cli_path: str = field(default_factory=lambda: os.getenv("CLAUDE_CLI_PATH", "claude"))

    @property
    def is_configured(self) -> bool:
        return bool(self.cli_path)


@dataclass
class CodexConfig:
    """Codex CLI configuration (OpenAI subscription mode)."""

    cli_path: str = field(default_factory=lambda: os.getenv("CODEX_CLI_PATH", "codex"))

    @property
    def is_configured(self) -> bool:
        return bool(self.cli_path)


@dataclass
class AIConfig:
    """AI provider configuration."""

    # Provider: "claude", "codex", or "auto"
    provider: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "auto"))

    def get_provider(self) -> str:
        """Get the actual provider to use."""
        if self.provider == "auto":
            # Auto-detect: prefer Claude if available, fallback to Codex
            return "claude"
        return self.provider


@dataclass
class Settings:
    """Main settings container."""

    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    codex: CodexConfig = field(default_factory=CodexConfig)
    ai: AIConfig = field(default_factory=AIConfig)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    load_dotenv(override=True)
    _settings = Settings()
    return _settings

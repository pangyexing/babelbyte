"""Configuration management for BabelByte."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class TwitterConfig:
    """Twitter API configuration using TwitterAPI.io."""

    twitterapi_io_key: str = field(default_factory=lambda: os.getenv("TWITTERAPI_IO_KEY", ""))

    @property
    def is_configured(self) -> bool:
        return bool(self.twitterapi_io_key)


@dataclass
class EmailConfig:
    """Email/SMTP configuration."""

    host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", "smtp.139.com"))
    port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "465")))
    user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    from_addr: str = field(default_factory=lambda: os.getenv("EMAIL_FROM", ""))
    to_addr: str = field(default_factory=lambda: os.getenv("EMAIL_TO", ""))

    def __post_init__(self):
        """Validate email configuration values."""
        # Validate port range
        if not 1 <= self.port <= 65535:
            raise ValueError(f"SMTP port must be 1-65535, got {self.port}")

        # Validate email format if provided
        if self.from_addr and "@" not in self.from_addr:
            raise ValueError(f"Invalid from_addr format: {self.from_addr}")
        if self.to_addr and "@" not in self.to_addr:
            raise ValueError(f"Invalid to_addr format: {self.to_addr}")

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

    def __post_init__(self):
        """Validate scheduler configuration values."""
        # Validate time format
        if ":" not in self.digest_send_time:
            raise ValueError(
                f"Invalid digest_send_time format: {self.digest_send_time}, expected HH:MM"
            )

        try:
            hour, minute = self.digest_send_time.split(":")
            hour_int = int(hour)
            minute_int = int(minute)
            if not (0 <= hour_int <= 23):
                raise ValueError(f"Hour must be 0-23, got {hour_int}")
            if not (0 <= minute_int <= 59):
                raise ValueError(f"Minute must be 0-59, got {minute_int}")
        except ValueError as e:
            raise ValueError(f"Invalid digest_send_time '{self.digest_send_time}': {e}") from e

        # Validate fetch interval
        if not 1 <= self.fetch_interval_hours <= 168:  # 1 hour to 1 week
            raise ValueError(f"fetch_interval_hours must be 1-168, got {self.fetch_interval_hours}")

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
class QualityGuardConfig:
    """Quality protection configuration for model tier selection."""

    unknown_domain_use_heavy: bool = True
    reprocess_high_score_from_light: bool = True
    reprocess_threshold: int = 7
    retain_key_points_for_light: bool = True


@dataclass
class ModelTierConfig:
    """Model tier configuration for AI cost optimization."""

    enabled: bool = True
    heavy_threshold: int = 7
    low_confidence_threshold: int = 5
    confidence_cutoff: float = 0.5

    # Claude models
    claude_heavy: str = "sonnet"
    claude_light: str = "haiku"

    # Codex models
    codex_heavy: str = "gpt-5.1-codex"
    codex_light: str = "gpt-5.1-codex-mini"

    # Quality protection
    quality_guard: QualityGuardConfig = field(default_factory=QualityGuardConfig)

    # Task-level overrides
    task_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate model tier configuration values."""
        # Validate thresholds are in valid range
        if not 1 <= self.heavy_threshold <= 10:
            raise ValueError(f"heavy_threshold must be 1-10, got {self.heavy_threshold}")
        if not 1 <= self.low_confidence_threshold <= 10:
            raise ValueError(
                f"low_confidence_threshold must be 1-10, got {self.low_confidence_threshold}"
            )
        if not 0.0 <= self.confidence_cutoff <= 1.0:
            raise ValueError(f"confidence_cutoff must be 0.0-1.0, got {self.confidence_cutoff}")

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelTierConfig":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        tier_data = data.get("model_tiers", {})
        guard_data = data.get("quality_guard", {})
        task_overrides = data.get("task_overrides") or {}

        # Extract claude/codex nested config
        claude_config = tier_data.pop("claude", {}) if tier_data else {}
        codex_config = tier_data.pop("codex", {}) if tier_data else {}

        # Build config with defaults
        low_conf_threshold = tier_data.get("low_confidence_threshold", 5) if tier_data else 5
        codex_heavy_model = codex_config.get("heavy", "gpt-5.1-codex") or "gpt-5.1-codex"
        codex_light_model = codex_config.get("light", "gpt-5.1-codex-mini") or "gpt-5.1-codex-mini"
        quality = QualityGuardConfig(**guard_data) if guard_data else QualityGuardConfig()
        overrides = {k: v for k, v in task_overrides.items() if v is not None}

        return cls(
            enabled=tier_data.get("enabled", True) if tier_data else True,
            heavy_threshold=tier_data.get("heavy_threshold", 7) if tier_data else 7,
            low_confidence_threshold=low_conf_threshold,
            confidence_cutoff=tier_data.get("confidence_cutoff", 0.5) if tier_data else 0.5,
            claude_heavy=claude_config.get("heavy", "sonnet") or "sonnet",
            claude_light=claude_config.get("light", "haiku") or "haiku",
            codex_heavy=codex_heavy_model,
            codex_light=codex_light_model,
            quality_guard=quality,
            task_overrides=overrides,
        )


@dataclass
class EmbeddingConfig:
    """Embedding provider configuration."""

    # Provider: "sentence-transformers" (local, free) or "openai" (API, paid)
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"))

    # Model settings
    sentence_transformers_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_ST_MODEL", "all-MiniLM-L6-v2")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_OPENAI_MODEL", "text-embedding-3-small")
    )

    # Dimension (auto-detected based on model, but can be overridden)
    dimension: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "0")))

    # Caching
    cache_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_CACHE_SIZE", "1000")))

    # Hybrid similarity weights
    rule_weight: float = field(default_factory=lambda: float(os.getenv("EMBEDDING_RULE_WEIGHT", "0.4")))
    semantic_weight: float = field(
        default_factory=lambda: float(os.getenv("EMBEDDING_SEMANTIC_WEIGHT", "0.6"))
    )

    # Enable/disable embeddings
    enabled: bool = field(
        default_factory=lambda: os.getenv("EMBEDDING_ENABLED", "true").lower() == "true"
    )

    def __post_init__(self):
        """Validate embedding configuration."""
        valid_providers = {"sentence-transformers", "openai"}
        if self.provider not in valid_providers:
            raise ValueError(f"EMBEDDING_PROVIDER must be one of {valid_providers}, got '{self.provider}'")

        if not 0.0 <= self.rule_weight <= 1.0:
            raise ValueError(f"rule_weight must be 0.0-1.0, got {self.rule_weight}")
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError(f"semantic_weight must be 0.0-1.0, got {self.semantic_weight}")

        # Weights should sum to 1.0 (with small tolerance)
        if abs(self.rule_weight + self.semantic_weight - 1.0) > 0.01:
            raise ValueError(
                f"rule_weight + semantic_weight should equal 1.0, "
                f"got {self.rule_weight} + {self.semantic_weight} = {self.rule_weight + self.semantic_weight}"
            )


@dataclass
class AIConfig:
    """AI provider configuration."""

    # Provider: "claude", "codex", or "auto"
    provider: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "auto"))

    # Content length limits for token optimization
    max_content_length: int = field(
        default_factory=lambda: int(os.getenv("AI_MAX_CONTENT_LENGTH", "1500"))
    )
    max_title_length: int = field(
        default_factory=lambda: int(os.getenv("AI_MAX_TITLE_LENGTH", "200"))
    )

    # Batch processing sizes
    batch_size_short: int = field(
        default_factory=lambda: int(os.getenv("AI_BATCH_SIZE_SHORT", "12"))
    )
    batch_size_long: int = field(default_factory=lambda: int(os.getenv("AI_BATCH_SIZE_LONG", "6")))

    # Cache configuration
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("AI_CACHE_ENABLED", "true").lower() == "true"
    )
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("AI_CACHE_TTL", "86400")))

    # Model tier configuration
    model_tiers: ModelTierConfig = field(default_factory=ModelTierConfig)

    def __post_init__(self):
        """Validate AI configuration and load model tier config from YAML."""
        # Validate provider
        valid_providers = {"claude", "codex", "auto"}
        if self.provider not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}, got '{self.provider}'")

        # Validate content lengths
        if self.max_content_length < 100:
            raise ValueError(f"max_content_length must be >= 100, got {self.max_content_length}")
        if self.max_title_length < 20:
            raise ValueError(f"max_title_length must be >= 20, got {self.max_title_length}")

        # Validate batch sizes
        if not 1 <= self.batch_size_short <= 50:
            raise ValueError(f"batch_size_short must be 1-50, got {self.batch_size_short}")
        if not 1 <= self.batch_size_long <= 20:
            raise ValueError(f"batch_size_long must be 1-20, got {self.batch_size_long}")

        # Validate cache TTL (1 minute to 30 days)
        if self.cache_enabled and not 60 <= self.cache_ttl <= 2592000:
            raise ValueError(f"cache_ttl must be 60-2592000 seconds, got {self.cache_ttl}")

        # Load model tier config from YAML if not already set
        if isinstance(self.model_tiers, ModelTierConfig) and self.model_tiers.enabled:
            yaml_path = PROJECT_ROOT / "config" / "ai_models.yaml"
            if yaml_path.exists():
                self.model_tiers = ModelTierConfig.from_yaml(yaml_path)

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
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


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

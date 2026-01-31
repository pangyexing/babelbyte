"""Token usage tracking for AI calls."""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from src.storage.models import TokenUsage

logger = logging.getLogger(__name__)


class AICallType(Enum):
    """Types of AI calls with estimated token usage."""

    CONTENT_HEAVY = "content_heavy"  # Full processing with all fields (~825 tokens/item)
    CONTENT_LIGHT = "content_light"  # Light processing (~615 tokens/item)
    CONTENT_BATCH = "content_batch"  # Batch processing (~215 tokens/item)
    EVENT_CONFIRM = "event_confirm"  # Event confirmation (~220 tokens/call)
    EVENT_TITLE = "event_title"  # Title generation (~50 tokens/call)
    TIMELINE_SUMMARY = "timeline_summary"  # Timeline summary (~150 tokens/call)
    DIGEST_GENERATE = "digest_generate"  # Digest generation (~500 tokens/call)
    REPORT = "report"  # Report generation (~800 tokens/call)

    @property
    def estimated_tokens(self) -> int:
        """Get estimated token count for this call type."""
        estimates = {
            AICallType.CONTENT_HEAVY: 825,
            AICallType.CONTENT_LIGHT: 615,
            AICallType.CONTENT_BATCH: 215,
            AICallType.EVENT_CONFIRM: 220,
            AICallType.EVENT_TITLE: 50,
            AICallType.TIMELINE_SUMMARY: 150,
            AICallType.DIGEST_GENERATE: 500,
            AICallType.REPORT: 800,
        }
        return estimates.get(self, 500)


@dataclass
class AICall:
    """Record of a single AI call."""

    call_type: AICallType
    timestamp: datetime
    cached: bool = False
    input_chars: int = 0
    output_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class SessionStats:
    """Statistics for a tracking session."""

    start_time: datetime = field(default_factory=datetime.now)
    total_calls: int = 0
    actual_ai_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_duration_ms: int = 0
    calls_by_type: dict = field(default_factory=dict)
    errors: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.cache_hits / self.total_calls) * 100

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


class TokenTracker:
    """
    Tracks AI token usage across the application.

    Thread-safe singleton for tracking all AI calls during a session.
    """

    _instance: Optional["TokenTracker"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._calls: list[AICall] = []
        self._session_start = datetime.now()
        self._call_lock = threading.Lock()
        self._initialized = True

    def reset(self):
        """Reset tracker for a new session."""
        with self._call_lock:
            self._calls = []
            self._session_start = datetime.now()

    def record_call(
        self,
        call_type: AICallType,
        cached: bool = False,
        input_chars: int = 0,
        output_chars: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: int = 0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Record an AI call.

        Args:
            call_type: Type of AI call.
            cached: Whether the result was from cache.
            input_chars: Number of input characters (will estimate tokens if not provided).
            output_chars: Number of output characters.
            input_tokens: Actual input token count if known.
            output_tokens: Actual output token count if known.
            duration_ms: Call duration in milliseconds.
            success: Whether the call succeeded.
            error: Error message if failed.
        """
        # Estimate tokens from characters if not provided
        # Rough estimate: 1 token ≈ 4 characters for English, ≈ 2 for Chinese
        if input_tokens == 0 and input_chars > 0:
            input_tokens = max(input_chars // 3, 10)  # Conservative estimate
        if output_tokens == 0 and output_chars > 0:
            output_tokens = max(output_chars // 3, 10)

        # Use type estimates if no actual counts
        if input_tokens == 0:
            input_tokens = call_type.estimated_tokens

        call = AICall(
            call_type=call_type,
            timestamp=datetime.now(),
            cached=cached,
            input_chars=input_chars,
            output_chars=output_chars,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )

        with self._call_lock:
            self._calls.append(call)

        # Persist to database
        self._persist_to_db(call)

        if cached:
            logger.debug(f"Token tracker: {call_type.value} (cache hit)")
        else:
            logger.debug(
                f"Token tracker: {call_type.value} - {input_tokens}+{output_tokens} tokens"
            )

    def _persist_to_db(self, call: AICall) -> None:
        """Persist an AI call record to the database."""
        try:
            from src.storage.database import SyncDatabase

            usage = TokenUsage(
                call_type=call.call_type.value,
                timestamp=call.timestamp,
                cached=call.cached,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                duration_ms=call.duration_ms,
                success=call.success,
                error=call.error,
            )

            db = SyncDatabase()
            db.connect()
            try:
                db.record_token_usage(usage)
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to persist token usage to database: {e}")

    def get_session_summary(self) -> SessionStats:
        """
        Get summary statistics for the current session.

        Returns:
            SessionStats with aggregated metrics.
        """
        stats = SessionStats(start_time=self._session_start)

        with self._call_lock:
            for call in self._calls:
                stats.total_calls += 1

                if call.cached:
                    stats.cache_hits += 1
                else:
                    stats.actual_ai_calls += 1
                    stats.input_tokens += call.input_tokens
                    stats.output_tokens += call.output_tokens
                    stats.total_duration_ms += call.duration_ms

                if not call.success:
                    stats.errors += 1

                # Track by type
                type_name = call.call_type.value
                if type_name not in stats.calls_by_type:
                    stats.calls_by_type[type_name] = {
                        "total": 0,
                        "cached": 0,
                        "tokens": 0,
                    }
                stats.calls_by_type[type_name]["total"] += 1
                if call.cached:
                    stats.calls_by_type[type_name]["cached"] += 1
                else:
                    stats.calls_by_type[type_name]["tokens"] += (
                        call.input_tokens + call.output_tokens
                    )

        return stats

    def estimate_cost(self, model: str = "haiku") -> float:
        """
        Estimate cost based on token usage.

        Args:
            model: Model name for pricing ("haiku", "sonnet", "opus").

        Returns:
            Estimated cost in USD.
        """
        # Pricing per 1M tokens (as of early 2025)
        pricing = {
            "haiku": {"input": 0.25, "output": 1.25},
            "sonnet": {"input": 3.00, "output": 15.00},
            "opus": {"input": 15.00, "output": 75.00},
        }

        rates = pricing.get(model, pricing["haiku"])
        stats = self.get_session_summary()

        input_cost = (stats.input_tokens / 1_000_000) * rates["input"]
        output_cost = (stats.output_tokens / 1_000_000) * rates["output"]

        return input_cost + output_cost

    def get_calls_since(self, since: datetime) -> list[AICall]:
        """Get all calls since a specific time."""
        with self._call_lock:
            return [c for c in self._calls if c.timestamp >= since]

    def get_persistent_stats(self, since: Optional[datetime] = None) -> SessionStats:
        """
        Get token usage statistics from the database.

        Args:
            since: Only count usage since this time. If None, count all.

        Returns:
            SessionStats with aggregated metrics from the database.
        """
        try:
            from src.storage.database import SyncDatabase

            db = SyncDatabase()
            db.connect()
            try:
                db_stats = db.get_token_usage_stats(since)
            finally:
                db.close()

            stats = SessionStats(
                start_time=since or datetime.min,
                total_calls=db_stats["total_calls"],
                actual_ai_calls=db_stats["actual_ai_calls"],
                cache_hits=db_stats["cache_hits"],
                input_tokens=db_stats["input_tokens"],
                output_tokens=db_stats["output_tokens"],
                total_duration_ms=db_stats["total_duration_ms"],
                calls_by_type=db_stats["calls_by_type"],
                errors=db_stats["errors"],
            )
            return stats
        except Exception as e:
            logger.warning(f"Failed to get persistent stats from database: {e}")
            return SessionStats()

    def clear_persistent_stats(self, before: Optional[datetime] = None) -> int:
        """
        Clear token usage records from database.

        Args:
            before: If provided, only clear records before this time.
                    If None, clear all records.

        Returns:
            Number of records deleted.
        """
        try:
            from src.storage.database import SyncDatabase

            db = SyncDatabase()
            db.connect()
            try:
                return db.clear_token_usage(before)
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to clear persistent stats: {e}")
            return 0

    def format_summary(self) -> str:
        """Format session summary as a string for display."""
        stats = self.get_session_summary()

        lines = [
            "Token Usage Statistics",
            "=" * 40,
            f"Total Calls:      {stats.total_calls}",
            f"Actual AI Calls:  {stats.actual_ai_calls}",
            f"Cache Hits:       {stats.cache_hits}",
            f"Cache Hit Rate:   {stats.cache_hit_rate:.1f}%",
            f"Input Tokens:     {stats.input_tokens:,}",
            f"Output Tokens:    {stats.output_tokens:,}",
            f"Total Tokens:     {stats.total_tokens:,}",
            f"Est. Cost (Haiku): ${self.estimate_cost('haiku'):.4f}",
            "",
            "By Call Type:",
        ]

        for call_type, data in sorted(stats.calls_by_type.items()):
            cached = data["cached"]
            total = data["total"]
            tokens = data["tokens"]
            lines.append(f"  {call_type}: {total} calls, {cached} cached, {tokens:,} tokens")

        if stats.errors > 0:
            lines.append(f"\nErrors: {stats.errors}")

        return "\n".join(lines)


# Global tracker instance
_tracker: Optional[TokenTracker] = None


def get_tracker() -> TokenTracker:
    """Get the global TokenTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker


def record_ai_call(
    call_type: AICallType,
    cached: bool = False,
    input_chars: int = 0,
    output_chars: int = 0,
    **kwargs,
) -> None:
    """Convenience function to record an AI call."""
    get_tracker().record_call(
        call_type=call_type,
        cached=cached,
        input_chars=input_chars,
        output_chars=output_chars,
        **kwargs,
    )

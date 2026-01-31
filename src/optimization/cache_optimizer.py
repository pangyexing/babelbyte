"""Cache optimization utilities for BabelByte."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.storage.database import SyncDatabase

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Metrics for cache performance."""

    total_entries: int = 0
    valid_entries: int = 0
    expired_entries: int = 0
    oldest_entry: Optional[str] = None
    newest_entry: Optional[str] = None
    estimated_size_kb: float = 0.0

    @property
    def utilization_rate(self) -> float:
        """Percentage of valid entries vs total."""
        if self.total_entries == 0:
            return 0.0
        return (self.valid_entries / self.total_entries) * 100


@dataclass
class CacheEfficiency:
    """Analysis of cache efficiency patterns."""

    hit_count: int = 0
    miss_count: int = 0
    content_hits: int = 0
    event_confirm_hits: int = 0
    recommendations: list = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

    @property
    def hit_rate(self) -> float:
        """Overall hit rate percentage."""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return (self.hit_count / total) * 100


class CacheOptimizer:
    """
    Optimizes AI cache usage and performance.

    Provides metrics, cleanup, and efficiency analysis for the AI cache.
    """

    def __init__(self, db: SyncDatabase):
        self.db = db

    def get_cache_metrics(self) -> CacheMetrics:
        """
        Get current cache performance metrics.

        Returns:
            CacheMetrics with current cache state.
        """
        try:
            stats = self.db.get_ai_cache_stats()

            # Estimate size (rough: avg 500 bytes per entry)
            estimated_size = stats.get("valid_entries", 0) * 500 / 1024

            return CacheMetrics(
                total_entries=stats.get("total_entries", 0),
                valid_entries=stats.get("valid_entries", 0),
                expired_entries=stats.get("expired_entries", 0),
                oldest_entry=stats.get("oldest_entry"),
                newest_entry=stats.get("newest_entry"),
                estimated_size_kb=round(estimated_size, 2),
            )
        except Exception as e:
            logger.error(f"Failed to get cache metrics: {e}")
            return CacheMetrics()

    def cleanup_and_optimize(self) -> dict:
        """
        Clean up expired entries and optimize cache.

        Returns:
            Dict with cleanup results: expired_removed, space_freed_kb.
        """
        result = {
            "expired_removed": 0,
            "space_freed_kb": 0.0,
            "optimized": False,
        }

        try:
            # Get metrics before cleanup
            before = self.get_cache_metrics()

            # Remove expired entries
            removed = self.db.cleanup_expired_cache()
            result["expired_removed"] = removed

            # Get metrics after cleanup
            after = self.get_cache_metrics()

            # Estimate freed space
            freed = (before.total_entries - after.total_entries) * 500 / 1024
            result["space_freed_kb"] = round(freed, 2)
            result["optimized"] = True

            logger.info(f"Cache cleanup: removed {removed} entries, freed ~{freed:.2f} KB")

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

        return result

    def analyze_cache_efficiency(self) -> CacheEfficiency:
        """
        Analyze cache hit patterns and provide recommendations.

        Returns:
            CacheEfficiency with analysis and recommendations.
        """
        efficiency = CacheEfficiency()

        try:
            metrics = self.get_cache_metrics()

            # Analyze cache utilization
            if metrics.utilization_rate < 50:
                efficiency.recommendations.append(
                    "Low utilization rate - consider increasing cache TTL"
                )

            if metrics.expired_entries > metrics.valid_entries:
                efficiency.recommendations.append(
                    "Many expired entries - run cleanup to free space"
                )

            if metrics.valid_entries < 100:
                efficiency.recommendations.append(
                    "Cache is underutilized - verify cache is enabled in settings"
                )

            if metrics.estimated_size_kb > 10000:
                efficiency.recommendations.append(
                    "Cache is large (>10MB) - consider reducing TTL or running cleanup"
                )

            # Analyze entry age
            if metrics.oldest_entry:
                try:
                    oldest = datetime.fromisoformat(metrics.oldest_entry)
                    age_days = (datetime.now() - oldest).days
                    if age_days > 30:
                        efficiency.recommendations.append(
                            f"Oldest entry is {age_days} days old - "
                            "consider if long TTL is appropriate"
                        )
                except Exception:
                    pass

            if not efficiency.recommendations:
                efficiency.recommendations.append("Cache is operating efficiently")

        except Exception as e:
            logger.error(f"Cache analysis failed: {e}")
            efficiency.recommendations.append(f"Analysis failed: {e}")

        return efficiency

    def get_cache_summary(self) -> str:
        """
        Get a formatted summary of cache status.

        Returns:
            Formatted string with cache information.
        """
        metrics = self.get_cache_metrics()
        efficiency = self.analyze_cache_efficiency()

        lines = [
            "AI Cache Status",
            "=" * 40,
            f"Total Entries:    {metrics.total_entries:,}",
            f"Valid Entries:    {metrics.valid_entries:,}",
            f"Expired Entries:  {metrics.expired_entries:,}",
            f"Utilization:      {metrics.utilization_rate:.1f}%",
            f"Est. Size:        {metrics.estimated_size_kb:.2f} KB",
            "",
            "Age:",
            f"  Oldest: {metrics.oldest_entry or 'N/A'}",
            f"  Newest: {metrics.newest_entry or 'N/A'}",
            "",
            "Recommendations:",
        ]

        for rec in efficiency.recommendations:
            lines.append(f"  - {rec}")

        return "\n".join(lines)

    def preload_common_patterns(self) -> int:
        """
        Preload cache with common patterns (placeholder for future use).

        Returns:
            Number of patterns preloaded.
        """
        # This could be used to pre-cache common AI responses
        # For now, just return 0 as it's not implemented
        logger.debug("Preload not implemented - cache is populated on-demand")
        return 0

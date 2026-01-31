"""Data validation module for verifying database integrity."""

import logging
from dataclasses import dataclass, field
from enum import Enum

from src.storage.database import SyncDatabase
from src.validation.diagnostic_queries import DiagnosticQueries

logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Status of a validation check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    status: CheckStatus
    message: str
    count: int = 0
    details: list = field(default_factory=list)
    can_fix: bool = False


@dataclass
class ValidationResult:
    """Aggregate result of all validation checks."""

    checks: list[ValidationCheck] = field(default_factory=list)
    total_issues: int = 0
    fixed_issues: int = 0

    @property
    def passed(self) -> bool:
        """Return True if all checks passed."""
        return all(c.status in (CheckStatus.PASS, CheckStatus.SKIP) for c in self.checks)

    @property
    def summary(self) -> str:
        """Return a summary of validation results."""
        passed = sum(1 for c in self.checks if c.status == CheckStatus.PASS)
        failed = sum(1 for c in self.checks if c.status == CheckStatus.FAIL)
        warned = sum(1 for c in self.checks if c.status == CheckStatus.WARN)
        return f"Passed: {passed}, Failed: {failed}, Warnings: {warned}"


class DataValidator:
    """Validates data integrity across all database tables."""

    def __init__(self, db: SyncDatabase):
        self.db = db

    def run_all_checks(self, verbose: bool = False) -> ValidationResult:
        """Run all validation checks.

        Args:
            verbose: If True, include detailed information for each check.

        Returns:
            ValidationResult with all check results.
        """
        result = ValidationResult()

        # Run each check
        result.checks.append(self._check_orphan_content_items(verbose))
        result.checks.append(self._check_duplicate_external_ids(verbose))
        result.checks.append(self._check_processed_no_summary(verbose))
        result.checks.append(self._check_importance_range(verbose))
        result.checks.append(self._check_empty_clusters(verbose))
        result.checks.append(self._check_cluster_count_mismatch(verbose))
        result.checks.append(self._check_orphan_event_members(verbose))
        result.checks.append(self._check_fts_missing(verbose))
        result.checks.append(self._check_duplicate_memberships(verbose))
        result.checks.append(self._check_orphan_action_items(verbose))
        result.checks.append(self._check_orphan_topics(verbose))
        result.checks.append(self._check_expired_cache(verbose))

        # Count total issues
        result.total_issues = sum(c.count for c in result.checks if c.status == CheckStatus.FAIL)

        return result

    def fix_issues(self, result: ValidationResult) -> int:
        """Attempt to fix identified issues.

        Args:
            result: ValidationResult from run_all_checks.

        Returns:
            Number of issues fixed.
        """
        fixed = 0

        for check in result.checks:
            if check.status != CheckStatus.FAIL or not check.can_fix:
                continue

            if check.name == "cluster_count_mismatch":
                fixed += self._fix_cluster_counts()
            elif check.name == "empty_clusters":
                fixed += self._fix_empty_clusters()
            elif check.name == "duplicate_memberships":
                fixed += self._fix_duplicate_memberships()
            elif check.name == "fts_missing":
                fixed += self._fix_fts_missing()
            elif check.name == "expired_cache":
                fixed += self._fix_expired_cache()

        result.fixed_issues = fixed
        return fixed

    def _execute_query(self, query: str) -> list:
        """Execute a query and return results."""
        try:
            conn = self.db._async_db._connection
            if conn is None:
                return []
            cursor = self.db._run(conn.execute(query))
            rows = self.db._run(cursor.fetchall())
            return list(rows)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def _check_orphan_content_items(self, verbose: bool) -> ValidationCheck:
        """Check for content items with invalid subscription_id."""
        rows = self._execute_query(DiagnosticQueries.ORPHAN_CONTENT_ITEMS)
        count = len(rows)

        if count == 0:
            return ValidationCheck(
                name="orphan_content_items",
                status=CheckStatus.PASS,
                message="No orphan content items found",
            )

        details = [{"id": r["id"], "title": r["title"][:50]} for r in rows[:10]] if verbose else []
        return ValidationCheck(
            name="orphan_content_items",
            status=CheckStatus.FAIL,
            message=f"Found {count} content items with invalid subscription_id",
            count=count,
            details=details,
            can_fix=False,  # Requires manual investigation
        )

    def _check_duplicate_external_ids(self, verbose: bool) -> ValidationCheck:
        """Check for duplicate external_id per source_type."""
        rows = self._execute_query(DiagnosticQueries.DUPLICATE_EXTERNAL_IDS)
        count = len(rows)

        if count == 0:
            return ValidationCheck(
                name="duplicate_external_ids",
                status=CheckStatus.PASS,
                message="No duplicate external_ids found",
            )

        details = (
            [{"source": r["source_type"], "external_id": r["external_id"]} for r in rows[:10]]
            if verbose
            else []
        )
        return ValidationCheck(
            name="duplicate_external_ids",
            status=CheckStatus.WARN,
            message=f"Found {count} duplicate external_id entries (may be valid)",
            count=count,
            details=details,
            can_fix=False,
        )

    def _check_processed_no_summary(self, verbose: bool) -> ValidationCheck:
        """Check for processed items without summary."""
        rows = self._execute_query(DiagnosticQueries.PROCESSED_NO_SUMMARY)
        count = len(rows)

        if count == 0:
            return ValidationCheck(
                name="processed_no_summary",
                status=CheckStatus.PASS,
                message="All processed items have summaries",
            )

        details = [{"id": r["id"], "title": r["title"][:50]} for r in rows[:10]] if verbose else []
        return ValidationCheck(
            name="processed_no_summary",
            status=CheckStatus.WARN,
            message=f"Found {count} processed items without summary",
            count=count,
            details=details,
            can_fix=False,
        )

    def _check_importance_range(self, verbose: bool) -> ValidationCheck:
        """Check for importance scores outside 1-10 range."""
        rows = self._execute_query(DiagnosticQueries.IMPORTANCE_OUT_OF_RANGE)
        count = len(rows)

        if count == 0:
            return ValidationCheck(
                name="importance_range",
                status=CheckStatus.PASS,
                message="All importance scores are within 1-10 range",
            )

        details = (
            [{"id": r["id"], "score": r["importance_score"]} for r in rows[:10]] if verbose else []
        )
        return ValidationCheck(
            name="importance_range",
            status=CheckStatus.FAIL,
            message=f"Found {count} items with invalid importance score",
            count=count,
            details=details,
            can_fix=False,
        )

    def _check_empty_clusters(self, verbose: bool) -> ValidationCheck:
        """Check for event clusters with no members."""
        rows = self._execute_query(DiagnosticQueries.EMPTY_CLUSTERS)
        count = len(rows)

        if count == 0:
            return ValidationCheck(
                name="empty_clusters",
                status=CheckStatus.PASS,
                message="No empty event clusters found",
            )

        details = (
            [{"id": r["id"], "title": r["event_title"][:50]} for r in rows[:10]] if verbose else []
        )
        return ValidationCheck(
            name="empty_clusters",
            status=CheckStatus.FAIL,
            message=f"Found {count} empty event clusters",
            count=count,
            details=details,
            can_fix=True,
        )

    def _check_cluster_count_mismatch(self, verbose: bool) -> ValidationCheck:
        """Check for mismatched article_count vs actual member count."""
        rows = self._execute_query(DiagnosticQueries.CLUSTER_COUNT_MISMATCH)
        count = len(rows)

        if count == 0:
            return ValidationCheck(
                name="cluster_count_mismatch",
                status=CheckStatus.PASS,
                message="All cluster article counts are correct",
            )

        details = (
            [
                {"id": r["id"], "stored": r["stored_count"], "actual": r["actual_count"]}
                for r in rows[:10]
            ]
            if verbose
            else []
        )
        return ValidationCheck(
            name="cluster_count_mismatch",
            status=CheckStatus.FAIL,
            message=f"Found {count} clusters with incorrect article_count",
            count=count,
            details=details,
            can_fix=True,
        )

    def _check_orphan_event_members(self, verbose: bool) -> ValidationCheck:
        """Check for event members with invalid foreign keys."""
        rows_cluster = self._execute_query(DiagnosticQueries.ORPHAN_EVENT_MEMBERS_CLUSTER)
        rows_content = self._execute_query(DiagnosticQueries.ORPHAN_EVENT_MEMBERS_CONTENT)
        count = len(rows_cluster) + len(rows_content)

        if count == 0:
            return ValidationCheck(
                name="orphan_event_members",
                status=CheckStatus.PASS,
                message="No orphan event members found",
            )

        details = []
        if verbose:
            for r in rows_cluster[:5]:
                details.append({"type": "cluster", "cluster_id": r["event_cluster_id"]})
            for r in rows_content[:5]:
                details.append({"type": "content", "content_id": r["content_item_id"]})

        return ValidationCheck(
            name="orphan_event_members",
            status=CheckStatus.FAIL,
            message=f"Found {count} orphan event member entries",
            count=count,
            details=details,
            can_fix=False,
        )

    def _check_fts_missing(self, verbose: bool) -> ValidationCheck:
        """Check for processed items not in FTS index."""
        rows = self._execute_query(DiagnosticQueries.FTS_MISSING_ENTRIES)
        count = len(rows)

        if count == 0:
            return ValidationCheck(
                name="fts_missing",
                status=CheckStatus.PASS,
                message="All processed items are in FTS index",
            )

        details = [{"id": r["id"], "title": r["title"][:50]} for r in rows[:10]] if verbose else []
        return ValidationCheck(
            name="fts_missing",
            status=CheckStatus.WARN,
            message=f"Found {count} processed items missing from FTS index",
            count=count,
            details=details,
            can_fix=True,
        )

    def _check_duplicate_memberships(self, verbose: bool) -> ValidationCheck:
        """Check for items in multiple clusters (should not happen)."""
        rows = self._execute_query(DiagnosticQueries.DUPLICATE_CLUSTER_MEMBERSHIPS)
        count = len(rows)

        if count == 0:
            return ValidationCheck(
                name="duplicate_memberships",
                status=CheckStatus.PASS,
                message="No duplicate cluster memberships found",
            )

        details = []
        if verbose:
            for r in rows[:10]:
                details.append(
                    {"content_id": r["content_item_id"], "cluster_count": r["cluster_count"]}
                )
        return ValidationCheck(
            name="duplicate_memberships",
            status=CheckStatus.FAIL,
            message=f"Found {count} items in multiple clusters",
            count=count,
            details=details,
            can_fix=True,
        )

    def _check_orphan_action_items(self, verbose: bool) -> ValidationCheck:
        """Check for action items with invalid content_item_id."""
        rows = self._execute_query(DiagnosticQueries.ORPHAN_ACTION_ITEMS)
        count = len(rows)

        if count == 0:
            return ValidationCheck(
                name="orphan_action_items",
                status=CheckStatus.PASS,
                message="No orphan action items found",
            )

        details = (
            [{"id": r["id"], "description": r["description"][:50]} for r in rows[:10]]
            if verbose
            else []
        )
        return ValidationCheck(
            name="orphan_action_items",
            status=CheckStatus.WARN,
            message=f"Found {count} orphan action items",
            count=count,
            details=details,
            can_fix=False,
        )

    def _check_orphan_topics(self, verbose: bool) -> ValidationCheck:
        """Check for orphan topic associations and snapshots."""
        rows_ct = self._execute_query(DiagnosticQueries.ORPHAN_CONTENT_TOPICS)
        rows_ts = self._execute_query(DiagnosticQueries.ORPHAN_TOPIC_SNAPSHOTS)
        count = len(rows_ct) + len(rows_ts)

        if count == 0:
            return ValidationCheck(
                name="orphan_topics",
                status=CheckStatus.PASS,
                message="No orphan topic data found",
            )

        return ValidationCheck(
            name="orphan_topics",
            status=CheckStatus.WARN,
            message=f"Found {count} orphan topic associations/snapshots",
            count=count,
            can_fix=False,
        )

    def _check_expired_cache(self, verbose: bool) -> ValidationCheck:
        """Check for expired AI cache entries."""
        rows = self._execute_query(DiagnosticQueries.EXPIRED_AI_CACHE)
        count = rows[0]["count"] if rows else 0

        if count == 0:
            return ValidationCheck(
                name="expired_cache",
                status=CheckStatus.PASS,
                message="No expired cache entries",
            )

        return ValidationCheck(
            name="expired_cache",
            status=CheckStatus.WARN,
            message=f"Found {count} expired AI cache entries",
            count=count,
            can_fix=True,
        )

    # Fix methods

    def _fix_cluster_counts(self) -> int:
        """Fix mismatched cluster article counts."""
        try:
            query = """
                UPDATE event_clusters SET article_count = (
                    SELECT COUNT(*) FROM event_members
                    WHERE event_cluster_id = event_clusters.id
                )
            """
            conn = self.db._async_db._connection
            self.db._run(conn.execute(query))
            self.db._run(conn.commit())
            logger.info("Fixed cluster article counts")
            return 1
        except Exception as e:
            logger.error(f"Failed to fix cluster counts: {e}")
            return 0

    def _fix_empty_clusters(self) -> int:
        """Delete empty event clusters."""
        try:
            query = """
                DELETE FROM event_clusters
                WHERE id NOT IN (SELECT DISTINCT event_cluster_id FROM event_members)
            """
            conn = self.db._async_db._connection
            cursor = self.db._run(conn.execute(query))
            self.db._run(conn.commit())
            deleted = cursor.rowcount
            logger.info(f"Deleted {deleted} empty clusters")
            return deleted
        except Exception as e:
            logger.error(f"Failed to fix empty clusters: {e}")
            return 0

    def _fix_duplicate_memberships(self) -> int:
        """Remove duplicate cluster memberships."""
        try:
            removed = self.db.cleanup_duplicate_cluster_memberships()
            logger.info(f"Removed {removed} duplicate memberships")
            return removed
        except Exception as e:
            logger.error(f"Failed to fix duplicate memberships: {e}")
            return 0

    def _fix_fts_missing(self) -> int:
        """Rebuild FTS index."""
        try:
            count = self.db.rebuild_fts_index()
            logger.info(f"Rebuilt FTS index with {count} entries")
            return count
        except Exception as e:
            logger.error(f"Failed to rebuild FTS index: {e}")
            return 0

    def _fix_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        try:
            cleaned = self.db.cleanup_expired_cache()
            logger.info(f"Cleaned up {cleaned} expired cache entries")
            return cleaned
        except Exception as e:
            logger.error(f"Failed to clean up cache: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get database statistics for diagnostics."""
        stats = {}

        # Content stats
        content_rows = self._execute_query(DiagnosticQueries.CONTENT_STATS)
        if content_rows:
            r = content_rows[0]
            stats["content"] = {
                "total": r["total"],
                "processed": r["processed"],
                "delivered": r["delivered"],
                "avg_importance": round(r["avg_importance"] or 0, 2),
            }

        # Cluster stats
        cluster_rows = self._execute_query(DiagnosticQueries.CLUSTER_STATS)
        if cluster_rows:
            r = cluster_rows[0]
            stats["clusters"] = {
                "total": r["total_clusters"],
                "total_members": r["total_members"],
                "avg_members": round(r["avg_members_per_cluster"] or 0, 2),
            }

        # Cache stats
        cache_rows = self._execute_query(DiagnosticQueries.AI_CACHE_STATS)
        if cache_rows:
            r = cache_rows[0]
            stats["cache"] = {
                "total": r["total_entries"],
                "valid": r["valid_entries"],
                "expired": r["expired_entries"],
            }

        # Categories
        category_rows = self._execute_query(DiagnosticQueries.ITEMS_BY_CATEGORY)
        stats["categories"] = {r["category"]: r["count"] for r in category_rows}

        return stats

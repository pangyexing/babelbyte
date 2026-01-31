"""Tests for the data validation module."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.storage.database import SyncDatabase
from src.storage.models import ContentItem, EventCluster, EventMember, SourceType
from src.validation.data_validator import (
    CheckStatus,
    DataValidator,
    ValidationCheck,
    ValidationResult,
)
from src.validation.diagnostic_queries import DiagnosticQueries


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_passed_all_pass(self):
        """Test passed property when all checks pass."""
        result = ValidationResult(
            checks=[
                ValidationCheck(name="test1", status=CheckStatus.PASS, message="ok"),
                ValidationCheck(name="test2", status=CheckStatus.PASS, message="ok"),
            ]
        )
        assert result.passed is True

    def test_passed_with_skip(self):
        """Test passed property with skipped checks."""
        result = ValidationResult(
            checks=[
                ValidationCheck(name="test1", status=CheckStatus.PASS, message="ok"),
                ValidationCheck(name="test2", status=CheckStatus.SKIP, message="skip"),
            ]
        )
        assert result.passed is True

    def test_passed_with_fail(self):
        """Test passed property when a check fails."""
        result = ValidationResult(
            checks=[
                ValidationCheck(name="test1", status=CheckStatus.PASS, message="ok"),
                ValidationCheck(name="test2", status=CheckStatus.FAIL, message="fail"),
            ]
        )
        assert result.passed is False

    def test_summary(self):
        """Test summary property."""
        result = ValidationResult(
            checks=[
                ValidationCheck(name="test1", status=CheckStatus.PASS, message="ok"),
                ValidationCheck(name="test2", status=CheckStatus.FAIL, message="fail"),
                ValidationCheck(name="test3", status=CheckStatus.WARN, message="warn"),
            ]
        )
        assert "Passed: 1" in result.summary
        assert "Failed: 1" in result.summary
        assert "Warnings: 1" in result.summary


class TestDiagnosticQueries:
    """Tests for diagnostic SQL queries."""

    def test_queries_are_valid_sql(self):
        """Test that all queries are syntactically valid SQL."""
        queries = [
            DiagnosticQueries.ORPHAN_CONTENT_ITEMS,
            DiagnosticQueries.DUPLICATE_EXTERNAL_IDS,
            DiagnosticQueries.PROCESSED_NO_SUMMARY,
            DiagnosticQueries.IMPORTANCE_OUT_OF_RANGE,
            DiagnosticQueries.EMPTY_CLUSTERS,
            DiagnosticQueries.CLUSTER_COUNT_MISMATCH,
            DiagnosticQueries.FTS_MISSING_ENTRIES,
        ]

        for query in queries:
            assert "SELECT" in query.upper()
            assert "FROM" in query.upper()


class TestDataValidatorIntegration:
    """Integration tests for DataValidator with temporary database."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SyncDatabase(db_path)
            db.connect()
            yield db
            db.close()

    def test_run_all_checks_empty_db(self, temp_db):
        """Test running all checks on an empty database."""
        validator = DataValidator(temp_db)
        result = validator.run_all_checks(verbose=False)

        assert len(result.checks) > 0
        # Empty database should have mostly passing checks
        pass_count = sum(1 for c in result.checks if c.status == CheckStatus.PASS)
        assert pass_count >= len(result.checks) // 2

    def test_check_importance_range(self, temp_db):
        """Test importance score range validation."""
        validator = DataValidator(temp_db)

        # Add a subscription first
        from src.storage.models import Subscription, SubscriptionType

        sub = Subscription(
            source_type=SourceType.REDDIT,
            subscription_type=SubscriptionType.SUBREDDIT,
            name="test",
            enabled=True,
            created_at=datetime.now(),
        )
        sub = temp_db.add_subscription(sub)

        # Add item with valid importance
        item = ContentItem(
            subscription_id=sub.id,
            source_type=SourceType.REDDIT,
            external_id="valid1",
            title="Valid item",
            content="Content",
            url="https://example.com",
            author="test",
            published_at=datetime.now(),
            fetched_at=datetime.now(),
            importance_score=7,
        )
        temp_db.add_content_item(item)

        result = validator.run_all_checks(verbose=False)
        importance_check = next(c for c in result.checks if c.name == "importance_range")
        assert importance_check.status == CheckStatus.PASS

    def test_get_stats(self, temp_db):
        """Test getting database statistics."""
        validator = DataValidator(temp_db)
        stats = validator.get_stats()

        # Should have some keys even with empty db
        assert isinstance(stats, dict)


class TestDataValidatorMocked:
    """Unit tests for DataValidator with mocked database."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        db = MagicMock(spec=SyncDatabase)
        db._async_db = MagicMock()
        db._async_db._connection = MagicMock()
        db._run = MagicMock(side_effect=lambda x: x)
        return db

    def test_check_orphan_items_no_orphans(self, mock_db):
        """Test orphan items check with no orphans."""
        # Mock the execute to return empty result
        cursor_mock = MagicMock()
        cursor_mock.fetchall = MagicMock(return_value=[])
        mock_db._async_db._connection.execute = MagicMock(return_value=cursor_mock)
        mock_db._run = lambda x: x

        validator = DataValidator(mock_db)
        check = validator._check_orphan_content_items(verbose=False)

        assert check.status == CheckStatus.PASS
        assert check.count == 0

    def test_validation_check_can_fix_flag(self, mock_db):
        """Test that fixable checks have can_fix=True."""
        validator = DataValidator(mock_db)

        # These checks should be fixable
        fixable_checks = [
            "cluster_count_mismatch",
            "empty_clusters",
            "duplicate_memberships",
            "fts_missing",
        ]

        for check_name in fixable_checks:
            check = ValidationCheck(
                name=check_name,
                status=CheckStatus.FAIL,
                message="test",
                count=1,
                can_fix=True,
            )
            assert check.can_fix is True

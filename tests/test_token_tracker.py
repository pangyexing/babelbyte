"""Tests for the token tracker module."""

from datetime import datetime, timedelta

import pytest

from src.analytics.token_tracker import (
    AICall,
    AICallType,
    SessionStats,
    TokenTracker,
    get_tracker,
    record_ai_call,
)


class TestAICallType:
    """Tests for AICallType enum."""

    def test_estimated_tokens(self):
        """Test that all call types have estimated tokens."""
        for call_type in AICallType:
            assert call_type.estimated_tokens > 0

    def test_content_heavy_tokens(self):
        """Test content heavy token estimate."""
        assert AICallType.CONTENT_HEAVY.estimated_tokens == 825

    def test_event_confirm_tokens(self):
        """Test event confirm token estimate."""
        assert AICallType.EVENT_CONFIRM.estimated_tokens == 220


class TestSessionStats:
    """Tests for SessionStats dataclass."""

    def test_cache_hit_rate_zero_calls(self):
        """Test cache hit rate with zero calls."""
        stats = SessionStats()
        assert stats.cache_hit_rate == 0.0

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        stats = SessionStats(total_calls=10, cache_hits=3)
        assert stats.cache_hit_rate == 30.0

    def test_total_tokens(self):
        """Test total tokens calculation."""
        stats = SessionStats(input_tokens=1000, output_tokens=500)
        assert stats.total_tokens == 1500


class TestTokenTracker:
    """Tests for TokenTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for testing."""
        tracker = TokenTracker()
        tracker.reset()
        return tracker

    def test_singleton_pattern(self):
        """Test that TokenTracker is a singleton."""
        tracker1 = TokenTracker()
        tracker2 = TokenTracker()
        assert tracker1 is tracker2

    def test_record_call_basic(self, tracker):
        """Test basic call recording."""
        tracker.record_call(
            call_type=AICallType.CONTENT_HEAVY,
            cached=False,
            input_chars=1000,
            output_chars=500,
        )

        stats = tracker.get_session_summary()
        assert stats.total_calls == 1
        assert stats.actual_ai_calls == 1
        assert stats.cache_hits == 0

    def test_record_cached_call(self, tracker):
        """Test recording a cached call."""
        tracker.record_call(
            call_type=AICallType.CONTENT_HEAVY,
            cached=True,
            input_chars=1000,
        )

        stats = tracker.get_session_summary()
        assert stats.total_calls == 1
        assert stats.actual_ai_calls == 0
        assert stats.cache_hits == 1

    def test_record_multiple_calls(self, tracker):
        """Test recording multiple calls of different types."""
        tracker.record_call(call_type=AICallType.CONTENT_HEAVY, cached=False)
        tracker.record_call(call_type=AICallType.CONTENT_LIGHT, cached=False)
        tracker.record_call(call_type=AICallType.EVENT_CONFIRM, cached=True)

        stats = tracker.get_session_summary()
        assert stats.total_calls == 3
        assert stats.actual_ai_calls == 2
        assert stats.cache_hits == 1

    def test_calls_by_type(self, tracker):
        """Test tracking calls by type."""
        tracker.record_call(call_type=AICallType.CONTENT_HEAVY, cached=False)
        tracker.record_call(call_type=AICallType.CONTENT_HEAVY, cached=True)
        tracker.record_call(call_type=AICallType.EVENT_CONFIRM, cached=False)

        stats = tracker.get_session_summary()
        assert "content_heavy" in stats.calls_by_type
        assert stats.calls_by_type["content_heavy"]["total"] == 2
        assert stats.calls_by_type["content_heavy"]["cached"] == 1

    def test_record_call_with_error(self, tracker):
        """Test recording a failed call."""
        tracker.record_call(
            call_type=AICallType.CONTENT_HEAVY,
            cached=False,
            success=False,
            error="Test error",
        )

        stats = tracker.get_session_summary()
        assert stats.errors == 1

    def test_estimate_cost_haiku(self, tracker):
        """Test cost estimation for Haiku model."""
        tracker.record_call(
            call_type=AICallType.CONTENT_HEAVY,
            cached=False,
            input_tokens=1000,
            output_tokens=500,
        )

        cost = tracker.estimate_cost("haiku")
        assert cost > 0
        # Haiku is the cheapest model
        assert cost < tracker.estimate_cost("sonnet")
        assert cost < tracker.estimate_cost("opus")

    def test_estimate_cost_different_models(self, tracker):
        """Test that cost increases with model tier."""
        tracker.record_call(
            call_type=AICallType.CONTENT_HEAVY,
            cached=False,
            input_tokens=10000,
            output_tokens=5000,
        )

        haiku_cost = tracker.estimate_cost("haiku")
        sonnet_cost = tracker.estimate_cost("sonnet")
        opus_cost = tracker.estimate_cost("opus")

        assert haiku_cost < sonnet_cost < opus_cost

    def test_reset(self, tracker):
        """Test resetting the tracker."""
        tracker.record_call(call_type=AICallType.CONTENT_HEAVY, cached=False)
        tracker.reset()

        stats = tracker.get_session_summary()
        assert stats.total_calls == 0

    def test_get_calls_since(self, tracker):
        """Test getting calls since a specific time."""
        old_time = datetime.now() - timedelta(hours=1)
        tracker.record_call(call_type=AICallType.CONTENT_HEAVY, cached=False)

        calls = tracker.get_calls_since(old_time)
        assert len(calls) == 1

        future_time = datetime.now() + timedelta(hours=1)
        calls = tracker.get_calls_since(future_time)
        assert len(calls) == 0

    def test_format_summary(self, tracker):
        """Test formatting the summary string."""
        tracker.record_call(call_type=AICallType.CONTENT_HEAVY, cached=False)
        tracker.record_call(call_type=AICallType.EVENT_CONFIRM, cached=True)

        summary = tracker.format_summary()
        assert "Token Usage Statistics" in summary
        assert "Total Calls" in summary
        assert "Cache Hit" in summary

    def test_token_estimation_from_chars(self, tracker):
        """Test that tokens are estimated from characters."""
        tracker.record_call(
            call_type=AICallType.CONTENT_HEAVY,
            cached=False,
            input_chars=3000,  # Should estimate ~1000 tokens
            output_chars=1500,  # Should estimate ~500 tokens
        )

        stats = tracker.get_session_summary()
        # Tokens should be estimated from chars
        assert stats.input_tokens > 0


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_get_tracker_returns_singleton(self):
        """Test that get_tracker returns the same instance."""
        tracker1 = get_tracker()
        tracker2 = get_tracker()
        assert tracker1 is tracker2

    def test_record_ai_call_convenience(self):
        """Test the convenience function for recording calls."""
        tracker = get_tracker()
        tracker.reset()

        record_ai_call(
            call_type=AICallType.EVENT_CONFIRM,
            cached=True,
            input_chars=500,
        )

        stats = tracker.get_session_summary()
        assert stats.total_calls >= 1

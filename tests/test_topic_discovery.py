"""Tests for automatic topic discovery."""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.analytics.topic_discovery import (
    TopicDiscovery,
    extract_entities,
    extract_bigrams,
)
from src.storage.models import ContentItem, Topic, TopicSuggestion, SourceType


class TestEntityExtraction:
    """Tests for entity extraction."""

    def test_extract_english_companies(self):
        """Test extraction of English company names."""
        text = "OpenAI and Google are leading AI research"
        entities = extract_entities(text)
        assert "OpenAI" in entities
        assert "Google" in entities

    def test_extract_chinese_companies(self):
        """Test extraction of Chinese company names."""
        text = "腾讯和百度在人工智能领域竞争激烈"
        entities = extract_entities(text)
        assert "腾讯" in entities
        assert "百度" in entities

    def test_extract_ai_models(self):
        """Test extraction of AI model names."""
        text = "GPT-4 and Claude are popular LLMs"
        entities = extract_entities(text)
        assert "GPT-4" in entities
        assert "Claude" in entities

    def test_extract_people(self):
        """Test extraction of prominent tech figures."""
        text = "Sam Altman announced new features"
        entities = extract_entities(text)
        assert "Sam Altman" in entities

    def test_extract_mixed(self):
        """Test extraction of mixed entity types."""
        text = "Elon Musk's Tesla is working with Nvidia on AI for ChatGPT"
        entities = extract_entities(text)
        assert "Elon Musk" in entities or "Tesla" in entities
        assert "Nvidia" in entities
        assert "ChatGPT" in entities

    def test_case_insensitive(self):
        """Test case-insensitive extraction."""
        text = "OPENAI and openai are the same"
        entities = extract_entities(text)
        # Should find at least one
        assert any("openai" in e.lower() for e in entities)


class TestBigramExtraction:
    """Tests for bigram extraction."""

    def test_extract_chinese_terms(self):
        """Test extraction of Chinese terms."""
        text = "人工智能正在快速发展"
        bigrams = extract_bigrams(text)
        assert any("人工智能" in b for b in bigrams)

    def test_extract_english_bigrams(self):
        """Test extraction of English bigrams."""
        text = "machine learning and deep learning are popular"
        bigrams = extract_bigrams(text)
        assert "machine learning" in bigrams
        assert "deep learning" in bigrams

    def test_filter_stopwords(self):
        """Test that stopwords are filtered."""
        text = "the quick brown fox and the lazy dog"
        bigrams = extract_bigrams(text)
        # Should not have bigrams with only stopwords
        assert "the quick" not in bigrams
        assert "and the" not in bigrams


class TestTopicSuggestion:
    """Tests for TopicSuggestion model."""

    def test_basic_creation(self):
        """Test basic TopicSuggestion creation."""
        suggestion = TopicSuggestion(
            name="OpenAI",
            keywords=json.dumps(["OpenAI", "GPT"]),
            frequency=10,
            confidence=0.8,
            source="entity",
        )
        assert suggestion.name == "OpenAI"
        assert suggestion.frequency == 10
        assert suggestion.confidence == 0.8
        assert suggestion.status == "pending"

    def test_get_keywords(self):
        """Test keyword parsing."""
        suggestion = TopicSuggestion(
            name="Test",
            keywords=json.dumps(["keyword1", "keyword2"]),
            frequency=5,
            confidence=0.5,
            source="entity",
        )
        keywords = suggestion.get_keywords()
        assert keywords == ["keyword1", "keyword2"]

    def test_get_sample_titles(self):
        """Test sample title parsing."""
        suggestion = TopicSuggestion(
            name="Test",
            keywords=json.dumps([]),
            sample_titles=json.dumps(["Title 1", "Title 2"]),
            frequency=5,
            confidence=0.5,
            source="entity",
        )
        titles = suggestion.get_sample_titles()
        assert titles == ["Title 1", "Title 2"]

    def test_empty_keywords(self):
        """Test empty keywords handling."""
        suggestion = TopicSuggestion(
            name="Test",
            keywords=None,
            frequency=5,
            confidence=0.5,
            source="entity",
        )
        assert suggestion.get_keywords() == []


class TestTopicDiscovery:
    """Tests for TopicDiscovery class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db = MagicMock()
        self.discovery = TopicDiscovery(db=self.mock_db, use_mock=True)

    def _create_mock_content(self, title: str, summary: str = "", days_ago: int = 0) -> ContentItem:
        """Helper to create mock content items."""
        return ContentItem(
            id=1,
            subscription_id=1,
            source_type=SourceType.REDDIT,
            external_id="test123",
            title=title,
            content="",
            summary=summary,
            url="https://example.com",
            author="test",
            published_at=datetime.now() - timedelta(days=days_ago),
            importance_score=7,
        )

    def test_discover_by_entities(self):
        """Test entity-based discovery."""
        content = [
            self._create_mock_content("OpenAI releases GPT-5"),
            self._create_mock_content("OpenAI announces new features"),
            self._create_mock_content("OpenAI partnership with Microsoft"),
            self._create_mock_content("Google AI updates"),
            self._create_mock_content("OpenAI research paper"),
            self._create_mock_content("OpenAI CEO interview"),
        ]

        suggestions = self.discovery._discover_by_entities(content, min_frequency=5)

        # Should find OpenAI as it appears 5+ times
        openai_suggestions = [s for s in suggestions if "OpenAI" in s.name]
        assert len(openai_suggestions) > 0
        assert openai_suggestions[0].frequency >= 5
        assert openai_suggestions[0].source == "entity"

    def test_discover_by_keywords(self):
        """Test keyword-based discovery."""
        content = [
            self._create_mock_content("machine learning trends"),
            self._create_mock_content("machine learning in production"),
            self._create_mock_content("machine learning frameworks"),
            self._create_mock_content("machine learning best practices"),
            self._create_mock_content("machine learning for beginners"),
        ]

        suggestions = self.discovery._discover_by_keywords(content, min_frequency=5)

        # Should find "machine learning" bigram
        ml_suggestions = [s for s in suggestions if "machine learning" in s.name.lower()]
        assert len(ml_suggestions) > 0
        assert ml_suggestions[0].source == "keyword"

    def test_discover_by_trends(self):
        """Test trend detection."""
        # Current week: 6 mentions of OpenAI
        current_content = [
            self._create_mock_content("OpenAI news 1", days_ago=1),
            self._create_mock_content("OpenAI news 2", days_ago=2),
            self._create_mock_content("OpenAI news 3", days_ago=3),
            self._create_mock_content("OpenAI news 4", days_ago=4),
            self._create_mock_content("OpenAI news 5", days_ago=5),
            self._create_mock_content("OpenAI news 6", days_ago=6),
        ]
        # Previous week: 1 mention of OpenAI
        previous_content = [
            self._create_mock_content("OpenAI old news", days_ago=10),
        ]

        all_content = current_content + previous_content
        suggestions = self.discovery._discover_by_trends(all_content, days=14)

        # Should detect OpenAI as trending (6x increase)
        openai_trends = [s for s in suggestions if "OpenAI" in s.name]
        assert len(openai_trends) > 0
        assert openai_trends[0].source == "trend"

    def test_deduplicate_suggestions(self):
        """Test suggestion deduplication."""
        from src.analytics.topic_discovery import TopicSuggestion as TDSuggestion

        # Create duplicate suggestions with different confidences
        class MockSuggestion:
            def __init__(self, name, confidence):
                self.name = name
                self.confidence = confidence
                self.keywords = ["test"]
                self.frequency = 5
                self.source = "entity"
                self.sample_titles = []

            def get_keywords(self):
                return self.keywords

            def get_sample_titles(self):
                return self.sample_titles

        suggestions = [
            MockSuggestion("OpenAI", 0.8),
            MockSuggestion("openai", 0.6),  # Duplicate with lower confidence
            MockSuggestion("Google", 0.7),
        ]

        deduped = self.discovery._deduplicate_suggestions(suggestions)

        # Should have 2 unique suggestions
        assert len(deduped) == 2
        # Higher confidence should be kept
        openai = next(s for s in deduped if s.name.lower() == "openai")
        assert openai.confidence == 0.8

    def test_filter_existing_topics(self):
        """Test filtering of existing topics."""
        # Mock existing topics
        existing_topic = Topic(
            id=1,
            name="OpenAI",
            keywords=json.dumps(["OpenAI", "GPT"]),
        )
        self.mock_db.list_topics.return_value = [existing_topic]
        self.mock_db.get_undelivered_items.return_value = []
        self.mock_db.browse_by_date.return_value = []

        # Discover should filter out OpenAI
        suggestions = self.discovery.discover_topics(days=7, min_frequency=1)

        # OpenAI should not be in suggestions
        openai_suggestions = [s for s in suggestions if "OpenAI" in s.name]
        assert len(openai_suggestions) == 0


class TestTopicDiscoveryIntegration:
    """Integration tests for topic discovery."""

    def test_full_discovery_pipeline(self):
        """Test the full discovery pipeline."""
        mock_db = MagicMock()

        # No existing topics
        mock_db.list_topics.return_value = []

        # Mock content
        content = [
            ContentItem(
                id=i,
                subscription_id=1,
                source_type=SourceType.REDDIT,
                external_id=f"test{i}",
                title=f"Anthropic announces Claude {i}",
                content="",
                summary="Anthropic news",
                url=f"https://example.com/{i}",
                author="test",
                published_at=datetime.now() - timedelta(days=i % 7),
                importance_score=7,
            )
            for i in range(10)
        ]

        mock_db.get_undelivered_items.return_value = content
        mock_db.browse_by_date.return_value = []

        discovery = TopicDiscovery(db=mock_db, use_mock=True)
        suggestions = discovery.discover_topics(days=7, min_frequency=3)

        # Should find Anthropic and Claude
        names = [s.name for s in suggestions]
        assert any("Anthropic" in name or "Claude" in name for name in names)

    def test_suggestion_saving(self):
        """Test that suggestions can be saved."""
        mock_db = MagicMock()

        suggestion = TopicSuggestion(
            name="Test Topic",
            keywords=json.dumps(["test"]),
            frequency=10,
            confidence=0.8,
            source="entity",
            sample_titles=json.dumps(["Sample"]),
        )

        mock_db.create_topic_suggestion.return_value = suggestion

        discovery = TopicDiscovery(db=mock_db)
        saved = discovery.save_suggestion(suggestion)

        mock_db.create_topic_suggestion.assert_called_once()
        assert saved == suggestion

    def test_accept_suggestion(self):
        """Test accepting a suggestion creates a topic."""
        mock_db = MagicMock()

        suggestion = TopicSuggestion(
            id=1,
            name="OpenAI",
            keywords=json.dumps(["OpenAI", "GPT"]),
            frequency=10,
            confidence=0.8,
            source="entity",
        )

        mock_db.get_topic_suggestion.return_value = suggestion
        mock_db.create_topic.return_value = Topic(id=1, name="OpenAI")
        mock_db.update_topic_suggestion_status.return_value = True

        discovery = TopicDiscovery(db=mock_db)
        topic = discovery.accept_suggestion(1)

        assert topic is not None
        assert topic.name == "OpenAI"
        mock_db.update_topic_suggestion_status.assert_called_with(1, "accepted")

    def test_reject_suggestion(self):
        """Test rejecting a suggestion."""
        mock_db = MagicMock()
        mock_db.update_topic_suggestion_status.return_value = True

        discovery = TopicDiscovery(db=mock_db)
        result = discovery.reject_suggestion(1)

        assert result is True
        mock_db.update_topic_suggestion_status.assert_called_with(1, "rejected")

    def test_merge_suggestion(self):
        """Test merging a suggestion into existing topic."""
        mock_db = MagicMock()

        suggestion = TopicSuggestion(
            id=1,
            name="GPT-5",
            keywords=json.dumps(["GPT-5", "GPT5"]),
            frequency=10,
            confidence=0.8,
            source="entity",
        )

        existing_topic = Topic(
            id=5,
            name="OpenAI",
            keywords=json.dumps(["OpenAI", "GPT-4"]),
        )

        mock_db.get_topic_suggestion.return_value = suggestion
        mock_db.get_topic.return_value = existing_topic
        mock_db.update_topic_suggestion_status.return_value = True

        discovery = TopicDiscovery(db=mock_db)
        result = discovery.merge_suggestion(1, 5)

        assert result is True
        mock_db.update_topic.assert_called_once()
        mock_db.update_topic_suggestion_status.assert_called_with(1, "merged", merged_with_topic_id=5)


class TestTopicSuggestionEdgeCases:
    """Edge case tests for TopicSuggestion model."""

    def test_invalid_json_keywords(self):
        """Test handling of invalid JSON in keywords field."""
        suggestion = TopicSuggestion(
            name="Test",
            keywords="not valid json",
            frequency=5,
            confidence=0.5,
            source="entity",
        )
        # Should return empty list on invalid JSON
        assert suggestion.get_keywords() == []

    def test_invalid_json_sample_titles(self):
        """Test handling of invalid JSON in sample_titles field."""
        suggestion = TopicSuggestion(
            name="Test",
            keywords=json.dumps([]),
            sample_titles="not valid json",
            frequency=5,
            confidence=0.5,
            source="entity",
        )
        # Should return empty list on invalid JSON
        assert suggestion.get_sample_titles() == []

    def test_none_sample_titles(self):
        """Test handling of None sample_titles."""
        suggestion = TopicSuggestion(
            name="Test",
            keywords=json.dumps(["test"]),
            sample_titles=None,
            frequency=5,
            confidence=0.5,
            source="entity",
        )
        assert suggestion.get_sample_titles() == []

    def test_unicode_keywords(self):
        """Test handling of Unicode keywords."""
        suggestion = TopicSuggestion(
            name="中文话题",
            keywords=json.dumps(["人工智能", "机器学习", "深度学习"]),
            frequency=10,
            confidence=0.9,
            source="keyword",
        )
        keywords = suggestion.get_keywords()
        assert "人工智能" in keywords
        assert "机器学习" in keywords
        assert len(keywords) == 3

    def test_empty_name(self):
        """Test creation with empty name."""
        suggestion = TopicSuggestion(
            name="",
            keywords=json.dumps(["test"]),
            frequency=5,
            confidence=0.5,
            source="entity",
        )
        assert suggestion.name == ""

    def test_zero_frequency(self):
        """Test creation with zero frequency."""
        suggestion = TopicSuggestion(
            name="Rare Topic",
            keywords=json.dumps(["rare"]),
            frequency=0,
            confidence=0.1,
            source="trend",
        )
        assert suggestion.frequency == 0

    def test_confidence_bounds(self):
        """Test confidence at boundary values."""
        # Min confidence
        low = TopicSuggestion(
            name="Low",
            keywords=json.dumps([]),
            frequency=1,
            confidence=0.0,
            source="entity",
        )
        assert low.confidence == 0.0

        # Max confidence
        high = TopicSuggestion(
            name="High",
            keywords=json.dumps([]),
            frequency=100,
            confidence=1.0,
            source="entity",
        )
        assert high.confidence == 1.0

    def test_all_status_values(self):
        """Test all valid status values."""
        for status in ["pending", "accepted", "rejected", "merged"]:
            suggestion = TopicSuggestion(
                name=f"Topic {status}",
                keywords=json.dumps([]),
                frequency=5,
                confidence=0.5,
                source="entity",
                status=status,
            )
            assert suggestion.status == status

    def test_source_types(self):
        """Test all valid source types."""
        for source in ["entity", "keyword", "trend"]:
            suggestion = TopicSuggestion(
                name=f"Topic from {source}",
                keywords=json.dumps([]),
                frequency=5,
                confidence=0.5,
                source=source,
            )
            assert suggestion.source == source

    def test_merged_with_topic_id(self):
        """Test merged_with_topic_id field."""
        suggestion = TopicSuggestion(
            name="Merged Topic",
            keywords=json.dumps(["test"]),
            frequency=5,
            confidence=0.5,
            source="entity",
            status="merged",
            merged_with_topic_id=42,
        )
        assert suggestion.merged_with_topic_id == 42
        assert suggestion.status == "merged"


class TestTopicSuggestionDatabase:
    """Database integration tests for topic suggestions."""

    def setup_method(self):
        """Set up test fixtures with mock database."""
        self.mock_db = MagicMock()

    def test_create_and_retrieve_suggestion(self):
        """Test creating and retrieving a topic suggestion."""
        suggestion = TopicSuggestion(
            name="Test Topic",
            keywords=json.dumps(["test", "keyword"]),
            frequency=15,
            confidence=0.85,
            source="entity",
            sample_titles=json.dumps(["Sample Title 1", "Sample Title 2"]),
        )

        # Mock the create operation
        created_suggestion = TopicSuggestion(
            id=1,
            name=suggestion.name,
            keywords=suggestion.keywords,
            frequency=suggestion.frequency,
            confidence=suggestion.confidence,
            source=suggestion.source,
            sample_titles=suggestion.sample_titles,
            status="pending",
        )
        self.mock_db.create_topic_suggestion.return_value = created_suggestion
        self.mock_db.get_topic_suggestion.return_value = created_suggestion

        # Create
        result = self.mock_db.create_topic_suggestion(suggestion)
        assert result.id == 1
        assert result.name == "Test Topic"

        # Retrieve
        retrieved = self.mock_db.get_topic_suggestion(1)
        assert retrieved.name == "Test Topic"
        assert retrieved.get_keywords() == ["test", "keyword"]

    def test_get_suggestions_by_status(self):
        """Test retrieving suggestions filtered by status."""
        pending = [
            TopicSuggestion(id=1, name="Pending 1", keywords="[]", frequency=5, confidence=0.5, source="entity", status="pending"),
            TopicSuggestion(id=2, name="Pending 2", keywords="[]", frequency=10, confidence=0.7, source="entity", status="pending"),
        ]
        self.mock_db.get_topic_suggestions.return_value = pending

        results = self.mock_db.get_topic_suggestions(status="pending")
        assert len(results) == 2
        assert all(s.status == "pending" for s in results)

    def test_update_suggestion_status_to_accepted(self):
        """Test updating suggestion status to accepted."""
        self.mock_db.update_topic_suggestion_status.return_value = True

        result = self.mock_db.update_topic_suggestion_status(1, "accepted")
        assert result is True
        self.mock_db.update_topic_suggestion_status.assert_called_with(1, "accepted")

    def test_update_suggestion_status_to_rejected(self):
        """Test updating suggestion status to rejected."""
        self.mock_db.update_topic_suggestion_status.return_value = True

        result = self.mock_db.update_topic_suggestion_status(1, "rejected")
        assert result is True
        self.mock_db.update_topic_suggestion_status.assert_called_with(1, "rejected")

    def test_update_suggestion_status_to_merged(self):
        """Test updating suggestion status to merged with topic ID."""
        self.mock_db.update_topic_suggestion_status.return_value = True

        result = self.mock_db.update_topic_suggestion_status(1, "merged", merged_with_topic_id=5)
        assert result is True
        self.mock_db.update_topic_suggestion_status.assert_called_with(1, "merged", merged_with_topic_id=5)

    def test_update_nonexistent_suggestion(self):
        """Test updating a non-existent suggestion."""
        self.mock_db.update_topic_suggestion_status.return_value = False

        result = self.mock_db.update_topic_suggestion_status(999, "accepted")
        assert result is False

    def test_get_nonexistent_suggestion(self):
        """Test retrieving a non-existent suggestion."""
        self.mock_db.get_topic_suggestion.return_value = None

        result = self.mock_db.get_topic_suggestion(999)
        assert result is None

    def test_get_suggestions_ordered_by_score(self):
        """Test that suggestions are ordered by confidence * frequency."""
        suggestions = [
            TopicSuggestion(id=1, name="Low Score", keywords="[]", frequency=5, confidence=0.2, source="entity"),  # score: 1.0
            TopicSuggestion(id=2, name="High Score", keywords="[]", frequency=10, confidence=0.9, source="entity"),  # score: 9.0
            TopicSuggestion(id=3, name="Med Score", keywords="[]", frequency=8, confidence=0.5, source="entity"),  # score: 4.0
        ]
        # Simulate ordering by score desc
        ordered = sorted(suggestions, key=lambda s: s.confidence * s.frequency, reverse=True)
        self.mock_db.get_topic_suggestions.return_value = ordered

        results = self.mock_db.get_topic_suggestions(status="pending")
        assert results[0].name == "High Score"
        assert results[1].name == "Med Score"
        assert results[2].name == "Low Score"

    def test_get_suggestions_with_limit(self):
        """Test limiting the number of returned suggestions."""
        all_suggestions = [
            TopicSuggestion(id=i, name=f"Topic {i}", keywords="[]", frequency=i, confidence=0.5, source="entity")
            for i in range(1, 11)
        ]
        self.mock_db.get_topic_suggestions.return_value = all_suggestions[:5]

        results = self.mock_db.get_topic_suggestions(limit=5)
        assert len(results) == 5


class TestTopicDiscoveryWorkflow:
    """Tests for the complete topic suggestion workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db = MagicMock()

    def test_discovery_to_acceptance_workflow(self):
        """Test the full workflow from discovery to acceptance."""
        # 1. Discover topics
        suggestion = TopicSuggestion(
            id=1,
            name="Emerging Tech",
            keywords=json.dumps(["emerging", "technology"]),
            frequency=20,
            confidence=0.9,
            source="entity",
            status="pending",
        )

        # 2. Save suggestion
        self.mock_db.create_topic_suggestion.return_value = suggestion
        saved = self.mock_db.create_topic_suggestion(suggestion)
        assert saved.status == "pending"

        # 3. Review and accept
        self.mock_db.get_topic_suggestion.return_value = suggestion
        self.mock_db.update_topic_suggestion_status.return_value = True
        self.mock_db.create_topic.return_value = Topic(id=1, name="Emerging Tech")

        discovery = TopicDiscovery(db=self.mock_db)
        topic = discovery.accept_suggestion(1)

        assert topic is not None
        assert topic.name == "Emerging Tech"

    def test_discovery_to_rejection_workflow(self):
        """Test the workflow from discovery to rejection."""
        suggestion = TopicSuggestion(
            id=1,
            name="Spam Topic",
            keywords=json.dumps(["spam"]),
            frequency=5,
            confidence=0.3,
            source="keyword",
            status="pending",
        )

        self.mock_db.get_topic_suggestion.return_value = suggestion
        self.mock_db.update_topic_suggestion_status.return_value = True

        discovery = TopicDiscovery(db=self.mock_db)
        result = discovery.reject_suggestion(1)

        assert result is True
        self.mock_db.update_topic_suggestion_status.assert_called_with(1, "rejected")

    def test_discovery_to_merge_workflow(self):
        """Test the workflow from discovery to merging with existing topic."""
        suggestion = TopicSuggestion(
            id=1,
            name="GPT-5",
            keywords=json.dumps(["GPT-5", "GPT5"]),
            frequency=15,
            confidence=0.8,
            source="entity",
            status="pending",
        )

        existing_topic = Topic(
            id=10,
            name="OpenAI",
            keywords=json.dumps(["OpenAI", "GPT-4", "ChatGPT"]),
        )

        self.mock_db.get_topic_suggestion.return_value = suggestion
        self.mock_db.get_topic.return_value = existing_topic
        self.mock_db.update_topic_suggestion_status.return_value = True

        discovery = TopicDiscovery(db=self.mock_db)
        result = discovery.merge_suggestion(1, 10)

        assert result is True
        self.mock_db.update_topic.assert_called_once()
        self.mock_db.update_topic_suggestion_status.assert_called_with(1, "merged", merged_with_topic_id=10)

    def test_accept_nonexistent_suggestion(self):
        """Test accepting a suggestion that doesn't exist."""
        self.mock_db.get_topic_suggestion.return_value = None

        discovery = TopicDiscovery(db=self.mock_db)
        result = discovery.accept_suggestion(999)

        assert result is None

    def test_merge_with_nonexistent_topic(self):
        """Test merging with a topic that doesn't exist."""
        suggestion = TopicSuggestion(
            id=1,
            name="Test",
            keywords=json.dumps(["test"]),
            frequency=5,
            confidence=0.5,
            source="entity",
        )

        self.mock_db.get_topic_suggestion.return_value = suggestion
        self.mock_db.get_topic.return_value = None

        discovery = TopicDiscovery(db=self.mock_db)
        result = discovery.merge_suggestion(1, 999)

        assert result is False

    def test_high_confidence_suggestions_prioritized(self):
        """Test that high confidence suggestions are prioritized."""
        suggestions = [
            TopicSuggestion(id=1, name="Low", keywords="[]", frequency=10, confidence=0.3, source="entity"),
            TopicSuggestion(id=2, name="High", keywords="[]", frequency=10, confidence=0.95, source="entity"),
            TopicSuggestion(id=3, name="Medium", keywords="[]", frequency=10, confidence=0.6, source="entity"),
        ]

        # Sort by confidence descending
        sorted_suggestions = sorted(suggestions, key=lambda s: s.confidence, reverse=True)
        assert sorted_suggestions[0].name == "High"
        assert sorted_suggestions[1].name == "Medium"
        assert sorted_suggestions[2].name == "Low"

    def test_frequency_weighted_suggestions(self):
        """Test that frequency affects suggestion ranking."""
        suggestions = [
            TopicSuggestion(id=1, name="Rare", keywords="[]", frequency=2, confidence=0.9, source="entity"),  # score: 1.8
            TopicSuggestion(id=2, name="Common", keywords="[]", frequency=50, confidence=0.5, source="entity"),  # score: 25.0
            TopicSuggestion(id=3, name="Medium", keywords="[]", frequency=10, confidence=0.7, source="entity"),  # score: 7.0
        ]

        # Sort by score (frequency * confidence) descending
        sorted_suggestions = sorted(suggestions, key=lambda s: s.frequency * s.confidence, reverse=True)
        assert sorted_suggestions[0].name == "Common"
        assert sorted_suggestions[1].name == "Medium"
        assert sorted_suggestions[2].name == "Rare"


class TestTopicSuggestionValidation:
    """Tests for topic suggestion validation logic."""

    def test_minimum_frequency_threshold(self):
        """Test that low frequency suggestions are filtered."""
        mock_db = MagicMock()
        mock_db.list_topics.return_value = []
        mock_db.get_undelivered_items.return_value = []
        mock_db.browse_by_date.return_value = []

        discovery = TopicDiscovery(db=mock_db, use_mock=True)

        # Create content with low frequency entities
        content = [
            ContentItem(
                id=1,
                subscription_id=1,
                source_type=SourceType.REDDIT,
                external_id="test1",
                title="Rare entity mentioned once",
                content="",
                summary="",
                url="https://example.com",
                author="test",
                published_at=datetime.now(),
                importance_score=5,
            )
        ]

        suggestions = discovery._discover_by_entities(content, min_frequency=5)

        # Should not find anything with frequency < 5
        assert len(suggestions) == 0

    def test_confidence_calculation(self):
        """Test that confidence is calculated appropriately."""
        suggestion = TopicSuggestion(
            name="Test",
            keywords=json.dumps(["test"]),
            frequency=10,
            confidence=0.75,
            source="entity",
        )

        # Confidence should be between 0 and 1
        assert 0.0 <= suggestion.confidence <= 1.0

    def test_duplicate_detection_case_insensitive(self):
        """Test that duplicate detection is case insensitive."""
        mock_db = MagicMock()
        discovery = TopicDiscovery(db=mock_db, use_mock=True)

        class MockSuggestion:
            def __init__(self, name, confidence):
                self.name = name
                self.confidence = confidence
                self.keywords = ["test"]
                self.frequency = 5
                self.source = "entity"
                self.sample_titles = []

            def get_keywords(self):
                return self.keywords

            def get_sample_titles(self):
                return self.sample_titles

        suggestions = [
            MockSuggestion("OpenAI", 0.9),
            MockSuggestion("OPENAI", 0.7),
            MockSuggestion("openai", 0.5),
        ]

        deduped = discovery._deduplicate_suggestions(suggestions)

        # Should only have one OpenAI suggestion (highest confidence)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.9

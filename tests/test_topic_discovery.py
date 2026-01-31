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

"""Automatic topic discovery through frequency analysis and trend detection."""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from config.settings import get_settings
from src.storage.database import SyncDatabase
from src.storage.models import ContentItem, Topic

logger = logging.getLogger(__name__)


@dataclass
class TopicSuggestion:
    """A suggested topic discovered from content analysis."""

    id: Optional[int] = None
    name: str = ""
    keywords: list[str] = field(default_factory=list)
    frequency: int = 0  # How many times it appeared
    confidence: float = 0.0  # 0-1, how confident we are this is a real topic
    source: str = ""  # 'entity', 'keyword', 'trend'
    sample_titles: list[str] = field(default_factory=list)
    status: str = "pending"  # 'pending', 'accepted', 'rejected', 'merged'
    suggested_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    merged_with_topic_id: Optional[int] = None


# Entity patterns for extraction
_ENTITY_PATTERNS = [
    # Companies (English)
    r"\b(OpenAI|Google|Microsoft|Meta|Apple|Amazon|Tesla|Anthropic|Nvidia|AMD|Intel)\b",
    r"\b(DeepMind|Stability\s*AI|Hugging\s*Face|Cohere|Mistral|xAI|Perplexity)\b",
    r"\b(Netflix|Spotify|Adobe|Salesforce|Oracle|IBM|Cisco|Dell|HP)\b",
    # Companies (Chinese)
    r"(阿里巴巴|腾讯|百度|字节跳动|华为|小米|京东|美团|拼多多|滴滴)",
    r"(商汤|旷视|智谱|月之暗面|零一万物|百川智能|MiniMax)",
    # AI Models
    r"\b(GPT-\d+[a-z]*|Claude|Gemini|Llama\s*\d*|ChatGPT)\b",
    r"\b(Sora|Grok|Flux|DeepSeek|Qwen|Kimi|DALL-E|Midjourney)\b",
    # Products
    r"\b(iPhone|iPad|MacBook|Vision\s*Pro|Copilot|GitHub)\b",
    # People (prominent tech figures)
    r"\b(Sam\s*Altman|Elon\s*Musk|Mark\s*Zuckerberg|Sundar\s*Pichai|Jensen\s*Huang)\b",
    r"(马斯克|扎克伯格|马化腾|李彦宏|雷军|任正非)",
]

# Compiled patterns
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _ENTITY_PATTERNS]


def extract_entities(text: str) -> list[str]:
    """Extract named entities from text."""
    entities = []
    for pattern in _COMPILED_PATTERNS:
        matches = pattern.findall(text)
        for m in matches:
            if isinstance(m, str):
                entities.append(m.strip())
            elif isinstance(m, tuple):
                for g in m:
                    if g:
                        entities.append(g.strip())
                        break
    return entities


def extract_bigrams(text: str) -> list[str]:
    """Extract meaningful bigrams (two-word phrases) from text."""
    # Chinese: extract 2-4 character terms
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,4}", text)

    # English: extract two-word phrases
    words = re.findall(r"[a-zA-Z]+", text.lower())
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
        "her", "this", "that", "these", "those", "what", "which", "who",
        "how", "why", "and", "or", "but", "if", "so", "just", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "can", "may", "might", "must", "shall",
    }

    english_bigrams = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if len(w1) > 2 and len(w2) > 2 and w1 not in stopwords and w2 not in stopwords:
            english_bigrams.append(f"{w1} {w2}")

    return chinese_terms + english_bigrams


class TopicDiscovery:
    """
    Discovers topics automatically from content analysis.

    Methods:
    1. Entity Frequency Analysis - High-frequency company/product names
    2. Keyword Clustering - Co-occurring bigrams/trigrams
    3. Trend Detection - 3x week-over-week increase
    """

    def __init__(self, db: SyncDatabase, use_mock: bool = False):
        self.db = db
        self.use_mock = use_mock
        self.settings = get_settings()

    def discover_topics(
        self,
        days: int = 14,
        min_frequency: int = 5,
        min_confidence: float = 0.5,
    ) -> list[TopicSuggestion]:
        """
        Discover potential topics from recent content.

        Args:
            days: Look back this many days
            min_frequency: Minimum occurrences to suggest
            min_confidence: Minimum confidence score

        Returns:
            List of topic suggestions
        """
        # Get recent content
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        content = self._get_recent_content(from_date, days)

        if not content:
            logger.info("No content to analyze for topic discovery")
            return []

        suggestions = []

        # Method 1: Entity frequency analysis
        entity_suggestions = self._discover_by_entities(content, min_frequency)
        suggestions.extend(entity_suggestions)

        # Method 2: Keyword clustering
        keyword_suggestions = self._discover_by_keywords(content, min_frequency)
        suggestions.extend(keyword_suggestions)

        # Method 3: Trend detection
        trend_suggestions = self._discover_by_trends(content, days)
        suggestions.extend(trend_suggestions)

        # Filter duplicates and low confidence
        suggestions = self._deduplicate_suggestions(suggestions)
        suggestions = [s for s in suggestions if s.confidence >= min_confidence]

        # Filter out suggestions that match existing topics
        existing_topics = self.db.list_topics()
        existing_names = {t.name.lower() for t in existing_topics}
        existing_keywords = set()
        for t in existing_topics:
            for kw in t.get_keywords():
                existing_keywords.add(kw.lower())

        filtered = []
        for s in suggestions:
            if s.name.lower() not in existing_names:
                # Check if any keyword matches existing topic
                has_match = any(kw.lower() in existing_keywords for kw in s.keywords)
                if not has_match:
                    filtered.append(s)

        # Sort by confidence * frequency
        filtered.sort(key=lambda s: s.confidence * s.frequency, reverse=True)

        logger.info(f"Discovered {len(filtered)} topic suggestions")
        return filtered[:20]  # Top 20

    def _get_recent_content(self, from_date: str, days: int) -> list[ContentItem]:
        """Get recent processed content."""
        items = []
        # Get undelivered items (recently processed)
        undelivered = self.db.get_undelivered_items(min_importance=1, limit=500)
        items.extend(undelivered)

        # Also get recently delivered items via browse
        for d in range(days):
            date = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")
            day_items = self.db.browse_by_date(date, limit=100)
            items.extend(day_items)

        # Deduplicate by ID
        seen = set()
        unique = []
        for item in items:
            if item.id not in seen:
                seen.add(item.id)
                unique.append(item)

        return unique

    def _discover_by_entities(
        self, content: list[ContentItem], min_frequency: int
    ) -> list[TopicSuggestion]:
        """Discover topics by analyzing entity frequency."""
        entity_counts = Counter()
        entity_samples: dict[str, list[str]] = defaultdict(list)

        for item in content:
            text = f"{item.title} {item.summary or ''}"
            entities = extract_entities(text)
            for entity in entities:
                entity_norm = entity.strip()
                entity_counts[entity_norm] += 1
                if len(entity_samples[entity_norm]) < 3:
                    entity_samples[entity_norm].append(item.title)

        suggestions = []
        for entity, count in entity_counts.most_common(50):
            if count >= min_frequency:
                # Confidence based on frequency relative to total content
                confidence = min(1.0, count / (len(content) * 0.1))
                confidence = max(0.3, confidence)  # Minimum 0.3 for entities

                suggestions.append(TopicSuggestion(
                    name=entity,
                    keywords=[entity],
                    frequency=count,
                    confidence=confidence,
                    source="entity",
                    sample_titles=entity_samples[entity],
                ))

        return suggestions

    def _discover_by_keywords(
        self, content: list[ContentItem], min_frequency: int
    ) -> list[TopicSuggestion]:
        """Discover topics by analyzing keyword co-occurrence."""
        bigram_counts = Counter()
        bigram_samples: dict[str, list[str]] = defaultdict(list)

        for item in content:
            text = f"{item.title} {item.summary or ''}"
            bigrams = extract_bigrams(text)
            for bigram in bigrams:
                bigram_counts[bigram] += 1
                if len(bigram_samples[bigram]) < 3:
                    bigram_samples[bigram].append(item.title)

        suggestions = []
        for bigram, count in bigram_counts.most_common(50):
            if count >= min_frequency:
                # Lower confidence for keyword-based discovery
                confidence = min(0.8, count / (len(content) * 0.15))
                confidence = max(0.2, confidence)

                suggestions.append(TopicSuggestion(
                    name=bigram.title(),
                    keywords=[bigram],
                    frequency=count,
                    confidence=confidence,
                    source="keyword",
                    sample_titles=bigram_samples[bigram],
                ))

        return suggestions

    def _discover_by_trends(
        self, content: list[ContentItem], days: int
    ) -> list[TopicSuggestion]:
        """Discover topics by detecting sudden trend increases."""
        # Split content into current week and previous week
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        two_weeks_ago = now - timedelta(days=14)

        current_entities = Counter()
        previous_entities = Counter()

        for item in content:
            text = f"{item.title} {item.summary or ''}"
            entities = extract_entities(text)

            if item.published_at >= week_ago:
                for entity in entities:
                    current_entities[entity] += 1
            elif item.published_at >= two_weeks_ago:
                for entity in entities:
                    previous_entities[entity] += 1

        suggestions = []
        for entity, current_count in current_entities.items():
            previous_count = previous_entities.get(entity, 0)

            # Detect 3x increase (trend)
            if current_count >= 3 and (previous_count == 0 or current_count / max(previous_count, 1) >= 3):
                # High confidence for trending topics
                confidence = min(0.9, 0.5 + (current_count / 20))

                suggestions.append(TopicSuggestion(
                    name=entity,
                    keywords=[entity],
                    frequency=current_count,
                    confidence=confidence,
                    source="trend",
                    sample_titles=[],  # Could add samples later
                ))

        return suggestions

    def _deduplicate_suggestions(
        self, suggestions: list[TopicSuggestion]
    ) -> list[TopicSuggestion]:
        """Remove duplicate suggestions, keeping the highest confidence one."""
        seen: dict[str, TopicSuggestion] = {}

        for s in suggestions:
            key = s.name.lower()
            if key in seen:
                if s.confidence > seen[key].confidence:
                    seen[key] = s
            else:
                seen[key] = s

        return list(seen.values())

    def save_suggestion(self, suggestion: TopicSuggestion) -> TopicSuggestion:
        """Save a topic suggestion to the database."""
        return self.db.create_topic_suggestion(suggestion)

    def get_pending_suggestions(self, limit: int = 20) -> list[TopicSuggestion]:
        """Get pending topic suggestions for review."""
        return self.db.get_topic_suggestions(status="pending", limit=limit)

    def accept_suggestion(self, suggestion_id: int) -> Optional[Topic]:
        """Accept a suggestion and create a topic from it."""
        suggestion = self.db.get_topic_suggestion(suggestion_id)
        if not suggestion:
            return None

        # Create the topic
        topic = Topic(
            name=suggestion.name,
            description=f"Auto-discovered via {suggestion.source} analysis",
            keywords=json.dumps(suggestion.keywords, ensure_ascii=False),
            created_at=datetime.now(),
        )
        topic = self.db.create_topic(topic)

        # Update suggestion status
        self.db.update_topic_suggestion_status(suggestion_id, "accepted")

        return topic

    def reject_suggestion(self, suggestion_id: int) -> bool:
        """Reject a topic suggestion."""
        return self.db.update_topic_suggestion_status(suggestion_id, "rejected")

    def merge_suggestion(self, suggestion_id: int, target_topic_id: int) -> bool:
        """Merge a suggestion's keywords into an existing topic."""
        suggestion = self.db.get_topic_suggestion(suggestion_id)
        if not suggestion:
            return False

        topic = self.db.get_topic(target_topic_id)
        if not topic:
            return False

        # Add keywords to existing topic
        existing_keywords = topic.get_keywords()
        new_keywords = list(set(existing_keywords + suggestion.get_keywords()))
        topic.keywords = json.dumps(new_keywords, ensure_ascii=False)
        self.db.update_topic(topic)

        # Update suggestion status
        self.db.update_topic_suggestion_status(
            suggestion_id, "merged", merged_with_topic_id=target_topic_id
        )

        return True


def run_topic_discovery(
    db: SyncDatabase,
    days: int = 14,
    min_frequency: int = 5,
    save_suggestions: bool = True,
) -> list[TopicSuggestion]:
    """
    Run topic discovery and optionally save suggestions.

    Args:
        db: Database connection
        days: Days to look back
        min_frequency: Minimum frequency for suggestions
        save_suggestions: Whether to save to database

    Returns:
        List of topic suggestions
    """
    discovery = TopicDiscovery(db)
    suggestions = discovery.discover_topics(days=days, min_frequency=min_frequency)

    if save_suggestions:
        for suggestion in suggestions:
            discovery.save_suggestion(suggestion)
        logger.info(f"Saved {len(suggestions)} topic suggestions")

    return suggestions

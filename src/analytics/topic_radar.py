"""Topic Radar for tracking and analyzing topics over time."""

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from config.settings import get_settings
from src.storage.database import SyncDatabase
from src.storage.models import ContentItem, Topic, TopicSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TopicMatch:
    """A match between content and a topic."""

    topic_id: int
    topic_name: str
    relevance: float  # 0-1


@dataclass
class TrendAnalysis:
    """Trend analysis for a topic."""

    direction: str  # 'up', 'down', 'stable'
    change_percent: float
    key_drivers: list[str]


class TopicRadar:
    """
    Analyzes content against defined topics and tracks topic trends.

    Features:
    - Keyword-based topic matching
    - AI-enhanced relevance scoring
    - Trend detection over time
    - Weekly/monthly snapshots
    """

    def __init__(self, db: SyncDatabase, use_mock: bool = False):
        self.db = db
        self.use_mock = use_mock
        self.settings = get_settings()

    def match_content_to_topics(self, item: ContentItem) -> list[TopicMatch]:
        """
        Match a content item to relevant topics.

        Args:
            item: Content item to match

        Returns:
            List of topic matches with relevance scores
        """
        if not item.id:
            return []

        topics = self.db.list_topics()
        if not topics:
            return []

        matches = []
        text = f"{item.title} {item.summary or ''} {item.content or ''}".lower()

        for topic in topics:
            relevance = self._calculate_relevance(text, topic)
            if relevance > 0.3:  # Minimum threshold
                matches.append(TopicMatch(
                    topic_id=topic.id,
                    topic_name=topic.name,
                    relevance=relevance,
                ))

                # Save to database
                self.db.add_content_topic(item.id, topic.id, relevance)

        return matches

    def _calculate_relevance(self, text: str, topic: Topic) -> float:
        """Calculate relevance score between text and topic."""
        keywords = topic.get_keywords()
        if not keywords:
            return 0.0

        # Count keyword matches
        match_count = 0
        weighted_score = 0.0

        for i, keyword in enumerate(keywords):
            keyword_lower = keyword.lower()
            # Exact match
            if keyword_lower in text:
                # Earlier keywords are more important
                weight = 1.0 - (i * 0.05)  # 5% decay per position
                weight = max(weight, 0.3)  # Minimum weight
                weighted_score += weight
                match_count += 1

        if match_count == 0:
            return 0.0

        # Normalize score (cap at 1.0)
        base_score = min(weighted_score / len(keywords), 1.0)

        # Boost for multiple keyword matches
        if match_count >= 3:
            base_score = min(base_score * 1.2, 1.0)

        return round(base_score, 3)

    def generate_topic_snapshot(self, topic_id: int, period_days: int = 7) -> Optional[TopicSnapshot]:
        """
        Generate a snapshot summarizing topic activity over a period.

        Args:
            topic_id: Topic to analyze
            period_days: Number of days to look back

        Returns:
            Generated snapshot or None
        """
        topic = self.db.get_topic(topic_id)
        if not topic:
            return None

        # Get content related to this topic from the period
        content = self.db.get_topic_content(topic_id, min_relevance=0.4, limit=100)

        # Filter by date
        cutoff = datetime.now() - timedelta(days=period_days)
        recent_content = [c for c in content if c.published_at >= cutoff]

        if not recent_content:
            return None

        # Extract key entities
        entities = self._extract_entities(recent_content)

        # Calculate metrics
        metrics = {
            "article_count": len(recent_content),
            "avg_importance": sum(c.importance_score or 5 for c in recent_content) / len(recent_content),
            "high_importance_count": len([c for c in recent_content if c.importance_score and c.importance_score >= 7]),
        }

        # Detect trend
        trend_analysis = self.detect_trend(topic_id, period_days)
        trend = trend_analysis.direction if trend_analysis else "stable"

        # Generate summary
        summary = self._generate_topic_summary(topic, recent_content, entities)

        snapshot = TopicSnapshot(
            topic_id=topic_id,
            snapshot_date=datetime.now().strftime("%Y-%m-%d"),
            summary=summary,
            key_entities=json.dumps(entities, ensure_ascii=False),
            metrics=json.dumps(metrics, ensure_ascii=False),
            trend=trend,
        )

        return self.db.add_topic_snapshot(snapshot)

    def detect_trend(self, topic_id: int, period_days: int = 7) -> Optional[TrendAnalysis]:
        """
        Detect trend for a topic by comparing current vs previous period.

        Args:
            topic_id: Topic to analyze
            period_days: Period length in days

        Returns:
            Trend analysis or None
        """
        topic = self.db.get_topic(topic_id)
        if not topic:
            return None

        # Get all content for this topic
        all_content = self.db.get_topic_content(topic_id, min_relevance=0.3, limit=200)

        now = datetime.now()
        current_start = now - timedelta(days=period_days)
        previous_start = current_start - timedelta(days=period_days)

        # Count articles in each period
        current_count = len([c for c in all_content if c.published_at >= current_start])
        previous_count = len([c for c in all_content if previous_start <= c.published_at < current_start])

        # Calculate change
        if previous_count == 0:
            if current_count > 0:
                return TrendAnalysis(direction="up", change_percent=100.0, key_drivers=[])
            return TrendAnalysis(direction="stable", change_percent=0.0, key_drivers=[])

        change_percent = ((current_count - previous_count) / previous_count) * 100

        # Determine direction
        if change_percent > 20:
            direction = "up"
        elif change_percent < -20:
            direction = "down"
        else:
            direction = "stable"

        # Identify key drivers (most common entities in current period)
        current_content = [c for c in all_content if c.published_at >= current_start]
        key_drivers = self._extract_entities(current_content)[:5]

        return TrendAnalysis(
            direction=direction,
            change_percent=round(change_percent, 1),
            key_drivers=key_drivers,
        )

    def _extract_entities(self, content_list: list[ContentItem]) -> list[str]:
        """Extract key entities from content list."""
        entity_counts: dict[str, int] = {}

        # Common entity patterns
        patterns = [
            r"\b(OpenAI|Google|Microsoft|Meta|Apple|Amazon|Tesla|Anthropic|Nvidia)\b",
            r"(阿里巴巴|腾讯|百度|字节跳动|华为|小米)",
            r"\b(GPT-\d+|Claude|Gemini|Llama|ChatGPT)\b",
        ]

        for item in content_list:
            text = f"{item.title} {item.summary or ''}"
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = match if isinstance(match, str) else match[0]
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1

        # Sort by count and return top entities
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        return [e[0] for e in sorted_entities[:10]]

    def _generate_topic_summary(
        self, topic: Topic, content: list[ContentItem], entities: list[str]
    ) -> str:
        """Generate a summary for the topic snapshot."""
        if self.use_mock:
            return f"主题「{topic.name}」本周有{len(content)}篇相关文章"

        try:
            # Build context
            top_content = sorted(
                content,
                key=lambda x: x.importance_score or 5,
                reverse=True
            )[:5]

            content_summaries = "\n".join([
                f"- [{c.importance_score}/10] {c.summary or c.title}"
                for c in top_content
            ])

            prompt = f"""总结主题「{topic.name}」本周动态（50字内）：

主要文章：
{content_summaries}

涉及实体：{', '.join(entities[:5])}

要求：概括主要进展和变化"""

            cli_path = self.settings.ai.get_cli_path()
            result = subprocess.run(
                [cli_path, "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                summary = result.stdout.strip()
                if 10 <= len(summary) <= 200:
                    return summary

        except Exception as e:
            logger.warning(f"Topic summary generation failed: {e}")

        # Fallback
        return f"主题「{topic.name}」本周有{len(content)}篇相关文章，主要涉及{', '.join(entities[:3])}"


def process_content_topics(db: SyncDatabase, use_mock: bool = False, limit: int = 100) -> int:
    """
    Process recent content and match to topics.

    Args:
        db: Database connection
        use_mock: Use mock mode
        limit: Maximum items to process

    Returns:
        Number of topic associations created
    """
    radar = TopicRadar(db=db, use_mock=use_mock)

    # Get recently processed content that might not have topic associations
    content = db.get_undelivered_items(min_importance=1, limit=limit)

    total_matches = 0
    for item in content:
        matches = radar.match_content_to_topics(item)
        total_matches += len(matches)

    logger.info(f"Created {total_matches} topic associations from {len(content)} items")
    return total_matches


def generate_all_snapshots(db: SyncDatabase, use_mock: bool = False) -> int:
    """
    Generate snapshots for all topics.

    Args:
        db: Database connection
        use_mock: Use mock mode

    Returns:
        Number of snapshots created
    """
    radar = TopicRadar(db=db, use_mock=use_mock)
    topics = db.list_topics()

    created = 0
    for topic in topics:
        snapshot = radar.generate_topic_snapshot(topic.id)
        if snapshot:
            created += 1
            logger.info(f"Generated snapshot for topic '{topic.name}': {snapshot.trend}")

    return created

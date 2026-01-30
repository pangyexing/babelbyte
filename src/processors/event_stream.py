"""Event Stream processor for clustering related content items."""

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from config.settings import get_settings
from src.storage.database import SyncDatabase
from src.storage.models import ContentItem, EventCluster, EventMember, EventTimeline

logger = logging.getLogger(__name__)


@dataclass
class ClusterCandidate:
    """A candidate match for event clustering."""

    cluster_id: int
    cluster_title: str
    score: float  # 0-1 similarity score
    method: str  # 'keyword' or 'entity'


class EventStreamProcessor:
    """
    Processes content items to identify and cluster related events.

    Pipeline:
    1. Rule-based pre-filtering (keywords, time window, entities)
    2. AI confirmation for borderline cases
    3. Event timeline generation
    """

    def __init__(self, db: SyncDatabase, use_mock: bool = False):
        self.db = db
        self.use_mock = use_mock
        self.settings = get_settings()

        # Common entity patterns for extraction
        self.entity_patterns = [
            # Company names (English)
            r"\b(OpenAI|Google|Microsoft|Meta|Apple|Amazon|Tesla|Anthropic|Nvidia|AMD|Intel)\b",
            # Chinese company names
            r"(阿里巴巴|腾讯|百度|字节跳动|华为|小米|京东|美团|拼多多)",
            # Tech terms
            r"\b(GPT-\d+|Claude|Gemini|Llama|ChatGPT|Copilot|DALL-E|Midjourney|Stable\s*Diffusion)\b",
            # Common event markers
            r"(发布|上线|宣布|收购|融资|IPO|裁员|合并|开源)",
        ]

    def find_cluster_candidates(
        self, item: ContentItem, time_window_days: int = 3
    ) -> list[ClusterCandidate]:
        """
        Find potential cluster matches using rule-based pre-filtering.

        Args:
            item: Content item to find clusters for
            time_window_days: Look for clusters updated within this many days

        Returns:
            List of potential cluster matches with scores
        """
        candidates = []

        # Get recent clusters
        recent_clusters = self.db.get_recent_event_clusters(
            days=time_window_days, category=item.category, limit=50
        )

        if not recent_clusters:
            return candidates

        # Extract key entities from the item
        item_entities = self._extract_entities(item.title + " " + (item.content or ""))
        item_keywords = self._extract_keywords(item.title)

        for cluster in recent_clusters:
            score = 0.0
            method = "keyword"

            # Check entity overlap
            cluster_entities = self._extract_entities(cluster.event_title)
            entity_overlap = item_entities & cluster_entities
            if entity_overlap:
                score += 0.4 * min(len(entity_overlap), 3) / 3  # Max 0.4 for entities
                method = "entity"

            # Check keyword overlap in title
            cluster_keywords = self._extract_keywords(cluster.event_title)
            keyword_overlap = item_keywords & cluster_keywords
            if keyword_overlap:
                score += 0.3 * min(len(keyword_overlap), 5) / 5  # Max 0.3 for keywords

            # Check category match
            if item.category and cluster.category == item.category:
                score += 0.2

            # Time proximity bonus
            time_diff = abs((datetime.now() - cluster.last_updated_at).days)
            if time_diff <= 1:
                score += 0.1

            if score >= 0.3:  # Minimum threshold
                candidates.append(ClusterCandidate(
                    cluster_id=cluster.id,
                    cluster_title=cluster.event_title,
                    score=score,
                    method=method,
                ))

        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:5]  # Top 5 candidates

    def confirm_same_event_with_ai(
        self, item: ContentItem, candidates: list[ClusterCandidate]
    ) -> Optional[ClusterCandidate]:
        """
        Use AI to confirm if the item belongs to any candidate cluster.

        Args:
            item: Content item to classify
            candidates: Pre-filtered cluster candidates

        Returns:
            Best matching cluster or None
        """
        if not candidates:
            return None

        if self.use_mock:
            # In mock mode, accept the top candidate if score > 0.5
            return candidates[0] if candidates[0].score > 0.5 else None

        # Build prompt for AI confirmation
        candidate_list = "\n".join([
            f"{i+1}. [{c.cluster_title}] (匹配度: {c.score:.2f})"
            for i, c in enumerate(candidates)
        ])

        prompt = f"""判断这篇文章是否属于以下某个事件：

文章标题：{item.title}
文章摘要：{item.summary or item.content[:200] if item.content else ''}

候选事件：
{candidate_list}

只返回一个数字（1-{len(candidates)}表示属于该事件，0表示不属于任何事件）："""

        try:
            cli_path = self.settings.claude.cli_path
            result = subprocess.run(
                [cli_path, "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                response = result.stdout.strip()
                # Extract number from response
                match = re.search(r"\d+", response)
                if match:
                    idx = int(match.group())
                    if 1 <= idx <= len(candidates):
                        return candidates[idx - 1]

        except Exception as e:
            logger.warning(f"AI confirmation failed: {e}")

        # Fallback: accept high-confidence matches
        if candidates[0].score >= 0.7:
            return candidates[0]

        return None

    def process_item_for_clustering(self, item: ContentItem) -> Optional[EventCluster]:
        """
        Process a single item and assign it to a cluster or create a new one.

        Args:
            item: Content item to process

        Returns:
            The cluster the item was assigned to, or None
        """
        if not item.id:
            logger.warning("Cannot cluster item without ID")
            return None

        # Find candidates
        candidates = self.find_cluster_candidates(item)

        if candidates:
            # Try AI confirmation for borderline cases
            confirmed = self.confirm_same_event_with_ai(item, candidates)
            if confirmed:
                # Add to existing cluster
                member = EventMember(
                    event_cluster_id=confirmed.cluster_id,
                    content_item_id=item.id,
                    similarity_score=confirmed.score,
                    detection_method=confirmed.method,
                )
                self.db.add_event_member(member)

                cluster = self.db.get_event_cluster(confirmed.cluster_id)
                logger.info(f"Added item {item.id} to cluster '{confirmed.cluster_title}'")
                return cluster

        # Create new cluster for high-importance items
        if item.importance_score and item.importance_score >= 6:
            cluster = EventCluster(
                event_title=self._generate_event_title(item),
                category=item.category or "其他",
                first_seen_at=item.published_at,
                last_updated_at=datetime.now(),
                article_count=1,
            )
            cluster = self.db.create_event_cluster(cluster)

            # Add this item as first member
            member = EventMember(
                event_cluster_id=cluster.id,
                content_item_id=item.id,
                similarity_score=1.0,
                detection_method="initial",
            )
            self.db.add_event_member(member)

            logger.info(f"Created new cluster '{cluster.event_title}' for item {item.id}")
            return cluster

        return None

    def analyze_event_evolution(self, cluster_id: int) -> Optional[EventTimeline]:
        """
        Analyze an event cluster and generate a timeline entry.

        Args:
            cluster_id: ID of the cluster to analyze

        Returns:
            Generated timeline entry or None
        """
        cluster = self.db.get_event_cluster(cluster_id)
        if not cluster:
            return None

        members = self.db.get_event_members(cluster_id)
        if not members:
            return None

        # Get today's and yesterday's articles
        today = datetime.now().strftime("%Y-%m-%d")
        today_items = [m for m in members if m.published_at.strftime("%Y-%m-%d") == today]
        recent_items = members[:5]  # Most recent 5

        if not today_items:
            return None

        # Generate summary of today's developments
        if self.use_mock:
            summary = f"事件「{cluster.event_title}」今日有{len(today_items)}篇相关报道"
            consensus = "high"
        else:
            summary, consensus = self._generate_timeline_summary(cluster, today_items, recent_items)

        timeline = EventTimeline(
            event_cluster_id=cluster_id,
            entry_date=today,
            summary=summary,
            consensus_level=consensus,
        )
        self.db.add_event_timeline(timeline)

        return timeline

    def _extract_entities(self, text: str) -> set[str]:
        """Extract named entities from text."""
        entities = set()
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update(m.lower() if isinstance(m, str) else m[0].lower() for m in matches)
        return entities

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract key words from title."""
        # Remove punctuation and split
        words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())
        # Filter short words and stopwords
        stopwords = {"的", "是", "在", "了", "和", "与", "a", "the", "is", "are", "to", "of", "in"}
        return {w for w in words if len(w) > 1 and w not in stopwords}

    def _generate_event_title(self, item: ContentItem) -> str:
        """Generate a concise event title from content item."""
        # Use AI to generate a concise title if available
        if not self.use_mock:
            try:
                prompt = f"用10个字以内概括这个事件的主题：{item.title}"
                cli_path = self.settings.claude.cli_path
                result = subprocess.run(
                    [cli_path, "-p", prompt, "--output-format", "text"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0:
                    title = result.stdout.strip()
                    if 3 <= len(title) <= 30:
                        return title
            except Exception:
                pass

        # Fallback: truncate original title
        title = item.title.split("：")[0].split(":")[0][:30]
        return title if title else item.title[:30]

    def _generate_timeline_summary(
        self, cluster: EventCluster, today_items: list[ContentItem], recent_items: list[ContentItem]
    ) -> tuple[str, str]:
        """Generate a timeline summary comparing today vs recent developments."""
        try:
            # Build context from items
            today_summaries = "\n".join([f"- {i.summary or i.title}" for i in today_items[:3]])
            recent_summaries = "\n".join([f"- {i.summary or i.title}" for i in recent_items[:3]])

            prompt = f"""分析事件进展并总结今日变化：

事件：{cluster.event_title}

今日报道：
{today_summaries}

此前报道：
{recent_summaries}

返回JSON格式：{{"summary": "50字总结今日进展", "consensus": "high/conflicted"}}
high表示各来源一致，conflicted表示有分歧"""

            cli_path = self.settings.claude.cli_path
            result = subprocess.run(
                [cli_path, "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                response = result.stdout.strip()
                # Find JSON in response
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return data.get("summary", ""), data.get("consensus", "high")

        except Exception as e:
            logger.warning(f"Timeline summary generation failed: {e}")

        # Fallback
        return f"事件「{cluster.event_title}」今日有{len(today_items)}篇更新", "high"


def cluster_unprocessed_items(db: SyncDatabase, use_mock: bool = False, limit: int = 100) -> int:
    """
    Cluster all processed but unclustered items.

    Args:
        db: Database connection
        use_mock: Use mock mode
        limit: Maximum items to process

    Returns:
        Number of items clustered
    """
    processor = EventStreamProcessor(db=db, use_mock=use_mock)

    # Get processed items that might need clustering
    # (high importance items from the last few days)
    items = db.get_undelivered_items(min_importance=6, limit=limit)

    clustered = 0
    for item in items:
        result = processor.process_item_for_clustering(item)
        if result:
            clustered += 1

    logger.info(f"Clustered {clustered}/{len(items)} items into events")
    return clustered

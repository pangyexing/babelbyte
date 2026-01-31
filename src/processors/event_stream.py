"""Event Stream processor for clustering related content items."""

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
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


# Common entity patterns for extraction (module-level for caching)
_ENTITY_PATTERNS = [
    # Company names (English)
    r"\b(OpenAI|Google|Microsoft|Meta|Apple|Amazon|Tesla|Anthropic|Nvidia|AMD|Intel)\b",
    # Chinese company names
    r"(阿里巴巴|腾讯|百度|字节跳动|华为|小米|京东|美团|拼多多)",
    # Tech terms
    r"\b(GPT-\d+|Claude|Gemini|Llama|ChatGPT|Copilot|DALL-E|Midjourney|Stable\s*Diffusion)\b",
    # Common event markers
    r"(发布|上线|宣布|收购|融资|IPO|裁员|合并|开源)",
]


@lru_cache(maxsize=1024)
def _extract_entities_cached(text: str) -> frozenset[str]:
    """Extract named entities from text (cached)."""
    entities = set()
    for pattern in _ENTITY_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.update(m.lower() if isinstance(m, str) else m[0].lower() for m in matches)
    return frozenset(entities)


@lru_cache(maxsize=1024)
def _extract_keywords_cached(text: str) -> frozenset[str]:
    """Extract key words from title (cached)."""
    # Remove punctuation and split
    words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())
    # Filter short words and stopwords
    stopwords = {"的", "是", "在", "了", "和", "与", "a", "the", "is", "are", "to", "of", "in"}
    return frozenset(w for w in words if len(w) > 1 and w not in stopwords)


class EventStreamProcessor:
    """
    Processes content items to identify and cluster related events.

    Pipeline:
    1. Rule-based pre-filtering (keywords, time window, entities)
    2. AI confirmation for borderline cases
    3. Event timeline generation

    Caching:
    - Entity/keyword extraction: LRU cache (module-level)
    - Recent clusters query: TTL cache (class-level, 60s)
    - AI confirmation: Per-session cache (instance-level)
    """

    # Class-level TTL cache for recent clusters: {cache_key: (timestamp, clusters)}
    _clusters_cache: dict[str, tuple[float, list]] = {}
    _clusters_cache_ttl: float = float(os.environ.get("CLUSTER_CACHE_TTL", 60.0))

    def __init__(self, db: SyncDatabase, use_mock: bool = False):
        self.db = db
        self.use_mock = use_mock
        self.settings = get_settings()
        # Instance-level cache for AI confirmation: {(item_id, cluster_ids_tuple): result}
        self._ai_confirm_cache: dict[tuple, Optional[ClusterCandidate]] = {}

    def _build_event_confirm_cache_key(self, item_id: int, candidates: list[ClusterCandidate]) -> str:
        """Build a stable cache key for event confirmation results."""
        signature = "|".join(
            f"{c.cluster_id}:{c.cluster_title}" for c in candidates
        )
        digest = hashlib.sha256(f"{item_id}|{signature}".encode("utf-8")).hexdigest()
        return f"event_confirm:{digest}"

    def _get_persistent_event_confirm(
        self, item_id: int, candidates: list[ClusterCandidate]
    ) -> tuple[bool, Optional[ClusterCandidate]]:
        """Get cached event confirmation result from persistent cache."""
        if not self.settings.ai.cache_enabled or not isinstance(self.db, SyncDatabase):
            return False, None
        cache_key = self._build_event_confirm_cache_key(item_id, candidates)
        cached_json = self.db.get_ai_cache(cache_key)
        if not cached_json:
            return False, None
        try:
            data = json.loads(cached_json)
            cluster_id = data.get("cluster_id")
            if cluster_id in (None, 0):
                return True, None
            for candidate in candidates:
                if candidate.cluster_id == cluster_id:
                    return True, candidate
        except Exception:
            return False, None
        return False, None

    def _set_persistent_event_confirm(
        self, item_id: int, candidates: list[ClusterCandidate], result: Optional[ClusterCandidate]
    ) -> None:
        """Store event confirmation result in persistent cache."""
        if not self.settings.ai.cache_enabled or not isinstance(self.db, SyncDatabase):
            return
        cache_key = self._build_event_confirm_cache_key(item_id, candidates)
        payload = {"cluster_id": result.cluster_id if result else 0}
        self.db.set_ai_cache(cache_key, json.dumps(payload), self.settings.ai.cache_ttl)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the clusters cache (useful for testing)."""
        cls._clusters_cache.clear()
        _extract_entities_cached.cache_clear()
        _extract_keywords_cached.cache_clear()

    def _get_recent_clusters_cached(
        self, time_window_days: int, category: Optional[str]
    ) -> list[EventCluster]:
        """Get recent clusters with TTL caching."""
        cache_key = f"{time_window_days}:{category or ''}"
        now = time.time()

        if cache_key in self._clusters_cache:
            cached_time, cached_clusters = self._clusters_cache[cache_key]
            if now - cached_time < self._clusters_cache_ttl:
                logger.debug(f"Using cached clusters for key={cache_key}")
                return cached_clusters

        # Query database and cache result
        clusters = self.db.get_recent_event_clusters(
            days=time_window_days, category=category, limit=50
        )
        self._clusters_cache[cache_key] = (now, clusters)
        return clusters

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

        # Get recent clusters (cached)
        recent_clusters = self._get_recent_clusters_cached(time_window_days, item.category)

        if not recent_clusters:
            return candidates

        # Extract key entities from the item (uses LRU cached functions)
        item_entities = _extract_entities_cached(item.title + " " + (item.content or ""))
        item_keywords = _extract_keywords_cached(item.title)

        for cluster in recent_clusters:
            score = 0.0
            method = "keyword"

            # Check entity overlap (uses LRU cached functions)
            cluster_entities = _extract_entities_cached(cluster.event_title)
            entity_overlap = item_entities & cluster_entities
            if entity_overlap:
                score += 0.4 * min(len(entity_overlap), 3) / 3  # Max 0.4 for entities
                method = "entity"

            # Check keyword overlap in title (uses LRU cached functions)
            cluster_keywords = _extract_keywords_cached(cluster.event_title)
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
                candidates.append(
                    ClusterCandidate(
                        cluster_id=cluster.id,
                        cluster_title=cluster.event_title,
                        score=score,
                        method=method,
                    )
                )

        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:5]  # Top 5 candidates

    def confirm_same_event_with_ai(
        self, item: ContentItem, candidates: list[ClusterCandidate]
    ) -> Optional[ClusterCandidate]:
        """
        Use AI to confirm if the item belongs to any candidate cluster.

        Uses light model for this simple yes/no classification task.
        Results are cached per (item_id, cluster_ids) to avoid redundant AI calls.

        Args:
            item: Content item to classify
            candidates: Pre-filtered cluster candidates

        Returns:
            Best matching cluster or None
        """
        if not candidates:
            return None

        # Check persistent cache first (cross-run)
        if item.id is not None:
            cache_hit, cached = self._get_persistent_event_confirm(item.id, candidates)
            if cache_hit:
                return cached

        # Auto-accept very high confidence matches without AI call
        # This is a performance optimization - score >= 0.8 indicates strong
        # entity/keyword overlap that doesn't need AI confirmation
        if candidates[0].score >= 0.8:
            logger.info(
                f"Auto-accepting high-confidence match (score={candidates[0].score:.2f}) "
                f"for item {item.id} -> cluster '{candidates[0].cluster_title}'"
            )
            if item.id is not None:
                self._set_persistent_event_confirm(item.id, candidates, candidates[0])
            return candidates[0]

        # Check cache first
        cache_key = (item.id, tuple(c.cluster_id for c in candidates))
        if cache_key in self._ai_confirm_cache:
            logger.debug(f"Using cached AI confirmation for item {item.id}")
            return self._ai_confirm_cache[cache_key]

        if self.use_mock:
            # In mock mode, accept the top candidate if score > 0.5
            result = candidates[0] if candidates[0].score > 0.5 else None
            self._ai_confirm_cache[cache_key] = result
            if item.id is not None:
                self._set_persistent_event_confirm(item.id, candidates, result)
            return result

        # Build prompt for AI confirmation
        candidate_list = "\n".join(
            [
                f"{i+1}. [{c.cluster_title}] (匹配度: {c.score:.2f})"
                for i, c in enumerate(candidates)
            ]
        )

        prompt = f"""判断这篇文章是否属于以下某个事件：

文章标题：{item.title}
文章摘要：{item.summary or item.content[:200] if item.content else ''}

候选事件：
{candidate_list}

只返回一个数字（1-{len(candidates)}表示属于该事件，0表示不属于任何事件）："""

        try:
            cli_path = self.settings.claude.cli_path
            config = self.settings.ai.model_tiers

            # Build command with light model for simple confirmation task
            cmd = [cli_path, "-p", prompt, "--output-format", "text"]
            if config.enabled:
                model = config.task_overrides.get("event_confirm", config.claude_light)
                if model:
                    cmd.extend(["--model", model])
                    logger.debug(f"Event confirm using model: {model}")

            result = subprocess.run(
                cmd,
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
                        confirmed = candidates[idx - 1]
                        self._ai_confirm_cache[cache_key] = confirmed
                        if item.id is not None:
                            self._set_persistent_event_confirm(item.id, candidates, confirmed)
                        return confirmed

        except Exception as e:
            logger.warning(f"AI confirmation failed: {e}")

        # Fallback: accept high-confidence matches
        if candidates[0].score >= 0.7:
            result = candidates[0]
            self._ai_confirm_cache[cache_key] = result
            if item.id is not None:
                self._set_persistent_event_confirm(item.id, candidates, result)
            return result

        self._ai_confirm_cache[cache_key] = None
        if item.id is not None:
            self._set_persistent_event_confirm(item.id, candidates, None)
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

        if item.id is not None and isinstance(self.db, SyncDatabase):
            self.db.mark_cluster_attempted(item.id)

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

    def _generate_event_title(self, item: ContentItem) -> str:
        """Generate a concise event title from content item using light model."""
        # Use AI to generate a concise title if available
        if not self.use_mock:
            try:
                prompt = f"用10个字以内概括这个事件的主题：{item.title}"
                cli_path = self.settings.claude.cli_path
                config = self.settings.ai.model_tiers

                # Build command with light model for simple title generation
                cmd = [cli_path, "-p", prompt, "--output-format", "text"]
                if config.enabled:
                    model = config.task_overrides.get("event_title", config.claude_light)
                    if model:
                        cmd.extend(["--model", model])
                        logger.debug(f"Event title using model: {model}")

                result = subprocess.run(
                    cmd,
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
        """Generate a timeline summary comparing today vs recent developments using light model."""
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
            config = self.settings.ai.model_tiers

            # Build command with light model for timeline summary
            cmd = [cli_path, "-p", prompt, "--output-format", "text"]
            if config.enabled:
                model = config.task_overrides.get("timeline", config.claude_light)
                if model:
                    cmd.extend(["--model", model])
                    logger.debug(f"Timeline summary using model: {model}")

            result = subprocess.run(
                cmd,
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


def cluster_unprocessed_items(
    db: SyncDatabase,
    use_mock: bool = False,
    limit: int = 100,
    progress_callback: Optional[callable] = None,
) -> int:
    """
    Cluster all processed but unclustered items (sequential version).

    Args:
        db: Database connection
        use_mock: Use mock mode
        limit: Maximum items to process
        progress_callback: Optional callback(current, total, clustered) for progress updates

    Returns:
        Number of items clustered
    """
    processor = EventStreamProcessor(db=db, use_mock=use_mock)

    # Get unclustered items only - this skips items already in clusters
    # for better performance (optimization: skip already-clustered items)
    retry_after_hours = int(os.environ.get("CLUSTER_RETRY_HOURS", "24"))
    items = db.get_unclustered_items(
        min_importance=6,
        limit=limit,
        retry_after_hours=retry_after_hours if retry_after_hours > 0 else None,
    )

    clustered = 0
    total = len(items)

    for i, item in enumerate(items):
        result = processor.process_item_for_clustering(item)
        if result:
            clustered += 1

        if progress_callback:
            progress_callback(i + 1, total, clustered)

    logger.info(f"Clustered {clustered}/{total} items into events")
    return clustered


def cluster_unprocessed_items_parallel(
    db: SyncDatabase,
    use_mock: bool = False,
    limit: int = 100,
    max_workers: int = 4,
    progress_callback: Optional[callable] = None,
) -> int:
    """
    Cluster items using parallel AI processing with ThreadPoolExecutor.

    This is significantly faster than sequential processing when AI calls are needed,
    as multiple subprocess calls can run concurrently.

    Note: Each worker thread gets its own database connection to avoid
    event loop conflicts with asyncio.

    Args:
        db: Database connection (used for initial query only)
        use_mock: Use mock mode
        limit: Maximum items to process
        max_workers: Number of parallel workers (default 4)
        progress_callback: Optional callback(current, total, clustered) for progress updates

    Returns:
        Number of items clustered
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Get unclustered items using the main db connection
    retry_after_hours = int(os.environ.get("CLUSTER_RETRY_HOURS", "24"))
    items = db.get_unclustered_items(
        min_importance=6,
        limit=limit,
        retry_after_hours=retry_after_hours if retry_after_hours > 0 else None,
    )

    if not items:
        logger.info("No unclustered items to process")
        return 0

    total = len(items)
    clustered = 0
    completed = 0
    lock = threading.Lock()

    # Thread-local storage for per-thread database connections
    thread_local = threading.local()

    def get_thread_db() -> SyncDatabase:
        """Get or create a database connection for the current thread."""
        if not hasattr(thread_local, "db"):
            # Check if db has a valid db_path (not a MagicMock or missing)
            db_path = getattr(db, "db_path", None)
            if db_path is None or not isinstance(db_path, (str, Path)):
                # In mock/test mode, just reuse the original mock db
                thread_local.db = db
            else:
                # Each thread gets its own SyncDatabase instance
                # This ensures each thread has its own event loop
                thread_local.db = SyncDatabase(db_path)
                thread_local.db.connect()
        return thread_local.db

    def get_thread_processor() -> EventStreamProcessor:
        """Get or create a processor for the current thread."""
        if not hasattr(thread_local, "processor"):
            thread_local.processor = EventStreamProcessor(
                db=get_thread_db(), use_mock=use_mock
            )
        return thread_local.processor

    def process_single_item(item: "ContentItem") -> bool:
        """Process a single item for clustering (thread-safe)."""
        nonlocal clustered, completed

        try:
            processor = get_thread_processor()
            result = processor.process_item_for_clustering(item)

            with lock:
                completed += 1
                if result:
                    clustered += 1
                current_clustered = clustered
                current_completed = completed

            if progress_callback:
                progress_callback(current_completed, total, current_clustered)

            return result is not None
        except Exception as e:
            with lock:
                completed += 1
                current_completed = completed

            if progress_callback:
                progress_callback(current_completed, total, clustered)

            raise e

    # Process items in parallel
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_item, item): item for item in items}

            # Wait for completion (results are tracked via side effects)
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    item = futures[future]
                    logger.warning(f"Error processing item {item.id}: {e}")
    finally:
        # Clean up thread-local database connections
        # Note: ThreadPoolExecutor reuses threads, so we can't easily close
        # connections here. They will be closed when the threads are destroyed.
        pass

    logger.info(
        f"Clustered {clustered}/{total} items into events (parallel, {max_workers} workers)"
    )
    return clustered

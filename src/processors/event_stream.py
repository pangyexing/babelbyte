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
from src.analytics.token_tracker import AICallType, record_ai_call
from src.storage.database import SyncDatabase
from src.storage.models import ContentItem, EventCluster, EventMember, EventTimeline

# Optional numpy import for embeddings
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ClusterCandidate:
    """A candidate match for event clustering."""

    cluster_id: int
    cluster_title: str
    score: float  # 0-1 similarity score
    method: str  # 'keyword', 'entity', 'semantic', or 'hybrid'
    rule_score: float = 0.0  # Rule-based component
    semantic_score: float = 0.0  # Semantic/embedding component


# HN discussion prefix patterns for special handling
_HN_DISCUSSION_PREFIXES = [
    r"^Ask HN[:\s]+",
    r"^Show HN[:\s]+",
    r"^Tell HN[:\s]+",
    r"^Launch HN[:\s]+",
]

# Common entity patterns for extraction (module-level for caching)
_ENTITY_PATTERNS = [
    # Company names (English)
    r"\b(OpenAI|Google|Microsoft|Meta|Apple|Amazon|Tesla|Anthropic|Nvidia|AMD|Intel)\b",
    r"\b(DeepMind|Stability|Hugging\s*Face|Cohere|Mistral|xAI|Perplexity)\b",
    # Chinese company names
    r"(阿里巴巴|腾讯|百度|字节跳动|华为|小米|京东|美团|拼多多|商汤|旷视|智谱)",
    # Tech terms / AI models
    r"\b(GPT-\d+|Claude|Gemini|Llama|ChatGPT|Copilot|DALL-E|Midjourney|Stable\s*Diffusion)\b",
    r"\b(Sora|Grok|Flux|DeepSeek|Qwen|Yi|Kimi)\b",
    # Twitter/X usernames (capture the username part)
    r"@([A-Za-z0-9_]{1,15})\b",
    # Hashtags
    r"#([A-Za-z0-9_]+)\b",
    # Common event markers (Chinese)
    r"(发布|上线|宣布|收购|融资|IPO|裁员|合并|开源|升级|更新)",
    # Common event markers (English)
    r"\b(launch|release|announce|acquire|funding|merger|layoff|update)\b",
]


@lru_cache(maxsize=1024)
def _extract_entities_cached(text: str) -> frozenset[str]:
    """Extract named entities from text (cached).

    Extracts:
    - Company names (OpenAI, Google, etc.)
    - AI model names (GPT-4, Claude, etc.)
    - Twitter usernames (@username)
    - Hashtags (#topic)
    - Event markers (launch, release, etc.)
    """
    entities = set()
    for pattern in _ENTITY_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            if isinstance(m, str):
                entities.add(m.lower())
            elif isinstance(m, tuple):
                # For patterns with groups, take the first non-empty group
                for g in m:
                    if g:
                        entities.add(g.lower())
                        break
            else:
                entities.add(str(m).lower())
    return frozenset(entities)


@lru_cache(maxsize=1024)
def _extract_keywords_cached(text: str) -> frozenset[str]:
    """Extract key words from title (cached)."""
    # Remove punctuation and split
    words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())
    # Filter short words and stopwords
    stopwords = {
        "的",
        "是",
        "在",
        "了",
        "和",
        "与",
        "我",
        "你",
        "他",
        "她",
        "它",
        "这",
        "那",
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "my",
        "your",
        "his",
        "her",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "how",
        "why",
        "and",
        "or",
        "but",
        "if",
        "so",
        "just",
        "have",
        "has",
        "had",
        "do",
        "does",
    }
    return frozenset(w for w in words if len(w) > 1 and w not in stopwords)


def _get_hn_discussion_type(title: str) -> Optional[str]:
    """Get HN discussion type from title.

    Returns 'ask', 'show', 'tell', 'launch' if the title matches a pattern,
    or None if it's not a HN discussion post.
    """
    if not title:
        return None
    title_lower = title.lower()
    if title_lower.startswith("ask hn"):
        return "ask"
    if title_lower.startswith("show hn"):
        return "show"
    if title_lower.startswith("tell hn"):
        return "tell"
    if title_lower.startswith("launch hn"):
        return "launch"
    return None


@lru_cache(maxsize=1024)
def _normalize_title(text: str) -> str:
    """Normalize title for comparison (cached).

    Removes common prefixes like RT, HN discussion prefixes, hashtags formatting,
    and normalizes whitespace.
    """
    if not text:
        return ""
    # Remove RT prefix
    text = re.sub(r"^RT\s+", "", text, flags=re.IGNORECASE)
    # Remove HN discussion prefixes to focus on actual topic
    for pattern in _HN_DISCUSSION_PREFIXES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Remove @mentions at the start
    text = re.sub(r"^@\w+[:\s]+", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip().lower()


def _calculate_title_similarity(title1: str, title2: str) -> float:
    """Calculate Jaccard similarity between two titles.

    Returns a score between 0 and 1.
    """
    if not title1 or not title2:
        return 0.0

    # Normalize titles
    norm1 = _normalize_title(title1)
    norm2 = _normalize_title(title2)

    # If normalized titles are identical, return 1.0
    if norm1 == norm2:
        return 1.0

    # Extract keywords
    keywords1 = _extract_keywords_cached(norm1)
    keywords2 = _extract_keywords_cached(norm2)

    if not keywords1 or not keywords2:
        return 0.0

    # Jaccard similarity
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)

    if union == 0:
        return 0.0

    return intersection / union


class EventStreamProcessor:
    """
    Processes content items to identify and cluster related events.

    Pipeline:
    1. Rule-based pre-filtering (keywords, time window, entities)
    2. Semantic similarity using embeddings (optional)
    3. Hybrid scoring combining rule-based and semantic scores
    4. AI confirmation for borderline cases
    5. Event timeline generation

    Caching:
    - Entity/keyword extraction: LRU cache (module-level)
    - Recent clusters query: TTL cache (class-level, 60s)
    - AI confirmation: Per-session cache (instance-level)
    - Embeddings: Managed by EmbeddingManager
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
        # Embedding manager for semantic similarity (lazy loaded)
        self._embedding_manager = None
        # Cache for cluster centroids: {cluster_id: embedding}
        self._centroid_cache: dict = {}

    def _get_embedding_manager(self):
        """Get embedding manager if enabled."""
        if not self.settings.embedding.enabled:
            return None
        if self._embedding_manager is None:
            try:
                from src.processors.embeddings import EmbeddingManager

                self._embedding_manager = EmbeddingManager.get_instance()
            except ImportError as e:
                logger.warning(f"Embeddings disabled: {e}")
                return None
        return self._embedding_manager

    def _get_cluster_centroid(self, cluster_id: int):
        """Get cluster centroid embedding with caching."""
        if cluster_id in self._centroid_cache:
            return self._centroid_cache[cluster_id]

        result = self.db.get_cluster_centroid(cluster_id)
        if result:
            centroid_bytes, member_count = result
            manager = self._get_embedding_manager()
            if manager:
                from src.processors.embeddings import bytes_to_embedding

                centroid = bytes_to_embedding(centroid_bytes, manager.dimension)
                self._centroid_cache[cluster_id] = centroid
                return centroid
        return None

    def _compute_semantic_similarity(self, item: ContentItem, cluster: EventCluster) -> float:
        """Compute semantic similarity between item and cluster using embeddings."""
        manager = self._get_embedding_manager()
        if not manager:
            return 0.0

        try:
            # Get item embedding
            item_text = f"{item.title} {item.summary or ''}"[:500]
            item_embedding = manager.get_embedding(item_text)

            # Get cluster centroid
            centroid = self._get_cluster_centroid(cluster.id)
            if centroid is not None:
                similarity = manager.cosine_similarity(item_embedding, centroid)
                # Map from [-1, 1] to [0, 1]
                return max(0.0, (similarity + 1) / 2)

            # Fallback: compare with cluster title
            cluster_embedding = manager.get_embedding(cluster.event_title)
            similarity = manager.cosine_similarity(item_embedding, cluster_embedding)
            return max(0.0, (similarity + 1) / 2)

        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            return 0.0

    def _build_event_confirm_cache_key(
        self, item_id: int, candidates: list[ClusterCandidate]
    ) -> str:
        """Build a stable cache key for event confirmation results."""
        signature = "|".join(f"{c.cluster_id}:{c.cluster_title}" for c in candidates)
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
        except (json.JSONDecodeError, KeyError, TypeError):
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

    def _update_cluster_centroid(
        self, cluster_id: int, item: ContentItem, is_initial: bool = False
    ) -> None:
        """Update cluster centroid after adding a new member.

        Uses incremental centroid update formula:
        new_centroid = (old_centroid * n + new_embedding) / (n + 1)

        Args:
            cluster_id: Cluster ID to update
            item: New content item being added
            is_initial: If True, this is the first member (create centroid)
        """
        manager = self._get_embedding_manager()
        if not manager:
            return

        if not NUMPY_AVAILABLE:
            return

        try:
            from src.processors.embeddings import bytes_to_embedding, embedding_to_bytes

            # Get item embedding
            item_text = f"{item.title} {item.summary or ''}"[:500]
            item_embedding = manager.get_embedding(item_text)

            if is_initial:
                # First member - use item embedding as centroid
                centroid_bytes = embedding_to_bytes(item_embedding)
                self.db.save_cluster_centroid(cluster_id, centroid_bytes, 1)
                self._centroid_cache[cluster_id] = item_embedding
                logger.debug(f"Initialized centroid for cluster {cluster_id}")
            else:
                # Get existing centroid
                existing = self.db.get_cluster_centroid(cluster_id)
                if existing:
                    centroid_bytes, member_count = existing
                    old_centroid = bytes_to_embedding(centroid_bytes, manager.dimension)

                    # Incremental update
                    new_centroid = (old_centroid * member_count + item_embedding) / (
                        member_count + 1
                    )

                    # Normalize
                    norm = np.linalg.norm(new_centroid)
                    if norm > 0:
                        new_centroid = new_centroid / norm

                    new_centroid_bytes = embedding_to_bytes(new_centroid)
                    self.db.save_cluster_centroid(cluster_id, new_centroid_bytes, member_count + 1)
                    self._centroid_cache[cluster_id] = new_centroid
                    logger.debug(
                        f"Updated centroid for cluster {cluster_id} (now {member_count + 1} members)"
                    )
                else:
                    # No existing centroid - create one
                    centroid_bytes = embedding_to_bytes(item_embedding)
                    self.db.save_cluster_centroid(cluster_id, centroid_bytes, 1)
                    self._centroid_cache[cluster_id] = item_embedding
                    logger.debug(f"Created centroid for cluster {cluster_id}")

        except Exception as e:
            logger.warning(f"Failed to update cluster centroid: {e}")

    def _get_recent_clusters_cached(
        self, time_window_days: int, category: Optional[str] = None
    ) -> list[EventCluster]:
        """Get recent clusters with TTL caching.

        Args:
            time_window_days: Look back this many days
            category: Optional category filter. If None, returns all categories.
        """
        cache_key = f"{time_window_days}:{category or 'all'}"
        now = time.time()

        if cache_key in self._clusters_cache:
            cached_time, cached_clusters = self._clusters_cache[cache_key]
            if now - cached_time < self._clusters_cache_ttl:
                logger.debug(f"Using cached clusters for key={cache_key}")
                return cached_clusters

        # Query database and cache result
        # Pass None to get clusters from ALL categories for better cross-category matching
        clusters = self.db.get_recent_event_clusters(
            days=time_window_days, category=category, limit=100
        )
        self._clusters_cache[cache_key] = (now, clusters)
        return clusters

    def find_cluster_candidates(
        self, item: ContentItem, time_window_days: int = 3
    ) -> list[ClusterCandidate]:
        """
        Find potential cluster matches using hybrid similarity (rule-based + semantic).

        Rule-based scoring breakdown (max 1.0):
        - Title similarity (Jaccard): up to 0.5 (highest weight for content match)
        - Entity overlap: up to 0.25
        - Keyword overlap: up to 0.15
        - Category match: 0.05 (reduced - don't want to block cross-category)
        - Time proximity: 0.05

        When embeddings are enabled, final score is:
        - 40% rule-based score + 60% semantic score (configurable)

        Special handling:
        - HN discussion posts (Ask HN, Show HN, etc.) with different types are penalized
          to prevent incorrect clustering

        Args:
            item: Content item to find clusters for
            time_window_days: Look for clusters updated within this many days

        Returns:
            List of potential cluster matches with scores
        """
        candidates = []

        # Get recent clusters from ALL categories (not filtered by item.category)
        # This enables cross-category clustering for the same event
        recent_clusters = self._get_recent_clusters_cached(time_window_days, category=None)

        if not recent_clusters:
            return candidates

        # Extract key entities from the item (uses LRU cached functions)
        item_text = item.title + " " + (item.content or "")
        item_entities = _extract_entities_cached(item_text)
        item_keywords = _extract_keywords_cached(item.title)

        # Check HN discussion type for the item
        item_hn_type = _get_hn_discussion_type(item.title)

        # Check if embeddings are enabled
        use_embeddings = (
            self.settings.embedding.enabled and self._get_embedding_manager() is not None
        )

        for cluster in recent_clusters:
            rule_score = 0.0
            method = "keyword"

            # Check HN type mismatch penalty
            # If item is a HN discussion post and cluster is a different HN type, heavily penalize
            cluster_hn_type = _get_hn_discussion_type(cluster.event_title)
            if item_hn_type and cluster_hn_type and item_hn_type != cluster_hn_type:
                # Different HN discussion types should not cluster together
                continue

            # 1. Title similarity (most important - up to 0.5)
            title_sim = _calculate_title_similarity(item.title, cluster.event_title)
            if title_sim > 0:
                rule_score += 0.5 * title_sim
                if title_sim >= 0.8:
                    method = "title"

            # 2. Entity overlap (up to 0.25)
            cluster_entities = _extract_entities_cached(cluster.event_title)
            entity_overlap = item_entities & cluster_entities
            if entity_overlap:
                entity_score = 0.25 * min(len(entity_overlap), 3) / 3
                rule_score += entity_score
                if entity_score > 0.1:
                    method = "entity"

            # 3. Keyword overlap (up to 0.15)
            cluster_keywords = _extract_keywords_cached(cluster.event_title)
            keyword_overlap = item_keywords & cluster_keywords
            if keyword_overlap:
                rule_score += 0.15 * min(len(keyword_overlap), 5) / 5

            # 4. Category match (small bonus - 0.05)
            if item.category and cluster.category == item.category:
                rule_score += 0.05

            # 5. Time proximity bonus (0.05)
            time_diff = abs((datetime.now() - cluster.last_updated_at).days)
            if time_diff <= 1:
                rule_score += 0.05

            # Compute final score with hybrid similarity if embeddings enabled
            semantic_score = 0.0
            if use_embeddings:
                semantic_score = self._compute_semantic_similarity(item, cluster)
                from src.processors.embeddings import compute_hybrid_similarity

                # Detect if embeddings are actually available (non-zero semantic score
                # indicates embeddings were computed; zero might mean no centroids yet)
                embeddings_actually_available = semantic_score > 0.0
                final_score = compute_hybrid_similarity(
                    rule_score,
                    semantic_score,
                    self.settings.embedding.rule_weight,
                    self.settings.embedding.semantic_weight,
                    embeddings_available=embeddings_actually_available,
                )
                if semantic_score > 0.7:
                    method = "semantic"
                elif semantic_score > 0.5 and rule_score > 0.1:
                    method = "hybrid"
            else:
                final_score = rule_score

            # Threshold: 0.40 (raised from 0.35 to reduce false positives)
            # For keyword-only matches, require higher threshold to reduce low-quality matches
            threshold = 0.50 if method == "keyword" else 0.40
            if final_score >= threshold:
                candidates.append(
                    ClusterCandidate(
                        cluster_id=cluster.id,
                        cluster_title=cluster.event_title,
                        score=final_score,
                        method=method,
                        rule_score=rule_score,
                        semantic_score=semantic_score,
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
                logger.debug(f"Event confirmation cache hit for item {item.id}")
                record_ai_call(
                    call_type=AICallType.EVENT_CONFIRM,
                    cached=True,
                    input_chars=len(item.title),
                )
                return cached

        # Auto-accept high confidence matches without AI call
        # Score >= 0.75 indicates strong title/entity overlap (raised from 0.6)
        # With the new scoring system:
        # - Title similarity 0.8+ alone gives 0.4+ score
        # - Title 0.5 + 2 entities + same category = 0.25 + 0.17 + 0.05 = 0.47
        # - So 0.75 means very confident match
        if candidates[0].score >= 0.75:
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
            # In mock mode, accept the top candidate if score > 0.35
            # (lowered threshold for better clustering in mock mode)
            result = candidates[0] if candidates[0].score > 0.35 else None
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

        start_time = time.time()

        try:
            cli_path = self.settings.ai.get_cli_path()
            config = self.settings.ai.model_tiers

            # Build command with light model for simple confirmation task
            cmd = [cli_path, "-p", prompt, "--output-format", "text"]
            if config.enabled:
                model = config.task_overrides.get(
                    "event_confirm", self.settings.ai.get_light_model()
                )
                if model:
                    cmd.extend(["--model", model])
                    logger.debug(f"Event confirm using model: {model}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            if result.returncode == 0:
                response = result.stdout.strip()

                # Record successful AI call
                record_ai_call(
                    call_type=AICallType.EVENT_CONFIRM,
                    cached=False,
                    input_chars=len(prompt),
                    output_chars=len(response),
                    duration_ms=duration_ms,
                    success=True,
                )

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
            else:
                # Record failed AI call
                record_ai_call(
                    call_type=AICallType.EVENT_CONFIRM,
                    cached=False,
                    input_chars=len(prompt),
                    duration_ms=duration_ms,
                    success=False,
                    error=result.stderr[:100] if result.stderr else "Non-zero exit",
                )

        except subprocess.TimeoutExpired:
            logger.warning("AI confirmation timed out")
            record_ai_call(
                call_type=AICallType.EVENT_CONFIRM,
                cached=False,
                input_chars=len(prompt),
                success=False,
                error="Timeout",
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as e:
            logger.warning(f"AI confirmation failed: {e}")
            record_ai_call(
                call_type=AICallType.EVENT_CONFIRM,
                cached=False,
                input_chars=len(prompt),
                success=False,
                error=str(e)[:100],
            )

        # Fallback: accept moderately confident matches when AI fails
        if candidates[0].score >= 0.55:
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

        # Check if item is already in a cluster (prevent duplicates)
        if self.db.is_item_in_cluster(item.id):
            logger.debug(f"Item {item.id} already in a cluster, skipping")
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

                # Update cluster centroid if embeddings enabled
                self._update_cluster_centroid(confirmed.cluster_id, item)

                cluster = self.db.get_event_cluster(confirmed.cluster_id)
                logger.info(f"Added item {item.id} to cluster '{confirmed.cluster_title}'")
                return cluster

        # Create new cluster for important items
        # Threshold 6 balances coverage vs single-member cluster proliferation
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

            # Initialize cluster centroid if embeddings enabled
            self._update_cluster_centroid(cluster.id, item, is_initial=True)

            # Invalidate cluster cache since we created a new cluster
            cache_key = f"3:{None or 'all'}"
            if cache_key in self._clusters_cache:
                del self._clusters_cache[cache_key]

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
        """Generate a concise event title from content item.

        Uses the original English title (with HN prefixes removed) instead of
        AI translation. This ensures consistent formatting and saves tokens.
        The Chinese context is provided via the one_liner field instead.
        """
        title = item.title

        # Remove HN discussion prefixes
        for pattern in _HN_DISCUSSION_PREFIXES:
            title = re.sub(pattern, "", title, flags=re.IGNORECASE)

        title = title.strip()

        # Truncate to 50 characters at word boundary
        if len(title) > 50:
            truncated = title[:50]
            last_space = truncated.rfind(" ")
            if last_space > 30:
                title = truncated[:last_space]
            else:
                title = truncated

        return title.strip() if title.strip() else item.title[:50]

    def _generate_timeline_summary(
        self, cluster: EventCluster, today_items: list[ContentItem], recent_items: list[ContentItem]
    ) -> tuple[str, str]:
        """Generate a timeline summary comparing today vs recent developments using light model."""
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

        start_time = time.time()

        try:
            cli_path = self.settings.ai.get_cli_path()
            config = self.settings.ai.model_tiers

            # Build command with light model for timeline summary
            cmd = [cli_path, "-p", prompt, "--output-format", "text"]
            if config.enabled:
                model = config.task_overrides.get("timeline", self.settings.ai.get_light_model())
                if model:
                    cmd.extend(["--model", model])
                    logger.debug(f"Timeline summary using model: {model}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            if result.returncode == 0:
                response = result.stdout.strip()

                record_ai_call(
                    call_type=AICallType.TIMELINE_SUMMARY,
                    cached=False,
                    input_chars=len(prompt),
                    output_chars=len(response),
                    duration_ms=duration_ms,
                    success=True,
                )

                # Find JSON in response
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return data.get("summary", ""), data.get("consensus", "high")
            else:
                record_ai_call(
                    call_type=AICallType.TIMELINE_SUMMARY,
                    cached=False,
                    input_chars=len(prompt),
                    duration_ms=duration_ms,
                    success=False,
                )

        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            OSError,
            subprocess.SubprocessError,
        ) as e:
            logger.warning(f"Timeline summary CLI failed: {e}")
            record_ai_call(
                call_type=AICallType.TIMELINE_SUMMARY,
                cached=False,
                input_chars=len(prompt),
                success=False,
                error=str(e)[:50],
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Timeline summary JSON parse failed: {e}")
            record_ai_call(
                call_type=AICallType.TIMELINE_SUMMARY,
                cached=False,
                input_chars=len(prompt),
                success=False,
                error="JSON parse error",
            )

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
    # min_importance=5 allows matching to existing clusters; creating new requires >= 6
    retry_after_hours = int(os.environ.get("CLUSTER_RETRY_HOURS", "24"))
    items = db.get_unclustered_items(
        min_importance=5,
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
    # min_importance=5 allows matching to existing clusters; creating new requires >= 6
    retry_after_hours = int(os.environ.get("CLUSTER_RETRY_HOURS", "24"))
    items = db.get_unclustered_items(
        min_importance=5,
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
            thread_local.processor = EventStreamProcessor(db=get_thread_db(), use_mock=use_mock)
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

    # Track thread-local resources for cleanup (use set to avoid duplicates)
    thread_resources: set = set()
    resources_lock = threading.Lock()

    def register_thread_resources():
        """Register current thread's resources for later cleanup."""
        if hasattr(thread_local, "db") and thread_local.db is not db:
            with resources_lock:
                thread_resources.add(thread_local.db)

    def process_single_item_with_registration(item: "ContentItem") -> bool:
        """Process a single item and register resources for cleanup."""
        result = process_single_item(item)
        register_thread_resources()
        return result

    # Process items in parallel
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_item_with_registration, item): item for item in items
            }

            # Wait for completion (results are tracked via side effects)
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    item = futures[future]
                    logger.warning(f"Error processing item {item.id}: {e}")
    finally:
        # Clean up all thread-local database connections
        for thread_db in thread_resources:
            try:
                thread_db.close()
            except Exception:
                pass

    logger.info(
        f"Clustered {clustered}/{total} items into events (parallel, {max_workers} workers)"
    )
    return clustered

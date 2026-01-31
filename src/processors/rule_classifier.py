"""Rule-based classifier for content pre-classification to save AI tokens."""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml

from src.storage.models import ContentItem

logger = logging.getLogger(__name__)

# Path to classification config
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "classifications.yaml"


@dataclass
class ImportanceEstimate:
    """Result of importance estimation before AI processing."""

    score: int  # Estimated score 1-10
    confidence: float  # Confidence level 0-1
    reason: str  # Why this estimate was made


@dataclass
class RuleClassificationResult:
    """Result of rule-based classification."""

    category: str
    importance_score: int
    summary: Optional[str] = None  # Optional pre-generated summary
    reason: str = ""  # Why this rule matched


@lru_cache(maxsize=1)
def _load_classifications() -> dict:
    """Load classifications from YAML config (cached)."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Classifications config not found at {CONFIG_PATH}, using defaults")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse classifications config: {e}")
        return {}


def _get_domain_categories() -> dict[str, tuple[str, int]]:
    """Get domain to category mappings from config."""
    config = _load_classifications()
    domain_config = config.get("domain_categories", {})

    result = {}
    for domain, data in domain_config.items():
        if isinstance(data, dict):
            category = data.get("category", "其他")
            importance = data.get("importance", 5)
            result[domain] = (category, importance)

    return result


def _get_title_keywords() -> list[tuple[re.Pattern, str, int]]:
    """Get title keyword patterns from config."""
    config = _load_classifications()
    keywords_config = config.get("title_keywords", [])

    result = []
    for item in keywords_config:
        if not isinstance(item, dict):
            continue

        pattern_str = item.get("pattern", "")
        if not pattern_str:
            continue

        category = item.get("category", "其他")
        boost = item.get("boost", 0)
        case_insensitive = item.get("case_insensitive", False)

        try:
            flags = re.IGNORECASE if case_insensitive else 0
            pattern = re.compile(pattern_str, flags)
            result.append((pattern, category, boost))
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern_str}': {e}")

    return result


def _get_importance_patterns(key: str) -> list[tuple[str, int, float]]:
    """Get importance estimation patterns from config."""
    config = _load_classifications()
    patterns_config = config.get(key, [])

    result = []
    for item in patterns_config:
        if not isinstance(item, dict):
            continue

        pattern = item.get("pattern", "")
        if not pattern:
            continue

        boost = item.get("boost", 0)
        confidence = item.get("confidence", 0.5)
        result.append((pattern, boost, confidence))

    return result


class RuleClassifier:
    """
    Rule-based content classifier to reduce AI token usage.

    Classifies content based on:
    - Domain patterns in URLs
    - Title keywords
    - Content patterns

    Configuration is loaded from config/classifications.yaml
    """

    def __init__(self):
        """Initialize the classifier with config-based mappings."""
        self._domain_categories: Optional[dict[str, tuple[str, int]]] = None
        self._title_keywords: Optional[list[tuple[re.Pattern, str, int]]] = None

    @property
    def domain_categories(self) -> dict[str, tuple[str, int]]:
        """Get domain categories (lazy loaded)."""
        if self._domain_categories is None:
            self._domain_categories = _get_domain_categories()
        return self._domain_categories

    @property
    def title_keywords(self) -> list[tuple[re.Pattern, str, int]]:
        """Get title keywords (lazy loaded)."""
        if self._title_keywords is None:
            self._title_keywords = _get_title_keywords()
        return self._title_keywords

    def classify(self, item: ContentItem) -> Optional[RuleClassificationResult]:
        """
        Attempt to classify content using rules.

        Args:
            item: The content item to classify.

        Returns:
            RuleClassificationResult if rules matched, None if AI processing needed.
        """
        # Try domain-based classification first
        result = self._classify_by_domain(item)
        if result:
            # Adjust importance based on title keywords
            result = self._adjust_by_keywords(item, result)
            logger.debug(
                f"Rule classified: {item.title[:40]}... -> {result.category} ({result.reason})"
            )
            return result

        # Try keyword-only classification for strong signals
        result = self._classify_by_keywords_only(item)
        if result:
            logger.debug(
                f"Keyword classified: {item.title[:40]}... -> {result.category} ({result.reason})"
            )
            return result

        return None

    def _classify_by_domain(self, item: ContentItem) -> Optional[RuleClassificationResult]:
        """Classify based on URL domain."""
        if not item.url:
            return None

        try:
            parsed = urlparse(item.url)
            domain = parsed.netloc.lower()

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Check exact match first
            if domain in self.domain_categories:
                category, importance = self.domain_categories[domain]
                return RuleClassificationResult(
                    category=category,
                    importance_score=importance,
                    reason=f"domain:{domain}",
                )

            # Check subdomain matches (e.g., blog.github.com)
            for known_domain, (category, importance) in self.domain_categories.items():
                if domain.endswith("." + known_domain) or domain == known_domain:
                    return RuleClassificationResult(
                        category=category,
                        importance_score=importance,
                        reason=f"domain:{known_domain}",
                    )

        except (ValueError, AttributeError):
            pass

        return None

    def _adjust_by_keywords(
        self, item: ContentItem, result: RuleClassificationResult
    ) -> RuleClassificationResult:
        """Adjust classification based on title keywords."""
        text = f"{item.title} {item.content[:500] if item.content else ''}"

        importance_boost = 0
        matched_keywords = []

        for pattern, _, boost in self.title_keywords:
            if pattern.search(text):
                importance_boost = max(importance_boost, boost)
                matched_keywords.append(pattern.pattern[:20])

        if importance_boost > 0:
            result.importance_score = min(10, result.importance_score + importance_boost)
            if matched_keywords:
                result.reason += f"+keywords:{','.join(matched_keywords[:2])}"

        return result

    def _classify_by_keywords_only(self, item: ContentItem) -> Optional[RuleClassificationResult]:
        """Classify based on strong keyword signals only (no domain match)."""
        text = f"{item.title} {item.content[:500] if item.content else ''}"

        # Count category signals
        category_scores: dict[str, int] = {}

        for pattern, category, boost in self.title_keywords:
            if pattern.search(text):
                if category not in category_scores:
                    category_scores[category] = 0
                category_scores[category] += 1 + boost

        # Classify if we have moderate signals (2+ score for same category)
        if category_scores:
            best_category = max(category_scores, key=lambda k: category_scores[k])
            if category_scores[best_category] >= 2:
                return RuleClassificationResult(
                    category=best_category,
                    importance_score=6,  # Default medium importance
                    reason=f"keywords:{best_category}",
                )

        return None

    def _classify_by_domain_with_url(self, url: str) -> Optional[RuleClassificationResult]:
        """Classify based on URL domain only (used by estimate_importance)."""
        if not url:
            return None

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Check exact match first
            if domain in self.domain_categories:
                category, importance = self.domain_categories[domain]
                return RuleClassificationResult(
                    category=category,
                    importance_score=importance,
                    reason=f"domain:{domain}",
                )

            # Check subdomain matches
            for known_domain, (category, importance) in self.domain_categories.items():
                if domain.endswith("." + known_domain) or domain == known_domain:
                    return RuleClassificationResult(
                        category=category,
                        importance_score=importance,
                        reason=f"domain:{known_domain}",
                    )

        except (ValueError, AttributeError):
            pass

        return None


def estimate_importance(item: ContentItem) -> ImportanceEstimate:
    """
    Estimate importance score before AI processing for model tier selection.

    Returns score with confidence level to guide heavy/light model decision.

    Args:
        item: Content item to estimate.

    Returns:
        ImportanceEstimate with score, confidence, and reason.
    """
    score = 5
    confidence = 0.3  # Default low confidence
    reason = "default"

    classifier = RuleClassifier()

    # 1. Domain matching - high confidence
    domain_result = classifier._classify_by_domain_with_url(item.url)
    if domain_result:
        score = domain_result.importance_score
        confidence = 0.85
        domain_name = domain_result.reason.split(":")[1] if ":" in domain_result.reason else ""
        reason = f"domain:{domain_name}" if domain_name else domain_result.reason
        return ImportanceEstimate(score, confidence, reason)

    # 2. Strong keyword patterns - medium-high confidence
    text = f"{item.title} {item.content[:500] if item.content else ''}"

    high_value_patterns = _get_importance_patterns("high_value_patterns")
    for pattern, boost, conf in high_value_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score = min(10, score + boost)
            confidence = conf
            reason = f"strong_keyword:{pattern[:25]}"
            return ImportanceEstimate(score, confidence, reason)

    # 3. Weak keyword patterns - medium confidence
    weak_patterns = _get_importance_patterns("weak_patterns")
    for pattern, boost, conf in weak_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score = min(10, score + boost)
            if conf > confidence:
                confidence = conf
                reason = f"weak_keyword:{pattern[:20]}"

    # 4. Unknown domain + no keyword matches - low confidence
    if confidence < 0.4:
        reason = "unknown"

    return ImportanceEstimate(score, confidence, reason)


def should_skip_ai_processing(item: ContentItem) -> tuple[bool, str]:
    """
    Check if content should skip AI processing entirely.

    Returns:
        (should_skip, reason) tuple
    """
    content = item.content or ""
    title = item.title or ""

    # Reddit link-posts with no real content
    if len(content) < 100 and "[link]" in content.lower():
        return True, "Reddit link-post with no content"

    # Very short content (likely just a link or single sentence)
    if len(content) < 50 and len(title) < 30:
        return True, "Content too short (<50 chars)"

    # Content is just "submitted by" boilerplate
    if content.strip().startswith("submitted by") and len(content) < 150:
        return True, "Reddit submission boilerplate only"

    # Empty or whitespace-only content
    if not content.strip():
        return True, "Empty content"

    return False, ""


def create_skip_result(item: ContentItem, reason: str) -> tuple[str, str, int]:
    """
    Create default values for skipped items.

    Returns:
        (summary, category, importance_score) tuple
    """
    # Use title as summary for skipped items
    title_preview = item.title[:40] + "..." if len(item.title) > 40 else item.title
    summary = f"[Skipped] {title_preview}"
    return summary, "其他", 1  # Low importance for skipped items


def reload_classifications() -> None:
    """Force reload of classifications config."""
    _load_classifications.cache_clear()

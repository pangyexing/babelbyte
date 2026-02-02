"""Rule-based classifier for content pre-classification to save AI tokens."""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml

from src.processors.base import ProcessingResult, KeyPointResult, ImpactResult
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

    # 4. Source-based confidence boost (for unmatched content)
    # Quality sources get higher confidence to use LIGHT_PROMPT instead of SIMPLE_PROMPT
    if confidence < 0.4:
        source_confidence = {
            "reddit": (0.5, "source:reddit"),      # Medium confidence → LIGHT_PROMPT
            "twitter": (0.5, "source:twitter"),
            "hackernews": (0.6, "source:hackernews"),
        }
        source_type = getattr(item, "source_type", None)
        if source_type in source_confidence:
            conf_boost, src_reason = source_confidence[source_type]
            confidence = conf_boost
            reason = src_reason
        else:
            reason = "unknown"

    return ImportanceEstimate(score, confidence, reason)


def should_skip_ai_processing(item: ContentItem) -> tuple[bool, str]:
    """
    Check if content should skip AI processing entirely.

    Configuration (env vars):
    - SKIP_ENABLED: Enable/disable skip processing (default: true)

    Returns:
        (should_skip, reason) tuple
    """
    from config.settings import get_settings
    settings = get_settings().rule_optimization

    # Check if skip processing is enabled
    if not settings.skip_enabled:
        return False, ""

    content = item.content or ""
    title = item.title or ""
    content_lower = content.lower()
    title_lower = title.lower()

    # Empty or whitespace-only content
    if not content.strip():
        return True, "Empty content"

    # Very short content (likely just a link or single sentence)
    if len(content) < 100 and len(title) < 50:
        return True, "Content too short"

    # Reddit link-posts with no real content
    if len(content) < 100 and "[link]" in content_lower:
        return True, "Reddit link-post with no content"

    # Content is just "submitted by" boilerplate
    if content.strip().startswith("submitted by") and len(content) < 150:
        return True, "Reddit submission boilerplate only"

    # Twitter/X retweet with no added content (avg RT imp only 4.8, safe to skip up to 500 chars)
    if title_lower.startswith("rt @") and len(content) < 500:
        return True, "Retweet with no added content"

    # Art posts - typically low value for tech intelligence (title patterns like "medium, year")
    art_patterns = [
        r",\s*(digital|oil|acrylic|watercolor|charcoal|graphite|mixed media)\s*,?\s*\d{4}",
        r"\[OC\]\s*$",
        r",\s*\d+\s*x\s*\d+\s*(cm|in|px|mm)?",
    ]
    for pattern in art_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return True, "Art post"

    # Job postings / hiring posts
    job_patterns = [
        r"\b(we.?re hiring|job opening|career opportunity|join our team)\b",
        r"\b(招聘|招人|求职|职位|岗位空缺)\b",
        r"\b(apply now|submit.*resume|send.*cv)\b",
    ]
    for pattern in job_patterns:
        if re.search(pattern, content_lower, re.IGNORECASE):
            return True, "Job posting"

    # Promotional / spam patterns
    spam_patterns = [
        r"\b(click here|subscribe now|limited time|act now|don.?t miss)\b",
        r"\b(免费领取|限时优惠|点击领取|扫码关注)\b",
        r"(🔥|💰|🎁|💯){2,}",  # Multiple promotional emojis
        r"\b(giveaway|airdrop|free.*tokens?)\b",
    ]
    for pattern in spam_patterns:
        if re.search(pattern, content_lower, re.IGNORECASE):
            return True, "Promotional/spam content"

    # Duplicate/cross-post indicators
    if re.search(r"(cross-?posted?|x-?post|originally posted)", content_lower):
        return True, "Cross-post reference"

    # Auto-generated bot content
    bot_patterns = [
        r"^(I am a bot|This is an automated|Auto-generated)",
        r"\[bot\]|\[automated\]",
        r"(This action was performed automatically)",
    ]
    for pattern in bot_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True, "Bot-generated content"

    # Social media follow/like requests
    social_patterns = [
        r"\b(follow me|like and subscribe|hit that.*button)\b",
        r"\b(关注我|点赞|转发|求关注)\b",
        r"(follow|subscribe|like).*(@|#)",
    ]
    for pattern in social_patterns:
        if re.search(pattern, content_lower, re.IGNORECASE):
            if len(content) < 200:  # Only skip if it's mostly just the request
                return True, "Social media engagement request"

    # Newsletter subscription prompts
    if re.search(r"(subscribe.*newsletter|sign up.*updates|join.*mailing list)", content_lower):
        if len(content) < 300:
            return True, "Newsletter subscription prompt"

    # Content that's just a list of links/references
    url_count = len(re.findall(r"https?://\S+", content))
    if url_count >= 5 and len(content) < 500:
        # Mostly URLs with little actual content
        non_url_content = re.sub(r"https?://\S+", "", content).strip()
        if len(non_url_content) < 100:
            return True, "Link aggregation with minimal content"

    # Repetitive content (same phrases repeated)
    words = content_lower.split()
    if len(words) >= 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # Less than 30% unique words
            return True, "Repetitive/spam content"

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


# High-confidence domains that can skip AI processing entirely
HIGH_CONFIDENCE_DOMAINS = {
    # Major AI companies - very predictable content structure
    "openai.com": ("AI", 8),
    "anthropic.com": ("AI", 8),
    "deepmind.com": ("AI", 8),
    "mistral.ai": ("AI", 8),
    "huggingface.co": ("AI", 7),
    "developer.nvidia.com": ("AI", 7),
    # Top science journals
    "nature.com": ("科学", 8),
    "science.org": ("科学", 8),
    "arxiv.org": ("科学", 7),
    # Official company blogs (predictable announcements)
    "blog.google": ("技术", 7),
    "engineering.fb.com": ("编程", 7),
    "aws.amazon.com/blogs": ("技术", 7),
    # Programming language official sites
    "pytorch.org": ("编程", 7),
    "tensorflow.org": ("编程", 7),
    "rust-lang.org": ("编程", 6),
    "python.org": ("编程", 6),
}


def try_rule_only_processing(item: ContentItem) -> Optional[ProcessingResult]:
    """
    Attempt to fully process content using rules only, skipping AI.

    This is for high-confidence cases where:
    1. Domain is well-known with predictable content
    2. Strong keyword signals confirm the classification
    3. We can generate a reasonable summary from title

    Quality Protection:
    - Requires keyword confirmation (prevents wrong category)
    - Only works for content < max_content_length (configurable, default 1000)
    - Returns metadata indicating rule-based processing for tracking

    Configuration (env vars):
    - RULE_ONLY_ENABLED: Enable/disable rule-only processing (default: true)
    - RULE_ONLY_MAX_CONTENT: Max content length for rule-only (default: 1000)
    - RULE_ONLY_MIN_BOOST: Min keyword boost required (default: 1)

    Args:
        item: ContentItem to process

    Returns:
        ProcessingResult if rules can handle it, None if AI is needed.
    """
    from config.settings import get_settings
    settings = get_settings().rule_optimization

    # Check if rule-only processing is enabled
    if not settings.rule_only_enabled:
        return None

    if not item.url or not item.title:
        return None

    # Quality gate: Long content needs AI for proper analysis
    content_len = len(item.content) if item.content else 0
    if content_len > settings.rule_only_max_content_length:
        logger.debug(f"Content too long for rule-only: {content_len} chars")
        return None

    # Check domain
    try:
        from urllib.parse import urlparse
        parsed = urlparse(item.url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
    except (ValueError, AttributeError):
        return None

    # Check high-confidence domains
    domain_match = None
    for known_domain, (category, importance) in HIGH_CONFIDENCE_DOMAINS.items():
        if domain == known_domain or domain.endswith("." + known_domain):
            domain_match = (category, importance)
            break

    if not domain_match:
        return None

    category, base_importance = domain_match

    # Verify with keyword signals
    text = f"{item.title} {item.content[:300] if item.content else ''}"
    keyword_signals = _check_keyword_signals(text, category)

    if not keyword_signals:
        # Domain matched but no confirming keywords - let AI handle
        return None

    # Check minimum boost requirement
    if keyword_signals.get("boost", 0) < settings.rule_only_min_keyword_boost:
        logger.debug(f"Keyword boost too low: {keyword_signals.get('boost', 0)}")
        return None

    # Adjust importance based on keywords
    importance = min(10, base_importance + keyword_signals.get("boost", 0))

    # Generate summary from title (simple extraction)
    summary = _generate_rule_summary(item.title, category)
    one_liner = _generate_one_liner(item.title, category)

    # Extract key points from title
    key_points = _extract_key_points_from_title(item.title)

    logger.info(
        f"Rule-only processing: {item.title[:40]}... -> {category} "
        f"(importance={importance}, domain={domain})"
    )

    return ProcessingResult(
        summary=summary,
        category=category,
        importance_score=importance,
        success=True,
        one_liner=one_liner,
        key_points=key_points,
        impact_assessment=ImpactResult(
            short_term="待观察",
            long_term="待评估",
            certainty="uncertain"
        ),
        actionable_items=[],
    )


def _check_keyword_signals(text: str, expected_category: str) -> Optional[dict]:
    """Check if text contains confirming keyword signals for the category."""
    category_keywords = {
        "AI": [
            (r"\b(GPT|Claude|Gemini|LLM|model|AI|机器学习)\b", 1),
            (r"\b(launch|release|announce|发布|上线|更新)\b", 1),
            (r"\b(研究|论文|paper|research)\b", 0),
        ],
        "科学": [
            (r"\b(研究|论文|paper|study|research)\b", 1),
            (r"\b(发现|breakthrough|discovery)\b", 2),
            (r"\b(实验|experiment|trial)\b", 0),
        ],
        "技术": [
            (r"\b(技术|tech|engineering|架构)\b", 0),
            (r"\b(更新|update|release|版本)\b", 1),
            (r"\b(开源|open.?source)\b", 1),
        ],
        "编程": [
            (r"\b(code|coding|程序|开发|developer)\b", 0),
            (r"\b(API|SDK|库|library|framework)\b", 1),
            (r"\b(bug|fix|feature|PR|commit)\b", 0),
        ],
    }

    patterns = category_keywords.get(expected_category, [])
    if not patterns:
        return None

    total_boost = 0
    matched = False
    for pattern, boost in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matched = True
            total_boost += boost

    if matched:
        return {"boost": min(total_boost, 2)}  # Cap boost at 2
    return None


def _generate_rule_summary(title: str, category: str) -> str:
    """Generate a simple summary from title for rule-based processing."""
    # Clean title
    title = re.sub(r"^(RT\s+)?@\w+[:\s]*", "", title)  # Remove RT and mentions
    title = re.sub(r"https?://\S+", "", title)  # Remove URLs
    title = title.strip()

    if len(title) <= 50:
        return title

    # Truncate at sentence boundary
    for sep in ["。", "！", "？", ".", "!", "?", "，", ",", " - ", " | "]:
        pos = title.find(sep)
        if 10 < pos < 50:
            return title[:pos + 1].strip()

    return title[:47] + "..."


def _generate_one_liner(title: str, category: str) -> str:
    """Generate a one-liner conclusion from title."""
    category_prefixes = {
        "AI": "AI动态",
        "科学": "科研进展",
        "技术": "技术更新",
        "编程": "开发资讯",
        "产品": "产品动态",
        "创业": "创业动态",
        "商业": "商业资讯",
    }
    prefix = category_prefixes.get(category, "资讯")

    # Extract key subject from title
    title_clean = re.sub(r"^(RT\s+)?@\w+[:\s]*", "", title)
    title_clean = re.sub(r"https?://\S+", "", title_clean).strip()

    if len(title_clean) <= 30:
        return f"{prefix}：{title_clean}"

    # Find key part
    for sep in ["：", ":", " - ", " | "]:
        if sep in title_clean:
            parts = title_clean.split(sep)
            return f"{prefix}：{parts[0].strip()[:30]}"

    return f"{prefix}：{title_clean[:30]}..."


def _extract_key_points_from_title(title: str) -> list[KeyPointResult]:
    """Extract key points from title using patterns."""
    key_points = []

    # Extract entities (companies, products)
    entity_patterns = [
        (r"\b(OpenAI|Anthropic|Google|Meta|Microsoft|Apple|Nvidia)\b", "实体"),
        (r"\b(GPT-?\d+|Claude|Gemini|Llama|ChatGPT)\b", "实体"),
        (r"(阿里|腾讯|百度|字节跳动|华为|小米)", "实体"),
    ]
    for pattern, ptype in entity_patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match and len(key_points) < 3:
            key_points.append(KeyPointResult(
                type=ptype,
                value=match.group(1) if match.lastindex else match.group(),
                impact="相关主体"
            ))

    # Extract numbers (funding, metrics)
    number_patterns = [
        (r"\$(\d+(?:\.\d+)?[BMK]?)", "数字", "金额"),
        (r"(\d+(?:\.\d+)?%)", "数字", "比例"),
        (r"(\d+(?:,\d+)*)\s*(?:用户|users?|downloads?)", "数字", "规模"),
    ]
    for pattern, ptype, impact in number_patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match and len(key_points) < 3:
            key_points.append(KeyPointResult(
                type=ptype,
                value=match.group(1) if match.lastindex else match.group(),
                impact=impact
            ))

    return key_points[:3]  # Max 3 key points

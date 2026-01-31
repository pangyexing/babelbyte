"""Rule-based classifier for content pre-classification to save AI tokens."""

import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from src.storage.models import ContentItem

logger = logging.getLogger(__name__)


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


class RuleClassifier:
    """
    Rule-based content classifier to reduce AI token usage.

    Classifies content based on:
    - Domain patterns in URLs
    - Title keywords
    - Content patterns
    """

    # Domain -> (category, base_importance)
    DOMAIN_CATEGORIES: dict[str, tuple[str, int]] = {
        # Programming
        "github.com": ("编程", 6),
        "gitlab.com": ("编程", 6),
        "stackoverflow.com": ("编程", 5),
        "dev.to": ("编程", 5),
        "hackernews.com": ("技术", 6),
        "news.ycombinator.com": ("技术", 7),
        "medium.com": ("技术", 5),
        "substack.com": ("技术", 5),
        "hashnode.dev": ("编程", 5),
        "codepen.io": ("编程", 5),
        "replit.com": ("编程", 5),
        "codesandbox.io": ("编程", 5),
        # Science
        "arxiv.org": ("科学", 7),
        "nature.com": ("科学", 8),
        "science.org": ("科学", 8),
        "sciencedirect.com": ("科学", 7),
        "pubmed.ncbi.nlm.nih.gov": ("科学", 7),
        "biorxiv.org": ("科学", 7),
        "medrxiv.org": ("科学", 7),
        # AI
        "openai.com": ("AI", 8),
        "anthropic.com": ("AI", 8),
        "deepmind.com": ("AI", 8),
        "huggingface.co": ("AI", 7),
        "replicate.com": ("AI", 6),
        "stability.ai": ("AI", 7),
        "midjourney.com": ("AI", 6),
        # Tech news
        "techcrunch.com": ("技术", 6),
        "theverge.com": ("技术", 5),
        "wired.com": ("技术", 5),
        "arstechnica.com": ("技术", 6),
        "engadget.com": ("技术", 5),
        "thenextweb.com": ("技术", 5),
        "zdnet.com": ("技术", 5),
        "cnet.com": ("技术", 5),
        # Product
        "producthunt.com": ("产品", 6),
        "indiegogo.com": ("产品", 5),
        "kickstarter.com": ("产品", 5),
        "betalist.com": ("产品", 5),
        # Business/Startup
        "crunchbase.com": ("创业", 6),
        "ycombinator.com": ("创业", 7),
        "techstars.com": ("创业", 6),
        "bloomberg.com": ("商业", 6),
        "wsj.com": ("商业", 6),
        "ft.com": ("商业", 6),
        "techcrunch.com/tag/funding": ("创业", 7),
        "venturebeat.com": ("创业", 6),
    }

    # Title keyword patterns -> (category, importance_boost)
    TITLE_KEYWORDS: list[tuple[re.Pattern, str, int]] = [
        # AI keywords
        (re.compile(r"\b(GPT-?[45]|Claude|Gemini|LLM|ChatGPT)\b", re.I), "AI", 2),
        (re.compile(r"\b(machine learning|deep learning|neural network)\b", re.I), "AI", 1),
        (re.compile(r"\b(transformer|diffusion|RLHF)\b", re.I), "AI", 1),
        (re.compile(r"\b(AI|artificial intelligence|ML model)\b", re.I), "AI", 1),
        (re.compile(r"\b(RAG|fine-?tun|embeddings?|vector)\b", re.I), "AI", 1),
        # Programming keywords
        (re.compile(r"\b(Python|Rust|Go|JavaScript|TypeScript)\b"), "编程", 0),
        (re.compile(r"\b(API|SDK|framework|library)\b", re.I), "编程", 0),
        (re.compile(r"\b(React|Vue|Angular|Node\.?js|Django|Flask)\b", re.I), "编程", 0),
        (re.compile(r"\b(database|SQL|PostgreSQL|MongoDB|Redis)\b", re.I), "编程", 0),
        (re.compile(r"\b(Docker|Kubernetes|K8s|DevOps|CI/CD)\b", re.I), "编程", 1),
        # Product launches
        (re.compile(r"\b(launch|announce|release|introducing)\b", re.I), "产品", 1),
        (re.compile(r"\b(new feature|update|v\d+\.\d+)\b", re.I), "产品", 0),
        # Funding/business
        (re.compile(r"\$\d+[MBK]|\d+ million|\d+ billion", re.I), "商业", 1),
        (re.compile(r"\b(Series [A-Z]|seed round|IPO|acquisition)\b", re.I), "创业", 2),
        (re.compile(r"\b(startup|founder|YC|Y Combinator)\b", re.I), "创业", 1),
        # Question posts (common on Reddit)
        (
            re.compile(r"^(How|What|Why|When|Where|Who|Which|Can|Should|Is|Are|Does|Do)\b", re.I),
            "其他",
            0,
        ),  # noqa: E501
        (re.compile(r"\?$"), "其他", 0),
        # Science
        (re.compile(r"\b(research|study|paper|published)\b", re.I), "科学", 0),
        (re.compile(r"\b(breakthrough|discovery|experiment)\b", re.I), "科学", 1),
    ]

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
            if domain in self.DOMAIN_CATEGORIES:
                category, importance = self.DOMAIN_CATEGORIES[domain]
                return RuleClassificationResult(
                    category=category,
                    importance_score=importance,
                    reason=f"domain:{domain}",
                )

            # Check subdomain matches (e.g., blog.github.com)
            for known_domain, (category, importance) in self.DOMAIN_CATEGORIES.items():
                if domain.endswith("." + known_domain) or domain == known_domain:
                    return RuleClassificationResult(
                        category=category,
                        importance_score=importance,
                        reason=f"domain:{known_domain}",
                    )

        except Exception:
            pass

        return None

    def _adjust_by_keywords(
        self, item: ContentItem, result: RuleClassificationResult
    ) -> RuleClassificationResult:
        """Adjust classification based on title keywords."""
        text = f"{item.title} {item.content[:500]}"

        importance_boost = 0
        matched_keywords = []

        for pattern, _, boost in self.TITLE_KEYWORDS:
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
        text = f"{item.title} {item.content[:500]}"

        # Count category signals
        category_scores: dict[str, int] = {}

        for pattern, category, boost in self.TITLE_KEYWORDS:
            if pattern.search(text):
                if category not in category_scores:
                    category_scores[category] = 0
                category_scores[category] += 1 + boost

        # Classify if we have moderate signals (2+ score for same category)
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] >= 2:  # Lowered threshold for better coverage
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
            if domain in self.DOMAIN_CATEGORIES:
                category, importance = self.DOMAIN_CATEGORIES[domain]
                return RuleClassificationResult(
                    category=category,
                    importance_score=importance,
                    reason=f"domain:{domain}",
                )

            # Check subdomain matches
            for known_domain, (category, importance) in self.DOMAIN_CATEGORIES.items():
                if domain.endswith("." + known_domain) or domain == known_domain:
                    return RuleClassificationResult(
                        category=category,
                        importance_score=importance,
                        reason=f"domain:{known_domain}",
                    )

        except Exception:
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

    high_value_patterns = [
        (r"(OpenAI|Anthropic|Google|Meta)\s+(announces?|launches?|releases?)", 3, 0.8),
        (r"\$\d+[BM]\b", 2, 0.75),
        (r"(breakthrough|major|significant)\s+(discovery|advancement|update)", 2, 0.7),
        (r"(Series [C-Z]|IPO|acqui)", 2, 0.7),
        (r"(GPT-?5|Claude\s+\d|Gemini\s+\d)", 2, 0.75),
    ]

    for pattern, boost, conf in high_value_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score = min(10, score + boost)
            confidence = conf
            reason = f"strong_keyword:{pattern[:25]}"
            return ImportanceEstimate(score, confidence, reason)

    # 3. Weak keyword patterns - medium confidence
    weak_patterns = [
        (r"\b(GPT|Claude|Gemini|LLM|ChatGPT)\b", 1, 0.55),
        (r"\b(AI|ML|机器学习|深度学习)\b", 1, 0.5),
        (r"\b(launch|announce|release|发布|上线)\b", 1, 0.45),
    ]

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
    summary = f"[跳过] {item.title[:40]}..." if len(item.title) > 40 else f"[跳过] {item.title}"
    return summary, "其他", 1  # Low importance for skipped items

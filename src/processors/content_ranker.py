"""Content pre-ranking algorithm based on historical AI scoring data.

This module provides priority scoring for content items to optimize AI processing order.
Lower scores indicate higher priority (processed first).
"""

import re
from typing import Optional

# Source quality tiers based on historical AI average scores
# Tier 1: avg 6.5+ (priority 10) - highest quality sources
# Tier 2: avg 5.5-6.5 (priority 20) - good quality
# Tier 3: avg 4.5-5.5 (priority 30) - medium quality
# Tier 4: avg <4.5 (priority 50) - lower quality
SOURCE_TIERS = {
    # Tier 1: avg 6.5+ (priority 10)
    "reddit/hardware": 10,
    "reddit/technology": 10,
    "reddit/programming": 10,
    "reddit/localllama": 10,
    "reddit/futurology": 10,
    # Tier 2: avg 5.5-6.5 (priority 20)
    "reddit/artificial": 20,
    "reddit/python": 20,
    "reddit/ycombinator": 20,
    "hackernews": 20,
    "reddit/reinforcementlearning": 20,
    "reddit/indiehackers": 20,
    "reddit/datascience": 20,
    "reddit/mlquestions": 20,
    # Academic/Research sources (priority 15) - high quality but separate handling
    "arxiv": 15,
    "nature": 15,
    "science": 15,
    # Tier 3: avg 4.5-5.5 (priority 30)
    "twitter": 30,
    "reddit/machinelearning": 30,
    "reddit/sideproject": 30,
    "reddit/stablediffusion": 30,
    "reddit/startups": 30,
    "reddit/learnmachinelearning": 30,
    "reddit/statistics": 30,
    # Tier 4: avg <4.5 (priority 50)
    "reddit/entrepreneur": 50,
    "reddit/design": 50,
    "reddit/creativecoding": 50,
    "reddit/art": 50,
    # RSS feeds - default medium priority
    "rss": 35,
}

# Twitter high-value authors (negative bonus = higher priority)
HIGH_VALUE_AUTHORS = {
    "@anthropicai": -15,
    "@openai": -15,
    "@a16z": -10,
    "@paperswithcode": -10,
    "@karpathy": -10,
    "@googledeepmind": -5,
    "@stanfordhai": -5,
}

# Source-specific length configuration for penalty calculation
# avg: average content length for this source
# penalty_below: content shorter than this gets +10 penalty
SOURCE_LENGTH_CONFIG = {
    "twitter": {"avg": 361, "penalty_below": 108},  # short tweets penalized
    "reddit": {"avg": 1320, "penalty_below": 396},  # very short posts penalized
    "hackernews": {"avg": 2674, "penalty_below": None},  # short content NOT penalized (still high quality)
    "arxiv": {"avg": 1500, "penalty_below": None},  # academic abstracts - no length penalty
    "nature": {"avg": 1500, "penalty_below": None},  # academic - no length penalty
    "science": {"avg": 1500, "penalty_below": None},  # academic - no length penalty
}

# Default priority for unknown sources
DEFAULT_SOURCE_PRIORITY = 40


def extract_source_key(url: str) -> str:
    """Extract source identifier from URL.

    Args:
        url: Content URL

    Returns:
        Source key like "twitter", "hackernews", "arxiv", or "reddit/subreddit"
    """
    if not url:
        return "other"

    url_lower = url.lower()

    if "twitter.com" in url_lower or "x.com" in url_lower:
        return "twitter"

    if "news.ycombinator.com" in url_lower:
        return "hackernews"

    # Academic/Research sources
    if "arxiv.org" in url_lower:
        return "arxiv"
    if "nature.com" in url_lower:
        return "nature"
    if "science.org" in url_lower:
        return "science"

    # Reddit: extract subreddit name
    match = re.search(r"reddit\.com/r/([^/]+)", url_lower)
    if match:
        return f"reddit/{match.group(1)}"

    return "other"


def calculate_priority_score(
    url: str,
    content: Optional[str],
    author: Optional[str],
    source_type: Optional[str] = None,
) -> int:
    """Calculate content priority score (lower = higher priority).

    The algorithm considers:
    1. Source quality tier (based on historical AI scores)
    2. Author reputation (Twitter high-value authors)
    3. Content length (penalize very short content)

    Args:
        url: Content URL
        content: Content text (for length analysis)
        author: Content author (for Twitter author bonus)
        source_type: Optional source type string (e.g., "rss") as fallback

    Returns:
        Priority score, typically 0-70 range. Lower = higher priority.
    """
    score = 0
    content_len = len(content) if content else 0

    # 1. Source tier score
    source_key = extract_source_key(url)

    # Fallback to source_type for "other" sources (e.g., RSS feeds)
    if source_key == "other" and source_type:
        source_key = source_type.lower()

    source_score = SOURCE_TIERS.get(source_key, DEFAULT_SOURCE_PRIORITY)
    score += source_score

    # 2. Twitter author bonus (only for Twitter sources)
    if source_key == "twitter" and author:
        # Normalize author name to @username format
        author_normalized = author.lower().strip()
        if not author_normalized.startswith("@"):
            author_normalized = f"@{author_normalized}"
        author_bonus = HIGH_VALUE_AUTHORS.get(author_normalized, 0)
        score += author_bonus

    # 3. Length penalty
    # Extremely short content (all sources) - likely low quality
    if content_len < 50:
        score += 20
    else:
        # Source-specific length penalty
        base_source = source_key.split("/")[0]  # twitter, reddit, hackernews, arxiv
        config = SOURCE_LENGTH_CONFIG.get(base_source)

        if config and config.get("penalty_below"):
            if content_len < config["penalty_below"]:
                score += 10

    return score


def rank_content_items(items: list, limit: Optional[int] = None) -> list:
    """Sort content items by priority score.

    Args:
        items: List of ContentItem objects
        limit: Optional limit on returned items

    Returns:
        Sorted list of ContentItem objects (highest priority first)
    """
    # Calculate priority scores and attach to items
    scored_items = []
    for item in items:
        # Get source_type as fallback for RSS items
        source_type = None
        if hasattr(item, "source_type") and item.source_type:
            source_type = item.source_type.value if hasattr(item.source_type, "value") else str(item.source_type)

        priority_score = calculate_priority_score(
            item.url or "",
            item.content,
            item.author,
            source_type,
        )
        scored_items.append((priority_score, item))

    # Sort by priority score (ascending), then by published_at (descending)
    scored_items.sort(
        key=lambda x: (
            x[0],  # priority score (lower = higher priority)
            -(x[1].published_at.timestamp() if x[1].published_at else 0),  # newer first
        )
    )

    # Extract items
    result = [item for _, item in scored_items]

    if limit:
        return result[:limit]
    return result

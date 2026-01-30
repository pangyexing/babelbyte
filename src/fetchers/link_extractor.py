"""Link content extractor for fetching external URL content."""

import html
import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Result of link content extraction."""

    content: str
    success: bool = True
    error_message: Optional[str] = None


class LinkExtractor:
    """Extracts main content from external URLs."""

    # Domains to skip (videos, images, etc.)
    SKIP_DOMAINS = {
        "youtube.com",
        "youtu.be",
        "vimeo.com",
        "twitch.tv",
        "imgur.com",
        "i.imgur.com",
        "gfycat.com",
        "reddit.com",
        "i.redd.it",
        "v.redd.it",
        "preview.redd.it",
        "twitter.com",
        "x.com",
    }

    # File extensions to skip
    SKIP_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".webm", ".mov", ".pdf"}

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self, timeout: float = 15.0, max_content_length: int = 5000):
        self.timeout = timeout
        self.max_content_length = max_content_length

    def should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped (video, image, etc.)."""
        if not url:
            return True

        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")

        # Check domain
        if domain in self.SKIP_DOMAINS:
            return True

        # Check file extension
        path_lower = parsed.path.lower()
        for ext in self.SKIP_EXTENSIONS:
            if path_lower.endswith(ext):
                return True

        return False

    async def extract(self, url: str) -> ExtractedContent:
        """
        Extract main content from a URL.

        Args:
            url: The URL to extract content from.

        Returns:
            ExtractedContent with the extracted text.
        """
        if self.should_skip_url(url):
            return ExtractedContent(
                content="",
                success=False,
                error_message=f"Skipped URL type: {url}",
            )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": self.USER_AGENT},
                    follow_redirects=True,
                )
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type:
                    return ExtractedContent(
                        content="",
                        success=False,
                        error_message=f"Not HTML content: {content_type}",
                    )

                html_content = response.text
                extracted = self._extract_text(html_content)

                if not extracted:
                    return ExtractedContent(
                        content="",
                        success=False,
                        error_message="Failed to extract content",
                    )

                return ExtractedContent(
                    content=extracted[: self.max_content_length],
                    success=True,
                )

        except httpx.TimeoutException:
            return ExtractedContent(
                content="",
                success=False,
                error_message=f"Timeout fetching URL: {url}",
            )
        except httpx.HTTPStatusError as e:
            return ExtractedContent(
                content="",
                success=False,
                error_message=f"HTTP error {e.response.status_code}",
            )
        except Exception as e:
            return ExtractedContent(
                content="",
                success=False,
                error_message=f"Error: {str(e)}",
            )

    def _extract_text(self, html_content: str) -> str:
        """Extract main text content from HTML."""
        # Try trafilatura first (best quality)
        try:
            import trafilatura

            extracted = trafilatura.extract(html_content, include_comments=False)
            if extracted:
                return extracted
        except ImportError:
            pass

        # Fallback: simple extraction
        return self._simple_extract(html_content)

    def _simple_extract(self, html_content: str) -> str:
        """Simple HTML text extraction fallback."""
        # Remove script and style elements
        html_content = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.I)
        html_content = re.sub(r"<style[^>]*>.*?</style>", "", html_content, flags=re.DOTALL | re.I)
        html_content = re.sub(r"<nav[^>]*>.*?</nav>", "", html_content, flags=re.DOTALL | re.I)
        html_content = re.sub(r"<header[^>]*>.*?</header>", "", html_content, flags=re.DOTALL | re.I)
        html_content = re.sub(r"<footer[^>]*>.*?</footer>", "", html_content, flags=re.DOTALL | re.I)

        # Try to find article or main content
        article_match = re.search(r"<article[^>]*>(.*?)</article>", html_content, re.DOTALL | re.I)
        if article_match:
            html_content = article_match.group(1)
        else:
            main_match = re.search(r"<main[^>]*>(.*?)</main>", html_content, re.DOTALL | re.I)
            if main_match:
                html_content = main_match.group(1)

        # Decode HTML entities
        text = html.unescape(html_content)

        # Remove remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

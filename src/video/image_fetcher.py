"""Image fetcher for video backgrounds.

Fetches images from content URLs (Open Graph images) or provides
default background images for video generation.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Default backgrounds by category
DEFAULT_BACKGROUNDS = {
    "AI": (30, 60, 120),  # Deep blue
    "技术": (40, 80, 100),  # Teal
    "金融": (80, 60, 40),  # Brown/gold
    "创业": (60, 40, 80),  # Purple
    "科技": (40, 60, 80),  # Steel blue
    "default": (30, 30, 50),  # Dark purple-gray
}

# Cache directory for downloaded images
CACHE_DIR = Path("./videos/cache/images")


class ImageFetcher:
    """Fetches and caches images for video backgrounds."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; BabelByte/1.0)"
        })

    def _get_cache_path(self, url: str) -> Path:
        """Get cache path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        return self.cache_dir / f"{url_hash}.jpg"

    def fetch_og_image(self, url: str, timeout: int = 10) -> Optional[str]:
        """Fetch Open Graph image URL from a webpage.

        Args:
            url: The webpage URL to fetch OG image from.
            timeout: Request timeout in seconds.

        Returns:
            The OG image URL if found, None otherwise.
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            # Look for og:image meta tag
            og_patterns = [
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
                r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
            ]

            for pattern in og_patterns:
                match = re.search(pattern, response.text, re.IGNORECASE)
                if match:
                    image_url = match.group(1)
                    # Handle relative URLs
                    if not image_url.startswith(("http://", "https://")):
                        image_url = urljoin(url, image_url)
                    return image_url

            return None

        except Exception as e:
            logger.debug(f"Failed to fetch OG image from {url}: {e}")
            return None

    def download_image(self, image_url: str, timeout: int = 15) -> Optional[Path]:
        """Download an image and cache it.

        Args:
            image_url: URL of the image to download.
            timeout: Request timeout in seconds.

        Returns:
            Path to the cached image, or None if download failed.
        """
        cache_path = self._get_cache_path(image_url)

        # Check cache first
        if cache_path.exists():
            return cache_path

        try:
            response = self.session.get(image_url, timeout=timeout, stream=True)
            response.raise_for_status()

            # Verify it's an image
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                logger.debug(f"Not an image: {content_type}")
                return None

            # Save to cache
            with open(cache_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify it's a valid image
            try:
                with Image.open(cache_path) as img:
                    img.verify()
                return cache_path
            except Exception:
                cache_path.unlink(missing_ok=True)
                return None

        except Exception as e:
            logger.debug(f"Failed to download image from {image_url}: {e}")
            return None

    def get_background_for_content(
        self,
        url: Optional[str] = None,
        category: Optional[str] = None,
        width: int = 1080,
        height: int = 1920,
    ) -> Image.Image:
        """Get a background image for content.

        Tries to fetch OG image from URL, falls back to gradient background.

        Args:
            url: Content URL to fetch OG image from.
            category: Content category for default color.
            width: Target width.
            height: Target height.

        Returns:
            PIL Image object for background.
        """
        bg_image = None

        # Try to fetch from URL
        if url:
            og_url = self.fetch_og_image(url)
            if og_url:
                image_path = self.download_image(og_url)
                if image_path:
                    try:
                        bg_image = Image.open(image_path)
                        bg_image = self._prepare_background(bg_image, width, height)
                    except Exception as e:
                        logger.debug(f"Failed to open image: {e}")
                        bg_image = None

        # Fall back to gradient background
        if bg_image is None:
            bg_image = self._create_gradient_background(width, height, category)

        return bg_image

    def _prepare_background(
        self,
        image: Image.Image,
        width: int,
        height: int,
    ) -> Image.Image:
        """Prepare an image for use as background.

        Resizes/crops to target dimensions and applies darkening overlay.

        Args:
            image: Source image.
            width: Target width.
            height: Target height.

        Returns:
            Prepared background image.
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Calculate scaling to cover the target dimensions
        img_ratio = image.width / image.height
        target_ratio = width / height

        if img_ratio > target_ratio:
            # Image is wider, scale by height
            new_height = height
            new_width = int(height * img_ratio)
        else:
            # Image is taller, scale by width
            new_width = width
            new_height = int(width / img_ratio)

        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Crop to center
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        image = image.crop((left, top, left + width, top + height))

        # Apply darkening overlay for better text readability
        image = self._apply_dark_overlay(image, opacity=0.6)

        return image

    def _apply_dark_overlay(
        self,
        image: Image.Image,
        opacity: float = 0.5,
    ) -> Image.Image:
        """Apply a dark overlay to an image.

        Args:
            image: Source image.
            opacity: Overlay opacity (0-1).

        Returns:
            Image with dark overlay.
        """
        # Create dark overlay
        overlay = Image.new("RGB", image.size, (0, 0, 0))

        # Blend
        return Image.blend(image, overlay, opacity)

    def _create_gradient_background(
        self,
        width: int,
        height: int,
        category: Optional[str] = None,
    ) -> Image.Image:
        """Create a gradient background using numpy vectorization.

        Args:
            width: Image width.
            height: Image height.
            category: Category for color selection.

        Returns:
            Gradient background image.
        """
        base_color = np.array(
            DEFAULT_BACKGROUNDS.get(category, DEFAULT_BACKGROUNDS["default"]),
            dtype=np.float64,
        )

        # Brightness curve: darker at top and bottom, lighter in middle
        ratios = np.linspace(0, 1, height)
        brightness = 1.0 - 0.3 * np.abs(ratios - 0.4)

        # Shape (height, 1, 3) * (height, 1, 1) -> broadcast to (height, width, 3)
        arr = (base_color.reshape(1, 1, 3) * brightness.reshape(-1, 1, 1)).astype(np.uint8)
        arr = np.broadcast_to(arr, (height, width, 3)).copy()

        return Image.fromarray(arr, "RGB")


# Singleton instance
_fetcher: Optional[ImageFetcher] = None


def get_image_fetcher() -> ImageFetcher:
    """Get the singleton ImageFetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = ImageFetcher()
    return _fetcher

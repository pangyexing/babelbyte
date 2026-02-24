"""Douyin cover image generator.

Generates 1080x1920 cover images with 3-text-line layout:
category tag, headline, and hook text on a FLUX dark background.
"""

import re
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


class DouyinCoverGenerator:
    """Generate Douyin 3-text-line cover images.

    Layout on FLUX dark background:
    - Top: category pill badge
    - Center: headline in large stroked text (max 2 lines)
    - Bottom: hook text in accent color
    """

    # Category accent colors (matches BulletinTemplate)
    CATEGORY_COLORS = {
        "AI": (100, 120, 255),
        "科技": (100, 120, 255),
        "金融": (255, 180, 60),
        "创业": (80, 200, 160),
        "创新": (80, 200, 160),
        "技术": (140, 100, 255),
        "default": (140, 140, 255),
    }

    TEXT_WHITE = (255, 255, 255)
    COMIC_BLACK = (20, 20, 20)

    def __init__(self, width: int = 1080, height: int = 1920):
        self.width = width
        self.height = height
        self._font_cache = {}

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font with caching."""
        if size in self._font_cache:
            return self._font_cache[size]

        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "C:/Windows/Fonts/msyh.ttc",
        ]

        for path in font_paths:
            try:
                font = ImageFont.truetype(path, size)
                self._font_cache[size] = font
                return font
            except (OSError, IOError):
                continue

        font = ImageFont.load_default()
        self._font_cache[size] = font
        return font

    def _get_accent_color(self, category: str) -> tuple:
        """Get accent color for category."""
        return self.CATEGORY_COLORS.get(category, self.CATEGORY_COLORS["default"])

    # CJK punctuation that must not start a new line
    _NO_BREAK_BEFORE = set("。，！？；：、）》」】…—·～")

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """Wrap text respecting English word boundaries and CJK punctuation rules."""
        tokens = re.findall(r"[a-zA-Z0-9]+(?:[.\-_][a-zA-Z0-9]+)*|.", text)

        lines: list[str] = []
        current = ""
        for token in tokens:
            test = current + token
            if font.getbbox(test)[2] <= max_width:
                current = test
            elif not current:
                for ch in token:
                    test = current + ch
                    if font.getbbox(test)[2] <= max_width:
                        current = test
                    else:
                        if current:
                            lines.append(current)
                        current = ch
            elif token in self._NO_BREAK_BEFORE:
                lines.append(current + token)
                current = ""
            else:
                lines.append(current)
                current = token

        if current:
            lines.append(current)

        return lines

    def _draw_stroked_text(
        self,
        draw: ImageDraw.ImageDraw,
        position: tuple,
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: tuple,
        stroke_color: tuple = (20, 20, 20),
        stroke_width: int = 3,
    ):
        """Draw text with 8-directional stroke outline."""
        x, y = position
        for dx, dy in [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ]:
            draw.text(
                (x + dx * stroke_width, y + dy * stroke_width),
                text, font=font, fill=stroke_color,
            )
        draw.text((x, y), text, font=font, fill=fill)

    def generate(
        self,
        category: str,
        headline: str,
        hook: str,
        background: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Generate a Douyin cover image.

        Args:
            category: Event category for pill badge.
            headline: Main title (max 2 lines, 100px).
            hook: Hook text below headline (60px, accent color).
            background: Optional background image. Uses dark gradient if None.

        Returns:
            1080x1920 cover image.
        """
        accent = self._get_accent_color(category)

        # Background
        if background is not None:
            img = background.copy()
            if img.size != (self.width, self.height):
                img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        else:
            img = self._create_dark_gradient()

        draw = ImageDraw.Draw(img)

        caption_font = self._get_font(36)
        headline_font = self._get_font(100)
        hook_font = self._get_font(60)

        padding = 60

        # 1. Category pill badge at top
        tag_text = f"#{category}"
        tag_bbox = caption_font.getbbox(tag_text)
        tag_padding = 12
        tag_height = tag_bbox[3] + tag_padding * 2
        tag_x = (self.width - tag_bbox[2] - tag_padding * 2) // 2
        tag_y = 400

        draw.rounded_rectangle(
            [tag_x, tag_y, tag_x + tag_bbox[2] + tag_padding * 2, tag_y + tag_height],
            radius=tag_height // 2,
            fill=accent,
        )
        draw.text(
            (tag_x + tag_padding, tag_y + tag_padding - 2),
            tag_text, font=caption_font, fill=self.COMIC_BLACK,
        )

        # 2. Headline — large centered text
        max_text_width = self.width - 2 * padding - 40
        headline_lines = self._wrap_text(headline, headline_font, max_text_width)
        line_h = int(100 * 1.4)
        headline_y = 750

        for line in headline_lines[:2]:
            line_bbox = headline_font.getbbox(line)
            line_x = (self.width - line_bbox[2]) // 2
            self._draw_stroked_text(
                draw, (line_x, headline_y), line, headline_font,
                fill=self.TEXT_WHITE,
                stroke_color=self.COMIC_BLACK,
                stroke_width=4,
            )
            headline_y += line_h

        # 3. Hook text — accent color below headline
        hook_lines = self._wrap_text(hook, hook_font, max_text_width)
        hook_y = 1200

        for line in hook_lines[:2]:
            line_bbox = hook_font.getbbox(line)
            line_x = (self.width - line_bbox[2]) // 2
            draw.text((line_x, hook_y), line, font=hook_font, fill=accent)
            hook_y += int(60 * 1.4)

        # Brand at bottom
        brand_font = self._get_font(32)
        brand_text = "巴别情报站"
        brand_bbox = brand_font.getbbox(brand_text)
        brand_x = (self.width - brand_bbox[2]) // 2
        brand_y = self.height - 200
        draw.text((brand_x, brand_y), brand_text, font=brand_font, fill=(150, 155, 170))

        return img

    def _create_dark_gradient(self) -> Image.Image:
        """Create a dark gradient background as fallback."""
        import numpy as np

        top = np.array([25, 25, 45], dtype=np.float64)
        bottom = np.array([10, 10, 20], dtype=np.float64)

        ratios = np.linspace(0, 1, self.height).reshape(-1, 1, 1)
        arr = (top * (1 - ratios) + bottom * ratios).astype(np.uint8)
        arr = np.broadcast_to(arr, (self.height, self.width, 3)).copy()

        return Image.fromarray(arr, "RGB")

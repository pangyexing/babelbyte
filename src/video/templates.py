"""Video templates for different content types."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


class TemplateType(Enum):
    """Available video template types."""

    NEWS_BRIEF = "news_brief"  # 新闻速报 (15-30秒)
    KEY_POINTS = "key_points"  # 要点列表 (20-40秒)
    DEEP_ANALYSIS = "deep_analysis"  # 深度解读 (40-60秒)
    DATA_CARD = "data_card"  # 数据卡片 (15-30秒)
    BULLETIN = "bulletin"  # 新闻快报 - 抖音科技风格 (60-90秒)


@dataclass
class TemplateConfig:
    """Template configuration."""

    # Canvas size (9:16 for Douyin, 6:7 for 视频号)
    width: int = 1080
    height: int = 1920  # 9:16 default

    # Colors - More vibrant palette
    bg_color: tuple = (15, 15, 25)  # Deep dark blue-black
    bg_gradient_top: tuple = (25, 25, 45)  # Gradient top (slightly lighter)
    bg_gradient_bottom: tuple = (10, 10, 20)  # Gradient bottom (darker)
    text_color: tuple = (255, 255, 255)  # White text
    accent_color: tuple = (0, 220, 180)  # Bright cyan/teal accent
    accent_secondary: tuple = (100, 100, 255)  # Purple-blue for variety
    secondary_color: tuple = (200, 200, 210)  # Lighter gray for better readability
    highlight_color: tuple = (255, 120, 80)  # Warm orange for highlights
    card_bg_color: tuple = (30, 30, 50)  # Card background
    card_border_color: tuple = (60, 60, 100)  # Card border

    # Fonts - Larger for mobile viewing
    title_font_size: int = 72  # Was 56, now bigger
    body_font_size: int = 48  # Was 40, now bigger
    caption_font_size: int = 36  # Was 32, now bigger
    big_number_size: int = 120  # For data cards

    # Layout
    padding: int = 60
    content_padding: int = 40  # Inner padding for cards
    line_spacing: float = 1.4
    card_radius: int = 24  # Rounded corners for cards

    # Animation
    fps: int = 24
    transition_duration: float = 0.5


@dataclass
class SlideContent:
    """Content for a single slide/frame."""

    title: str = ""
    body: str = ""
    bullet_points: list[str] = field(default_factory=list)
    category: str = ""
    source: str = ""
    timestamp: str = ""
    importance: int = 0
    duration: float = 3.0  # seconds
    highlights: list[str] = field(default_factory=list)  # Keywords to highlight
    background_image: Optional[Image.Image] = None  # Custom background image
    content_url: str = ""  # URL for fetching OG image


class VideoTemplate:
    """Base class for video templates."""

    def __init__(self, config: Optional[TemplateConfig] = None):
        self.config = config or TemplateConfig()
        self._font_cache = {}

    def _create_gradient_background(self, category: Optional[str] = None) -> Image.Image:
        """Create a gradient background image using numpy vectorization."""
        cfg = self.config
        top = np.array(cfg.bg_gradient_top, dtype=np.float64)
        bottom = np.array(cfg.bg_gradient_bottom, dtype=np.float64)

        ratios = np.linspace(0, 1, cfg.height).reshape(-1, 1, 1)
        arr = (top * (1 - ratios) + bottom * ratios).astype(np.uint8)
        # Broadcast to full width
        arr = np.broadcast_to(arr, (cfg.height, cfg.width, 3)).copy()

        return Image.fromarray(arr, "RGB")

    def _get_background(self, slide: "SlideContent") -> Image.Image:
        """Get background image for a slide.

        Uses provided background image, fetches from URL, or creates gradient.

        Args:
            slide: SlideContent with optional background_image or content_url.

        Returns:
            Background image.
        """
        cfg = self.config

        # Use provided background image
        if slide.background_image is not None:
            bg = slide.background_image.copy()
            if bg.size != (cfg.width, cfg.height):
                bg = bg.resize((cfg.width, cfg.height), Image.Resampling.LANCZOS)
            return bg

        # Try to fetch from URL
        if slide.content_url:
            try:
                from src.video.image_fetcher import get_image_fetcher

                fetcher = get_image_fetcher()
                return fetcher.get_background_for_content(
                    url=slide.content_url,
                    category=slide.category,
                    width=cfg.width,
                    height=cfg.height,
                )
            except Exception:
                pass

        # Fall back to gradient
        return self._create_gradient_background(slide.category)

    def _draw_glass_card(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        coords: tuple,
        opacity: float = 0.7,
        tint: tuple = (20, 20, 35),
        blur_radius: int = 15,
    ) -> Image.Image:
        """Draw a frosted glass effect card with blur.

        Crops the card region, blurs it, overlays a semi-transparent tint,
        applies a rounded corner mask, then composites back.

        Args:
            img: Base image to draw on (modified in place and returned).
            draw: ImageDraw object (will be recreated after composite).
            coords: (x1, y1, x2, y2) card coordinates.
            opacity: Card tint opacity (0-1).
            tint: Tint color RGB.
            blur_radius: Gaussian blur radius.

        Returns:
            Modified image with glass card applied.
        """
        x1, y1, x2, y2 = coords
        radius = self.config.card_radius
        card_w = x2 - x1
        card_h = y2 - y1

        # Crop card region and blur
        card_region = img.crop((x1, y1, x2, y2))
        blurred = card_region.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Create tinted overlay
        alpha = int(255 * opacity)
        tint_overlay = Image.new("RGBA", (card_w, card_h), (*tint, alpha))
        blurred_rgba = blurred.convert("RGBA")
        blurred_with_tint = Image.alpha_composite(blurred_rgba, tint_overlay)

        # Create rounded corner mask
        mask = Image.new("L", (card_w, card_h), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle([0, 0, card_w, card_h], radius=radius, fill=255)

        # Paste back with mask
        img.paste(blurred_with_tint.convert("RGB"), (x1, y1), mask)
        return img

    def _draw_card(
        self,
        draw: ImageDraw.ImageDraw,
        img: Image.Image,
        coords: tuple,
        fill: Optional[tuple] = None,
        border: Optional[tuple] = None,
        border_width: int = 2,
    ):
        """Draw a card with rounded corners, optional fill and border."""
        cfg = self.config
        x1, y1, x2, y2 = coords
        radius = cfg.card_radius
        fill = fill or cfg.card_bg_color
        border = border or cfg.card_border_color

        # Draw filled rounded rectangle
        self._draw_rounded_rect(draw, coords, radius, fill)

        # Draw border (by drawing slightly larger rect behind)
        if border and border_width > 0:
            # Top border
            draw.rectangle([x1 + radius, y1, x2 - radius, y1 + border_width], fill=border)
            # Bottom border
            draw.rectangle([x1 + radius, y2 - border_width, x2 - radius, y2], fill=border)
            # Left border
            draw.rectangle([x1, y1 + radius, x1 + border_width, y2 - radius], fill=border)
            # Right border
            draw.rectangle([x2 - border_width, y1 + radius, x2, y2 - radius], fill=border)

    def _draw_accent_line(
        self,
        draw: ImageDraw.ImageDraw,
        y: int,
        width_ratio: float = 0.3,
        color: Optional[tuple] = None,
        thickness: int = 4,
    ):
        """Draw a decorative accent line."""
        cfg = self.config
        color = color or cfg.accent_color
        line_width = int(cfg.width * width_ratio)
        x_start = cfg.padding
        draw.rectangle(
            [x_start, y, x_start + line_width, y + thickness],
            fill=color,
        )

    def _get_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Get font with caching. Uses system fonts."""
        cache_key = (size, bold)
        if cache_key not in self._font_cache:
            # Try common Chinese fonts
            font_paths = [
                # Linux
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                # macOS
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                # Windows
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
            ]

            font = None
            for path in font_paths:
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except (OSError, IOError):
                    continue

            if font is None:
                # Fallback to default
                font = ImageFont.load_default()

            self._font_cache[cache_key] = font

        return self._font_cache[cache_key]

    # CJK punctuation that must not start a new line
    _NO_BREAK_BEFORE = set("。，！？；：、）》」】…—·～")

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """Wrap text respecting English word boundaries and CJK punctuation rules.

        - ASCII words/numbers are kept together (never split "DeepMind")
        - CJK punctuation (。，！etc.) is absorbed into the previous line
          rather than starting a new line
        """
        # Tokenize: ASCII word/number runs stay together, everything else is individual
        tokens = re.findall(r"[a-zA-Z0-9]+(?:[.\-_][a-zA-Z0-9]+)*|.", text)

        lines: list[str] = []
        current = ""
        for token in tokens:
            test = current + token
            if font.getbbox(test)[2] <= max_width:
                current = test
            elif not current:
                # Single token exceeds width — force char-by-char split
                for ch in token:
                    test = current + ch
                    if font.getbbox(test)[2] <= max_width:
                        current = test
                    else:
                        if current:
                            lines.append(current)
                        current = ch
            elif token in self._NO_BREAK_BEFORE:
                # Absorb punctuation into current line (slight overflow OK)
                lines.append(current + token)
                current = ""
            else:
                lines.append(current)
                current = token

        if current:
            lines.append(current)

        return lines

    def _draw_rounded_rect(
        self,
        draw: ImageDraw.ImageDraw,
        coords: tuple,
        radius: int,
        fill: tuple,
    ):
        """Draw a rounded rectangle."""
        x1, y1, x2, y2 = coords
        draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
        draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
        draw.ellipse([x1, y1, x1 + 2 * radius, y1 + 2 * radius], fill=fill)
        draw.ellipse([x2 - 2 * radius, y1, x2, y1 + 2 * radius], fill=fill)
        draw.ellipse([x1, y2 - 2 * radius, x1 + 2 * radius, y2], fill=fill)
        draw.ellipse([x2 - 2 * radius, y2 - 2 * radius, x2, y2], fill=fill)

    def _draw_text_with_highlights(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        position: tuple,
        font: ImageFont.FreeTypeFont,
        default_color: tuple,
        highlights: list[str],
        highlight_color: Optional[tuple] = None,
    ) -> int:
        """Draw text with keyword highlighting.

        Scans the text for keywords and draws them in a different color.
        Since PIL doesn't support rich text, this draws text segment by segment.

        Args:
            draw: PIL ImageDraw object.
            text: Text to draw.
            position: (x, y) position to start drawing.
            font: Font to use.
            default_color: Default text color.
            highlights: List of keywords to highlight.
            highlight_color: Color for highlighted keywords. Defaults to config.highlight_color.

        Returns:
            Total width of drawn text.
        """
        if highlight_color is None:
            highlight_color = self.config.highlight_color

        if not highlights:
            # No highlights, draw normally
            draw.text(position, text, font=font, fill=default_color)
            bbox = font.getbbox(text)
            return bbox[2] if bbox else 0

        x, y = position
        total_width = 0

        # Build a pattern to split text while preserving delimiters
        # Escape special regex characters in highlights
        import re

        escaped_highlights = [re.escape(h) for h in highlights if h]
        if not escaped_highlights:
            draw.text(position, text, font=font, fill=default_color)
            bbox = font.getbbox(text)
            return bbox[2] if bbox else 0

        # Create pattern that captures the highlights
        pattern = f"({'|'.join(escaped_highlights)})"

        # Split text, keeping the delimiters
        segments = re.split(pattern, text, flags=re.IGNORECASE)

        # Create lowercase set for matching
        highlights_lower = {h.lower() for h in highlights}

        for segment in segments:
            if not segment:
                continue

            # Check if this segment is a highlight
            is_highlight = segment.lower() in highlights_lower
            color = highlight_color if is_highlight else default_color

            # Draw segment
            draw.text((x, y), segment, font=font, fill=color)

            # Update x position
            bbox = font.getbbox(segment)
            segment_width = bbox[2] if bbox else 0
            x += segment_width
            total_width += segment_width

        return total_width

    def render_slide(self, slide: SlideContent) -> Image.Image:
        """Render a single slide. Override in subclasses."""
        raise NotImplementedError

    def render_slides(self, slides: list[SlideContent]) -> list[Image.Image]:
        """Render multiple slides."""
        return [self.render_slide(slide) for slide in slides]


class NewsBriefTemplate(VideoTemplate):
    """News brief template - card overlay on background image."""

    def render_slide(self, slide: SlideContent) -> Image.Image:
        """Render a news brief slide with background image and card overlay."""
        cfg = self.config

        # Get background (from image, URL, or gradient)
        img = self._get_background(slide)
        draw = ImageDraw.Draw(img)

        # Fonts
        title_font = self._get_font(cfg.title_font_size, bold=True)
        body_font = self._get_font(cfg.body_font_size)
        caption_font = self._get_font(cfg.caption_font_size)
        small_font = self._get_font(cfg.caption_font_size - 4)

        # Pre-calculate text dimensions
        title_lines = []
        if slide.title:
            title_lines = self._wrap_text(
                slide.title,
                title_font,
                cfg.width - 2 * cfg.padding - 60,
            )

        body_lines = []
        if slide.body:
            body_lines = self._wrap_text(
                slide.body,
                body_font,
                cfg.width - 2 * cfg.padding - 60,
            )

        # Calculate card dimensions
        card_padding = 30
        card_content_height = card_padding * 2  # Top and bottom padding

        if slide.category:
            card_content_height += 60  # Category tag height

        if title_lines:
            card_content_height += (
                len(title_lines[:3]) * int(cfg.title_font_size * cfg.line_spacing) + 30
            )

        if body_lines:
            card_content_height += len(body_lines[:5]) * int(cfg.body_font_size * cfg.line_spacing)

        # Position card in center-upper area
        card_margin = 40
        card_top = cfg.height // 2 - card_content_height // 2 - 100
        card_top = max(cfg.padding + 80, card_top)
        card_bottom = card_top + card_content_height
        card_left = card_margin
        card_right = cfg.width - card_margin

        # Draw semi-transparent card background
        card_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        card_draw = ImageDraw.Draw(card_overlay)

        # Main card (dark semi-transparent)
        card_draw.rounded_rectangle(
            [card_left, card_top, card_right, card_bottom],
            radius=cfg.card_radius,
            fill=(15, 15, 30, 200),  # Semi-transparent dark
        )

        # Left accent stripe
        card_draw.rectangle(
            [card_left, card_top + 20, card_left + 6, card_bottom - 20],
            fill=(*cfg.accent_color, 255),
        )

        # Composite card onto background
        img = Image.alpha_composite(img.convert("RGBA"), card_overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw content on card
        y_pos = card_top + card_padding

        # Category tag
        if slide.category:
            tag_text = f"#{slide.category}"
            tag_bbox = caption_font.getbbox(tag_text)
            tag_width = tag_bbox[2] + 30
            tag_height = tag_bbox[3] + 14

            self._draw_rounded_rect(
                draw,
                (card_left + 25, y_pos, card_left + 25 + tag_width, y_pos + tag_height),
                radius=tag_height // 2,
                fill=cfg.accent_color,
            )
            draw.text(
                (card_left + 40, y_pos + 4),
                tag_text,
                font=caption_font,
                fill=(15, 15, 30),
            )
            y_pos += tag_height + 20

        # Title
        if title_lines:
            for line in title_lines[:3]:
                draw.text(
                    (card_left + 25, y_pos),
                    line,
                    font=title_font,
                    fill=cfg.text_color,
                )
                y_pos += int(cfg.title_font_size * cfg.line_spacing)
            y_pos += 15

        # Body text with highlights
        if body_lines:
            for line in body_lines[:5]:
                if slide.highlights:
                    self._draw_text_with_highlights(
                        draw,
                        line,
                        (card_left + 25, y_pos),
                        body_font,
                        cfg.secondary_color,
                        slide.highlights,
                    )
                else:
                    draw.text(
                        (card_left + 25, y_pos),
                        line,
                        font=body_font,
                        fill=cfg.secondary_color,
                    )
                y_pos += int(cfg.body_font_size * cfg.line_spacing)

        # Bottom info bar (outside main card)
        bottom_bar_y = cfg.height - cfg.padding - 120

        # Draw bottom info card
        bottom_card = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bottom_draw = ImageDraw.Draw(bottom_card)
        bottom_draw.rounded_rectangle(
            [card_margin, bottom_bar_y - 20, cfg.width - card_margin, bottom_bar_y + 80],
            radius=16,
            fill=(15, 15, 30, 180),
        )
        img = Image.alpha_composite(img.convert("RGBA"), bottom_card).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Source info (left)
        if slide.source:
            draw.text(
                (card_margin + 20, bottom_bar_y),
                slide.source,
                font=caption_font,
                fill=cfg.accent_color,
            )
        if slide.timestamp:
            draw.text(
                (card_margin + 20, bottom_bar_y + 40),
                slide.timestamp,
                font=small_font,
                fill=cfg.secondary_color,
            )

        # Importance badge (right)
        if slide.importance > 0:
            badge_x = cfg.width - card_margin - 80
            badge_y = bottom_bar_y - 5

            color = cfg.highlight_color if slide.importance >= 7 else cfg.accent_color
            draw.ellipse(
                [badge_x, badge_y, badge_x + 70, badge_y + 70],
                fill=color,
            )

            badge_text = str(slide.importance)
            badge_font = self._get_font(cfg.title_font_size - 10)
            bbox = badge_font.getbbox(badge_text)
            text_x = badge_x + (70 - bbox[2]) // 2
            text_y = badge_y + (70 - bbox[3]) // 2 - 5
            draw.text((text_x, text_y), badge_text, font=badge_font, fill=(15, 15, 30))

        # Top accent line
        draw.rectangle(
            [cfg.padding, cfg.padding, cfg.padding + 120, cfg.padding + 6],
            fill=cfg.accent_color,
        )

        return img


class KeyPointsTemplate(VideoTemplate):
    """Key points template - bullet list with card-based design."""

    def render_slide(self, slide: SlideContent) -> Image.Image:
        """Render a key points slide with card-based bullet points."""
        cfg = self.config

        # Get background (from image, URL, or gradient)
        img = self._get_background(slide)
        draw = ImageDraw.Draw(img)

        title_font = self._get_font(cfg.title_font_size, bold=True)
        body_font = self._get_font(cfg.body_font_size)
        caption_font = self._get_font(cfg.caption_font_size)
        number_font = self._get_font(cfg.title_font_size + 10, bold=True)

        y_pos = cfg.padding + 60

        # Header background (semi-transparent)
        header_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        header_draw = ImageDraw.Draw(header_overlay)
        header_draw.rounded_rectangle(
            [cfg.padding - 10, y_pos - 20, cfg.width - cfg.padding + 10, y_pos + 100],
            radius=16,
            fill=(15, 15, 30, 180),
        )
        img = Image.alpha_composite(img.convert("RGBA"), header_overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Top accent
        self._draw_accent_line(draw, y_pos - 15, width_ratio=0.12, thickness=6)

        # Section header
        header_text = "核心要点"
        draw.text(
            (cfg.padding + 10, y_pos + 10),
            header_text,
            font=title_font,
            fill=cfg.text_color,
        )
        y_pos += int(cfg.title_font_size * cfg.line_spacing) + 50

        # Calculate card heights
        num_points = min(len(slide.bullet_points), 4)
        if num_points == 0:
            num_points = 1  # Placeholder

        available_height = cfg.height - y_pos - cfg.padding - 150  # Leave space for CTA
        card_height = min(180, available_height // num_points - 20)
        card_spacing = 25

        # Bullet points as semi-transparent cards
        point_colors = [
            cfg.accent_color,
            cfg.accent_secondary,
            cfg.highlight_color,
            (100, 200, 100),  # Green
        ]

        for i, point in enumerate(slide.bullet_points[:4]):
            card_y = y_pos + i * (card_height + card_spacing)
            point_color = point_colors[i % len(point_colors)]

            # Semi-transparent card background
            card_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            card_overlay_draw = ImageDraw.Draw(card_overlay)
            card_overlay_draw.rounded_rectangle(
                [cfg.padding, card_y, cfg.width - cfg.padding, card_y + card_height],
                radius=cfg.card_radius,
                fill=(20, 20, 40, 200),
            )
            img = Image.alpha_composite(img.convert("RGBA"), card_overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Left accent stripe
            draw.rectangle(
                [cfg.padding, card_y + 20, cfg.padding + 8, card_y + card_height - 20],
                fill=point_color,
            )

            # Number badge
            badge_size = 60
            badge_x = cfg.padding + 30
            badge_y = card_y + (card_height - badge_size) // 2
            draw.ellipse(
                [badge_x, badge_y, badge_x + badge_size, badge_y + badge_size],
                fill=point_color,
            )
            num_text = str(i + 1)
            num_bbox = number_font.getbbox(num_text)
            num_x = badge_x + (badge_size - num_bbox[2]) // 2
            num_y = badge_y + (badge_size - num_bbox[3]) // 2 - 5
            draw.text(
                (num_x, num_y),
                num_text,
                font=number_font,
                fill=cfg.bg_color,
            )

            # Point text
            text_x = cfg.padding + 110
            text_max_width = cfg.width - text_x - cfg.padding - 20
            point_lines = self._wrap_text(point, body_font, text_max_width)

            text_y = (
                card_y + (card_height - len(point_lines[:2]) * int(cfg.body_font_size * 1.3)) // 2
            )
            for line in point_lines[:2]:
                if slide.highlights:
                    self._draw_text_with_highlights(
                        draw,
                        line,
                        (text_x, text_y),
                        body_font,
                        cfg.text_color,
                        slide.highlights,
                    )
                else:
                    draw.text(
                        (text_x, text_y),
                        line,
                        font=body_font,
                        fill=cfg.text_color,
                    )
                text_y += int(cfg.body_font_size * 1.3)

        # CTA at bottom
        cta_y = cfg.height - cfg.padding - 80
        cta_text = "关注了解更多"
        bbox = caption_font.getbbox(cta_text)
        x = (cfg.width - bbox[2]) // 2

        # CTA pill background (semi-transparent then solid pill)
        cta_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        cta_draw = ImageDraw.Draw(cta_overlay)
        cta_draw.rounded_rectangle(
            [x - 40, cta_y - 15, x + bbox[2] + 40, cta_y + bbox[3] + 20],
            radius=30,
            fill=(*cfg.accent_color, 255),
        )
        img = Image.alpha_composite(img.convert("RGBA"), cta_overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        draw.text(
            (x, cta_y),
            cta_text,
            font=caption_font,
            fill=(15, 15, 30),
        )

        return img


class DeepAnalysisTemplate(VideoTemplate):
    """Deep analysis template - structured breakdown with visual sections."""

    def render_slide(self, slide: SlideContent) -> Image.Image:
        """Render a deep analysis slide with section cards."""
        cfg = self.config

        # Create gradient background
        img = self._create_gradient_background()
        draw = ImageDraw.Draw(img)

        title_font = self._get_font(cfg.title_font_size, bold=True)
        body_font = self._get_font(cfg.body_font_size)
        caption_font = self._get_font(cfg.caption_font_size)
        small_font = self._get_font(cfg.caption_font_size - 4)

        # Top accent
        self._draw_accent_line(draw, cfg.padding + 40, width_ratio=0.12, thickness=6)

        y_pos = cfg.padding + 80

        # Header with icon
        header_icon = "🔍"
        header_text = "深度解读"
        draw.text((cfg.padding, y_pos), header_icon, font=title_font, fill=cfg.accent_color)
        draw.text((cfg.padding + 80, y_pos), header_text, font=title_font, fill=cfg.text_color)
        y_pos += int(cfg.title_font_size * cfg.line_spacing) + 20

        # Title card
        if slide.title:
            title_card_top = y_pos
            title_lines = self._wrap_text(slide.title, body_font, cfg.width - 2 * cfg.padding - 60)
            title_card_height = (
                len(title_lines[:2]) * int(cfg.body_font_size * cfg.line_spacing) + 40
            )

            # Card with accent border on left
            self._draw_rounded_rect(
                draw,
                (
                    cfg.padding,
                    title_card_top,
                    cfg.width - cfg.padding,
                    title_card_top + title_card_height,
                ),
                radius=cfg.card_radius,
                fill=cfg.card_bg_color,
            )
            # Left accent
            draw.rectangle(
                [
                    cfg.padding,
                    title_card_top + 15,
                    cfg.padding + 6,
                    title_card_top + title_card_height - 15,
                ],
                fill=cfg.accent_color,
            )

            for line in title_lines[:2]:
                draw.text(
                    (cfg.padding + 30, y_pos + 15),
                    line,
                    font=body_font,
                    fill=cfg.text_color,
                )
                y_pos += int(cfg.body_font_size * cfg.line_spacing)

            y_pos = title_card_top + title_card_height + 30

        # Divider line
        self._draw_accent_line(draw, y_pos, width_ratio=0.7, color=cfg.accent_color, thickness=2)
        y_pos += 40

        # Analysis sections as 2x2 grid
        sections = [
            ("背景", "📋", cfg.accent_color),
            ("原因", "🔎", cfg.accent_secondary),
            ("影响", "📊", cfg.highlight_color),
            ("行动", "✅", (100, 200, 100)),
        ]

        grid_cols = 2
        grid_rows = 2
        card_margin = 15
        card_width = (cfg.width - 2 * cfg.padding - card_margin) // grid_cols
        available_height = cfg.height - y_pos - cfg.padding - 80
        card_height = (available_height - card_margin) // grid_rows

        for idx, (section_name, icon, color) in enumerate(sections):
            col = idx % grid_cols
            row = idx // grid_cols

            card_x = cfg.padding + col * (card_width + card_margin)
            card_y = y_pos + row * (card_height + card_margin)

            # Section card
            self._draw_rounded_rect(
                draw,
                (card_x, card_y, card_x + card_width, card_y + card_height),
                radius=cfg.card_radius,
                fill=cfg.card_bg_color,
            )

            # Top accent bar
            draw.rectangle(
                [card_x + 15, card_y, card_x + card_width - 15, card_y + 5],
                fill=color,
            )

            # Icon and header
            icon_y = card_y + 25
            draw.text((card_x + 20, icon_y), icon, font=body_font, fill=color)
            draw.text(
                (card_x + 75, icon_y + 5), section_name, font=caption_font, fill=cfg.text_color
            )

            # Content placeholder (in real use, this would come from impact_assessment)
            content_y = icon_y + 70
            placeholder_text = f"点击查看{section_name}分析"
            draw.text(
                (card_x + 20, content_y),
                placeholder_text,
                font=small_font,
                fill=cfg.secondary_color,
            )

        # Bottom accent line
        draw.rectangle(
            [
                cfg.padding,
                cfg.height - cfg.padding,
                cfg.width - cfg.padding,
                cfg.height - cfg.padding + 4,
            ],
            fill=cfg.accent_color,
        )

        return img


class DataCardTemplate(VideoTemplate):
    """Data card template - metrics focused with visual impact."""

    def render_slide(self, slide: SlideContent) -> Image.Image:
        """Render a data card slide with prominent metrics."""
        cfg = self.config

        # Create gradient background
        img = self._create_gradient_background()
        draw = ImageDraw.Draw(img)

        title_font = self._get_font(cfg.title_font_size, bold=True)
        big_number_font = self._get_font(cfg.big_number_size, bold=True)
        body_font = self._get_font(cfg.body_font_size)
        caption_font = self._get_font(cfg.caption_font_size)

        # Top accent
        self._draw_accent_line(draw, cfg.padding + 40, width_ratio=0.12, thickness=6)

        y_pos = cfg.padding + 80

        # Header with icon
        header_icon = "📊"
        header_text = "数据速览"
        draw.text((cfg.padding, y_pos), header_icon, font=title_font, fill=cfg.accent_color)
        draw.text((cfg.padding + 80, y_pos), header_text, font=title_font, fill=cfg.text_color)
        y_pos += int(cfg.title_font_size * cfg.line_spacing) + 30

        # Title in card
        if slide.title:
            title_card_top = y_pos
            title_lines = self._wrap_text(slide.title, body_font, cfg.width - 2 * cfg.padding - 60)

            # Title card
            title_card_height = (
                len(title_lines[:2]) * int(cfg.body_font_size * cfg.line_spacing) + 40
            )
            self._draw_rounded_rect(
                draw,
                (
                    cfg.padding,
                    title_card_top,
                    cfg.width - cfg.padding,
                    title_card_top + title_card_height,
                ),
                radius=cfg.card_radius,
                fill=cfg.card_bg_color,
            )

            for line in title_lines[:2]:
                draw.text(
                    (cfg.padding + 30, y_pos + 15),
                    line,
                    font=body_font,
                    fill=cfg.text_color,
                )
                y_pos += int(cfg.body_font_size * cfg.line_spacing)

            y_pos = title_card_top + title_card_height + 50

        # Central big number with glow effect (simulated)
        center_y = y_pos + 50

        # Number background circle (large)
        circle_radius = 150
        circle_x = cfg.width // 2
        circle_y = center_y + circle_radius

        # Outer glow (larger, dimmer circle)
        draw.ellipse(
            [
                circle_x - circle_radius - 30,
                circle_y - circle_radius - 30,
                circle_x + circle_radius + 30,
                circle_y + circle_radius + 30,
            ],
            fill=(cfg.accent_color[0] // 4, cfg.accent_color[1] // 4, cfg.accent_color[2] // 4),
        )

        # Main circle
        draw.ellipse(
            [
                circle_x - circle_radius,
                circle_y - circle_radius,
                circle_x + circle_radius,
                circle_y + circle_radius,
            ],
            fill=cfg.accent_color,
        )

        # Big number
        number_text = str(slide.importance) if slide.importance else "—"
        bbox = big_number_font.getbbox(number_text)
        num_x = circle_x - bbox[2] // 2
        num_y = circle_y - bbox[3] // 2 - 10
        draw.text((num_x, num_y), number_text, font=big_number_font, fill=cfg.bg_color)

        # Label below number
        label_y = circle_y + circle_radius + 30
        label_text = "重要性评分"
        label_bbox = body_font.getbbox(label_text)
        label_x = (cfg.width - label_bbox[2]) // 2
        draw.text((label_x, label_y), label_text, font=body_font, fill=cfg.secondary_color)

        # Key points as metric cards at bottom
        metrics_y = label_y + 80
        num_points = min(len(slide.bullet_points), 3)

        if num_points > 0:
            card_width = (cfg.width - 2 * cfg.padding - (num_points - 1) * 20) // num_points
            card_height = 120

            for i, point in enumerate(slide.bullet_points[:3]):
                card_x = cfg.padding + i * (card_width + 20)

                # Metric card
                self._draw_rounded_rect(
                    draw,
                    (card_x, metrics_y, card_x + card_width, metrics_y + card_height),
                    radius=16,
                    fill=cfg.card_bg_color,
                )

                # Top accent on card
                draw.rectangle(
                    [card_x + 20, metrics_y, card_x + card_width - 20, metrics_y + 4],
                    fill=cfg.accent_secondary if i % 2 else cfg.accent_color,
                )

                # Point text (truncated)
                point_text = point[:20] + "..." if len(point) > 20 else point
                point_lines = self._wrap_text(point_text, caption_font, card_width - 30)
                text_y = metrics_y + 25
                for line in point_lines[:2]:
                    draw.text(
                        (card_x + 15, text_y),
                        line,
                        font=caption_font,
                        fill=cfg.text_color,
                    )
                    text_y += int(cfg.caption_font_size * 1.2)

        # Bottom accent line
        draw.rectangle(
            [
                cfg.padding,
                cfg.height - cfg.padding,
                cfg.width - cfg.padding,
                cfg.height - cfg.padding + 4,
            ],
            fill=cfg.accent_color,
        )

        return img


class BulletinTemplate(VideoTemplate):
    """Bulletin template - Clean Tech style.

    Visual style: clean, minimal, tech-forward with AI-generated backgrounds.
    Uses category-based accent colors, frosted glass cards, and subtle shadows
    instead of neon glow effects. Numpy-vectorized rendering for performance.
    """

    # Category accent colors (restrained, used as highlights only)
    CATEGORY_COLORS = {
        "AI": (100, 120, 255),  # Blue
        "科技": (100, 120, 255),  # Blue
        "金融": (255, 180, 60),  # Amber
        "创业": (80, 200, 160),  # Teal
        "创新": (80, 200, 160),  # Teal
        "default": (140, 140, 255),  # Light purple
    }

    DARK_BG = (12, 12, 24)
    TEXT_WHITE = (255, 255, 255)
    TEXT_SECONDARY = (190, 195, 210)
    SHADOW_COLOR = (0, 0, 0)

    # Comic/manga style colors
    CARD_WHITE = (250, 250, 245)  # Cream-white comic paper
    COMIC_BLACK = (20, 20, 20)  # Ink black for outlines/borders
    TEXT_DARK = (30, 30, 40)  # Dark text on white card

    def __init__(self, config: Optional[TemplateConfig] = None, image_generator=None):
        super().__init__(config)
        self._image_generator = image_generator

    def _get_accent_color(self, category: str) -> tuple:
        """Get accent color for a category."""
        return self.CATEGORY_COLORS.get(category, self.CATEGORY_COLORS["default"])

    def _create_category_gradient(self, category: Optional[str] = None) -> Image.Image:
        """Create a dark gradient background tinted by category.

        Uses numpy vectorization instead of putpixel loops.
        """
        cfg = self.config
        accent = np.array(self._get_accent_color(category or "default"), dtype=np.float64)
        bg = np.array(self.DARK_BG, dtype=np.float64)

        # Vertical gradient: subtle category tint at top, pure dark at bottom
        ratios = np.linspace(0.12, 0.0, cfg.height).reshape(-1, 1, 1)
        arr = (bg * (1 - ratios) + accent * ratios).astype(np.uint8)
        arr = np.broadcast_to(arr, (cfg.height, cfg.width, 3)).copy()

        return Image.fromarray(arr, "RGB")

    def _get_ai_background(self, category: str, headline: str = "") -> Optional[Image.Image]:
        """Get an AI-generated background image if available.

        Falls back to None (caller should use gradient).
        """
        if self._image_generator is None:
            return None

        try:
            return self._image_generator.generate_background(
                category=category,
                headline=headline,
                width=self.config.width,
                height=self.config.height,
            )
        except Exception:
            return None

    def _get_bulletin_background(
        self, category: str = "default", headline: str = ""
    ) -> Image.Image:
        """Get background: AI-generated if available, otherwise category gradient."""
        bg = self._get_ai_background(category, headline)
        if bg is not None:
            return bg
        return self._create_category_gradient(category)

    def _draw_text_shadow(
        self,
        draw: ImageDraw.ImageDraw,
        position: tuple,
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: tuple,
        shadow_offset: int = 2,
    ):
        """Draw text with a subtle drop shadow for readability."""
        x, y = position
        # Shadow
        shadow = tuple(min(255, c // 4) for c in self.SHADOW_COLOR)
        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow)
        # Main text
        draw.text((x, y), text, font=font, fill=fill)

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
        """Draw text with comic-style outline/stroke.

        Draws text at 8 directional offsets in stroke_color, then the main text
        on top for bold outlined lettering.
        """
        x, y = position
        # 8 directional offsets: N, NE, E, SE, S, SW, W, NW
        for dx, dy in [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ]:
            draw.text(
                (x + dx * stroke_width, y + dy * stroke_width),
                text,
                font=font,
                fill=stroke_color,
            )
        draw.text((x, y), text, font=font, fill=fill)

    def _draw_halftone_overlay(
        self,
        img: Image.Image,
        coords: tuple,
        color: tuple,
        dot_radius: int = 3,
        spacing: int = 18,
        opacity: int = 40,
    ) -> Image.Image:
        """Draw a halftone dot pattern overlay for comic paper texture.

        Creates an RGBA overlay with a grid of small circles in the accent color,
        then composites it onto the image within the given bounds.

        Returns:
            Modified image with halftone overlay applied.
        """
        x1, y1, x2, y2 = coords

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)

        dot_color = (*color[:3], opacity)
        for row_y in range(y1 + spacing, y2, spacing):
            # Offset every other row for classic halftone stagger
            offset = spacing // 2 if ((row_y - y1) // spacing) % 2 else 0
            for col_x in range(x1 + spacing + offset, x2, spacing):
                ov_draw.ellipse(
                    [
                        col_x - dot_radius,
                        row_y - dot_radius,
                        col_x + dot_radius,
                        row_y + dot_radius,
                    ],
                    fill=dot_color,
                )

        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        return img

    def _draw_speech_bubble(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: tuple = (255, 255, 255),
        border_color: tuple = (20, 20, 20),
    ) -> tuple:
        """Draw a speech bubble with rounded body and triangular tail.

        Returns:
            (img, draw, new_y) tuple — image/draw are refreshed due to alpha composite.
        """
        text_lines = self._wrap_text(text, font, width - 50)
        line_h = int(font.getbbox("A")[3] * 1.4)
        body_h = len(text_lines[:3]) * line_h + 30
        tail_h = 20

        total_h = body_h + tail_h
        bubble_right = x + width

        # Build bubble on RGBA overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bub_draw = ImageDraw.Draw(overlay)

        # Body: rounded rectangle
        border_w = 3
        body_coords = (x, y, bubble_right, y + body_h)
        bub_draw.rounded_rectangle(
            body_coords,
            radius=18,
            fill=(*fill, 255),
            outline=(*border_color, 255),
            width=border_w,
        )

        # Triangular tail at bottom-left
        tail_x = x + 50
        tail_points = [
            (tail_x, y + body_h - 2),
            (tail_x + 30, y + body_h - 2),
            (tail_x + 5, y + body_h + tail_h),
        ]
        bub_draw.polygon(tail_points, fill=(*fill, 255), outline=(*border_color, 255))
        # Cover the outline along the body bottom where tail meets body
        bub_draw.line(
            [(tail_x + 2, y + body_h - 1), (tail_x + 28, y + body_h - 1)],
            fill=(*fill, 255),
            width=border_w + 1,
        )

        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw text inside bubble
        text_y = y + 15
        for line in text_lines[:3]:
            draw.text((x + 25, text_y), line, font=font, fill=self.TEXT_DARK)
            text_y += line_h

        return img, draw, y + total_h

    def _draw_speed_lines(
        self,
        draw: ImageDraw.ImageDraw,
        center_x: int,
        y_start: int,
        y_end: int,
        width: int,
        color: tuple,
        num_lines: int = 12,
    ):
        """Draw radiating speed/action lines behind headline for dramatic emphasis.

        Short lines emanate outward from center in semi-transparent accent color.
        """
        for i in range(num_lines):
            # Distribute lines on left and right sides
            side = -1 if i % 2 == 0 else 1
            # Vary vertical position
            frac = (i / max(num_lines - 1, 1)) * 0.8 + 0.1
            line_y = y_start + int(frac * (y_end - y_start))

            # Lines go from card edge outward
            inner_x = center_x + side * (width // 2 - 40)
            outer_x = center_x + side * (width // 2 + 10)

            draw.line(
                [(inner_x, line_y), (outer_x, line_y)],
                fill=color,
                width=2,
            )

    def _draw_starburst(
        self,
        draw: ImageDraw.ImageDraw,
        center_x: int,
        center_y: int,
        text: str,
        font: ImageFont.FreeTypeFont,
        color: tuple,
        text_color: tuple = (20, 20, 20),
    ):
        """Draw a starburst badge with text centered inside.

        12-point starburst polygon with alternating inner/outer radii.
        """
        import math

        # Size based on text
        text_bbox = font.getbbox(text)
        text_w = text_bbox[2]
        outer_r = max(text_w // 2 + 30, 60)
        inner_r = int(outer_r * 0.65)
        num_points = 12

        points = []
        for i in range(num_points * 2):
            angle = math.pi * i / num_points - math.pi / 2
            r = outer_r if i % 2 == 0 else inner_r
            px = center_x + int(r * math.cos(angle))
            py = center_y + int(r * math.sin(angle))
            points.append((px, py))

        draw.polygon(points, fill=color, outline=self.COMIC_BLACK)

        # Center text
        tx = center_x - text_w // 2
        ty = center_y - text_bbox[3] // 2
        draw.text((tx, ty), text, font=font, fill=text_color)

    def _create_comic_background(self, category: str = "default") -> Image.Image:
        """Create a comic-style background: white with subtle halftone.

        Used by all slides for visual consistency.
        """
        cfg = self.config
        accent = self._get_accent_color(category)

        # Light background with subtle gradient tint
        top = np.array(self.CARD_WHITE, dtype=np.float64)
        bottom = np.array((235, 235, 230), dtype=np.float64)  # Slightly darker at bottom
        ratios = np.linspace(0, 1, cfg.height).reshape(-1, 1, 1)
        arr = (top * (1 - ratios) + bottom * ratios).astype(np.uint8)
        arr = np.broadcast_to(arr, (cfg.height, cfg.width, 3)).copy()
        img = Image.fromarray(arr, "RGB")

        # Full-page halftone for comic texture
        img = self._draw_halftone_overlay(
            img, (0, 0, cfg.width, cfg.height), accent, dot_radius=2, spacing=24, opacity=25
        )
        return img

    def _get_flux_background(
        self,
        date_str: str = "",
    ) -> tuple[Image.Image, bool]:
        """Get FLUX AI-generated dark background or fall back to comic background.

        All slides (opening, event cards, summary, closing) share the same
        FLUX background keyed by date_str. Event cards are mostly covered by
        a glass card, so a unique background per card wastes GPU time.

        Args:
            date_str: Date string for prompt variant selection.

        Returns:
            (image, is_dark_bg) — is_dark_bg=True when FLUX background was used.
        """
        if self._image_generator is None:
            return self._create_comic_background(), False

        try:
            bg = self._image_generator.generate_opening_background(
                date_str=date_str,
                width=self.config.width,
                height=self.config.height,
            )
            if bg is not None:
                return bg, True
        except Exception:
            pass

        return self._create_comic_background(), False

    def render_opening_slide(self, date_str: str, event_count: int) -> Image.Image:
        """Render comic-style opening slide with title and date.

        Args:
            date_str: Date string (e.g., "2月5日").
            event_count: Number of events in bulletin.

        Returns:
            Opening slide image.
        """
        cfg = self.config
        accent = self._get_accent_color("default")
        img, is_dark = self._get_flux_background(date_str=date_str)
        draw = ImageDraw.Draw(img)

        # Fonts
        title_font = self._get_font(cfg.title_font_size + 30, bold=True)
        date_font = self._get_font(cfg.body_font_size)
        caption_font = self._get_font(cfg.caption_font_size)

        center_y = cfg.height // 2

        # Panel frame
        frame_margin = 30
        frame_color = self.TEXT_SECONDARY if is_dark else self.COMIC_BLACK
        frame_width = 3 if is_dark else 5
        draw.rounded_rectangle(
            [frame_margin, frame_margin, cfg.width - frame_margin, cfg.height - frame_margin],
            radius=cfg.card_radius,
            outline=frame_color,
            width=frame_width,
        )

        # Speed lines behind title
        speed_color = (255, 255, 255, 50) if is_dark else (
            (*accent, 100) if len(accent) == 3 else accent
        )
        self._draw_speed_lines(
            draw,
            center_x=cfg.width // 2,
            y_start=center_y - 120,
            y_end=center_y + 40,
            width=cfg.width - 200,
            color=speed_color,
            num_lines=16,
        )

        # Main title — stroked comic text
        title_text = "巴别情报站"
        title_bbox = title_font.getbbox(title_text)
        title_x = (cfg.width - title_bbox[2]) // 2
        title_y = center_y - 100

        if is_dark:
            self._draw_stroked_text(
                draw, (title_x, title_y), title_text, title_font,
                fill=self.TEXT_WHITE,
                stroke_color=self.COMIC_BLACK,
                stroke_width=3,
            )
        else:
            self._draw_stroked_text(
                draw, (title_x, title_y), title_text, title_font,
                fill=self.COMIC_BLACK,
                stroke_color=(255, 255, 255),
                stroke_width=3,
            )

        # Date and count pill
        info_text = f"{date_str} · {event_count}件要闻"
        info_bbox = date_font.getbbox(info_text)
        info_x = (cfg.width - info_bbox[2]) // 2
        info_y = title_y + title_bbox[3] + 50

        pill_pad = 20
        pill_outline = self.COMIC_BLACK if not is_dark else accent
        draw.rounded_rectangle(
            [
                info_x - pill_pad * 2,
                info_y - pill_pad,
                info_x + info_bbox[2] + pill_pad * 2,
                info_y + info_bbox[3] + pill_pad,
            ],
            radius=30,
            fill=accent,
            outline=pill_outline,
            width=2,
        )
        draw.text((info_x, info_y), info_text, font=date_font, fill=self.COMIC_BLACK)

        # Starburst decoration at bottom
        self._draw_starburst(
            draw,
            cfg.width // 2,
            cfg.height - 250,
            "每日情报",
            caption_font,
            accent,
            text_color=self.COMIC_BLACK,
        )

        return img

    def render_event_card(
        self,
        headline: str,
        summary: str,
        category: str,
        source_count: int,
        one_liner: str = "",
        impact: str = "",
        actions: list[str] = None,
        illustration: Optional[Image.Image] = None,
        date_str: str = "",
        segment_index: int = -1,
    ) -> Image.Image:
        """Render a comic/manga-style event card slide.

        Args:
            headline: Short headline (10 chars).
            summary: Event summary (50 chars).
            category: Event category.
            source_count: Number of source articles.
            one_liner: One sentence conclusion.
            impact: Impact assessment text.
            actions: List of actionable items.
            illustration: Optional AI-generated illustration to display inside the card.
            date_str: Date string for shared FLUX background.

        Returns:
            Event card image.
        """
        if actions is None:
            actions = []
        cfg = self.config
        accent = self._get_accent_color(category)

        # Background: shared FLUX dark (by date) or comic white
        img, is_dark = self._get_flux_background(date_str=date_str)
        draw = ImageDraw.Draw(img)

        # Fonts — compact sizes to prevent overflow
        headline_font = self._get_font(cfg.title_font_size + 10, bold=True)
        summary_font = self._get_font(cfg.body_font_size)
        fact_font = self._get_font(cfg.body_font_size - 4)
        caption_font = self._get_font(cfg.caption_font_size)

        # --- Card panel ---
        card_margin = 40
        card_top = cfg.padding + 40
        card_bottom = cfg.height - cfg.padding - 120
        card_coords = (card_margin, card_top, cfg.width - card_margin, card_bottom)
        content_x = card_margin + 35
        content_width = cfg.width - 2 * card_margin - 70
        max_y = card_bottom - 25  # Hard stop: never render past this

        # Determine text/card colors based on background
        if is_dark:
            text_body = self.TEXT_WHITE
            text_secondary = self.TEXT_SECONDARY
            border_color = self.TEXT_SECONDARY
            border_width = 3
        else:
            text_body = self.TEXT_DARK
            text_secondary = self.TEXT_DARK
            border_color = self.COMIC_BLACK
            border_width = 4

        if is_dark:
            # Glass card (frosted semi-transparent)
            img = self._draw_glass_card(
                img, draw, card_coords,
                opacity=0.65, tint=(20, 20, 35), blur_radius=15,
            )
            draw = ImageDraw.Draw(img)
            # Subtle border on glass card
            draw.rounded_rectangle(
                card_coords, radius=cfg.card_radius,
                outline=border_color, width=border_width,
            )
        else:
            # White/cream card fill
            draw.rounded_rectangle(
                card_coords, radius=cfg.card_radius,
                fill=self.CARD_WHITE,
            )
            # Thick black panel border (4px)
            draw.rounded_rectangle(
                card_coords, radius=cfg.card_radius,
                outline=border_color, width=border_width,
            )
            # Halftone dot pattern overlay for comic paper texture
            img = self._draw_halftone_overlay(img, card_coords, accent)
            draw = ImageDraw.Draw(img)

        # Progress indicator (Douyin 5-segment bar at top)
        if segment_index >= 0:
            self._draw_progress_indicator(
                draw, segment_index, accent=accent, is_dark=is_dark,
            )

        # --- Content inside card ---
        content_y = card_top + 30

        # Progress bar: thin accent line at top of card
        if is_dark:
            draw.rectangle(
                [card_margin + 10, card_top, card_margin + 10 + content_width, card_top + 4],
                fill=accent,
            )

        # Category tag: accent fill
        tag_text = f"#{category}"
        tag_bbox = caption_font.getbbox(tag_text)
        tag_padding = 10
        tag_height = tag_bbox[3] + tag_padding * 2
        tag_coords = (
            content_x,
            content_y,
            content_x + tag_bbox[2] + tag_padding * 2,
            content_y + tag_height,
        )
        tag_outline = accent if is_dark else self.COMIC_BLACK
        draw.rounded_rectangle(
            tag_coords,
            radius=tag_height // 2,
            fill=accent,
            outline=tag_outline,
            width=2,
        )
        draw.text(
            (content_x + tag_padding, content_y + tag_padding - 2),
            tag_text,
            font=caption_font,
            fill=self.COMIC_BLACK,
        )

        content_y += tag_height + 20

        # Speed lines behind headline area
        head_line_h = int((cfg.title_font_size + 10) * cfg.line_spacing)
        headline_lines = self._wrap_text(headline, headline_font, content_width)
        max_head_lines = min(3, len(headline_lines))
        headline_block_h = max_head_lines * head_line_h
        speed_color = (255, 255, 255, 50) if is_dark else (
            (*accent, 120) if len(accent) == 3 else accent
        )
        self._draw_speed_lines(
            draw,
            center_x=cfg.width // 2,
            y_start=content_y,
            y_end=content_y + headline_block_h,
            width=content_width,
            color=speed_color,
        )

        # Headline with comic stroke/outline
        if is_dark:
            h_fill, h_stroke = self.TEXT_WHITE, self.COMIC_BLACK
        else:
            h_fill, h_stroke = self.COMIC_BLACK, (255, 255, 255)
        for line in headline_lines[:3]:
            self._draw_stroked_text(
                draw, (content_x, content_y), line, headline_font,
                fill=h_fill, stroke_color=h_stroke, stroke_width=2,
            )
            content_y += head_line_h

        content_y += 15

        # Illustration with border (adaptive height — larger on dark bg)
        if illustration is not None:
            remaining = max_y - content_y
            if is_dark:
                min_after_illust = 160
                max_illust_h_cap = 420
                min_illust_h = 200
            else:
                min_after_illust = 200
                max_illust_h_cap = 260
                min_illust_h = 140
            max_illust_h = min(max_illust_h_cap, remaining - min_after_illust)
            if max_illust_h >= min_illust_h:
                illust_width = content_width
                illust_height = min(max_illust_h, illustration.size[1])

                illust_resized = illustration.resize(
                    (illust_width, illust_height), Image.Resampling.LANCZOS
                )

                # Rounded corner mask
                illust_radius = 10
                illust_mask = Image.new("L", (illust_width, illust_height), 0)
                ImageDraw.Draw(illust_mask).rounded_rectangle(
                    [0, 0, illust_width, illust_height], radius=illust_radius, fill=255
                )

                # Border behind the image
                border_w = 3
                bordered_w = illust_width + border_w * 2
                bordered_h = illust_height + border_w * 2
                illust_border_color = border_color if is_dark else self.COMIC_BLACK
                border_img = Image.new("RGB", (bordered_w, bordered_h), illust_border_color)
                border_img.paste(illust_resized.convert("RGB"), (border_w, border_w), illust_mask)

                border_mask = Image.new("L", (bordered_w, bordered_h), 0)
                ImageDraw.Draw(border_mask).rounded_rectangle(
                    [0, 0, bordered_w, bordered_h],
                    radius=illust_radius + border_w,
                    fill=255,
                )

                img.paste(border_img, (content_x, content_y), border_mask)
                draw = ImageDraw.Draw(img)

                content_y += bordered_h + 15

        # Summary (overflow-aware)
        if summary and content_y < max_y - 60:
            sum_line_h = int(cfg.body_font_size * cfg.line_spacing)
            remaining = max_y - content_y
            max_sum_lines = min(6, max(1, (remaining - 100) // sum_line_h))
            summary_lines = self._wrap_text(summary, summary_font, content_width)
            for line in summary_lines[:max_sum_lines]:
                draw.text(
                    (content_x, content_y), line,
                    font=summary_font, fill=text_body,
                )
                content_y += sum_line_h

        content_y += 10

        # One-liner in speech bubble (overflow-aware) — white bubble on both themes
        if one_liner and content_y < max_y - 80:
            img, draw, content_y = self._draw_speech_bubble(
                img, draw, content_x, content_y, content_width,
                one_liner, fact_font,
                fill=(255, 255, 255),
                border_color=border_color if is_dark else self.COMIC_BLACK,
            )
            content_y += 10

        # Impact assessment with left bar (overflow-aware)
        fact_line_h = int((cfg.body_font_size - 4) * 1.3)
        if impact and content_y < max_y - fact_line_h:
            impact_text = f"影响：{impact}"
            impact_lines = self._wrap_text(impact_text, fact_font, content_width - 18)
            remaining = max_y - content_y
            max_imp_lines = min(2, max(1, remaining // fact_line_h))
            block_h = min(len(impact_lines), max_imp_lines) * fact_line_h
            draw.rectangle(
                [content_x, content_y, content_x + 5, content_y + block_h],
                fill=accent,
            )
            for line in impact_lines[:max_imp_lines]:
                draw.text(
                    (content_x + 18, content_y), line,
                    font=fact_font, fill=text_body,
                )
                content_y += fact_line_h
            content_y += 8

        # Actions with left bar (overflow-aware)
        if actions and content_y < max_y - fact_line_h:
            draw.rectangle(
                [content_x, content_y, content_x + 5, content_y + fact_line_h],
                fill=accent,
            )
            draw.text(
                (content_x + 18, content_y), "建议:",
                font=fact_font, fill=accent,
            )
            content_y += fact_line_h + 3

            for action in actions[:2]:
                if content_y >= max_y - fact_line_h:
                    break
                prefix = "→ "
                draw.text(
                    (content_x + 18, content_y), prefix,
                    font=fact_font, fill=accent,
                )
                prefix_width = fact_font.getbbox(prefix)[2]
                action_lines = self._wrap_text(action, fact_font, content_width - 18 - prefix_width)
                for line in action_lines[:2]:
                    if content_y >= max_y:
                        break
                    draw.text(
                        (content_x + 18 + prefix_width, content_y), line,
                        font=fact_font, fill=text_body,
                    )
                    content_y += fact_line_h
                content_y += 3

        # Source badge: starburst below the card (only if source_count > 0)
        if source_count > 0:
            badge_text = f"综合{source_count}家报道"
            badge_center_x = cfg.width // 2
            badge_center_y = card_bottom + 60
            self._draw_starburst(
                draw,
                badge_center_x,
                badge_center_y,
                badge_text,
                caption_font,
                accent,
                text_color=self.COMIC_BLACK,
            )

        return img

    def render_data_highlight(
        self,
        number: str,
        label: str,
        description: str,
        date_str: str = "",
    ) -> Image.Image:
        """Render a comic-style data highlight slide with large number.

        Args:
            number: The big number to display (e.g., "100%", "$5B").
            label: Label for the number.
            description: Brief description.
            date_str: Date string for shared FLUX background.

        Returns:
            Data highlight image.
        """
        cfg = self.config
        accent = self._get_accent_color("金融")
        img, is_dark = self._get_flux_background(date_str=date_str)
        draw = ImageDraw.Draw(img)

        # Fonts
        number_font = self._get_font(150, bold=True)
        label_font = self._get_font(cfg.title_font_size, bold=True)
        desc_font = self._get_font(cfg.body_font_size)

        center_y = cfg.height // 2

        # Panel frame
        frame_margin = 30
        frame_color = self.TEXT_SECONDARY if is_dark else self.COMIC_BLACK
        frame_width = 3 if is_dark else 5
        draw.rounded_rectangle(
            [frame_margin, frame_margin, cfg.width - frame_margin, cfg.height - frame_margin],
            radius=cfg.card_radius,
            outline=frame_color,
            width=frame_width,
        )

        # Big number in starburst
        self._draw_starburst(
            draw,
            cfg.width // 2,
            center_y - 60,
            number,
            number_font,
            accent,
            text_color=self.COMIC_BLACK,
        )

        # Label — stroked text below
        label_bbox = label_font.getbbox(label)
        label_x = (cfg.width - label_bbox[2]) // 2
        label_y = center_y + 100
        if is_dark:
            self._draw_stroked_text(
                draw, (label_x, label_y), label, label_font,
                fill=self.TEXT_WHITE,
                stroke_color=self.COMIC_BLACK,
                stroke_width=2,
            )
        else:
            self._draw_stroked_text(
                draw, (label_x, label_y), label, label_font,
                fill=self.COMIC_BLACK,
                stroke_color=(255, 255, 255),
                stroke_width=2,
            )

        # Description
        desc_color = self.TEXT_WHITE if is_dark else self.TEXT_DARK
        if description:
            desc_lines = self._wrap_text(description, desc_font, cfg.width - 2 * cfg.padding - 100)
            desc_y = label_y + label_bbox[3] + 40
            for line in desc_lines[:2]:
                line_bbox = desc_font.getbbox(line)
                line_x = (cfg.width - line_bbox[2]) // 2
                draw.text((line_x, desc_y), line, font=desc_font, fill=desc_color)
                desc_y += int(cfg.body_font_size * cfg.line_spacing)

        return img

    def render_summary_slide(
        self,
        headlines: list[str],
        date_str: str = "",
    ) -> Image.Image:
        """Render comic-style summary slide with all headlines.

        Args:
            headlines: List of event headlines.
            date_str: Date string for FLUX background selection.

        Returns:
            Summary slide image.
        """
        cfg = self.config
        img, is_dark = self._get_flux_background(date_str=date_str)
        draw = ImageDraw.Draw(img)

        # Fonts
        title_font = self._get_font(cfg.title_font_size, bold=True)
        item_font = self._get_font(cfg.body_font_size + 4)
        badge_number_font = self._get_font(cfg.body_font_size + 6, bold=True)

        # Panel frame
        frame_margin = 30
        frame_color = self.TEXT_SECONDARY if is_dark else self.COMIC_BLACK
        frame_width = 3 if is_dark else 5
        draw.rounded_rectangle(
            [frame_margin, frame_margin, cfg.width - frame_margin, cfg.height - frame_margin],
            radius=cfg.card_radius,
            outline=frame_color,
            width=frame_width,
        )

        y_pos = cfg.padding + 80

        # Header — stroked comic text
        header_text = "今日要点回顾"
        header_bbox = title_font.getbbox(header_text)
        header_x = (cfg.width - header_bbox[2]) // 2

        if is_dark:
            self._draw_stroked_text(
                draw, (header_x, y_pos), header_text, title_font,
                fill=self.TEXT_WHITE,
                stroke_color=self.COMIC_BLACK,
                stroke_width=2,
            )
        else:
            self._draw_stroked_text(
                draw, (header_x, y_pos), header_text, title_font,
                fill=self.COMIC_BLACK,
                stroke_color=(255, 255, 255),
                stroke_width=2,
            )
        y_pos += header_bbox[3] + 50

        # Divider line
        line_width = 200
        line_x = (cfg.width - line_width) // 2
        divider_color = self.TEXT_SECONDARY if is_dark else self.COMIC_BLACK
        draw.rectangle([line_x, y_pos, line_x + line_width, y_pos + 4], fill=divider_color)
        y_pos += 50

        # Text color for headlines
        text_color = self.TEXT_WHITE if is_dark else self.TEXT_DARK

        # Headlines list in comic panel style
        for i, headline in enumerate(headlines[:5]):
            color_keys = ["AI", "金融", "创业", "default", "科技"]
            color = self._get_accent_color(color_keys[i % len(color_keys)])

            # Number badge
            badge_size = 48
            badge_x = cfg.padding + 50
            badge_outline = color if is_dark else self.COMIC_BLACK
            draw.ellipse(
                [badge_x, y_pos + 3, badge_x + badge_size, y_pos + 3 + badge_size],
                fill=color,
                outline=badge_outline,
                width=2,
            )

            # Number text centered in badge
            num_text = str(i + 1)
            num_bbox = badge_number_font.getbbox(num_text)
            num_width = num_bbox[2] - num_bbox[0]
            num_height = num_bbox[3] - num_bbox[1]
            num_x = badge_x + (badge_size - num_width) // 2 - num_bbox[0]
            num_y = y_pos + 3 + (badge_size - num_height) // 2 - num_bbox[1]
            draw.text((num_x, num_y), num_text, font=badge_number_font, fill=self.COMIC_BLACK)

            # Headline text
            text_x = badge_x + badge_size + 20
            text_max_width = cfg.width - text_x - cfg.padding - 40
            text_lines = self._wrap_text(headline, item_font, text_max_width)
            line_height = int((cfg.body_font_size + 4) * 1.3)

            if len(text_lines) <= 1:
                headline_bbox = item_font.getbbox(text_lines[0] if text_lines else "A")
                text_height = headline_bbox[3] - headline_bbox[1]
                text_y = y_pos + 3 + (badge_size - text_height) // 2 - headline_bbox[1]
                if text_lines:
                    draw.text(
                        (text_x, text_y), text_lines[0],
                        font=item_font, fill=text_color,
                    )
                y_pos += badge_size + 30
            else:
                text_y = y_pos + 5
                for line in text_lines[:2]:
                    draw.text((text_x, text_y), line, font=item_font, fill=text_color)
                    text_y += line_height
                y_pos += max(badge_size, line_height * min(len(text_lines), 2)) + 20

            # Divider line between items
            if i < len(headlines[:5]) - 1:
                sep_color = (100, 100, 120) if is_dark else (200, 200, 195)
                draw.line(
                    [(badge_x, y_pos - 10), (cfg.width - cfg.padding - 50, y_pos - 10)],
                    fill=sep_color,
                    width=1,
                )

        return img

    def render_closing_slide(self, date_str: str = "") -> Image.Image:
        """Render comic-style closing slide with CTA.

        Args:
            date_str: Date string for FLUX background selection.

        Returns:
            Closing slide image.
        """
        cfg = self.config
        accent = self._get_accent_color("default")
        img, is_dark = self._get_flux_background(date_str=date_str)
        draw = ImageDraw.Draw(img)

        # Fonts
        cta_font = self._get_font(cfg.title_font_size, bold=True)
        subtitle_font = self._get_font(cfg.body_font_size)
        caption_font = self._get_font(cfg.caption_font_size)

        center_y = cfg.height // 2

        # Panel frame
        frame_margin = 30
        frame_color = self.TEXT_SECONDARY if is_dark else self.COMIC_BLACK
        frame_width = 3 if is_dark else 5
        draw.rounded_rectangle(
            [frame_margin, frame_margin, cfg.width - frame_margin, cfg.height - frame_margin],
            radius=cfg.card_radius,
            outline=frame_color,
            width=frame_width,
        )

        # CTA in a speech bubble — white bubble on both themes
        cta_text = "关注了解更多"
        bubble_width = cfg.width - 200
        bubble_x = (cfg.width - bubble_width) // 2
        bubble_border = frame_color if is_dark else self.COMIC_BLACK
        img, draw, new_y = self._draw_speech_bubble(
            img, draw, bubble_x, center_y - 60, bubble_width,
            cta_text, cta_font,
            fill=(255, 255, 255),
            border_color=bubble_border,
        )

        # Subtitle below — stroked text
        subtitle_text = "巴别情报站 · 快人一步"
        subtitle_bbox = subtitle_font.getbbox(subtitle_text)
        subtitle_x = (cfg.width - subtitle_bbox[2]) // 2
        subtitle_y = new_y + 40

        if is_dark:
            self._draw_stroked_text(
                draw, (subtitle_x, subtitle_y), subtitle_text, subtitle_font,
                fill=self.TEXT_WHITE,
                stroke_color=self.COMIC_BLACK,
                stroke_width=2,
            )
        else:
            self._draw_stroked_text(
                draw, (subtitle_x, subtitle_y), subtitle_text, subtitle_font,
                fill=self.COMIC_BLACK,
                stroke_color=(255, 255, 255),
                stroke_width=2,
            )

        # Starburst decoration at top
        self._draw_starburst(
            draw,
            cfg.width // 2,
            cfg.padding + 200,
            "THE END",
            caption_font,
            accent,
            text_color=self.COMIC_BLACK,
        )

        return img

    def render_cover_slide(
        self,
        date_str: str,
        event_count: int,
        top_headline: str = "",
    ) -> Image.Image:
        """Render comic-style cover slide (thumbnail) for the video.

        Args:
            date_str: Date string (e.g., "2月5日").
            event_count: Number of events in bulletin.
            top_headline: Top headline to preview (optional).

        Returns:
            Cover slide image.
        """
        cfg = self.config
        accent = self._get_accent_color("default")
        img, is_dark = self._get_flux_background(date_str=date_str)
        draw = ImageDraw.Draw(img)

        # Fonts
        brand_font = self._get_font(cfg.title_font_size + 50, bold=True)
        count_font = self._get_font(cfg.big_number_size, bold=True)
        headline_font = self._get_font(cfg.body_font_size + 10, bold=True)
        date_font = self._get_font(cfg.body_font_size)

        center_x = cfg.width // 2

        # Panel frame
        frame_margin = 30
        frame_color = self.TEXT_SECONDARY if is_dark else self.COMIC_BLACK
        frame_width = 3 if is_dark else 5
        draw.rounded_rectangle(
            [frame_margin, frame_margin, cfg.width - frame_margin, cfg.height - frame_margin],
            radius=cfg.card_radius,
            outline=frame_color,
            width=frame_width,
        )

        # Brand name — stroked comic text
        brand_text = "巴别情报站"
        brand_bbox = brand_font.getbbox(brand_text)
        brand_x = (cfg.width - brand_bbox[2]) // 2
        brand_y = cfg.padding + 150

        if is_dark:
            self._draw_stroked_text(
                draw, (brand_x, brand_y), brand_text, brand_font,
                fill=self.TEXT_WHITE,
                stroke_color=self.COMIC_BLACK,
                stroke_width=3,
            )
        else:
            self._draw_stroked_text(
                draw, (brand_x, brand_y), brand_text, brand_font,
                fill=self.COMIC_BLACK,
                stroke_color=(255, 255, 255),
                stroke_width=3,
            )

        # Decorative line below brand
        line_width = 250
        line_y = brand_y + brand_bbox[3] + 30
        line_x = (cfg.width - line_width) // 2
        line_color = self.TEXT_SECONDARY if is_dark else self.COMIC_BLACK
        draw.rectangle([line_x, line_y, line_x + line_width, line_y + 4], fill=line_color)

        # Big event count in starburst
        count_y = cfg.height // 2 - 30
        count_label = f"{event_count}件要闻"
        self._draw_starburst(
            draw,
            center_x,
            count_y,
            count_label,
            count_font,
            accent,
            text_color=self.COMIC_BLACK,
        )

        # Date below starburst — accent pill on dark, black pill on light
        date_y = count_y + 120
        date_bbox = date_font.getbbox(date_str)
        date_x = center_x - date_bbox[2] // 2

        pill_pad = 15
        pill_fill = accent if is_dark else self.COMIC_BLACK
        pill_text = self.COMIC_BLACK if is_dark else self.CARD_WHITE
        draw.rounded_rectangle(
            [
                date_x - pill_pad * 2,
                date_y - pill_pad,
                date_x + date_bbox[2] + pill_pad * 2,
                date_y + date_bbox[3] + pill_pad,
            ],
            radius=25,
            fill=pill_fill,
        )
        draw.text((date_x, date_y), date_str, font=date_font, fill=pill_text)

        # Top headline preview at bottom in speech bubble style
        if top_headline:
            card_margin = 60
            bubble_y = cfg.height - cfg.padding - 220
            bubble_width = cfg.width - 2 * card_margin
            bubble_border = frame_color if is_dark else self.COMIC_BLACK
            img, draw, _ = self._draw_speech_bubble(
                img, draw, card_margin, bubble_y, bubble_width,
                top_headline, headline_font,
                fill=(255, 255, 255),
                border_color=bubble_border,
            )

        return img

    def _draw_progress_indicator(
        self,
        draw: ImageDraw.ImageDraw,
        segment_index: int,
        total_segments: int = 5,
        accent: tuple = (100, 120, 255),
        is_dark: bool = True,
    ):
        """Draw a 5-segment progress bar at the top of a slide.

        Current and previous segments use accent color, future segments are dim.

        Args:
            draw: PIL ImageDraw object.
            segment_index: Current segment (0-based).
            total_segments: Total number of segments.
            accent: Accent color for active segments.
            is_dark: Whether background is dark (affects dim color).
        """
        cfg = self.config
        margin = 30
        bar_y = margin + 12
        bar_height = 4
        gap = 6
        total_width = cfg.width - 2 * margin
        seg_width = (total_width - gap * (total_segments - 1)) // total_segments

        dim_color = (60, 60, 80) if is_dark else (200, 200, 195)

        for i in range(total_segments):
            x1 = margin + i * (seg_width + gap)
            x2 = x1 + seg_width
            color = accent if i <= segment_index else dim_color
            draw.rounded_rectangle(
                [x1, bar_y, x2, bar_y + bar_height],
                radius=bar_height // 2,
                fill=color,
            )

    def render_hook_slide(
        self,
        hook_text: str,
        category: str,
        date_str: str = "",
        segment_index: int = 0,
    ) -> Image.Image:
        """Render Douyin hook slide — large text on FLUX dark background.

        Args:
            hook_text: 3-second hook text (8-15 chars).
            category: Event category for accent color.
            date_str: Date string for FLUX background selection.
            segment_index: Current segment index for progress bar.

        Returns:
            Hook slide image.
        """
        cfg = self.config
        accent = self._get_accent_color(category)
        img, is_dark = self._get_flux_background(date_str=date_str)
        draw = ImageDraw.Draw(img)

        hook_font = self._get_font(100, bold=True)
        brand_font = self._get_font(36)
        caption_font = self._get_font(cfg.caption_font_size)

        # Progress indicator
        self._draw_progress_indicator(
            draw, segment_index, accent=accent, is_dark=is_dark,
        )

        center_y = cfg.height // 2

        # Panel frame
        frame_margin = 30
        frame_color = self.TEXT_SECONDARY if is_dark else self.COMIC_BLACK
        frame_width = 3 if is_dark else 5
        draw.rounded_rectangle(
            [frame_margin, frame_margin, cfg.width - frame_margin, cfg.height - frame_margin],
            radius=cfg.card_radius,
            outline=frame_color,
            width=frame_width,
        )

        # Category pill badge at top
        tag_text = f"#{category}"
        tag_bbox = caption_font.getbbox(tag_text)
        tag_padding = 10
        tag_height = tag_bbox[3] + tag_padding * 2
        tag_x = (cfg.width - tag_bbox[2] - tag_padding * 2) // 2
        draw.rounded_rectangle(
            [tag_x, cfg.padding + 80, tag_x + tag_bbox[2] + tag_padding * 2,
             cfg.padding + 80 + tag_height],
            radius=tag_height // 2,
            fill=accent,
            outline=accent,
            width=2,
        )
        draw.text(
            (tag_x + tag_padding, cfg.padding + 80 + tag_padding - 2),
            tag_text, font=caption_font, fill=self.COMIC_BLACK,
        )

        # Hook text — large centered with stroke
        hook_lines = self._wrap_text(hook_text, hook_font, cfg.width - 2 * cfg.padding - 60)
        line_h = int(100 * 1.4)
        total_h = len(hook_lines[:3]) * line_h
        text_y = center_y - total_h // 2

        h_fill = self.TEXT_WHITE if is_dark else self.COMIC_BLACK
        h_stroke = self.COMIC_BLACK if is_dark else (255, 255, 255)
        for line in hook_lines[:3]:
            line_bbox = hook_font.getbbox(line)
            line_x = (cfg.width - line_bbox[2]) // 2
            self._draw_stroked_text(
                draw, (line_x, text_y), line, hook_font,
                fill=h_fill, stroke_color=h_stroke, stroke_width=3,
            )
            text_y += line_h

        # Brand name at bottom
        brand_text = "巴别情报站"
        brand_bbox = brand_font.getbbox(brand_text)
        brand_x = (cfg.width - brand_bbox[2]) // 2
        brand_y = cfg.height - cfg.padding - 120
        brand_color = self.TEXT_SECONDARY if is_dark else self.TEXT_DARK
        draw.text((brand_x, brand_y), brand_text, font=brand_font, fill=brand_color)

        return img

    def render_impact_slide(
        self,
        impact: str,
        category: str,
        date_str: str = "",
        title: str = "影响分析",
        segment_index: int = -1,
    ) -> Image.Image:
        """Render Douyin impact/body slide with optional title.

        Args:
            impact: Body text to display.
            category: Event category.
            date_str: Date string for FLUX background selection.
            title: Card title (pass empty string to hide).

        Returns:
            Impact slide image.
        """
        cfg = self.config
        accent = self._get_accent_color(category)
        img, is_dark = self._get_flux_background(date_str=date_str)
        draw = ImageDraw.Draw(img)

        title_font = self._get_font(cfg.title_font_size, bold=True)
        body_font = self._get_font(cfg.body_font_size + 2)

        center_y = cfg.height // 2

        # Panel frame
        frame_margin = 30
        frame_color = self.TEXT_SECONDARY if is_dark else self.COMIC_BLACK
        frame_width = 3 if is_dark else 5
        draw.rounded_rectangle(
            [frame_margin, frame_margin, cfg.width - frame_margin, cfg.height - frame_margin],
            radius=cfg.card_radius,
            outline=frame_color,
            width=frame_width,
        )

        # Progress indicator (Douyin 5-segment bar)
        if segment_index >= 0:
            self._draw_progress_indicator(
                draw, segment_index, accent=accent, is_dark=is_dark,
            )

        # Glass card for content
        card_margin = 60
        card_top = center_y - 240
        card_bottom = center_y + 240
        card_coords = (card_margin, card_top, cfg.width - card_margin, card_bottom)

        if is_dark:
            img = self._draw_glass_card(
                img, draw, card_coords,
                opacity=0.65, tint=(20, 20, 35), blur_radius=15,
            )
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle(
                card_coords, radius=cfg.card_radius,
                outline=self.TEXT_SECONDARY, width=3,
            )
        else:
            draw.rounded_rectangle(
                card_coords, radius=cfg.card_radius,
                fill=self.CARD_WHITE,
                outline=self.COMIC_BLACK, width=4,
            )

        content_x = card_margin + 35
        content_width = cfg.width - 2 * card_margin - 70

        # Accent left bar
        draw.rectangle(
            [card_margin + 10, card_top + 25, card_margin + 15, card_bottom - 25],
            fill=accent,
        )

        # Title (optional)
        text_y = card_top + 40
        if title:
            draw.text(
                (content_x + 20, text_y), title, font=title_font, fill=accent,
            )
            text_y += int(cfg.title_font_size * 1.5)

        # Body text
        text_color = self.TEXT_WHITE if is_dark else self.TEXT_DARK
        impact_lines = self._wrap_text(impact, body_font, content_width - 30)
        line_h = int((cfg.body_font_size + 2) * 1.4)
        for line in impact_lines[:6]:
            draw.text((content_x + 20, text_y), line, font=body_font, fill=text_color)
            text_y += line_h

        return img

    def render_detail_slide(
        self,
        detail_text: str,
        category: str,
        date_str: str = "",
        segment_index: int = -1,
    ) -> Image.Image:
        """Render Douyin detail slide — open text on background, no card frame.

        Visually distinct from impact_slide (which uses a glass card).
        Creates a rhythm of: card → open → card in the 5-slide sequence.

        Args:
            detail_text: Key detail/data text to display.
            category: Event category.
            date_str: Date string for FLUX background selection.
            segment_index: Current segment index for progress bar.

        Returns:
            Detail slide image.
        """
        cfg = self.config
        accent = self._get_accent_color(category)
        img, is_dark = self._get_flux_background(date_str=date_str)
        draw = ImageDraw.Draw(img)

        body_font = self._get_font(cfg.body_font_size + 4)

        center_y = cfg.height // 2

        # Panel frame
        frame_margin = 30
        frame_color = self.TEXT_SECONDARY if is_dark else self.COMIC_BLACK
        frame_width = 3 if is_dark else 5
        draw.rounded_rectangle(
            [frame_margin, frame_margin, cfg.width - frame_margin, cfg.height - frame_margin],
            radius=cfg.card_radius,
            outline=frame_color,
            width=frame_width,
        )

        # Progress indicator
        if segment_index >= 0:
            self._draw_progress_indicator(
                draw, segment_index, accent=accent, is_dark=is_dark,
            )

        # Horizontal accent bar (distinct from impact slide's vertical bar)
        bar_width = 120
        bar_x = (cfg.width - bar_width) // 2
        bar_y = center_y - 200
        draw.rectangle(
            [bar_x, bar_y, bar_x + bar_width, bar_y + 5],
            fill=accent,
        )

        # Body text — directly on background with stroke, no card
        content_width = cfg.width - 2 * cfg.padding - 60
        detail_lines = self._wrap_text(detail_text, body_font, content_width)
        line_h = int((cfg.body_font_size + 4) * 1.4)
        total_h = len(detail_lines[:7]) * line_h
        text_y = center_y - total_h // 2

        h_fill = self.TEXT_WHITE if is_dark else self.COMIC_BLACK
        h_stroke = self.COMIC_BLACK if is_dark else (255, 255, 255)
        for line in detail_lines[:7]:
            line_bbox = body_font.getbbox(line)
            line_x = (cfg.width - line_bbox[2]) // 2
            self._draw_stroked_text(
                draw, (line_x, text_y), line, body_font,
                fill=h_fill, stroke_color=h_stroke, stroke_width=2,
            )
            text_y += line_h

        # Bottom accent bar (mirror of top)
        draw.rectangle(
            [bar_x, text_y + 30, bar_x + bar_width, text_y + 35],
            fill=accent,
        )

        return img

    def render_cta_slide(
        self,
        cta_text: str,
        date_str: str = "",
        segment_index: int = -1,
    ) -> Image.Image:
        """Render Douyin ending slide with summary + engagement prompt.

        Args:
            cta_text: Ending text with summary and question (30-50 chars).
            date_str: Date string for FLUX background selection.
            segment_index: Current segment index for progress bar.

        Returns:
            CTA slide image.
        """
        cfg = self.config
        accent = self._get_accent_color("default")
        img, is_dark = self._get_flux_background(date_str=date_str)
        draw = ImageDraw.Draw(img)

        cta_font = self._get_font(60, bold=True)
        guide_font = self._get_font(cfg.body_font_size)
        caption_font = self._get_font(cfg.caption_font_size)

        center_y = cfg.height // 2

        # Panel frame
        frame_margin = 30
        frame_color = self.TEXT_SECONDARY if is_dark else self.COMIC_BLACK
        frame_width = 3 if is_dark else 5
        draw.rounded_rectangle(
            [frame_margin, frame_margin, cfg.width - frame_margin, cfg.height - frame_margin],
            radius=cfg.card_radius,
            outline=frame_color,
            width=frame_width,
        )

        # Progress indicator
        if segment_index >= 0:
            self._draw_progress_indicator(
                draw, segment_index, accent=accent, is_dark=is_dark,
            )

        # Ending text — centered with stroke
        cta_lines = self._wrap_text(cta_text, cta_font, cfg.width - 2 * cfg.padding - 60)
        line_h = int(60 * 1.4)
        total_h = len(cta_lines[:5]) * line_h
        text_y = center_y - total_h // 2 - 60

        h_fill = self.TEXT_WHITE if is_dark else self.COMIC_BLACK
        h_stroke = self.COMIC_BLACK if is_dark else (255, 255, 255)
        for line in cta_lines[:5]:
            line_bbox = cta_font.getbbox(line)
            line_x = (cfg.width - line_bbox[2]) // 2
            self._draw_stroked_text(
                draw, (line_x, text_y), line, cta_font,
                fill=h_fill, stroke_color=h_stroke, stroke_width=2,
            )
            text_y += line_h

        # Follow guide
        guide_text = "关注 @巴别情报站"
        guide_bbox = guide_font.getbbox(guide_text)
        guide_x = (cfg.width - guide_bbox[2]) // 2
        guide_y = text_y + 60
        guide_color = self.TEXT_SECONDARY if is_dark else self.TEXT_DARK
        draw.text((guide_x, guide_y), guide_text, font=guide_font, fill=accent)

        # Engagement hints
        hint_y = guide_y + int(cfg.body_font_size * 1.6)
        hint_text = "点赞 + 评论 + 关注"
        hint_bbox = caption_font.getbbox(hint_text)
        hint_x = (cfg.width - hint_bbox[2]) // 2
        hint_color = self.TEXT_SECONDARY if is_dark else self.TEXT_DARK
        draw.text((hint_x, hint_y), hint_text, font=caption_font, fill=hint_color)

        return img

    def render_slide(self, slide: SlideContent) -> Image.Image:
        """Render a generic slide (fallback for base class compatibility).

        For bulletins, use the specific render methods instead.
        """
        return self.render_event_card(
            headline=slide.title[:10] if slide.title else "新闻",
            summary=slide.body[:50] if slide.body else "",
            category=slide.category or "资讯",
            source_count=1,
        )


def get_template(
    template_type: TemplateType, config: Optional[TemplateConfig] = None
) -> VideoTemplate:
    """Factory function to get template instance."""
    templates = {
        TemplateType.NEWS_BRIEF: NewsBriefTemplate,
        TemplateType.KEY_POINTS: KeyPointsTemplate,
        TemplateType.DEEP_ANALYSIS: DeepAnalysisTemplate,
        TemplateType.DATA_CARD: DataCardTemplate,
        TemplateType.BULLETIN: BulletinTemplate,
    }
    template_class = templates.get(template_type, NewsBriefTemplate)
    return template_class(config)

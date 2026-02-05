"""Video templates for different content types."""

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

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """Wrap text to fit within max_width."""
        lines = []
        current_line = ""

        for char in text:
            test_line = current_line + char
            bbox = font.getbbox(test_line)
            if bbox[2] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char

        if current_line:
            lines.append(current_line)

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

    def _draw_progress_bar(self, draw: ImageDraw.ImageDraw, current: int, total: int, color: tuple):
        """Draw a thin progress bar at the top of the slide."""
        cfg = self.config
        bar_height = 4
        bar_y = 0
        full_width = cfg.width

        # Background bar (dim)
        draw.rectangle([0, bar_y, full_width, bar_y + bar_height], fill=(40, 40, 60))

        # Progress bar (colored)
        if total > 0:
            progress_width = int(full_width * current / total)
            draw.rectangle([0, bar_y, progress_width, bar_y + bar_height], fill=color)

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

    def render_opening_slide(self, date_str: str, event_count: int) -> Image.Image:
        """Render opening slide with title and date.

        Args:
            date_str: Date string (e.g., "2月5日").
            event_count: Number of events in bulletin.

        Returns:
            Opening slide image.
        """
        cfg = self.config
        accent = self._get_accent_color("default")
        img = self._get_bulletin_background("default", "巴别情报站")
        draw = ImageDraw.Draw(img)

        # Fonts
        title_font = self._get_font(cfg.title_font_size + 30, bold=True)
        date_font = self._get_font(cfg.body_font_size)

        center_y = cfg.height // 2

        # Main title - clean white, single shadow
        title_text = "巴别情报站"
        title_bbox = title_font.getbbox(title_text)
        title_x = (cfg.width - title_bbox[2]) // 2
        title_y = center_y - 100

        self._draw_text_shadow(draw, (title_x, title_y), title_text, title_font, self.TEXT_WHITE)

        # Date and count - accent color
        info_text = f"{date_str} · {event_count}件要闻"
        info_bbox = date_font.getbbox(info_text)
        info_x = (cfg.width - info_bbox[2]) // 2
        info_y = title_y + title_bbox[3] + 40

        draw.text((info_x, info_y), info_text, font=date_font, fill=accent)

        # Simple decorative line
        line_width = 200
        line_y = info_y + 60
        line_x = (cfg.width - line_width) // 2
        draw.rectangle([line_x, line_y, line_x + line_width, line_y + 3], fill=accent)

        return img

    def render_event_card(
        self,
        headline: str,
        summary: str,
        category: str,
        source_count: int,
        event_number: int,
        total_events: int,
        one_liner: str = "",
        impact: str = "",
        actions: list[str] = None,
    ) -> Image.Image:
        """Render an event card slide.

        Args:
            headline: Short headline (10 chars).
            summary: Event summary (50 chars).
            category: Event category.
            source_count: Number of source articles.
            event_number: Current event number (1-based).
            total_events: Total number of events.
            one_liner: One sentence conclusion.
            impact: Impact assessment text.
            actions: List of actionable items.

        Returns:
            Event card image.
        """
        if actions is None:
            actions = []
        cfg = self.config
        accent = self._get_accent_color(category)

        # Background: AI-generated per event (each event gets unique visual)
        img = self._get_bulletin_background(category, headline)
        draw = ImageDraw.Draw(img)

        # Progress bar at top
        self._draw_progress_bar(draw, event_number, total_events, accent)

        # Fonts
        headline_font = self._get_font(cfg.title_font_size + 20, bold=True)
        summary_font = self._get_font(cfg.body_font_size + 8)
        fact_font = self._get_font(cfg.body_font_size)
        caption_font = self._get_font(cfg.caption_font_size)

        # Main card area - frosted glass
        card_margin = 50
        card_top = cfg.padding + 80
        card_bottom = cfg.height - cfg.padding - 150
        card_coords = (card_margin, card_top, cfg.width - card_margin, card_bottom)

        # Frosted glass card
        img = self._draw_glass_card(img, draw, card_coords, opacity=0.65)
        draw = ImageDraw.Draw(img)

        # Simple rounded border in accent color
        draw.rounded_rectangle(
            card_coords,
            radius=cfg.card_radius,
            outline=(*accent, 180) if len(accent) == 3 else accent,
            width=2,
        )

        # Content inside card
        content_x = card_margin + 40
        content_y = card_top + 40

        # Category tag with accent background
        tag_text = f"#{category}"
        tag_bbox = caption_font.getbbox(tag_text)
        tag_padding = 12
        tag_height = tag_bbox[3] + tag_padding * 2

        self._draw_rounded_rect(
            draw,
            (
                content_x,
                content_y,
                content_x + tag_bbox[2] + tag_padding * 2,
                content_y + tag_height,
            ),
            radius=tag_height // 2,
            fill=accent,
        )
        draw.text(
            (content_x + tag_padding, content_y + tag_padding - 2),
            tag_text,
            font=caption_font,
            fill=self.DARK_BG,
        )

        content_y += tag_height + 30

        # Headline - white, clean
        headline_lines = self._wrap_text(headline, headline_font, cfg.width - 2 * card_margin - 80)
        for line in headline_lines[:2]:
            self._draw_text_shadow(
                draw, (content_x, content_y), line, headline_font, self.TEXT_WHITE
            )
            content_y += int((cfg.title_font_size + 20) * cfg.line_spacing)

        content_y += 20

        # Summary
        if summary:
            summary_lines = self._wrap_text(summary, summary_font, cfg.width - 2 * card_margin - 80)
            for line in summary_lines[:3]:
                draw.text((content_x, content_y), line, font=summary_font, fill=self.TEXT_SECONDARY)
                content_y += int((cfg.body_font_size + 8) * cfg.line_spacing)

        content_y += 20

        # One-liner (highlighted box)
        if one_liner:
            one_liner_lines = self._wrap_text(
                one_liner, fact_font, cfg.width - 2 * card_margin - 100
            )
            ol_height = len(one_liner_lines[:2]) * int(cfg.body_font_size * 1.3) + 30

            # Semi-transparent highlight box
            ol_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ol_draw = ImageDraw.Draw(ol_overlay)
            ol_draw.rounded_rectangle(
                (content_x, content_y, cfg.width - card_margin - 40, content_y + ol_height),
                radius=12,
                fill=(accent[0] // 6, accent[1] // 6, accent[2] // 6, 160),
            )
            img = Image.alpha_composite(img.convert("RGBA"), ol_overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Left accent bar
            draw.rectangle(
                [content_x, content_y + 8, content_x + 5, content_y + ol_height - 8],
                fill=accent,
            )

            ol_text_y = content_y + 12
            for line in one_liner_lines[:2]:
                draw.text((content_x + 20, ol_text_y), line, font=fact_font, fill=self.TEXT_WHITE)
                ol_text_y += int(cfg.body_font_size * 1.3)
            content_y += ol_height + 15

        # Impact assessment
        if impact:
            text_bbox = fact_font.getbbox("影响")
            text_height = text_bbox[3] - text_bbox[1]
            bullet_size = 10
            bullet_top = content_y + 2 + text_bbox[1] + (text_height - bullet_size) // 2
            draw.ellipse(
                [content_x, bullet_top, content_x + bullet_size, bullet_top + bullet_size],
                fill=accent,
            )
            impact_text = f"影响：{impact}"
            impact_lines = self._wrap_text(
                impact_text, fact_font, cfg.width - 2 * card_margin - 100
            )
            for line in impact_lines[:2]:
                draw.text(
                    (content_x + 20, content_y + 2), line, font=fact_font, fill=self.TEXT_SECONDARY
                )
                content_y += int(cfg.body_font_size * 1.3)
            content_y += 10

        # Actions
        if actions:
            text_bbox = fact_font.getbbox("建议")
            text_height = text_bbox[3] - text_bbox[1]
            bullet_size = 10
            bullet_top = content_y + 2 + text_bbox[1] + (text_height - bullet_size) // 2
            draw.ellipse(
                [content_x, bullet_top, content_x + bullet_size, bullet_top + bullet_size],
                fill=accent,
            )
            draw.text((content_x + 20, content_y + 2), "建议:", font=fact_font, fill=accent)
            content_y += int(cfg.body_font_size * 1.3) + 5

            arrow_indent = content_x + 20
            for action in actions[:2]:
                prefix = "→ "
                draw.text((arrow_indent, content_y + 2), prefix, font=fact_font, fill=accent)
                prefix_width = fact_font.getbbox(prefix)[2]
                action_lines = self._wrap_text(
                    action, fact_font, cfg.width - 2 * card_margin - 100 - prefix_width
                )
                for line in action_lines[:2]:
                    draw.text(
                        (arrow_indent + prefix_width, content_y + 2),
                        line,
                        font=fact_font,
                        fill=self.TEXT_WHITE,
                    )
                    content_y += int(cfg.body_font_size * 1.3)
                content_y += 5

        # Source badge at bottom right
        badge_text = f"综合{source_count}家报道"
        badge_bbox = caption_font.getbbox(badge_text)
        badge_x = cfg.width - card_margin - badge_bbox[2] - 60
        badge_y = card_bottom - 60

        self._draw_rounded_rect(
            draw,
            (badge_x, badge_y, badge_x + badge_bbox[2] + 30, badge_y + badge_bbox[3] + 16),
            radius=20,
            fill=accent,
        )
        draw.text((badge_x + 15, badge_y + 5), badge_text, font=caption_font, fill=self.DARK_BG)

        return img

    def render_data_highlight(
        self,
        number: str,
        label: str,
        description: str,
    ) -> Image.Image:
        """Render a data highlight slide with large number.

        Args:
            number: The big number to display (e.g., "100%", "$5B").
            label: Label for the number.
            description: Brief description.

        Returns:
            Data highlight image.
        """
        cfg = self.config
        accent = self._get_accent_color("金融")
        img = self._get_bulletin_background("金融")
        draw = ImageDraw.Draw(img)

        # Fonts
        number_font = self._get_font(150, bold=True)
        label_font = self._get_font(cfg.title_font_size, bold=True)
        desc_font = self._get_font(cfg.body_font_size)

        center_y = cfg.height // 2

        # Big number - clean white with shadow
        number_bbox = number_font.getbbox(number)
        number_x = (cfg.width - number_bbox[2]) // 2
        number_y = center_y - 100

        self._draw_text_shadow(
            draw, (number_x, number_y), number, number_font, self.TEXT_WHITE, shadow_offset=3
        )

        # Label
        label_bbox = label_font.getbbox(label)
        label_x = (cfg.width - label_bbox[2]) // 2
        label_y = number_y + number_bbox[3] + 30
        draw.text((label_x, label_y), label, font=label_font, fill=accent)

        # Description
        if description:
            desc_lines = self._wrap_text(description, desc_font, cfg.width - 2 * cfg.padding - 100)
            desc_y = label_y + label_bbox[3] + 40
            for line in desc_lines[:2]:
                line_bbox = desc_font.getbbox(line)
                line_x = (cfg.width - line_bbox[2]) // 2
                draw.text((line_x, desc_y), line, font=desc_font, fill=self.TEXT_SECONDARY)
                desc_y += int(cfg.body_font_size * cfg.line_spacing)

        return img

    def render_summary_slide(
        self,
        headlines: list[str],
    ) -> Image.Image:
        """Render summary slide with all headlines.

        Args:
            headlines: List of event headlines.

        Returns:
            Summary slide image.
        """
        cfg = self.config
        img = self._create_category_gradient()
        draw = ImageDraw.Draw(img)

        # Fonts
        title_font = self._get_font(cfg.title_font_size, bold=True)
        item_font = self._get_font(cfg.body_font_size + 4)
        badge_number_font = self._get_font(cfg.body_font_size + 6, bold=True)

        y_pos = cfg.padding + 80

        # Header
        header_text = "今日要点回顾"
        header_bbox = title_font.getbbox(header_text)
        header_x = (cfg.width - header_bbox[2]) // 2

        self._draw_text_shadow(draw, (header_x, y_pos), header_text, title_font, self.TEXT_WHITE)
        y_pos += header_bbox[3] + 50

        # Simple accent line
        accent = self._get_accent_color("default")
        line_width = 150
        line_x = (cfg.width - line_width) // 2
        draw.rectangle([line_x, y_pos, line_x + line_width, y_pos + 3], fill=accent)
        y_pos += 50

        # Headlines list - use per-item category colors for badge
        for i, headline in enumerate(headlines[:5]):
            # Cycle through category colors for visual variety
            color_keys = ["AI", "金融", "创业", "default", "科技"]
            color = self._get_accent_color(color_keys[i % len(color_keys)])

            # Number badge
            badge_size = 44
            badge_x = cfg.padding + 30
            draw.ellipse(
                [badge_x, y_pos + 3, badge_x + badge_size, y_pos + 3 + badge_size],
                fill=color,
            )

            # Number text centered in badge
            num_text = str(i + 1)
            num_bbox = badge_number_font.getbbox(num_text)
            num_width = num_bbox[2] - num_bbox[0]
            num_height = num_bbox[3] - num_bbox[1]
            num_x = badge_x + (badge_size - num_width) // 2 - num_bbox[0]
            num_y = y_pos + 3 + (badge_size - num_height) // 2 - num_bbox[1]
            draw.text((num_x, num_y), num_text, font=badge_number_font, fill=self.DARK_BG)

            # Headline text
            text_x = badge_x + badge_size + 20
            text_max_width = cfg.width - text_x - cfg.padding - 20
            text_lines = self._wrap_text(headline, item_font, text_max_width)
            line_height = int((cfg.body_font_size + 4) * 1.3)

            if len(text_lines) <= 1:
                # Single line: center vertically with badge
                headline_bbox = item_font.getbbox(text_lines[0] if text_lines else "A")
                text_height = headline_bbox[3] - headline_bbox[1]
                text_y = y_pos + 3 + (badge_size - text_height) // 2 - headline_bbox[1]
                if text_lines:
                    draw.text((text_x, text_y), text_lines[0], font=item_font, fill=self.TEXT_WHITE)
                y_pos += badge_size + 30
            else:
                # Multi-line: align top with badge
                text_y = y_pos + 5
                for line in text_lines[:2]:
                    draw.text((text_x, text_y), line, font=item_font, fill=self.TEXT_WHITE)
                    text_y += line_height
                y_pos += max(badge_size, line_height * min(len(text_lines), 2)) + 20

        return img

    def render_closing_slide(self) -> Image.Image:
        """Render closing slide with CTA.

        Returns:
            Closing slide image.
        """
        cfg = self.config
        accent = self._get_accent_color("default")
        img = self._get_bulletin_background("default")
        draw = ImageDraw.Draw(img)

        # Fonts
        cta_font = self._get_font(cfg.title_font_size, bold=True)
        subtitle_font = self._get_font(cfg.body_font_size)

        center_y = cfg.height // 2

        # CTA button - simple rounded rect with accent border
        cta_text = "关注了解更多"
        cta_bbox = cta_font.getbbox(cta_text)
        button_padding = 40
        button_width = cta_bbox[2] + button_padding * 2
        button_height = cta_bbox[3] + button_padding

        button_x = (cfg.width - button_width) // 2
        button_y = center_y - button_height // 2

        button_coords = (button_x, button_y, button_x + button_width, button_y + button_height)

        # Frosted glass button
        img = self._draw_glass_card(img, draw, button_coords, opacity=0.6)
        draw = ImageDraw.Draw(img)

        # Simple border
        draw.rounded_rectangle(button_coords, radius=cfg.card_radius, outline=accent, width=2)

        # CTA text
        text_x = (cfg.width - cta_bbox[2]) // 2
        text_y = button_y + (button_height - cta_bbox[3]) // 2 - 5
        draw.text((text_x, text_y), cta_text, font=cta_font, fill=self.TEXT_WHITE)

        # Subtitle below
        subtitle_text = "巴别情报站 · 快人一步"
        subtitle_bbox = subtitle_font.getbbox(subtitle_text)
        subtitle_x = (cfg.width - subtitle_bbox[2]) // 2
        subtitle_y = button_y + button_height + 40

        draw.text((subtitle_x, subtitle_y), subtitle_text, font=subtitle_font, fill=accent)

        return img

    def render_cover_slide(
        self,
        date_str: str,
        event_count: int,
        top_headline: str = "",
    ) -> Image.Image:
        """Render cover slide (thumbnail) for the video.

        Args:
            date_str: Date string (e.g., "2月5日").
            event_count: Number of events in bulletin.
            top_headline: Top headline to preview (optional).

        Returns:
            Cover slide image.
        """
        cfg = self.config
        accent = self._get_accent_color("default")
        img = self._get_bulletin_background("default", top_headline or "巴别情报站")
        draw = ImageDraw.Draw(img)

        # Fonts
        brand_font = self._get_font(cfg.title_font_size + 50, bold=True)
        count_font = self._get_font(cfg.big_number_size, bold=True)
        headline_font = self._get_font(cfg.body_font_size + 10, bold=True)
        date_font = self._get_font(cfg.body_font_size)
        caption_font = self._get_font(cfg.caption_font_size)

        center_x = cfg.width // 2
        center_y = cfg.height // 2

        # Brand name - clean white with shadow
        brand_text = "巴别情报站"
        brand_bbox = brand_font.getbbox(brand_text)
        brand_x = (cfg.width - brand_bbox[2]) // 2
        brand_y = cfg.padding + 150

        self._draw_text_shadow(
            draw, (brand_x, brand_y), brand_text, brand_font, self.TEXT_WHITE, shadow_offset=3
        )

        # Decorative line below brand
        line_width = 250
        line_y = brand_y + brand_bbox[3] + 30
        line_x = (cfg.width - line_width) // 2
        draw.rectangle([line_x, line_y, line_x + line_width, line_y + 4], fill=accent)

        # Big event count in center with circle
        count_y = center_y - 80
        circle_radius = 100

        # Outer glow circle (subtle)
        draw.ellipse(
            [
                center_x - circle_radius - 15,
                count_y - 15,
                center_x + circle_radius + 15,
                count_y + circle_radius * 2 + 15,
            ],
            fill=(accent[0] // 4, accent[1] // 4, accent[2] // 4),
        )

        # Main circle
        draw.ellipse(
            [
                center_x - circle_radius,
                count_y,
                center_x + circle_radius,
                count_y + circle_radius * 2,
            ],
            fill=accent,
        )

        # Event count number
        count_text = str(event_count)
        count_bbox = count_font.getbbox(count_text)
        count_text_x = center_x - count_bbox[2] // 2
        count_text_y = count_y + circle_radius - count_bbox[3] // 2 - 10
        draw.text((count_text_x, count_text_y), count_text, font=count_font, fill=self.DARK_BG)

        # "件要闻" label below
        label_text = "件要闻"
        label_bbox = caption_font.getbbox(label_text)
        label_x = center_x - label_bbox[2] // 2
        label_y = count_y + circle_radius * 2 + 20
        draw.text((label_x, label_y), label_text, font=date_font, fill=self.TEXT_WHITE)

        # Date below
        date_y = label_y + 50
        date_bbox = date_font.getbbox(date_str)
        date_x = center_x - date_bbox[2] // 2
        draw.text((date_x, date_y), date_str, font=date_font, fill=accent)

        # Top headline preview at bottom
        if top_headline:
            card_margin = 50
            card_top = cfg.height - cfg.padding - 200
            card_bottom = cfg.height - cfg.padding - 60
            card_coords = (card_margin, card_top, cfg.width - card_margin, card_bottom)

            # Frosted glass card
            img = self._draw_glass_card(img, draw, card_coords, opacity=0.7)
            draw = ImageDraw.Draw(img)

            # Left accent bar
            draw.rectangle(
                [card_margin, card_top + 15, card_margin + 5, card_bottom - 15],
                fill=accent,
            )

            # Headline text
            headline_lines = self._wrap_text(
                top_headline, headline_font, cfg.width - 2 * card_margin - 60
            )
            text_y = card_top + 25
            for line in headline_lines[:2]:
                draw.text(
                    (card_margin + 25, text_y), line, font=headline_font, fill=self.TEXT_WHITE
                )
                text_y += int((cfg.body_font_size + 10) * cfg.line_spacing)

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
            event_number=1,
            total_events=1,
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

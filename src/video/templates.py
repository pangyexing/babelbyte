"""Video templates for different content types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


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
        """Create a gradient background image."""
        cfg = self.config
        img = Image.new("RGB", (cfg.width, cfg.height), cfg.bg_color)

        # Create vertical gradient
        for y in range(cfg.height):
            ratio = y / cfg.height
            r = int(cfg.bg_gradient_top[0] * (1 - ratio) + cfg.bg_gradient_bottom[0] * ratio)
            g = int(cfg.bg_gradient_top[1] * (1 - ratio) + cfg.bg_gradient_bottom[1] * ratio)
            b = int(cfg.bg_gradient_top[2] * (1 - ratio) + cfg.bg_gradient_bottom[2] * ratio)
            for x in range(cfg.width):
                img.putpixel((x, y), (r, g, b))

        return img

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
    ) -> None:
        """Draw a frosted glass effect card.

        Args:
            img: Base image to draw on.
            draw: ImageDraw object.
            coords: (x1, y1, x2, y2) card coordinates.
            opacity: Card opacity (0-1).
        """
        x1, y1, x2, y2 = coords

        # Create semi-transparent overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Draw rounded rectangle with transparency
        alpha = int(255 * opacity)
        fill_color = (20, 20, 35, alpha)

        # Simple rectangle for now (PIL doesn't support rounded rect with alpha easily)
        overlay_draw.rectangle(coords, fill=fill_color)

        # Composite onto image
        img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))

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
    """Bulletin template - 抖音科技风格 (Douyin Tech Style).

    Visual style featuring neon gradients (purple → blue → cyan),
    dark backgrounds, large typography, and glowing accents.
    Designed for young tech-savvy audiences.
    """

    # Neon color palette
    NEON_PURPLE = (180, 100, 255)
    NEON_BLUE = (0, 150, 255)
    NEON_CYAN = (0, 255, 200)
    DARK_BG = (10, 10, 25)
    GLOW_WHITE = (255, 255, 255)

    def _create_neon_gradient_background(self) -> Image.Image:
        """Create a neon gradient background (purple → blue → cyan diagonal)."""
        cfg = self.config
        img = Image.new("RGB", (cfg.width, cfg.height), self.DARK_BG)

        # Create diagonal gradient
        for y in range(cfg.height):
            for x in range(cfg.width):
                # Diagonal ratio
                ratio = (x / cfg.width + y / cfg.height) / 2

                if ratio < 0.5:
                    # Purple to blue
                    local_ratio = ratio * 2
                    r = int(
                        self.NEON_PURPLE[0] * (1 - local_ratio) + self.NEON_BLUE[0] * local_ratio
                    )
                    g = int(
                        self.NEON_PURPLE[1] * (1 - local_ratio) + self.NEON_BLUE[1] * local_ratio
                    )
                    b = int(
                        self.NEON_PURPLE[2] * (1 - local_ratio) + self.NEON_BLUE[2] * local_ratio
                    )
                else:
                    # Blue to cyan
                    local_ratio = (ratio - 0.5) * 2
                    r = int(self.NEON_BLUE[0] * (1 - local_ratio) + self.NEON_CYAN[0] * local_ratio)
                    g = int(self.NEON_BLUE[1] * (1 - local_ratio) + self.NEON_CYAN[1] * local_ratio)
                    b = int(self.NEON_BLUE[2] * (1 - local_ratio) + self.NEON_CYAN[2] * local_ratio)

                # Apply to dark background with low opacity
                bg = self.DARK_BG
                opacity = 0.15
                final_r = int(bg[0] * (1 - opacity) + r * opacity)
                final_g = int(bg[1] * (1 - opacity) + g * opacity)
                final_b = int(bg[2] * (1 - opacity) + b * opacity)
                img.putpixel((x, y), (final_r, final_g, final_b))

        return img

    def _draw_glow_border(
        self,
        draw: ImageDraw.ImageDraw,
        coords: tuple,
        color: tuple,
        thickness: int = 3,
        glow_size: int = 6,
    ):
        """Draw a glowing border effect.

        Args:
            draw: ImageDraw object.
            coords: (x1, y1, x2, y2) rectangle coordinates.
            color: Border color (RGB).
            thickness: Border thickness.
            glow_size: Size of glow effect.
        """
        x1, y1, x2, y2 = coords
        radius = self.config.card_radius

        # Draw multiple layers for glow effect (outer to inner, fading)
        for i in range(glow_size, 0, -1):
            glow_color = (
                min(255, color[0] + 50),
                min(255, color[1] + 50),
                min(255, color[2] + 50),
            )
            offset = i
            draw.rounded_rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                radius=radius + offset,
                outline=glow_color,
                width=1,
            )

        # Main border
        draw.rounded_rectangle(
            coords,
            radius=radius,
            outline=color,
            width=thickness,
        )

    def _draw_scan_lines(self, img: Image.Image, opacity: float = 0.03):
        """Add subtle scan line effect for tech aesthetic.

        Args:
            img: Image to modify.
            opacity: Line opacity.
        """
        cfg = self.config
        draw = ImageDraw.Draw(img)

        # Draw horizontal lines every 4 pixels
        line_color = (255, 255, 255)
        for y in range(0, cfg.height, 4):
            if y % 8 == 0:
                draw.line(
                    [(0, y), (cfg.width, y)],
                    fill=(
                        int(line_color[0] * opacity),
                        int(line_color[1] * opacity),
                        int(line_color[2] * opacity),
                    ),
                    width=1,
                )

    def render_opening_slide(self, date_str: str, event_count: int) -> Image.Image:
        """Render opening slide with title and date.

        Args:
            date_str: Date string (e.g., "2月5日").
            event_count: Number of events in bulletin.

        Returns:
            Opening slide image.
        """
        cfg = self.config
        img = self._create_neon_gradient_background()
        draw = ImageDraw.Draw(img)

        # Add scan lines
        self._draw_scan_lines(img)

        # Fonts
        title_font = self._get_font(cfg.title_font_size + 30, bold=True)
        date_font = self._get_font(cfg.body_font_size)

        # Center position
        center_y = cfg.height // 2

        # Main title
        title_text = "巴别情报站"
        title_bbox = title_font.getbbox(title_text)
        title_x = (cfg.width - title_bbox[2]) // 2
        title_y = center_y - 100

        # Glow effect for title (draw multiple times with offset)
        for offset in range(4, 0, -1):
            glow_alpha = int(255 * (1 - offset / 4) * 0.3)
            glow_color = (
                min(255, self.NEON_CYAN[0] + glow_alpha),
                min(255, self.NEON_CYAN[1] + glow_alpha),
                min(255, self.NEON_CYAN[2] + glow_alpha),
            )
            draw.text(
                (title_x - offset // 2, title_y),
                title_text,
                font=title_font,
                fill=glow_color,
            )

        draw.text((title_x, title_y), title_text, font=title_font, fill=self.GLOW_WHITE)

        # Date and count
        info_text = f"{date_str} · {event_count}件要闻"
        info_bbox = date_font.getbbox(info_text)
        info_x = (cfg.width - info_bbox[2]) // 2
        info_y = title_y + title_bbox[3] + 40

        draw.text((info_x, info_y), info_text, font=date_font, fill=self.NEON_CYAN)

        # Decorative lines
        line_width = 200
        line_y = info_y + 60
        line_x = (cfg.width - line_width) // 2

        draw.rectangle(
            [line_x, line_y, line_x + line_width, line_y + 3],
            fill=self.NEON_PURPLE,
        )

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
        img = self._create_neon_gradient_background()
        draw = ImageDraw.Draw(img)

        # Add scan lines
        self._draw_scan_lines(img)

        # Fonts
        headline_font = self._get_font(cfg.title_font_size + 20, bold=True)
        summary_font = self._get_font(cfg.body_font_size + 8)
        fact_font = self._get_font(cfg.body_font_size)
        caption_font = self._get_font(cfg.caption_font_size)

        # Progress indicator at top
        progress_y = cfg.padding + 20
        progress_text = f"{event_number}/{total_events}"
        draw.text(
            (cfg.padding, progress_y),
            progress_text,
            font=caption_font,
            fill=self.NEON_PURPLE,
        )

        # Main card area
        card_margin = 50
        card_top = cfg.padding + 100
        card_bottom = cfg.height - cfg.padding - 150
        card_coords = (card_margin, card_top, cfg.width - card_margin, card_bottom)

        # Draw glowing border card
        self._draw_glow_border(draw, card_coords, self.NEON_BLUE, thickness=2)

        # Semi-transparent card fill
        card_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        card_draw = ImageDraw.Draw(card_overlay)
        card_draw.rounded_rectangle(
            card_coords,
            radius=cfg.card_radius,
            fill=(15, 15, 35, 220),
        )
        img = Image.alpha_composite(img.convert("RGBA"), card_overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Content inside card
        content_x = card_margin + 40
        content_y = card_top + 50

        # Category tag with neon background
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
            fill=self.NEON_PURPLE,
        )
        draw.text(
            (content_x + tag_padding, content_y + tag_padding - 2),
            tag_text,
            font=caption_font,
            fill=self.GLOW_WHITE,
        )

        content_y += tag_height + 30

        # Headline (large, white, with glow effect)
        headline_lines = self._wrap_text(headline, headline_font, cfg.width - 2 * card_margin - 80)
        for line in headline_lines[:2]:
            draw.text(
                (content_x, content_y),
                line,
                font=headline_font,
                fill=self.GLOW_WHITE,
            )
            content_y += int((cfg.title_font_size + 20) * cfg.line_spacing)

        content_y += 20

        # Summary
        if summary:
            summary_lines = self._wrap_text(summary, summary_font, cfg.width - 2 * card_margin - 80)
            for line in summary_lines[:3]:
                draw.text(
                    (content_x, content_y),
                    line,
                    font=summary_font,
                    fill=(200, 200, 220),
                )
                content_y += int((cfg.body_font_size + 8) * cfg.line_spacing)

        content_y += 20

        # One-liner (highlighted box)
        if one_liner:
            # Draw highlighted background for one-liner
            one_liner_lines = self._wrap_text(
                one_liner, fact_font, cfg.width - 2 * card_margin - 100
            )
            ol_height = len(one_liner_lines[:2]) * int(cfg.body_font_size * 1.3) + 30

            # Use alpha composite for semi-transparent background
            ol_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ol_draw = ImageDraw.Draw(ol_overlay)
            ol_draw.rounded_rectangle(
                (content_x, content_y, cfg.width - card_margin - 40, content_y + ol_height),
                radius=12,
                fill=(0, 80, 60, 180),  # Dark cyan tint
            )
            img = Image.alpha_composite(img.convert("RGBA"), ol_overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Left accent bar
            draw.rectangle(
                [content_x, content_y + 8, content_x + 6, content_y + ol_height - 8],
                fill=self.NEON_CYAN,
            )

            # One-liner text
            ol_text_y = content_y + 12
            for line in one_liner_lines[:2]:
                draw.text(
                    (content_x + 20, ol_text_y),
                    line,
                    font=fact_font,
                    fill=self.GLOW_WHITE,
                )
                ol_text_y += int(cfg.body_font_size * 1.3)
            content_y += ol_height + 15

        # Impact assessment - unified style with fact_font
        if impact:
            # Impact icon/bullet - vertically centered with text
            text_bbox = fact_font.getbbox("影响")
            text_height = text_bbox[3] - text_bbox[1]
            bullet_size = 12
            # Add text_bbox[1] to account for font top offset
            bullet_top = content_y + 2 + text_bbox[1] + (text_height - bullet_size) // 2
            draw.ellipse(
                [content_x, bullet_top, content_x + bullet_size, bullet_top + bullet_size],
                fill=self.NEON_PURPLE,
            )
            # Full impact text with label
            impact_text = f"影响：{impact}"
            impact_lines = self._wrap_text(
                impact_text, fact_font, cfg.width - 2 * card_margin - 100
            )
            for line in impact_lines[:2]:
                draw.text(
                    (content_x + 22, content_y + 2),
                    line,
                    font=fact_font,
                    fill=(200, 200, 220),
                )
                content_y += int(cfg.body_font_size * 1.3)
            content_y += 10

        # Actions - "建议:" header with bullet, then arrow list items
        if actions:
            # Header: bullet + "建议:" - vertically centered with text
            text_bbox = fact_font.getbbox("建议")
            text_height = text_bbox[3] - text_bbox[1]
            bullet_size = 12
            # Add text_bbox[1] to account for font top offset
            bullet_top = content_y + 2 + text_bbox[1] + (text_height - bullet_size) // 2
            draw.ellipse(
                [content_x, bullet_top, content_x + bullet_size, bullet_top + bullet_size],
                fill=self.NEON_CYAN,
            )
            draw.text(
                (content_x + 22, content_y + 2),
                "建议:",
                font=fact_font,
                fill=self.NEON_CYAN,
            )
            content_y += int(cfg.body_font_size * 1.3) + 5

            # Action items with arrow prefix (no bullet)
            arrow_indent = content_x + 22
            for i, action in enumerate(actions[:2]):
                arrow_color = self.NEON_CYAN if i == 0 else self.NEON_BLUE
                prefix = "→ "
                draw.text(
                    (arrow_indent, content_y + 2),
                    prefix,
                    font=fact_font,
                    fill=arrow_color,
                )
                prefix_width = fact_font.getbbox(prefix)[2]
                action_lines = self._wrap_text(
                    action, fact_font, cfg.width - 2 * card_margin - 100 - prefix_width
                )
                draw.text(
                    (arrow_indent + prefix_width, content_y + 2),
                    action_lines[0] if action_lines else "",
                    font=fact_font,
                    fill=self.GLOW_WHITE,
                )
                content_y += int(cfg.body_font_size * 1.3) + 5

        # Source badge at bottom right
        badge_text = f"综合{source_count}家报道"
        badge_bbox = caption_font.getbbox(badge_text)
        badge_x = cfg.width - card_margin - badge_bbox[2] - 60
        badge_y = card_bottom - 60

        self._draw_rounded_rect(
            draw,
            (badge_x, badge_y, badge_x + badge_bbox[2] + 30, badge_y + badge_bbox[3] + 16),
            radius=20,
            fill=self.NEON_CYAN,
        )
        draw.text(
            (badge_x + 15, badge_y + 5),
            badge_text,
            font=caption_font,
            fill=self.DARK_BG,
        )

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
        img = self._create_neon_gradient_background()
        draw = ImageDraw.Draw(img)

        # Add scan lines
        self._draw_scan_lines(img)

        # Fonts
        number_font = self._get_font(150, bold=True)
        label_font = self._get_font(cfg.title_font_size, bold=True)
        desc_font = self._get_font(cfg.body_font_size)

        center_y = cfg.height // 2

        # Big number with glow
        number_bbox = number_font.getbbox(number)
        number_x = (cfg.width - number_bbox[2]) // 2
        number_y = center_y - 100

        # Glow effect
        for offset in range(8, 0, -1):
            glow_color = (
                min(255, self.NEON_CYAN[0] + (8 - offset) * 15),
                min(255, self.NEON_CYAN[1] + (8 - offset) * 15),
                min(255, self.NEON_CYAN[2] + (8 - offset) * 15),
            )
            draw.text(
                (number_x - offset // 2, number_y - offset // 2),
                number,
                font=number_font,
                fill=glow_color,
            )

        draw.text((number_x, number_y), number, font=number_font, fill=self.GLOW_WHITE)

        # Label
        label_bbox = label_font.getbbox(label)
        label_x = (cfg.width - label_bbox[2]) // 2
        label_y = number_y + number_bbox[3] + 30

        draw.text((label_x, label_y), label, font=label_font, fill=self.NEON_CYAN)

        # Description
        if description:
            desc_lines = self._wrap_text(desc_font, cfg.width - 2 * cfg.padding - 100, description)
            desc_y = label_y + label_bbox[3] + 40
            for line in desc_lines[:2]:
                line_bbox = desc_font.getbbox(line)
                line_x = (cfg.width - line_bbox[2]) // 2
                draw.text((line_x, desc_y), line, font=desc_font, fill=(180, 180, 200))
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
        img = self._create_neon_gradient_background()
        draw = ImageDraw.Draw(img)

        # Add scan lines
        self._draw_scan_lines(img)

        # Fonts
        title_font = self._get_font(cfg.title_font_size, bold=True)
        item_font = self._get_font(cfg.body_font_size + 4)
        number_font = self._get_font(cfg.title_font_size - 10, bold=True)

        y_pos = cfg.padding + 80

        # Header
        header_text = "今日要点回顾"
        header_bbox = title_font.getbbox(header_text)
        header_x = (cfg.width - header_bbox[2]) // 2

        draw.text((header_x, y_pos), header_text, font=title_font, fill=self.GLOW_WHITE)
        y_pos += header_bbox[3] + 50

        # Decorative line
        line_width = 150
        line_x = (cfg.width - line_width) // 2
        draw.rectangle(
            [line_x, y_pos, line_x + line_width, y_pos + 3],
            fill=self.NEON_CYAN,
        )
        y_pos += 50

        # Headlines list
        colors = [self.NEON_CYAN, self.NEON_BLUE, self.NEON_PURPLE]
        # Use larger font for numbers
        badge_number_font = self._get_font(cfg.body_font_size + 6, bold=True)

        for i, headline in enumerate(headlines[:5]):
            color = colors[i % len(colors)]

            # Number badge - slightly larger
            badge_size = 44
            badge_x = cfg.padding + 30
            draw.ellipse(
                [badge_x, y_pos + 3, badge_x + badge_size, y_pos + 3 + badge_size],
                fill=color,
            )

            # Number text - centered in badge (subtract bbox offset for proper positioning)
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

            # Vertically center text with badge using actual font height
            headline_bbox = item_font.getbbox(text_lines[0] if text_lines else "A")
            text_height = headline_bbox[3] - headline_bbox[1]
            text_y = y_pos + 3 + (badge_size - text_height) // 2 - headline_bbox[1]
            for line in text_lines[:1]:
                draw.text((text_x, text_y), line, font=item_font, fill=self.GLOW_WHITE)

            y_pos += badge_size + 30

        return img

    def render_closing_slide(self) -> Image.Image:
        """Render closing slide with CTA.

        Returns:
            Closing slide image.
        """
        cfg = self.config
        img = self._create_neon_gradient_background()
        draw = ImageDraw.Draw(img)

        # Add scan lines
        self._draw_scan_lines(img)

        # Fonts
        cta_font = self._get_font(cfg.title_font_size, bold=True)
        subtitle_font = self._get_font(cfg.body_font_size)

        center_y = cfg.height // 2

        # CTA button with neon border
        cta_text = "关注了解更多"
        cta_bbox = cta_font.getbbox(cta_text)
        button_padding = 40
        button_width = cta_bbox[2] + button_padding * 2
        button_height = cta_bbox[3] + button_padding

        button_x = (cfg.width - button_width) // 2
        button_y = center_y - button_height // 2

        button_coords = (button_x, button_y, button_x + button_width, button_y + button_height)

        # Glowing border button
        self._draw_glow_border(draw, button_coords, self.NEON_CYAN, thickness=3, glow_size=10)

        # Semi-transparent fill
        card_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        card_draw = ImageDraw.Draw(card_overlay)
        card_draw.rounded_rectangle(
            button_coords,
            radius=cfg.card_radius,
            fill=(15, 15, 35, 200),
        )
        img = Image.alpha_composite(img.convert("RGBA"), card_overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # CTA text
        text_x = (cfg.width - cta_bbox[2]) // 2
        text_y = button_y + (button_height - cta_bbox[3]) // 2 - 5
        draw.text((text_x, text_y), cta_text, font=cta_font, fill=self.GLOW_WHITE)

        # Subtitle below
        subtitle_text = "巴别情报站 · 快人一步"
        subtitle_bbox = subtitle_font.getbbox(subtitle_text)
        subtitle_x = (cfg.width - subtitle_bbox[2]) // 2
        subtitle_y = button_y + button_height + 40

        draw.text(
            (subtitle_x, subtitle_y), subtitle_text, font=subtitle_font, fill=self.NEON_PURPLE
        )

        return img

    def render_cover_slide(
        self,
        date_str: str,
        event_count: int,
        top_headline: str = "",
    ) -> Image.Image:
        """Render cover slide (thumbnail) for the video.

        The cover is optimized for video thumbnails with eye-catching design:
        - Large brand name "巴别情报站"
        - Date and event count
        - Top headline preview to attract clicks

        Args:
            date_str: Date string (e.g., "2月5日").
            event_count: Number of events in bulletin.
            top_headline: Top headline to preview (optional).

        Returns:
            Cover slide image.
        """
        cfg = self.config
        img = self._create_neon_gradient_background()
        draw = ImageDraw.Draw(img)

        # Add scan lines
        self._draw_scan_lines(img)

        # Fonts
        brand_font = self._get_font(cfg.title_font_size + 50, bold=True)
        count_font = self._get_font(cfg.big_number_size, bold=True)
        headline_font = self._get_font(cfg.body_font_size + 10, bold=True)
        date_font = self._get_font(cfg.body_font_size)
        caption_font = self._get_font(cfg.caption_font_size)

        # Center position
        center_x = cfg.width // 2
        center_y = cfg.height // 2

        # Brand name "巴别情报站" at top with large glow effect
        brand_text = "巴别情报站"
        brand_bbox = brand_font.getbbox(brand_text)
        brand_x = (cfg.width - brand_bbox[2]) // 2
        brand_y = cfg.padding + 150

        # Strong glow effect for brand
        for offset in range(6, 0, -1):
            glow_alpha = int(255 * (1 - offset / 6) * 0.4)
            glow_color = (
                min(255, self.NEON_CYAN[0] + glow_alpha),
                min(255, self.NEON_CYAN[1] + glow_alpha),
                min(255, self.NEON_CYAN[2] + glow_alpha),
            )
            draw.text(
                (brand_x - offset // 2, brand_y),
                brand_text,
                font=brand_font,
                fill=glow_color,
            )
        draw.text((brand_x, brand_y), brand_text, font=brand_font, fill=self.GLOW_WHITE)

        # Decorative line below brand
        line_width = 250
        line_y = brand_y + brand_bbox[3] + 30
        line_x = (cfg.width - line_width) // 2
        draw.rectangle(
            [line_x, line_y, line_x + line_width, line_y + 4],
            fill=self.NEON_PURPLE,
        )

        # Big event count in center with circle background
        count_y = center_y - 80
        circle_radius = 100

        # Outer glow circle
        draw.ellipse(
            [
                center_x - circle_radius - 20,
                count_y - 20,
                center_x + circle_radius + 20,
                count_y + circle_radius * 2 + 20,
            ],
            fill=(self.NEON_CYAN[0] // 3, self.NEON_CYAN[1] // 3, self.NEON_CYAN[2] // 3),
        )

        # Main circle
        draw.ellipse(
            [
                center_x - circle_radius,
                count_y,
                center_x + circle_radius,
                count_y + circle_radius * 2,
            ],
            fill=self.NEON_CYAN,
        )

        # Event count number
        count_text = str(event_count)
        count_bbox = count_font.getbbox(count_text)
        count_text_x = center_x - count_bbox[2] // 2
        count_text_y = count_y + circle_radius - count_bbox[3] // 2 - 10
        draw.text((count_text_x, count_text_y), count_text, font=count_font, fill=self.DARK_BG)

        # "件要闻" label below the number
        label_text = "件要闻"
        label_bbox = caption_font.getbbox(label_text)
        label_x = center_x - label_bbox[2] // 2
        label_y = count_y + circle_radius * 2 + 20
        draw.text((label_x, label_y), label_text, font=date_font, fill=self.GLOW_WHITE)

        # Date below
        date_y = label_y + 50
        date_bbox = date_font.getbbox(date_str)
        date_x = center_x - date_bbox[2] // 2
        draw.text((date_x, date_y), date_str, font=date_font, fill=self.NEON_PURPLE)

        # Top headline preview at bottom (if provided)
        if top_headline:
            # Semi-transparent card for headline
            card_margin = 50
            card_top = cfg.height - cfg.padding - 200
            card_bottom = cfg.height - cfg.padding - 60

            card_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            card_draw = ImageDraw.Draw(card_overlay)
            card_draw.rounded_rectangle(
                [card_margin, card_top, cfg.width - card_margin, card_bottom],
                radius=cfg.card_radius,
                fill=(15, 15, 35, 220),
            )

            # Left accent bar
            card_draw.rectangle(
                [card_margin, card_top + 15, card_margin + 6, card_bottom - 15],
                fill=(*self.NEON_BLUE, 255),
            )

            img = Image.alpha_composite(img.convert("RGBA"), card_overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Headline text
            headline_lines = self._wrap_text(
                top_headline, headline_font, cfg.width - 2 * card_margin - 60
            )
            text_y = card_top + 25
            for line in headline_lines[:2]:
                draw.text(
                    (card_margin + 25, text_y),
                    line,
                    font=headline_font,
                    fill=self.GLOW_WHITE,
                )
                text_y += int((cfg.body_font_size + 10) * cfg.line_spacing)

        return img

    def render_slide(self, slide: SlideContent) -> Image.Image:
        """Render a generic slide (fallback for base class compatibility).

        For bulletins, use the specific render methods instead.
        """
        # Default to event card style
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

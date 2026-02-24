"""Subtitle renderer for burning subtitles onto video frames.

Renders synchronized subtitles using PIL, segmented by punctuation with
evenly distributed timing. Uses the same stroked text style as BulletinTemplate.
"""

import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SubtitleRenderer:
    """Burn subtitles onto video frames, synchronized with TTS audio.

    Segments script text by Chinese punctuation and distributes display time
    evenly across segments. Each frame gets the currently visible subtitle
    drawn at the bottom third of the screen.
    """

    # Chinese punctuation for sentence splitting
    SPLIT_PATTERN = re.compile(r"[，。？！；、,\.!\?;]+")

    def __init__(
        self,
        font_size: int = 60,
        color: tuple = (255, 255, 255),
        stroke_color: tuple = (0, 0, 0),
        stroke_width: int = 3,
        y_position: int = 1450,
        max_chars_per_line: int = 16,
        panel_enabled: bool = True,
    ):
        self.font_size = font_size
        self.color = color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.y_position = y_position
        self.max_chars_per_line = max_chars_per_line
        self.panel_enabled = panel_enabled
        self._font = None

    def _get_font(self) -> ImageFont.FreeTypeFont:
        """Get subtitle font (cached)."""
        if self._font is not None:
            return self._font

        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "C:/Windows/Fonts/msyh.ttc",
        ]

        for path in font_paths:
            try:
                self._font = ImageFont.truetype(path, self.font_size)
                return self._font
            except (OSError, IOError):
                continue

        self._font = ImageFont.load_default()
        return self._font

    # Regex to tokenize mixed CJK/ASCII: ASCII words stay together
    _TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[.\-_][a-zA-Z0-9]+)*|.")

    def _segment_text(self, script: str) -> list[str]:
        """Split script into display segments by punctuation.

        Respects English word boundaries when hard-splitting long segments.

        Args:
            script: Full TTS script text.

        Returns:
            List of subtitle segments.
        """
        # Split by Chinese punctuation
        parts = self.SPLIT_PATTERN.split(script.strip())
        segments = [p.strip() for p in parts if p.strip()]

        if not segments:
            return [script.strip()] if script.strip() else [""]

        # Further split segments that are too long (word-boundary aware)
        max_len = self.max_chars_per_line
        result = []
        for seg in segments:
            if len(seg) <= max_len:
                result.append(seg)
            else:
                # Tokenize to keep English words intact
                tokens = self._TOKEN_PATTERN.findall(seg)
                current = ""
                for token in tokens:
                    if len(current) + len(token) <= max_len:
                        current += token
                    else:
                        if current:
                            result.append(current)
                        # If single token exceeds max_len, force split it
                        if len(token) > max_len:
                            for i in range(0, len(token), max_len):
                                result.append(token[i : i + max_len])
                            current = ""
                        else:
                            current = token
                if current:
                    result.append(current)

        return result if result else [script.strip()]

    def _draw_stroked_text(
        self,
        draw: ImageDraw.ImageDraw,
        position: tuple,
        text: str,
        font: ImageFont.FreeTypeFont,
    ):
        """Draw text with 8-directional stroke outline."""
        x, y = position
        sw = self.stroke_width
        for dx, dy in [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ]:
            draw.text(
                (x + dx * sw, y + dy * sw), text,
                font=font, fill=self.stroke_color,
            )
        draw.text((x, y), text, font=font, fill=self.color)

    def render_subtitle_on_frame(
        self,
        frame_array: np.ndarray,
        text: str,
        width: int,
    ) -> np.ndarray:
        """Render a subtitle line onto a single frame.

        Args:
            frame_array: numpy array (H, W, 3) of the frame.
            text: Subtitle text to display.
            width: Frame width for centering.

        Returns:
            Modified frame array with subtitle burned in.
        """
        if not text:
            return frame_array

        img = Image.fromarray(frame_array)
        font = self._get_font()

        # Center horizontally
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2]
        text_height = text_bbox[3]
        x = (width - text_width) // 2

        # Semi-transparent rounded panel behind subtitle text
        if self.panel_enabled:
            panel_pad_x = 24
            panel_pad_y = 12
            panel_radius = 16
            panel_opacity = 140

            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(overlay)
            ov_draw.rounded_rectangle(
                [
                    x - panel_pad_x,
                    self.y_position - panel_pad_y,
                    x + text_width + panel_pad_x,
                    self.y_position + text_height + panel_pad_y,
                ],
                radius=panel_radius,
                fill=(0, 0, 0, panel_opacity),
            )
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(img)
        self._draw_stroked_text(draw, (x, self.y_position), text, font)

        return np.array(img)

    def create_subtitle_make_frame(
        self,
        base_array: np.ndarray,
        script: str,
        duration: float,
    ):
        """Create a make_frame function that overlays subtitles.

        Args:
            base_array: Base slide image as numpy array.
            script: Full TTS script for this slide.
            duration: Total duration of the slide in seconds.

        Returns:
            A make_frame(t) function for VideoClip.
        """
        segments = self._segment_text(script)
        n_segments = len(segments)
        segment_duration = duration / n_segments if n_segments > 0 else duration
        height, width = base_array.shape[:2]

        def make_frame(t):
            # Determine which segment to show
            if n_segments == 0:
                return base_array

            seg_idx = min(int(t / segment_duration), n_segments - 1)
            text = segments[seg_idx]

            return self.render_subtitle_on_frame(
                base_array.copy(), text, width,
            )

        return make_frame

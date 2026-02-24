"""Douyin short video generator.

Generates 15-30s single-event short videos optimized for Douyin:
- AI-generated hook (3s attention grab)
- 5-segment structure: hook → body×3 → ending
- Body split at sentence boundaries for 3 content slides
- Burned-in subtitles synced to TTS
- Ken Burns zoom + micro-pan motion effects
- 3-text-line Douyin cover image
- CRF 23 encoding (Douyin re-encodes, CRF 10 is wasteful)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    VideoClip,
    concatenate_audioclips,
    concatenate_videoclips,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

from config.settings import get_settings
from src.storage.models import ContentItem, EventCluster
from src.video.bulletin import BulletinGenerator, DouyinContent
from src.video.cover_generator import DouyinCoverGenerator
from src.video.generator import VideoConfig
from src.video.subtitle_renderer import SubtitleRenderer
from src.video.templates import BulletinTemplate, TemplateConfig
from src.video.tts import QwenTTS, get_tts

logger = logging.getLogger(__name__)


@dataclass
class DouyinVideoResult:
    """Result of Douyin short video generation."""

    video_path: Path
    cover_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    duration: float = 0.0
    success: bool = True
    error: Optional[str] = None


class DouyinVideoGenerator:
    """Generate Douyin-optimized short videos from single events.

    Each video is 15-30s with 5 segments:
    1. Hook slide (3s attention grab)
    2. Body segment 1 (event card with illustration)
    3. Body segment 2 (impact slide style)
    4. Body segment 3 (event card style)
    5. Ending (summary + engagement question)
    """

    def __init__(self, config: Optional[VideoConfig] = None, ai_bg: bool = True):
        self.config = config or VideoConfig()
        self.settings = get_settings()
        self.douyin_cfg = self.settings.douyin
        self.tts = get_tts(voice=self.config.voice, rate=self.config.speech_rate)

        # Template
        self._template_config = TemplateConfig(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
        )

        # Image generator
        image_generator = None
        if ai_bg:
            image_generator = self._create_image_generator()

        self.template = BulletinTemplate(
            self._template_config, image_generator=image_generator,
        )

        # Subtitle renderer — 50px fits ~20 CJK chars in 1080px width
        self.subtitle_renderer = SubtitleRenderer(
            font_size=50,
            y_position=int(self.config.height * 0.75),
            max_chars_per_line=20,
        )

        # Cover generator
        self.cover_generator = DouyinCoverGenerator(
            width=self.config.width, height=self.config.height,
        )

        # Content generator
        self.bulletin_gen = BulletinGenerator()

    def _create_image_generator(self):
        """Create ImageGenerator if configured."""
        try:
            from src.video.image_generator import ImageGenerator

            ig_cfg = self.settings.image_gen
            if ig_cfg.is_configured:
                return ImageGenerator(
                    model_id=ig_cfg.model_id,
                    num_inference_steps=ig_cfg.steps,
                    timeout=ig_cfg.timeout,
                )
        except Exception:
            pass
        return None

    def _auto_release_image_generator(self):
        """Release image generator GPU memory if auto_release is enabled."""
        try:
            ig = self.template._image_generator
            if ig is not None and self.settings.image_gen.auto_release:
                ig.unload()
        except Exception:
            pass

    def generate_from_event(
        self,
        cluster: EventCluster,
        members: list[ContentItem],
        output_name: Optional[str] = None,
    ) -> DouyinVideoResult:
        """Generate a single Douyin short video from an event.

        Args:
            cluster: Event cluster.
            members: Member content items.
            output_name: Optional output filename (without extension).

        Returns:
            DouyinVideoResult with paths and metadata.
        """
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if not output_name:
            timestamp = datetime.now().strftime("%Y%m%d")
            seq = len(list(output_dir.glob(f"douyin_{timestamp}_*.mp4"))) + 1
            output_name = f"douyin_{timestamp}_{seq:03d}"

        output_path = output_dir / f"{output_name}.mp4"
        audio_dir = output_dir / "temp" / output_name

        is_qwen = isinstance(self.tts, QwenTTS)
        audio_ext = ".wav" if is_qwen else ".mp3"

        # Disable auto_release for multi-segment generation
        saved_auto_release = None
        if is_qwen:
            saved_auto_release = self.tts._auto_release
            self.tts._auto_release = False

        try:
            # 1. Generate content
            logger.info(f"Generating Douyin content for: {cluster.event_title}")
            content = self.bulletin_gen.generate_douyin_content(cluster, members)
            if content is None:
                return DouyinVideoResult(
                    video_path=output_path, success=False,
                    error="Failed to generate content",
                )

            category = cluster.category or "资讯"
            date_str = datetime.now().strftime("%m月%d日")

            # 2. Generate illustration
            illustration = self._generate_illustration(
                category, content.headline, content.image_prompt,
            )

            # 3. Render 5 slides
            slides = self._render_slides(content, category, date_str, illustration)

            # 4. Generate TTS for each segment
            audio_dir.mkdir(parents=True, exist_ok=True)
            segment_audios = []
            segment_durations = []

            segments = self._split_body_segments(content.body, 3)
            tts_scripts = [content.hook] + segments + [content.ending]

            for i, script in enumerate(tts_scripts[:5]):
                seg_dir = audio_dir / f"segment_{i}"
                seg_audio, _ = self.tts.synthesize_with_timestamps(script, seg_dir)
                seg_duration = self._get_audio_duration(seg_audio)
                seg_duration = max(self.config.min_slide_duration, seg_duration)
                segment_audios.append(seg_audio)
                segment_durations.append(seg_duration)

            # Pad if fewer than 5 segments
            while len(segment_durations) < len(slides):
                segment_durations.append(3.0)

            # Concatenate audio
            audio_clips = [AudioFileClip(str(p)) for p in segment_audios]
            combined_audio = concatenate_audioclips(audio_clips)
            audio_path = audio_dir / f"voice{audio_ext}"
            combined_audio.write_audiofile(str(audio_path), logger=None)
            combined_audio.close()
            for clip in audio_clips:
                clip.close()

            # 5. Create video clips with effects
            video_clips = []
            for i, (slide_img, duration) in enumerate(
                zip(slides, segment_durations)
            ):
                script = tts_scripts[i] if i < len(tts_scripts) else ""
                clip = self._create_animated_clip(
                    slide_img, duration,
                    has_illustration=(i == 1 and illustration is not None),
                    subtitle_script=script if self.douyin_cfg.subtitle_enabled else "",
                )
                video_clips.append(clip)

            # 6. Combine and encode
            final_video = concatenate_videoclips(video_clips, method="compose")

            voice_audio = AudioFileClip(str(audio_path))
            if self.config.bg_music_path and self.config.bg_music_path.exists():
                bg_music = AudioFileClip(str(self.config.bg_music_path))
                bg_music = bg_music.with_volume_scaled(self.config.bg_music_volume)
                bg_music = bg_music.with_duration(final_video.duration)
                voice_audio = voice_audio.with_volume_scaled(self.config.voice_volume)
                final_audio = CompositeAudioClip([bg_music, voice_audio])
            else:
                final_audio = voice_audio.with_volume_scaled(self.config.voice_volume)

            final_video = final_video.with_audio(final_audio)

            crf = str(self.douyin_cfg.encoding_crf)
            preset = self.douyin_cfg.encoding_preset
            final_video.write_videofile(
                str(output_path),
                fps=self.config.fps,
                codec="libx264",
                audio_codec="aac",
                preset=preset,
                threads=4,
                logger="bar",
                ffmpeg_params=["-crf", crf],
            )

            video_duration = final_video.duration
            final_video.close()
            voice_audio.close()

            # 7. Generate cover
            cover_path = None
            if self.douyin_cfg.cover_enabled:
                cover_path = self._generate_cover(
                    content, category, date_str, output_dir, output_name,
                )

            # 8. Generate meta file
            meta_path = self._generate_meta(
                content, output_dir, output_name,
            )

            # Release GPU
            self._auto_release_image_generator()
            if is_qwen and saved_auto_release:
                self.tts._auto_release = saved_auto_release
                self.tts.unload()

            return DouyinVideoResult(
                video_path=output_path,
                cover_path=cover_path,
                meta_path=meta_path,
                audio_path=audio_path,
                duration=video_duration,
                success=True,
            )

        except Exception as e:
            logger.exception(f"Douyin video generation failed: {e}")
            self._auto_release_image_generator()
            if is_qwen:
                if saved_auto_release is not None:
                    self.tts._auto_release = saved_auto_release
                self.tts.unload()
            return DouyinVideoResult(
                video_path=output_path, success=False, error=str(e),
            )

    def generate_batch(
        self,
        events: list[tuple[EventCluster, list[ContentItem]]],
        output_dir: Optional[Path] = None,
    ) -> list[DouyinVideoResult]:
        """Generate multiple Douyin short videos.

        Args:
            events: List of (cluster, members) tuples.
            output_dir: Override output directory.

        Returns:
            List of DouyinVideoResult.
        """
        if output_dir:
            self.config.output_dir = output_dir

        results = []
        for cluster, members in events:
            result = self.generate_from_event(cluster, members)
            results.append(result)

        return results

    @staticmethod
    def _split_body_segments(body: str, n: int = 3) -> list[str]:
        """Split body text into n segments at sentence boundaries.

        Splits on Chinese sentence-ending punctuation (。！？；) and distributes
        sentences into roughly equal-length groups.

        Args:
            body: Full body text.
            n: Number of segments to produce.

        Returns:
            List of n text segments.
        """
        import re

        # Split on sentence-ending punctuation, keeping the delimiter
        parts = re.split(r"(?<=[。！？；])", body.strip())
        sentences = [s.strip() for s in parts if s.strip()]

        if not sentences:
            return [body] * n

        if len(sentences) <= n:
            # Not enough sentences — pad shorter segments
            result = []
            for i in range(n):
                result.append(sentences[i] if i < len(sentences) else sentences[-1])
            return result

        # Distribute sentences into n groups by character count
        total_chars = sum(len(s) for s in sentences)
        target = total_chars / n

        segments: list[str] = []
        current: list[str] = []
        current_len = 0

        for s in sentences:
            current.append(s)
            current_len += len(s)
            # Flush when we've exceeded the target and still have groups left
            remaining_groups = n - len(segments)
            if remaining_groups > 1 and current_len >= target:
                segments.append("".join(current))
                current = []
                current_len = 0

        # Last segment gets the rest
        if current:
            segments.append("".join(current))

        # Ensure exactly n segments
        while len(segments) < n:
            segments.append(segments[-1])

        return segments[:n]

    def _generate_illustration(
        self, category: str, headline: str, image_prompt: str = "",
    ) -> Optional["PILImage"]:
        """Generate illustration for event card."""
        ig = self.template._image_generator
        if ig is None:
            return None

        try:
            return ig.generate_illustration(
                category=category,
                headline=headline,
                prompt=image_prompt,
            )
        except Exception:
            return None

    def _render_slides(
        self,
        content: DouyinContent,
        category: str,
        date_str: str,
        illustration: Optional["PILImage"] = None,
    ) -> list["PILImage"]:
        """Render 5 slides for the Douyin video.

        Slides: hook → body segment 1 (with illustration) → body segment 2
        → body segment 3 → ending.

        Returns:
            List of 5 PIL images.
        """
        segments = self._split_body_segments(content.body, 3)

        slides = [
            # Slide 1: Hook
            self.template.render_hook_slide(
                content.hook, category, date_str,
            ),
            # Slide 2: Body segment 1 + illustration
            self.template.render_event_card(
                headline=content.headline,
                summary=segments[0],
                category=category,
                source_count=0,
                illustration=illustration,
                date_str=date_str,
            ),
            # Slide 3: Body segment 2 (impact slide style, no fixed title)
            self.template.render_impact_slide(
                segments[1], category, date_str, title="",
            ),
            # Slide 4: Body segment 3 (event card style, no headline)
            self.template.render_event_card(
                headline="",
                summary=segments[2],
                category=category,
                source_count=0,
                date_str=date_str,
            ),
            # Slide 5: Ending
            self.template.render_cta_slide(
                content.ending, date_str,
            ),
        ]

        return slides

    def _create_animated_clip(
        self,
        img: "PILImage",
        duration: float,
        has_illustration: bool = False,
        subtitle_script: str = "",
    ) -> VideoClip:
        """Create a video clip with Ken Burns motion and optional subtitles.

        Motion effects:
        - Ken Burns: slow zoom 1.0 → 1.04 (4% over duration)
        - Micro-pan: 8px horizontal drift

        Args:
            img: PIL Image for the slide.
            duration: Clip duration in seconds.
            has_illustration: Whether slide has illustration (stronger zoom).
            subtitle_script: TTS script for subtitle overlay.

        Returns:
            MoviePy VideoClip.
        """
        from PIL import Image as PILImageModule

        img_array = np.array(img)
        h, w = img_array.shape[:2]

        motion_enabled = self.douyin_cfg.motion_enabled
        subtitle_enabled = self.douyin_cfg.subtitle_enabled and subtitle_script

        if not motion_enabled and not subtitle_enabled:
            from moviepy import ImageClip
            return ImageClip(img_array).with_duration(duration)

        # Precompute subtitle segments
        sub_segments = []
        sub_seg_duration = duration
        if subtitle_enabled:
            sub_segments = self.subtitle_renderer._segment_text(subtitle_script)
            if sub_segments:
                sub_seg_duration = duration / len(sub_segments)

        max_zoom = 0.04 if has_illustration else 0.03
        max_dx = 8

        def make_frame(t):
            progress = t / max(duration, 0.001)

            if motion_enabled:
                scale = 1.0 + max_zoom * progress
                dx = int(max_dx * progress)

                new_w = int(w / scale)
                new_h = int(h / scale)
                x_offset = max(0, min((w - new_w) // 2 + dx, w - new_w))
                y_offset = max(0, min((h - new_h) // 2, h - new_h))

                pil_img = PILImageModule.fromarray(img_array)
                cropped = pil_img.crop((
                    x_offset, y_offset,
                    x_offset + new_w, y_offset + new_h,
                ))
                resized = cropped.resize((w, h), PILImageModule.Resampling.LANCZOS)
                frame = np.array(resized)
            else:
                frame = img_array.copy()

            if sub_segments:
                seg_idx = min(int(t / sub_seg_duration), len(sub_segments) - 1)
                frame = self.subtitle_renderer.render_subtitle_on_frame(
                    frame, sub_segments[seg_idx], w,
                )

            return frame

        return VideoClip(make_frame, duration=duration)

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        audio = AudioFileClip(str(audio_path))
        duration = audio.duration
        audio.close()
        return duration

    def _generate_cover(
        self,
        content: DouyinContent,
        category: str,
        date_str: str,
        output_dir: Path,
        output_name: str,
    ) -> Optional[Path]:
        """Generate Douyin cover image.

        Returns:
            Path to cover image, or None on failure.
        """
        try:
            # Try to get FLUX background for cover
            bg = None
            ig = self.template._image_generator
            if ig is not None:
                try:
                    bg = ig.generate_opening_background(
                        date_str=date_str,
                        width=self.config.width,
                        height=self.config.height,
                    )
                except Exception:
                    pass

            cover = self.cover_generator.generate(
                category=category,
                headline=content.headline,
                hook=content.hook,
                background=bg,
            )

            cover_path = output_dir / f"{output_name}_cover.jpg"
            cover.save(str(cover_path), "JPEG", quality=95)
            return cover_path

        except Exception as e:
            logger.warning(f"Cover generation failed: {e}")
            return None

    def _generate_meta(
        self,
        content: DouyinContent,
        output_dir: Path,
        output_name: str,
    ) -> Optional[Path]:
        """Generate Douyin publish metadata file.

        Returns:
            Path to meta file, or None on failure.
        """
        try:
            hashtags = " ".join(f"#{tag}" for tag in content.hashtags)
            meta_text = (
                f"【标题】{content.headline}\n"
                f"【话题】{hashtags}\n"
                f"【描述】{content.ending}\n"
            )

            meta_path = output_dir / f"{output_name}_meta.txt"
            meta_path.write_text(meta_text, encoding="utf-8")
            return meta_path

        except Exception as e:
            logger.warning(f"Meta generation failed: {e}")
            return None

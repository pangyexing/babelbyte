"""Video generator using MoviePy (open source)."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    ImageClip,
    concatenate_audioclips,
    concatenate_videoclips,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from src.video.bulletin import BulletinResult

from src.storage.models import ContentItem, EventCluster
from src.video.templates import (
    BulletinTemplate,
    SlideContent,
    TemplateConfig,
    TemplateType,
    VideoTemplate,
    get_template,
)
from src.video.tts import EdgeTTS, TTSConfig


@dataclass
class VideoConfig:
    """Video generation configuration."""

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./videos"))

    # Video specs
    width: int = 1080
    height: int = 1920  # 9:16 for Douyin
    fps: int = 24

    # Template
    template_type: TemplateType = TemplateType.NEWS_BRIEF
    auto_template: bool = False  # Auto-select template based on content

    # Script generation
    use_ai_script: bool = False  # Use AI to polish TTS script

    # TTS
    voice: str = "yunxi"  # Chinese male voice
    speech_rate: str = "+10%"

    # Audio
    bg_music_path: Optional[Path] = None
    bg_music_volume: float = 0.15
    voice_volume: float = 1.0

    # Timing
    min_slide_duration: float = 3.0
    max_slide_duration: float = 8.0
    transition_duration: float = 0.3

    # Platform presets
    platform: str = "douyin"  # "douyin" or "shipinhao"

    def __post_init__(self):
        """Adjust settings based on platform."""
        if self.platform == "shipinhao":
            # 视频号 uses 6:7 ratio
            self.height = 1260  # 1080 * 7/6


@dataclass
class VideoResult:
    """Result of video generation."""

    video_path: Path
    audio_path: Path
    duration: float
    slide_count: int
    script: str
    success: bool = True
    error: Optional[str] = None


class VideoGenerator:
    """Generate short videos from content items."""

    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self.tts = EdgeTTS(
            TTSConfig(
                voice=(
                    f"zh-CN-{self.config.voice.capitalize()}Neural"
                    if self.config.voice in ["yunxi", "yunyang", "xiaoxiao"]
                    else self.config.voice
                ),
                rate=self.config.speech_rate,
            )
        )

        # Initialize template config (template may be selected per-item if auto_template)
        self._template_config = TemplateConfig(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
        )
        self.template = get_template(self.config.template_type, self._template_config)

        # Script generator (lazy initialized)
        self._script_generator = None

    def _get_script_generator(self):
        """Get or create script generator."""
        if self._script_generator is None:
            from src.video.content_intelligence import ScriptGenerator

            self._script_generator = ScriptGenerator()
        return self._script_generator

    def _get_template_for_item(self, item: ContentItem) -> VideoTemplate:
        """Get the appropriate template for an item.

        If auto_template is enabled, selects template based on content.
        Otherwise, uses the configured template.

        Args:
            item: ContentItem to generate video for.

        Returns:
            VideoTemplate instance.
        """
        if self.config.auto_template:
            from src.video.content_intelligence import select_template

            template_type = select_template(item)
            return get_template(template_type, self._template_config)
        return self.template

    def generate_from_content(
        self,
        item: ContentItem,
        output_name: Optional[str] = None,
    ) -> VideoResult:
        """
        Generate video from a content item.

        Args:
            item: ContentItem to convert to video
            output_name: Optional output filename (without extension)

        Returns:
            VideoResult with paths and metadata
        """
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output name
        if not output_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in item.title[:30] if c.isalnum() or c in " _-")
            output_name = f"{timestamp}_{safe_title}"

        output_path = self.config.output_dir / f"{output_name}.mp4"
        audio_dir = self.config.output_dir / "temp" / output_name

        try:
            # 0. Select template for this item
            template = self._get_template_for_item(item)

            # 1. Extract key information (with highlights for keyword emphasis)
            slides = self._create_slides_from_content(item)

            # 2. Generate script (AI-polished if enabled)
            if self.config.use_ai_script:
                try:
                    script = self._get_script_generator().generate(item)
                except Exception:
                    # Fall back to simple script on AI failure
                    script = self._generate_script(item, slides)
            else:
                script = self._generate_script(item, slides)

            # 3. Generate TTS audio
            audio_path, timestamps = self.tts.synthesize_with_timestamps(
                script,
                audio_dir,
            )

            # 4. Calculate durations based on audio
            audio_duration = self._get_audio_duration(audio_path)
            slides = self._assign_durations(slides, audio_duration)

            # 5. Render video frames (using item-specific template)
            video_clips = self._render_video_clips(slides, template)

            # 6. Combine video clips
            final_video = concatenate_videoclips(video_clips, method="compose")

            # 7. Add audio
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

            # 8. Export
            final_video.write_videofile(
                str(output_path),
                fps=self.config.fps,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="medium",
                logger="bar",  # "bar" for progress bar, None for silent
            )

            # Cleanup
            final_video.close()
            voice_audio.close()

            return VideoResult(
                video_path=output_path,
                audio_path=audio_path,
                duration=final_video.duration,
                slide_count=len(slides),
                script=script,
                success=True,
            )

        except Exception as e:
            return VideoResult(
                video_path=output_path,
                audio_path=audio_dir / "voice.mp3",
                duration=0,
                slide_count=0,
                script="",
                success=False,
                error=str(e),
            )

    def _create_slides_from_content(self, item: ContentItem) -> list[SlideContent]:
        """Create slides from content item.

        Extracts key information and creates slide content with keywords
        for highlighting and content URL for background image.

        Args:
            item: ContentItem with AI processing results.

        Returns:
            List of SlideContent for video rendering.
        """
        slides = []

        # Parse enhanced data and extract highlights
        key_points = []
        highlights = []  # Keywords to highlight across slides
        if item.key_points:
            try:
                kp_data = json.loads(item.key_points)
                for kp in kp_data:
                    value = kp.get("value", "")
                    if value:
                        key_points.append(value)
                        # Use key point values as highlights (if not too long)
                        if len(value) <= 50:
                            highlights.append(value)
            except json.JSONDecodeError:
                pass

        # Common slide properties
        content_url = item.url or ""
        category = item.category or "资讯"

        # Slide 1: Title card with background image
        slides.append(
            SlideContent(
                title=item.title or "新闻速报",
                body=item.summary[:100] if item.summary else "",  # Short summary on title card
                category=category,
                source=self._get_source_name(item),
                timestamp=item.published_at.strftime("%m月%d日") if item.published_at else "",
                importance=item.importance_score or 5,
                duration=4.0,
                highlights=highlights,
                content_url=content_url,
            )
        )

        # Slide 2: Full summary (with highlights)
        if item.summary and len(item.summary) > 100:
            slides.append(
                SlideContent(
                    title="详情",
                    body=item.summary,
                    category=category,
                    duration=5.0,
                    highlights=highlights,
                    content_url=content_url,
                )
            )

        # Slide 3: Key points (if available, with highlights)
        if key_points:
            slides.append(
                SlideContent(
                    title="核心要点",
                    bullet_points=key_points[:4],
                    category=category,
                    duration=5.0,
                    highlights=highlights,
                    content_url=content_url,
                )
            )

        # Slide 4: One-liner conclusion
        if item.one_liner:
            slides.append(
                SlideContent(
                    title="总结",
                    body=item.one_liner,
                    category=category,
                    duration=3.0,
                    highlights=highlights,
                    content_url=content_url,
                )
            )

        # Slide 5: CTA
        slides.append(
            SlideContent(
                title="关注巴别情报站",
                body="获取更多资讯",
                category=category,
                duration=2.0,
                content_url=content_url,
            )
        )

        return slides

    def _generate_script(self, item: ContentItem, slides: list[SlideContent]) -> str:
        """Generate TTS script from content."""
        parts = []

        # Opening
        parts.append(f"欢迎收看巴别情报站。{item.title}")

        # Summary
        if item.summary:
            parts.append(item.summary)

        # Key points
        key_points = []
        if item.key_points:
            try:
                kp_data = json.loads(item.key_points)
                key_points = [kp.get("value", "") for kp in kp_data if kp.get("value")]
            except json.JSONDecodeError:
                pass

        if key_points:
            parts.append("关键要点：")
            for i, point in enumerate(key_points[:3], 1):
                parts.append(f"第{i}点，{point}")

        # One-liner
        if item.one_liner:
            parts.append(f"总结来说，{item.one_liner}")

        # Closing
        parts.append("感谢收看巴别情报站，关注我们获取更多资讯。")

        return "。".join(parts)

    def _get_source_name(self, item: ContentItem) -> str:
        """Get display name for source."""
        from src.storage.models import SourceType

        source_names = {
            SourceType.REDDIT: "Reddit",
            SourceType.TWITTER: "Twitter",
            SourceType.HACKERNEWS: "Hacker News",
            SourceType.RSS: "论文",
        }
        return source_names.get(item.source_type, "资讯")

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        audio = AudioFileClip(str(audio_path))
        duration = audio.duration
        audio.close()
        return duration

    def _assign_durations(
        self,
        slides: list[SlideContent],
        total_audio_duration: float,
    ) -> list[SlideContent]:
        """Assign durations to slides based on audio length."""
        # Calculate total requested duration
        total_requested = sum(s.duration for s in slides)

        # Scale if needed
        if total_requested > 0:
            scale = total_audio_duration / total_requested
            for slide in slides:
                slide.duration = max(
                    self.config.min_slide_duration,
                    min(self.config.max_slide_duration, slide.duration * scale),
                )

        return slides

    def _render_video_clips(
        self,
        slides: list[SlideContent],
        template: Optional[VideoTemplate] = None,
    ) -> list:
        """Render slides to video clips.

        Args:
            slides: List of SlideContent to render.
            template: Optional template to use. Defaults to self.template.

        Returns:
            List of video clips.
        """
        import numpy as np
        from moviepy import vfx

        use_template = template or self.template
        clips = []

        for slide in slides:
            # Render image
            img = use_template.render_slide(slide)

            # Convert PIL Image to numpy array for MoviePy 2.x
            img_array = np.array(img)

            # Convert to clip
            clip = ImageClip(img_array).with_duration(slide.duration)

            # Add fade transition
            if self.config.transition_duration > 0:
                clip = clip.with_effects([vfx.CrossFadeIn(self.config.transition_duration)])

            clips.append(clip)

        return clips

    def generate_batch(
        self,
        items: list[ContentItem],
        max_items: int = 10,
    ) -> list[VideoResult]:
        """Generate videos for multiple items."""
        results = []
        for item in items[:max_items]:
            result = self.generate_from_content(item)
            results.append(result)
        return results


class EventVideoGenerator(VideoGenerator):
    """Generate videos from event clusters."""

    def generate_from_event(
        self,
        cluster: EventCluster,
        members: list[ContentItem],
        output_name: Optional[str] = None,
    ) -> VideoResult:
        """
        Generate video from an event cluster.

        Args:
            cluster: EventCluster with event info
            members: List of content items in the cluster
            output_name: Optional output filename

        Returns:
            VideoResult with paths and metadata
        """
        # Use representative item (highest importance)
        sorted_members = sorted(
            members,
            key=lambda x: x.importance_score or 0,
            reverse=True,
        )
        representative = sorted_members[0] if sorted_members else None

        if not representative:
            return VideoResult(
                video_path=Path(),
                audio_path=Path(),
                duration=0,
                slide_count=0,
                script="",
                success=False,
                error="No members in cluster",
            )

        # Create enhanced content item
        enhanced_item = ContentItem(
            id=representative.id,
            subscription_id=representative.subscription_id,
            source_type=representative.source_type,
            external_id=representative.external_id,
            title=cluster.event_title,
            content=representative.content,
            url=representative.url,
            author=representative.author,
            published_at=cluster.first_seen_at,
            fetched_at=representative.fetched_at,
            summary=representative.summary,
            category=cluster.category,
            importance_score=max(m.importance_score or 0 for m in members),
            one_liner=f"事件涉及 {len(members)} 篇报道",
            key_points=representative.key_points,
            impact_assessment=representative.impact_assessment,
            actionable_items=representative.actionable_items,
        )

        return self.generate_from_content(enhanced_item, output_name)


@dataclass
class BulletinVideoResult:
    """Result of bulletin video generation."""

    video_path: Path
    audio_path: Path
    duration: float
    event_count: int
    script: str
    success: bool = True
    error: Optional[str] = None


class BulletinVideoGenerator:
    """Generate news bulletin videos from multiple event clusters.

    Creates a unified video with multiple segments:
    1. Opening slide with date and event count
    2. Event cards for each news item
    3. Summary slide with all headlines
    4. Closing slide with CTA
    """

    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self.tts = EdgeTTS(
            TTSConfig(
                voice=(
                    f"zh-CN-{self.config.voice.capitalize()}Neural"
                    if self.config.voice in ["yunxi", "yunyang", "xiaoxiao"]
                    else self.config.voice
                ),
                rate=self.config.speech_rate,
            )
        )

        # Template config
        self._template_config = TemplateConfig(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
        )
        self.template = BulletinTemplate(self._template_config)

    def generate_from_bulletin(
        self,
        bulletin_result: "BulletinResult",
        output_name: Optional[str] = None,
    ) -> BulletinVideoResult:
        """Generate video from a BulletinResult.

        Args:
            bulletin_result: BulletinResult from BulletinGenerator.
            output_name: Optional output filename (without extension).

        Returns:
            BulletinVideoResult with paths and metadata.
        """
        from src.video.bulletin import BulletinResult

        if not isinstance(bulletin_result, BulletinResult):
            return BulletinVideoResult(
                video_path=Path(),
                audio_path=Path(),
                duration=0,
                event_count=0,
                script="",
                success=False,
                error="Invalid bulletin result",
            )

        if not bulletin_result.items:
            return BulletinVideoResult(
                video_path=Path(),
                audio_path=Path(),
                duration=0,
                event_count=0,
                script="",
                success=False,
                error="No bulletin items to generate video from",
            )

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output name
        if not output_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"bulletin_{timestamp}"

        output_path = self.config.output_dir / f"{output_name}.mp4"
        audio_dir = self.config.output_dir / "temp" / output_name

        try:
            # 1. Render all slides (returns list of PIL images)
            slide_images = self._render_bulletin_slides_images(bulletin_result)

            # 2. Use segment scripts for precise sync, fallback to combined script
            segment_scripts = bulletin_result.segment_scripts
            if not segment_scripts or len(segment_scripts) != len(slide_images):
                # Fallback: use combined script with proportional timing
                script = bulletin_result.script
                audio_path, _ = self.tts.synthesize_with_timestamps(script, audio_dir)
                audio_duration = self._get_audio_duration(audio_path)

                # Assign durations proportionally
                slides_with_durations = [
                    (img, audio_duration / len(slide_images)) for img in slide_images
                ]
            else:
                # Generate TTS for each segment and get precise durations
                audio_dir.mkdir(parents=True, exist_ok=True)
                segment_audios = []
                slides_with_durations = []

                for i, (img, script_segment) in enumerate(
                    zip(slide_images, segment_scripts)
                ):
                    segment_audio_dir = audio_dir / f"segment_{i}"
                    segment_audio_path, _ = self.tts.synthesize_with_timestamps(
                        script_segment, segment_audio_dir
                    )
                    segment_duration = self._get_audio_duration(segment_audio_path)
                    # Ensure minimum duration
                    segment_duration = max(
                        self.config.min_slide_duration, segment_duration
                    )
                    segment_audios.append(segment_audio_path)
                    slides_with_durations.append((img, segment_duration))

                # Concatenate audio segments
                audio_clips = [AudioFileClip(str(p)) for p in segment_audios]
                combined_audio = concatenate_audioclips(audio_clips)
                audio_path = audio_dir / "voice.mp3"
                combined_audio.write_audiofile(str(audio_path), logger=None)
                combined_audio.close()
                for clip in audio_clips:
                    clip.close()

            # 3. Create video clips with precise durations
            video_clips = self._create_video_clips(slides_with_durations)

            # 4. Combine video clips
            final_video = concatenate_videoclips(video_clips, method="compose")

            # 5. Add audio
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

            # 6. Export
            final_video.write_videofile(
                str(output_path),
                fps=self.config.fps,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="medium",
                logger="bar",
            )

            # Cleanup
            final_video.close()
            voice_audio.close()

            return BulletinVideoResult(
                video_path=output_path,
                audio_path=audio_path,
                duration=final_video.duration,
                event_count=len(bulletin_result.items),
                script=bulletin_result.script,
                success=True,
            )

        except Exception as e:
            return BulletinVideoResult(
                video_path=output_path,
                audio_path=audio_dir / "voice.mp3",
                duration=0,
                event_count=0,
                script="",
                success=False,
                error=str(e),
            )

    def _render_bulletin_slides(
        self,
        bulletin_result: "BulletinResult",
    ) -> list[tuple["PILImage", float]]:
        """Render all slides for the bulletin.

        Args:
            bulletin_result: BulletinResult with items and date.

        Returns:
            List of (image, base_duration) tuples.
        """
        slides = []

        # 1. Opening slide
        date_str = bulletin_result.date.strftime("%m月%d日")
        event_count = len(bulletin_result.items)
        opening = self.template.render_opening_slide(date_str, event_count)
        slides.append((opening, 4.0))

        # 2. Event cards
        total_events = len(bulletin_result.items)
        for i, item in enumerate(bulletin_result.items, 1):
            card = self.template.render_event_card(
                headline=item.headline,
                summary=item.summary,
                category=item.cluster.category or "资讯",
                source_count=item.cluster.article_count,
                event_number=i,
                total_events=total_events,
                one_liner=item.one_liner,
                impact=item.impact,
                actions=item.actions,
            )
            # Duration based on content length (estimate reading time)
            # ~5 chars/sec for Chinese TTS (faster rate)
            content_len = len(item.headline) + len(item.summary)
            content_len += len(item.one_liner) if item.one_liner else 0
            content_len += len(item.impact) if item.impact else 0
            content_len += sum(len(a) for a in (item.actions or []))
            # Base 4s + reading time, capped at 12s per event
            base_duration = min(12.0, 4.0 + (content_len / 5.0))
            slides.append((card, base_duration))

        # 3. Summary slide
        headlines = [item.headline for item in bulletin_result.items]
        if len(headlines) > 1:
            summary = self.template.render_summary_slide(headlines)
            slides.append((summary, 4.0))

        # 4. Closing slide
        closing = self.template.render_closing_slide()
        slides.append((closing, 3.0))

        return slides

    def _render_bulletin_slides_images(
        self,
        bulletin_result: "BulletinResult",
    ) -> list["PILImage"]:
        """Render all slides for the bulletin (images only).

        Args:
            bulletin_result: BulletinResult with items and date.

        Returns:
            List of PIL images matching segment_scripts structure.
        """
        slides = []

        # 1. Opening slide
        date_str = bulletin_result.date.strftime("%m月%d日")
        event_count = len(bulletin_result.items)
        opening = self.template.render_opening_slide(date_str, event_count)
        slides.append(opening)

        # 2. Event cards
        total_events = len(bulletin_result.items)
        for i, item in enumerate(bulletin_result.items, 1):
            card = self.template.render_event_card(
                headline=item.headline,
                summary=item.summary,
                category=item.cluster.category or "资讯",
                source_count=item.cluster.article_count,
                event_number=i,
                total_events=total_events,
                one_liner=item.one_liner,
                impact=item.impact,
                actions=item.actions,
            )
            slides.append(card)

        # 3. Summary slide (only if more than 1 item)
        headlines = [item.headline for item in bulletin_result.items]
        if len(headlines) > 1:
            summary = self.template.render_summary_slide(headlines)
            slides.append(summary)

        # 4. Closing slide
        closing = self.template.render_closing_slide()
        slides.append(closing)

        return slides

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        audio = AudioFileClip(str(audio_path))
        duration = audio.duration
        audio.close()
        return duration

    def _assign_slide_durations(
        self,
        slides: list[tuple["PILImage", float]],
        total_audio_duration: float,
    ) -> list[tuple["PILImage", float]]:
        """Assign durations to slides based on audio length.

        Ensures total slide duration exactly matches audio duration,
        with minimum duration guaranteed for all slides.

        Args:
            slides: List of (image, base_duration) tuples.
            total_audio_duration: Total audio duration in seconds.

        Returns:
            List of (image, adjusted_duration) tuples.
        """
        if not slides:
            return []

        n_slides = len(slides)
        min_dur = self.config.min_slide_duration

        # Ensure we have enough time for all slides at minimum duration
        min_total = n_slides * min_dur
        if total_audio_duration < min_total:
            # Not enough time - distribute equally
            per_slide = total_audio_duration / n_slides
            return [(img, per_slide) for img, _ in slides]

        # Calculate total base duration
        total_base = sum(d for _, d in slides)

        if total_base <= 0:
            per_slide = total_audio_duration / n_slides
            return [(img, per_slide) for img, _ in slides]

        # First pass: scale proportionally
        scale = total_audio_duration / total_base
        preliminary = []
        for img, base_duration in slides:
            scaled = base_duration * scale
            preliminary.append((img, scaled))

        # Second pass: enforce minimum and redistribute excess
        # Count slides that would be below minimum
        below_min = [(i, min_dur - d) for i, (_, d) in enumerate(preliminary) if d < min_dur]
        excess_needed = sum(deficit for _, deficit in below_min)

        if excess_needed > 0:
            # Find slides above minimum that can give up time
            above_min = [(i, d - min_dur) for i, (_, d) in enumerate(preliminary) if d > min_dur]
            total_excess = sum(excess for _, excess in above_min)

            if total_excess >= excess_needed:
                # Proportionally reduce slides above minimum
                reduction_ratio = excess_needed / total_excess
                result = []
                for i, (img, dur) in enumerate(preliminary):
                    if dur < min_dur:
                        result.append((img, min_dur))
                    elif dur > min_dur:
                        reduction = (dur - min_dur) * reduction_ratio
                        result.append((img, dur - reduction))
                    else:
                        result.append((img, dur))
                return result

        # No redistribution needed
        result = []
        for img, dur in preliminary:
            result.append((img, max(min_dur, dur)))

        # Final adjustment: ensure total matches audio exactly
        total_assigned = sum(d for _, d in result)
        if abs(total_assigned - total_audio_duration) > 0.1:
            # Adjust last slide to match exactly
            img, dur = result[-1]
            adjustment = total_audio_duration - total_assigned
            result[-1] = (img, max(min_dur, dur + adjustment))

        return result

    def _create_video_clips(
        self,
        slides: list[tuple["PILImage", float]],
    ) -> list:
        """Create video clips from slides.

        Args:
            slides: List of (image, duration) tuples.

        Returns:
            List of video clips.
        """
        import numpy as np
        from moviepy import vfx

        clips = []

        for img, duration in slides:
            # Convert PIL Image to numpy array
            img_array = np.array(img)

            # Create clip
            clip = ImageClip(img_array).with_duration(duration)

            # Add fade transition
            if self.config.transition_duration > 0:
                clip = clip.with_effects([vfx.CrossFadeIn(self.config.transition_duration)])

            clips.append(clip)

        return clips

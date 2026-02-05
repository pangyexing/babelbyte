"""Video generation module for BabelByte."""

from src.video.generator import VideoGenerator, VideoConfig
from src.video.tts import EdgeTTS
from src.video.templates import VideoTemplate, TemplateType

__all__ = [
    "VideoGenerator",
    "VideoConfig",
    "EdgeTTS",
    "VideoTemplate",
    "TemplateType",
]

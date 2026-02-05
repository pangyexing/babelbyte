"""Text-to-Speech using edge-tts (completely free, no API key needed)."""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import edge_tts
from edge_tts.exceptions import NoAudioReceived

logger = logging.getLogger(__name__)

# Minimum interval between TTS requests (seconds) to avoid Microsoft throttling
_MIN_REQUEST_INTERVAL = 1.0


@dataclass
class TTSConfig:
    """TTS configuration."""

    voice: str = "zh-CN-YunxiNeural"  # 男声，适合新闻播报
    rate: str = "+10%"  # 语速稍快
    volume: str = "+0%"
    pitch: str = "+0Hz"
    max_retries: int = 3  # Retry on NoAudioReceived


# Available Chinese voices (edge-tts)
CHINESE_VOICES = {
    # 男声
    "yunxi": "zh-CN-YunxiNeural",  # 标准男声，适合新闻
    "yunyang": "zh-CN-YunyangNeural",  # 年轻男声
    "yunjian": "zh-CN-YunjianNeural",  # 成熟男声
    # 女声
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",  # 标准女声
    "xiaoyi": "zh-CN-XiaoyiNeural",  # 年轻女声
    "xiaomeng": "zh-CN-XiaomengNeural",  # 温柔女声
    # 粤语
    "hiugaai": "zh-HK-HiuGaaiNeural",  # 粤语女声
    # 台湾
    "hsiaoche": "zh-TW-HsiaoChenNeural",  # 台湾女声
}


def _has_speakable_content(text: str) -> bool:
    """Check if text has actual language content (not just punctuation/whitespace)."""
    clean = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return len(clean) > 0


class EdgeTTS:
    """Edge TTS wrapper for text-to-speech synthesis."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._last_request_time = 0.0

    async def _throttle(self):
        """Wait if needed to avoid hitting Microsoft's rate limit."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            await asyncio.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _resolve_voice(self, voice: Optional[str] = None) -> str:
        """Resolve voice name to full voice ID."""
        if voice:
            return CHINESE_VOICES.get(voice, voice)
        return self.config.voice

    def _create_communicate(self, text: str, voice_id: str) -> edge_tts.Communicate:
        """Create a new Communicate instance (each can only be streamed once)."""
        return edge_tts.Communicate(
            text=text,
            voice=voice_id,
            rate=self.config.rate,
            volume=self.config.volume,
            pitch=self.config.pitch,
        )

    async def synthesize_async(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
    ) -> Path:
        """
        Synthesize text to speech asynchronously.

        Args:
            text: Text to synthesize
            output_path: Output audio file path (.mp3)
            voice: Voice name (from CHINESE_VOICES keys) or full voice ID

        Returns:
            Path to generated audio file
        """
        if not _has_speakable_content(text):
            raise ValueError(f"Text has no speakable content: {text!r}")

        voice_id = self._resolve_voice(voice)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        last_error = None
        for attempt in range(self.config.max_retries):
            await self._throttle()
            try:
                communicate = self._create_communicate(text, voice_id)
                await communicate.save(str(output_path))
                return output_path
            except NoAudioReceived as e:
                last_error = e
                wait = 2**attempt
                logger.warning(
                    f"TTS attempt {attempt + 1}/{self.config.max_retries} failed "
                    f"(NoAudioReceived), retrying in {wait}s..."
                )
                await asyncio.sleep(wait)

        raise last_error

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
    ) -> Path:
        """
        Synthesize text to speech (sync wrapper).

        Args:
            text: Text to synthesize
            output_path: Output audio file path (.mp3)
            voice: Voice name or full voice ID

        Returns:
            Path to generated audio file
        """
        return asyncio.run(self.synthesize_async(text, output_path, voice))

    async def synthesize_with_timestamps_async(
        self,
        text: str,
        output_dir: Path,
        voice: Optional[str] = None,
    ) -> tuple[Path, list[dict]]:
        """
        Synthesize text and get word timestamps for subtitles.

        Args:
            text: Text to synthesize
            output_dir: Output directory
            voice: Voice name or full voice ID

        Returns:
            Tuple of (audio_path, word_timestamps)
        """
        if not _has_speakable_content(text):
            raise ValueError(f"Text has no speakable content: {text!r}")

        voice_id = self._resolve_voice(voice)
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / "voice.mp3"
        srt_path = output_dir / "voice.srt"

        last_error = None
        for attempt in range(self.config.max_retries):
            await self._throttle()
            try:
                communicate = self._create_communicate(text, voice_id)

                # Generate with subtitles
                submaker = edge_tts.SubMaker()
                with open(audio_path, "wb") as audio_file:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_file.write(chunk["data"])
                        elif chunk["type"] == "WordBoundary":
                            submaker.feed(chunk)

                # Save SRT file
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(submaker.get_srt())

                # Parse timestamps
                timestamps = self._parse_srt(srt_path)

                return audio_path, timestamps

            except NoAudioReceived as e:
                last_error = e
                wait = 2**attempt
                logger.warning(
                    f"TTS attempt {attempt + 1}/{self.config.max_retries} failed "
                    f"(NoAudioReceived), retrying in {wait}s..."
                )
                await asyncio.sleep(wait)

        raise last_error

    def synthesize_with_timestamps(
        self,
        text: str,
        output_dir: Path,
        voice: Optional[str] = None,
    ) -> tuple[Path, list[dict]]:
        """Sync wrapper for synthesize_with_timestamps_async."""
        return asyncio.run(self.synthesize_with_timestamps_async(text, output_dir, voice))

    def _parse_srt(self, srt_path: Path) -> list[dict]:
        """Parse SRT file to timestamp list."""
        timestamps = []
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = content.strip().split("\n\n")
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                # Parse timestamp line: 00:00:00,000 --> 00:00:01,500
                time_line = lines[1]
                start_str, end_str = time_line.split(" --> ")
                text = " ".join(lines[2:])

                timestamps.append(
                    {
                        "start": self._srt_time_to_seconds(start_str),
                        "end": self._srt_time_to_seconds(end_str),
                        "text": text,
                    }
                )

        return timestamps

    def _srt_time_to_seconds(self, time_str: str) -> float:
        """Convert SRT timestamp to seconds."""
        # Format: 00:00:00,000
        time_str = time_str.replace(",", ".")
        parts = time_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds


async def list_voices(language: str = "zh") -> list[dict]:
    """List available voices for a language."""
    voices = await edge_tts.list_voices()
    return [v for v in voices if v["Locale"].startswith(language)]

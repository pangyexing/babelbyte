"""Text-to-Speech providers: EdgeTTS (free, network) and QwenTTS (local GPU)."""

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

# Audio file extension per provider
AUDIO_EXT = {"edge": ".mp3", "qwen": ".wav"}

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


# Qwen3-TTS available speakers
# Character aliases:
#   "linhuaiyue" (林怀岳/冷面笑匠) → Uncle_Fu: 一本正经的反差幽默
#   "dushe" (毒舌主播) → Vivian: 明快犀利的毒舌风格
QWEN_SPEAKERS = {
    "vivian": "Vivian",
    "serena": "Serena",
    "uncle_fu": "Uncle_Fu",
    "dylan": "Dylan",
    "eric": "Eric",
    "ryan": "Ryan",
    "aiden": "Aiden",
    "linhuaiyue": "Uncle_Fu",
    "dushe": "Vivian",
}


class QwenTTS:
    """Qwen3-TTS wrapper for local GPU-based text-to-speech synthesis.

    Supports three model types:
    - custom_voice: Preset speakers with emotion control via instruct.
    - voice_design: Design a voice from scratch using natural language description.
    - voice_clone: Clone a voice from a reference audio file.

    Lazy-loads the model on first synthesis call, and can unload to free GPU
    memory for other models (Ollama, FLUX). Outputs .wav files via soundfile.
    """

    def __init__(
        self,
        speaker: str = "Vivian",
        language: str = "Chinese",
        instruct: str = "",
        model_type: str = "custom_voice",
        voice_design_instruct: str = "",
        voice_clone_ref_audio: str = "",
        voice_clone_ref_text: str = "",
        voice_clone_x_vector_only: bool = True,
    ):
        self.speaker = QWEN_SPEAKERS.get(speaker.lower(), speaker)
        self.language = language
        self.instruct = instruct  # emotion/style control for custom_voice
        self.voice_design_instruct = voice_design_instruct  # voice description for voice_design
        self.model_type = model_type
        self.voice_clone_ref_audio = voice_clone_ref_audio
        self.voice_clone_ref_text = voice_clone_ref_text
        self.voice_clone_x_vector_only = voice_clone_x_vector_only
        self._model = None
        self._voice_clone_prompt = None
        self._auto_release = True

    def _load_model(self):
        """Load Qwen3-TTS model to GPU (lazy, first call only)."""
        if self._model is not None:
            return

        self._unload_ollama_models()

        import torch
        from qwen_tts import Qwen3TTSModel

        from config.settings import get_settings

        cfg = get_settings().qwen_tts
        model_id = cfg.model_id

        # Derive model ID variant from CustomVoice base
        if self.model_type == "voice_design":
            model_id = model_id.replace("CustomVoice", "VoiceDesign")
        elif self.model_type == "voice_clone":
            model_id = model_id.replace("CustomVoice", "Base")

        logger.info(f"Loading Qwen3-TTS model: {model_id} (type={self.model_type})")
        self._model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        logger.info("Qwen3-TTS model loaded on GPU")

    def unload(self):
        """Unload model from GPU to free VRAM."""
        if self._model is None:
            return

        import gc

        import torch

        del self._model
        self._model = None
        self._voice_clone_prompt = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Qwen3-TTS model unloaded, GPU memory freed")

    def _unload_ollama_models(self):
        """Ask Ollama to unload all loaded models to free GPU memory."""
        try:
            import requests

            from config.settings import get_settings

            base_url = get_settings().ollama.base_url

            resp = requests.get(f"{base_url}/api/ps", timeout=5)
            if resp.status_code != 200:
                return

            models = resp.json().get("models", [])
            if not models:
                return

            for m in models:
                name = m.get("name", "")
                if name:
                    logger.info(f"Unloading Ollama model: {name}")
                    requests.post(
                        f"{base_url}/api/generate",
                        json={"model": name, "keep_alive": 0},
                        timeout=10,
                    )

            time.sleep(2)
            logger.info("Ollama models unloaded, GPU memory freed")
        except Exception as e:
            logger.debug(f"Ollama unload skipped: {e}")

    def _get_gen_params(self) -> dict:
        """Read generation parameters from config."""
        from config.settings import get_settings

        cfg = get_settings().qwen_tts
        return {
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "repetition_penalty": cfg.repetition_penalty,
        }

    def _resolve_ref_audio(self) -> str:
        """Resolve reference audio path — pick daily from directory.

        If voice_clone_ref_audio is a directory, selects one audio file
        deterministically by date hash (same day = same voice).
        If it's a file, returns it as-is.
        """
        import hashlib
        from datetime import date

        ref = Path(self.voice_clone_ref_audio)
        if not ref.is_dir():
            return self.voice_clone_ref_audio

        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        files = sorted(
            f for f in ref.iterdir()
            if f.is_file() and f.suffix.lower() in audio_exts
        )
        if not files:
            raise ValueError(
                f"No audio files found in {ref} "
                f"(supported: {', '.join(audio_exts)})"
            )

        today = date.today().isoformat()
        idx = int(hashlib.md5(today.encode()).hexdigest(), 16) % len(files)
        chosen = str(files[idx])
        logger.info(
            f"Daily voice clone: picked {files[idx].name} "
            f"({idx + 1}/{len(files)}) for {today}"
        )
        return chosen

    def _ensure_voice_clone_prompt(self):
        """Create and cache the voice clone prompt from the reference audio.

        The prompt is cached so it can be reused across multiple synthesize()
        calls (e.g. bulletin segments) without re-processing the reference.
        """
        if self._voice_clone_prompt is not None:
            return

        self._load_model()

        ref_audio = self._resolve_ref_audio()
        ref_text = self.voice_clone_ref_text or None
        x_vector_only = self.voice_clone_x_vector_only

        logger.info(
            f"Creating voice clone prompt from {ref_audio} "
            f"(x_vector_only={x_vector_only}, "
            f"ref_text={'yes' if ref_text else 'no'})"
        )

        kwargs = {
            "ref_audio": ref_audio,
            "x_vector_only_mode": x_vector_only,
        }
        if ref_text:
            kwargs["ref_text"] = ref_text

        self._voice_clone_prompt = (
            self._model.create_voice_clone_prompt(**kwargs)
        )
        logger.info("Voice clone prompt cached")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
    ) -> Path:
        """Synthesize text to speech.

        Args:
            text: Text to synthesize.
            output_path: Output audio file path (.wav).
            voice: Speaker name override (from QWEN_SPEAKERS keys).

        Returns:
            Path to generated audio file.
        """
        if not _has_speakable_content(text):
            raise ValueError(f"Text has no speakable content: {text!r}")

        import soundfile as sf

        self._load_model()

        output_path = output_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        gen_params = self._get_gen_params()

        if self.model_type == "voice_clone":
            # Voice clone: use cached prompt from reference audio
            self._ensure_voice_clone_prompt()
            logger.info(
                f"Qwen3-TTS voice_clone synthesizing ({len(text)} chars)"
            )
            wavs, sample_rate = self._model.generate_voice_clone(
                text=text,
                language=self.language,
                voice_clone_prompt=self._voice_clone_prompt,
                **gen_params,
            )
        elif self.model_type == "voice_design":
            # VoiceDesign: voice_design_instruct describes the desired voice characteristics
            vd_instruct = self.voice_design_instruct
            logger.info(
                f"Qwen3-TTS voice_design synthesizing ({len(text)} chars, "
                f"instruct={vd_instruct!r})"
            )
            gen_kwargs = {
                "text": text,
                "instruct": vd_instruct,
                "language": self.language,
                **gen_params,
            }
            wavs, sample_rate = self._model.generate_voice_design(**gen_kwargs)
        else:
            # CustomVoice: preset speakers with optional emotion instruct
            speaker = QWEN_SPEAKERS.get(voice.lower(), voice) if voice else self.speaker
            logger.info(
                f"Qwen3-TTS custom_voice synthesizing ({len(text)} chars, "
                f"speaker={speaker})"
            )
            gen_kwargs = {
                "text": text,
                "speaker": speaker,
                "language": self.language,
                **gen_params,
            }
            if self.instruct:
                gen_kwargs["instruct"] = self.instruct
            wavs, sample_rate = self._model.generate_custom_voice(**gen_kwargs)

        sf.write(str(output_path), wavs[0], sample_rate)

        if self._auto_release:
            self.unload()

        return output_path

    def synthesize_with_timestamps(
        self,
        text: str,
        output_dir: Path,
        voice: Optional[str] = None,
    ) -> tuple[Path, list[dict]]:
        """Synthesize text to speech (no timestamp support).

        Returns:
            Tuple of (audio_path, empty_list). Callers always discard timestamps.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / "voice.wav"
        self.synthesize(text, audio_path, voice)
        return audio_path, []


def get_tts(voice: str = "yunxi", rate: str = "+10%") -> EdgeTTS | QwenTTS:
    """Factory function to create the appropriate TTS provider.

    Reads TTS_PROVIDER from settings (qwen_tts.enabled).
    Returns QwenTTS for qwen provider, EdgeTTS for edge (default).
    """
    from config.settings import get_settings

    cfg = get_settings().qwen_tts

    if cfg.enabled:
        return QwenTTS(
            speaker=cfg.speaker,
            language=cfg.language,
            instruct=cfg.instruct,
            model_type=cfg.model_type,
            voice_design_instruct=cfg.voice_design_instruct,
            voice_clone_ref_audio=cfg.voice_clone_ref_audio,
            voice_clone_ref_text=cfg.voice_clone_ref_text,
            voice_clone_x_vector_only=cfg.voice_clone_x_vector_only,
        )

    # Default: EdgeTTS
    voice_id = CHINESE_VOICES.get(voice, voice)
    return EdgeTTS(TTSConfig(voice=voice_id, rate=rate))

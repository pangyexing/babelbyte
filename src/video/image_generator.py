"""AI image generation via diffusers pipeline (FLUX.2 Klein 4B).

Generates background images for video slides using a local diffusers pipeline
running directly on the host GPU. No Docker or API server required.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Cache directory for AI-generated images
AI_IMAGE_CACHE_DIR = Path("./videos/cache/ai_images")

# Category-specific prompt styles (dark, moody backgrounds for text overlay)
CATEGORY_STYLES = {
    "AI": (
        "abstract dark background, deep navy blue, "
        "subtle neural network patterns, glowing data streams"
    ),
    "科技": "dark tech background, faint circuit lines, deep midnight tones",
    "金融": "dark background, abstract golden geometric shapes, subtle grid lines",
    "创业": "dark background, subtle teal light streaks, smooth gradients",
    "创新": "dark background, subtle teal light streaks, smooth gradients",
    "default": (
        "dark abstract fluid art, deep midnight blue and dark purple, " "subtle light particles"
    ),
}

# Appended to all prompts to ensure usability as text background
PROMPT_SUFFIX = "dark background, moody atmosphere, cinematic lighting, no text, no objects"


class ImageGenerator:
    """AI image generation via local diffusers pipeline."""

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-4B",
        num_inference_steps: int = 4,
        timeout: int = 120,
        cache_dir: Optional[Path] = None,
    ):
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.timeout = timeout
        self.cache_dir = cache_dir or AI_IMAGE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = None  # Lazy loaded

    def _load_pipeline(self):
        """Load diffusers pipeline to GPU (lazy, first call only)."""
        if self._pipeline is not None:
            return

        # Free Ollama GPU memory before loading diffusers pipeline
        self._unload_ollama_models()

        import torch
        from diffusers import Flux2KleinPipeline

        logger.info(f"Loading diffusers pipeline: {self.model_id}")
        self._pipeline = Flux2KleinPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        self._pipeline.enable_model_cpu_offload()
        logger.info("Pipeline loaded with model_cpu_offload enabled")

    def _unload_ollama_models(self):
        """Ask Ollama to unload all loaded models to free GPU memory."""
        try:
            import requests

            from config.settings import get_settings

            base_url = get_settings().ollama.base_url

            # Get currently loaded models
            resp = requests.get(f"{base_url}/api/ps", timeout=5)
            if resp.status_code != 200:
                return

            models = resp.json().get("models", [])
            if not models:
                return

            # Unload each model (keep_alive=0)
            for m in models:
                name = m.get("name", "")
                if name:
                    logger.info(f"Unloading Ollama model: {name}")
                    requests.post(
                        f"{base_url}/api/generate",
                        json={"model": name, "keep_alive": 0},
                        timeout=10,
                    )

            import time
            time.sleep(2)
            logger.info("Ollama models unloaded, GPU memory freed")
        except Exception as e:
            logger.debug(f"Ollama unload skipped: {e}")

    def generate(self, prompt: str, width: int = 1080, height: int = 1920) -> Optional[Image.Image]:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate.
            width: Image width.
            height: Image height.

        Returns:
            PIL Image if successful, None otherwise.
        """
        # Check cache first
        cache_path = self._get_cache_path(prompt, width, height)
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception:
                cache_path.unlink(missing_ok=True)

        try:
            self._load_pipeline()

            result = self._pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=0.0,
            )
            img = result.images[0].convert("RGB")

            # Resize if needed
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.LANCZOS)

            # Cache the result
            img.save(cache_path, "JPEG", quality=90)

            return img

        except Exception as e:
            logger.warning(f"Image generation failed: {e}")
            return None

    def generate_background(
        self,
        category: str,
        headline: str,
        width: int = 1080,
        height: int = 1920,
    ) -> Optional[Image.Image]:
        """Generate a background image suitable for text overlay.

        The generated image is automatically darkened and blurred for readability.

        Args:
            category: Content category (AI, 科技, 金融, etc.).
            headline: Headline text (used to add context to prompt).
            width: Image width.
            height: Image height.

        Returns:
            Darkened background image, or None if generation fails.
        """
        prompt = self._build_prompt(category, headline)
        img = self.generate(prompt, width, height)

        if img is None:
            return None

        # Darken and blur for text readability
        img = self._prepare_as_background(img)
        return img

    def unload(self):
        """Unload the pipeline and release GPU memory for Ollama."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Pipeline unloaded, GPU memory released")

    def _build_prompt(self, category: str, headline: str) -> str:
        """Build an image generation prompt from category and headline.

        Args:
            category: Content category.
            headline: News headline.

        Returns:
            Full prompt string.
        """
        style = CATEGORY_STYLES.get(category, CATEGORY_STYLES["default"])

        # Add headline context (keep it abstract)
        if headline:
            # Take first few words for context, keep abstract
            context = headline[:20]
            prompt = f"{style}, inspired by: {context}, {PROMPT_SUFFIX}"
        else:
            prompt = f"{style}, {PROMPT_SUFFIX}"

        return prompt

    def _get_cache_path(self, prompt: str, width: int, height: int) -> Path:
        """Get cache file path for a prompt."""
        key = f"{prompt}_{width}x{height}"
        prompt_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{prompt_hash}.jpg"

    def _prepare_as_background(self, img: Image.Image) -> Image.Image:
        """Darken and slightly blur an image for use as text background.

        Args:
            img: Source image.

        Returns:
            Prepared background image.
        """
        # Apply slight blur to reduce visual noise
        img = img.filter(ImageFilter.GaussianBlur(radius=3))

        # Darken: blend with black overlay
        dark = Image.new("RGB", img.size, (0, 0, 0))
        img = Image.blend(img, dark, alpha=0.45)

        return img

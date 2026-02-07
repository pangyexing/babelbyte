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

# Category-specific background variants (dark, moody backgrounds for text overlay)
# Each category has 3 variants; selection uses headline hash to avoid repetition.
BACKGROUND_VARIANTS = {
    "AI": [
        ("abstract dark background, deep navy, " "neural network patterns, data streams"),
        "dark space, constellation of connected nodes, blue glow",
        "midnight black, holographic grid lines, subtle blue light",
    ],
    "科技": [
        "dark tech background, faint circuit lines, deep midnight tones",
        "dark background, satellite dish silhouette, starfield",
        "dark background, glowing fiber optic strands, cool blue",
    ],
    "金融": [
        "dark background, abstract golden geometric shapes, subtle grid lines",
        "dark background, stock chart silhouette, warm amber glow",
        "dark background, stacked coins silhouette, golden light rays",
    ],
    "创业": [
        "dark background, subtle teal light streaks, smooth gradients",
        "dark background, ascending staircase silhouette, teal glow",
        "dark background, sprouting seedling, warm teal highlight",
    ],
    "创新": [
        "dark background, interlocking gears silhouette, green accents",
        "dark background, lightbulb filament glow, warm orange highlight",
        "dark background, molecular structure, cyan glow",
    ],
    "技术": [
        "dark background, server rack silhouettes, subtle LED dots",
        "dark background, binary code rain, green on black",
        "dark background, ethernet cables and ports, blue indicators",
    ],
    "default": [
        ("dark abstract fluid art, deep midnight blue and dark purple, " "subtle light particles"),
        "dark background, soft bokeh circles, deep blue and violet",
        "dark background, ink wash texture, charcoal and indigo",
    ],
}

# Appended to all prompts to ensure usability as text background
PROMPT_SUFFIX = "dark background, moody atmosphere, cinematic lighting, no text, no objects"

# Opening/cover/summary/closing slide backgrounds (dark cinematic, date-hash selected)
OPENING_BACKGROUND_PROMPTS = [
    (
        "dark cinematic cityscape at night, neon reflections on wet streets, "
        "deep blue and purple atmosphere, moody fog"
    ),
    (
        "dark abstract cosmic scene, distant galaxies, nebula dust, "
        "deep indigo and violet tones, cinematic depth"
    ),
    (
        "dark futuristic control room, holographic displays, "
        "dim ambient lighting, teal and midnight blue"
    ),
    (
        "dark atmospheric ocean scene, bioluminescent waves, "
        "deep navy and cyan glow, cinematic mood"
    ),
    (
        "dark industrial interior, steel beams and glass, "
        "warm amber spotlights on dark background, cinematic shadows"
    ),
]

# Category-specific illustration styles (comic/manga art for bulletin cards)
ILLUSTRATION_STYLES = {
    "AI": (
        "comic book panel, robot head with glowing eyes, bold black ink outlines, "
        "halftone dot shading, blue and cyan pop art colors"
    ),
    "科技": (
        "manga style panel, futuristic device close-up, bold ink strokes, "
        "speed lines, screen glow, blue and white pop art"
    ),
    "金融": (
        "comic book panel, coins and rising arrow, bold black outlines, "
        "halftone dots, golden yellow and amber pop art colors"
    ),
    "创业": (
        "manga style panel, rocket launching upward, dynamic motion lines, "
        "bold ink outlines, teal and green pop art energy"
    ),
    "创新": (
        "comic book panel, gears and mechanical parts, bold black ink lines, "
        "halftone shading, green and cyan pop art colors"
    ),
    "技术": (
        "manga style panel, terminal screen with code, bold outlines, "
        "dramatic lighting, purple and blue pop art"
    ),
    "default": (
        "comic book panel, newspaper and megaphone, bold black ink outlines, "
        "halftone dot pattern, bright pop art colors"
    ),
}

# Suffix for LLM-generated prompts (style already specified by LLM)
ILLUSTRATION_SUFFIX = "no text, no words, no letters, no numbers"

# Full suffix for template-based fallback prompts (no LLM context)
ILLUSTRATION_FALLBACK_SUFFIX = (
    "comic book art, bold black ink outlines, halftone dots, flat vibrant colors, "
    "manga-inspired, clean composition, no text, no words, no letters"
)


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

            # Align to 16px multiples (FLUX requirement)
            gen_w = (width // 16) * 16
            gen_h = (height // 16) * 16

            result = self._pipeline(
                prompt=prompt,
                width=gen_w,
                height=gen_h,
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

    def generate_illustration(
        self,
        category: str,
        headline: str,
        width: int = 960,
        height: int = 528,
        prompt: str = "",
    ) -> Optional[Image.Image]:
        """Generate a bright illustration for display inside an event card.

        Unlike generate_background(), this produces vibrant, visible imagery
        without heavy darkening—suitable for inline card display.

        Args:
            category: Content category (AI, 科技, 金融, etc.).
            headline: Headline text (adds context to the prompt).
            width: Image width.
            height: Image height.
            prompt: Pre-generated prompt from LLM. Falls back to template if empty.

        Returns:
            Illustration image, or None if generation fails.
        """
        if prompt:
            full_prompt = f"{prompt}, {ILLUSTRATION_SUFFIX}"
        else:
            full_prompt = self._build_illustration_prompt(category, headline)
        img = self.generate(full_prompt, width, height)

        if img is None:
            return None

        # Light post-processing: gentle blur to soften noise, no darkening
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        return img

    def generate_opening_background(
        self,
        date_str: str,
        width: int = 1080,
        height: int = 1920,
    ) -> Optional[Image.Image]:
        """Generate a dark cinematic background for opening/cover/summary/closing slides.

        Uses date_str hash to select a prompt variant, ensuring the same date always
        gets the same background (and cache hits across slides sharing the same date).

        Args:
            date_str: Date string (e.g., "02月05日") used for variant selection.
            width: Image width.
            height: Image height.

        Returns:
            Darkened background image, or None if generation fails.
        """
        idx = int(hashlib.md5(date_str.encode()).hexdigest(), 16) % len(
            OPENING_BACKGROUND_PROMPTS
        )
        prompt = f"{OPENING_BACKGROUND_PROMPTS[idx]}, {PROMPT_SUFFIX}"
        img = self.generate(prompt, width, height)

        if img is None:
            return None

        return self._prepare_as_opening_background(img)

    def _prepare_as_opening_background(self, img: Image.Image) -> Image.Image:
        """Lightly blur and darken an image for opening/cover slides.

        Gentler than _prepare_as_background — keeps more visual detail.

        Args:
            img: Source image.

        Returns:
            Prepared background image.
        """
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        dark = Image.new("RGB", img.size, (0, 0, 0))
        img = Image.blend(img, dark, alpha=0.55)
        return img

    def _build_illustration_prompt(self, category: str, headline: str) -> str:
        """Build an illustration prompt from category and headline.

        Uses more headline context (50 chars) to differentiate images for
        events in the same category.

        Args:
            category: Content category.
            headline: News headline.

        Returns:
            Full prompt string.
        """
        style = ILLUSTRATION_STYLES.get(category, ILLUSTRATION_STYLES["default"])

        if headline:
            context = headline[:50]
            prompt = f"{style}, inspired by: {context}, " f"{ILLUSTRATION_FALLBACK_SUFFIX}"
        else:
            prompt = f"{style}, {ILLUSTRATION_FALLBACK_SUFFIX}"

        return prompt

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

        Uses headline hash to select among variants, ensuring different
        headlines in the same category get different backgrounds.

        Args:
            category: Content category.
            headline: News headline.

        Returns:
            Full prompt string.
        """
        variants = BACKGROUND_VARIANTS.get(category, BACKGROUND_VARIANTS["default"])
        idx = int(hashlib.md5(headline.encode()).hexdigest(), 16) % len(variants)
        style = variants[idx]

        # Add headline context (keep it abstract)
        if headline:
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

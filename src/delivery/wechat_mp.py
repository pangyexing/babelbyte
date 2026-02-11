"""WeChat Official Account (公众号) delivery module for BabelByte."""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from jinja2 import Environment, FileSystemLoader

from config.settings import PROJECT_ROOT, get_settings
from src.processors.digest_processor import DigestResult

logger = logging.getLogger(__name__)

# WeChat MP API error codes that indicate token expiry
TOKEN_EXPIRED_CODES = {40001, 40014, 42001}
# Daily mass send quota exhausted
QUOTA_EXHAUSTED_CODE = 45028
# Max retries for API calls
MAX_RETRIES = 3


@dataclass
class WechatMPResult:
    """Result of WeChat MP publishing operation."""

    success: bool
    message: str
    draft_media_id: Optional[str] = None
    publish_id: Optional[str] = None
    published_at: Optional[datetime] = None


class WechatMPTokenManager:
    """Manages access_token caching with 2h TTL and early refresh."""

    def __init__(self):
        self._token: Optional[str] = None
        self._expires_at: float = 0

    @property
    def token(self) -> Optional[str]:
        """Get cached token if still valid (with 200s safety margin)."""
        if self._token and time.time() < self._expires_at - 200:
            return self._token
        return None

    def update(self, token: str, expires_in: int) -> None:
        """Cache a new token with its TTL."""
        self._token = token
        self._expires_at = time.time() + expires_in

    def invalidate(self) -> None:
        """Force token refresh on next access."""
        self._token = None
        self._expires_at = 0


class WechatMPClient:
    """Low-level WeChat MP API client."""

    BASE_URL = "https://api.weixin.qq.com/cgi-bin"

    def __init__(self, appid: str, appsecret: str):
        self.appid = appid
        self.appsecret = appsecret
        self.token_manager = WechatMPTokenManager()
        self.session = requests.Session()

    def get_access_token(self) -> str:
        """Get a valid access_token, refreshing if needed."""
        cached = self.token_manager.token
        if cached:
            return cached

        url = f"{self.BASE_URL}/token"
        params = {
            "grant_type": "client_credential",
            "appid": self.appid,
            "secret": self.appsecret,
        }

        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if "access_token" not in data:
            errcode = data.get("errcode", "unknown")
            errmsg = data.get("errmsg", "unknown error")
            raise RuntimeError(f"Failed to get access_token: [{errcode}] {errmsg}")

        token = data["access_token"]
        expires_in = data.get("expires_in", 7200)
        self.token_manager.update(token, expires_in)
        logger.info("WeChat MP access_token refreshed, expires_in=%d", expires_in)
        return token

    def _api_call(self, method: str, path: str, **kwargs) -> dict:
        """Make an API call with automatic token refresh and retry.

        Args:
            method: HTTP method ("GET" or "POST").
            path: API path (appended to BASE_URL).
            **kwargs: Passed to requests (json, data, files, params, etc.)

        Returns:
            Parsed JSON response.
        """
        for attempt in range(MAX_RETRIES):
            token = self.get_access_token()

            # Merge access_token into params
            params = kwargs.pop("params", {})
            params["access_token"] = token

            url = f"{self.BASE_URL}/{path}"
            resp = self.session.request(
                method, url, params=params, timeout=30, **kwargs
            )
            resp.raise_for_status()
            data = resp.json()

            errcode = data.get("errcode", 0)
            if errcode == 0:
                return data

            # Token expired — refresh and retry
            if errcode in TOKEN_EXPIRED_CODES:
                logger.warning(
                    "Token expired (errcode=%d), refreshing (attempt %d/%d)",
                    errcode, attempt + 1, MAX_RETRIES,
                )
                self.token_manager.invalidate()
                time.sleep(1)
                continue

            # Quota exhausted — don't retry
            if errcode == QUOTA_EXHAUSTED_CODE:
                raise RuntimeError(
                    f"Daily mass send quota exhausted (errcode={errcode}). "
                    "Use --draft-only to create a draft instead."
                )

            # Other API error
            errmsg = data.get("errmsg", "unknown")
            raise RuntimeError(f"WeChat API error: [{errcode}] {errmsg}")

        raise RuntimeError("Max retries exceeded for WeChat API call")

    def upload_image(self, image_path: Path) -> str:
        """Upload an image for use in article content.

        Returns:
            The image URL that can be used in article HTML.
        """
        token = self.get_access_token()
        url = f"{self.BASE_URL}/media/uploadimg"
        params = {"access_token": token}

        with open(image_path, "rb") as f:
            files = {"media": (image_path.name, f, "image/jpeg")}
            resp = self.session.post(url, params=params, files=files, timeout=30)

        resp.raise_for_status()
        data = resp.json()

        if "url" not in data:
            errcode = data.get("errcode", "unknown")
            errmsg = data.get("errmsg", "unknown")
            raise RuntimeError(f"Image upload failed: [{errcode}] {errmsg}")

        return data["url"]

    def upload_thumb(self, image_path: Path) -> str:
        """Upload a thumb (cover) image as permanent material.

        Returns:
            The media_id for the thumb image.
        """
        token = self.get_access_token()
        url = f"{self.BASE_URL}/material/add_material"
        params = {"access_token": token, "type": "thumb"}

        with open(image_path, "rb") as f:
            files = {"media": (image_path.name, f, "image/jpeg")}
            resp = self.session.post(url, params=params, files=files, timeout=30)

        resp.raise_for_status()
        data = resp.json()

        if "media_id" not in data:
            errcode = data.get("errcode", "unknown")
            errmsg = data.get("errmsg", "unknown")
            raise RuntimeError(f"Thumb upload failed: [{errcode}] {errmsg}")

        return data["media_id"]

    def create_draft(
        self,
        title: str,
        content: str,
        author: str,
        thumb_media_id: str,
        digest: str = "",
    ) -> str:
        """Create a draft article.

        Returns:
            The media_id of the created draft.
        """
        articles = [
            {
                "title": title,
                "author": author,
                "content": content,
                "thumb_media_id": thumb_media_id,
                "digest": digest[:120] if digest else "",
                "show_cover_pic": 1,
                "need_open_comment": 0,
            }
        ]

        data = self._api_call("POST", "draft/add", json={"articles": articles})
        media_id = data.get("media_id")
        if not media_id:
            raise RuntimeError("Draft creation succeeded but no media_id returned")
        return media_id

    def publish_draft(self, media_id: str) -> str:
        """Submit a draft for publishing (freepublish).

        Returns:
            The publish_id for tracking publication status.
        """
        data = self._api_call(
            "POST", "freepublish/submit", json={"media_id": media_id}
        )
        publish_id = data.get("publish_id")
        if not publish_id:
            raise RuntimeError("Publish succeeded but no publish_id returned")
        return publish_id


class WechatMPPublisher:
    """High-level publisher: DigestResult -> WeChat MP article."""

    def __init__(self):
        settings = get_settings().wechat_mp
        self.client = WechatMPClient(settings.appid, settings.appsecret)
        self.author = settings.author
        self.default_thumb_path = Path(settings.default_thumb_path)

        # Jinja2 template environment
        templates_dir = PROJECT_ROOT / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=False,  # We handle sanitization ourselves
        )

    def publish_digest(
        self, digest: DigestResult, draft_only: bool = False
    ) -> WechatMPResult:
        """Publish a digest to WeChat MP.

        Args:
            digest: The digest result to publish.
            draft_only: If True, only create a draft without publishing.

        Returns:
            WechatMPResult with status and identifiers.
        """
        try:
            # 1. Render HTML content
            html = self._render_digest_html(digest)
            html = self._sanitize_html(html)

            # 2. Generate title and summary
            title = self._generate_title(digest)
            summary = self._generate_digest_summary(digest)

            # 3. Upload cover image
            thumb_media_id = self._get_thumb_media_id()

            # 4. Create draft
            logger.info("Creating WeChat MP draft: %s", title)
            draft_media_id = self.client.create_draft(
                title=title,
                content=html,
                author=self.author,
                thumb_media_id=thumb_media_id,
                digest=summary,
            )
            logger.info("Draft created: media_id=%s", draft_media_id)

            if draft_only:
                return WechatMPResult(
                    success=True,
                    message=f"Draft created: {title}",
                    draft_media_id=draft_media_id,
                )

            # 5. Publish
            logger.info("Publishing draft: %s", draft_media_id)
            publish_id = self.client.publish_draft(draft_media_id)
            logger.info("Published: publish_id=%s", publish_id)

            return WechatMPResult(
                success=True,
                message=f"Published: {title}",
                draft_media_id=draft_media_id,
                publish_id=publish_id,
                published_at=datetime.now(),
            )

        except RuntimeError as e:
            logger.error("WeChat MP publish failed: %s", e)
            return WechatMPResult(success=False, message=str(e))
        except requests.RequestException as e:
            logger.error("WeChat MP network error: %s", e)
            return WechatMPResult(success=False, message=f"Network error: {e}")

    def _render_digest_html(self, digest: DigestResult) -> str:
        """Render digest to WeChat-compatible HTML using Jinja2 template."""
        template = self.jinja_env.get_template("wechat_digest.html")
        # Estimate ~30s per item for reading time
        reading_time = max(1, round(digest.total_items * 0.5))
        return template.render(
            date=digest.generated_at.strftime("%Y年%m月%d日"),
            total_items=digest.total_items,
            event_count=len(digest.events),
            individual_count=len(digest.regular_items),
            paper_count=len(digest.papers),
            events=digest.events,
            events_by_category=digest.events_by_category,
            regular_items_by_category=digest.regular_items_by_category,
            papers_by_category=digest.papers_by_category,
            items=digest.items,
            reading_time=reading_time,
        )

    @staticmethod
    def _sanitize_html(html: str) -> str:
        """Remove elements forbidden by WeChat MP editor.

        Strips <style>, <script> blocks and class/id attributes.
        """
        # Remove <style>...</style>
        html = re.sub(r"<style[\s\S]*?</style>", "", html, flags=re.IGNORECASE)
        # Remove <script>...</script>
        html = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
        # Remove class attributes
        html = re.sub(r'\s+class="[^"]*"', "", html)
        html = re.sub(r"\s+class='[^']*'", "", html)
        # Remove id attributes
        html = re.sub(r'\s+id="[^"]*"', "", html)
        html = re.sub(r"\s+id='[^']*'", "", html)
        return html

    @staticmethod
    def _generate_title(digest: DigestResult) -> str:
        """Generate article title like 'BabelByte 02月09日 | 5个事件 20条资讯'."""
        date_str = digest.generated_at.strftime("%m月%d日")
        parts = []
        if digest.events:
            parts.append(f"{len(digest.events)}个事件")
        total_items = len(digest.regular_items) + len(digest.papers)
        if total_items:
            parts.append(f"{total_items}条资讯")
        detail = " ".join(parts) if parts else "每日摘要"
        return f"BabelByte {date_str} | {detail}"

    @staticmethod
    def _generate_digest_summary(digest: DigestResult) -> str:
        """Generate a short summary (max 120 chars for WeChat digest field)."""
        # Pick the top event or item title as summary hook
        parts = []
        if digest.events:
            top = digest.events[0]
            parts.append(top.event_title)
        if digest.regular_items:
            for item in digest.regular_items[:2]:
                parts.append(item.content_item.title[:30])

        summary = "; ".join(parts)
        if len(summary) > 64:
            summary = summary[:61] + "..."
        return summary

    def _get_thumb_media_id(self) -> str:
        """Get the thumb media_id for the cover image.

        Priority:
        1. User-provided static cover (default_thumb_path)
        2. FLUX AI-generated cover (if IMAGE_GEN_ENABLED)
        3. Simple Pillow-generated fallback
        """
        thumb_path = self.default_thumb_path
        if not thumb_path.is_absolute():
            thumb_path = PROJECT_ROOT / thumb_path

        if not thumb_path.exists():
            thumb_path = self._generate_cover()

        return self.client.upload_thumb(thumb_path)

    def _generate_cover(self) -> Path:
        """Generate a 900x383 cover image. Uses FLUX if available, else Pillow."""
        from PIL import Image, ImageDraw, ImageFilter, ImageFont

        width, height = 900, 383
        date_str = datetime.now().strftime("%m月%d日")
        cover_path = PROJECT_ROOT / "data" / "wechat_cover_generated.jpg"
        cover_path.parent.mkdir(parents=True, exist_ok=True)

        # Try FLUX generation
        bg = self._generate_flux_cover(width, height, date_str)
        if bg is None:
            # Fallback: solid gradient
            bg = Image.new("RGB", (width, height), color=(74, 144, 217))

        # Draw text overlay
        img = self._draw_cover_text(bg, date_str)
        img.save(cover_path, "JPEG", quality=92)
        logger.info("Generated cover: %s", cover_path)
        return cover_path

    @staticmethod
    def _generate_flux_cover(
        width: int, height: int, date_str: str
    ) -> "Optional[Image.Image]":
        """Try to generate a cover background via FLUX pipeline.

        Returns PIL Image on success, None if FLUX is not available.
        """
        try:
            from config.settings import get_settings
            from src.video.image_generator import ImageGenerator

            ig_cfg = get_settings().image_gen
            if not ig_cfg.is_configured:
                return None

            ig = ImageGenerator(
                model_id=ig_cfg.model_id,
                num_inference_steps=ig_cfg.steps,
                timeout=ig_cfg.timeout,
            )

            # Pick a prompt variant based on date for daily variety
            import hashlib
            prompts = [
                (
                    "dark cinematic landscape, neon city skyline at night, "
                    "deep blue and purple atmosphere, moody reflections"
                ),
                (
                    "dark abstract cosmic scene, distant galaxies and nebula, "
                    "deep indigo and violet tones, cinematic depth"
                ),
                (
                    "dark futuristic control room panorama, holographic displays, "
                    "dim ambient lighting, teal and midnight blue"
                ),
                (
                    "dark atmospheric ocean panorama, bioluminescent waves, "
                    "deep navy and cyan glow, cinematic wide angle"
                ),
                (
                    "dark abstract fluid art, deep midnight blue and dark purple, "
                    "subtle golden light particles, cinematic wide"
                ),
            ]
            idx = int(hashlib.md5(date_str.encode()).hexdigest(), 16) % len(prompts)
            prompt = (
                f"{prompts[idx]}, "
                "ultra wide angle, panoramic, cinematic lighting, "
                "no text, no objects, no people"
            )

            img = ig.generate(prompt, width, height)
            if img is not None:
                from PIL import ImageFilter, Image as PILImage

                # Darken for text readability
                img = img.filter(ImageFilter.GaussianBlur(radius=2))
                dark = PILImage.new("RGB", img.size, (0, 0, 0))
                img = PILImage.blend(img, dark, alpha=0.4)
                logger.info("FLUX cover generated for %s", date_str)

            # Release GPU memory
            if ig_cfg.auto_release:
                ig.unload()

            return img

        except Exception as e:
            logger.warning("FLUX cover generation failed, using fallback: %s", e)
            return None

    @staticmethod
    def _draw_cover_text(
        bg: "Image.Image", date_str: str
    ) -> "Image.Image":
        """Draw 'BabelByte' title and date on a background image."""
        from PIL import ImageDraw, ImageFont

        img = bg.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size

        title = "BabelByte"
        subtitle = f"每日摘要 {date_str}"

        # Try to use a good font, fall back to default
        try:
            font_large = ImageFont.truetype(
                "/System/Library/Fonts/PingFang.ttc", 56
            )
            font_small = ImageFont.truetype(
                "/System/Library/Fonts/PingFang.ttc", 24
            )
        except (OSError, IOError):
            try:
                font_large = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56
                )
                font_small = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
                )
            except (OSError, IOError):
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()

        # Center title
        bbox = draw.textbbox((0, 0), title, font=font_large)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (width - tw) / 2
        ty = (height - th) / 2 - 24

        # Text shadow for readability on any background
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 3)]:
            draw.text((tx + dx, ty + dy), title, fill=(0, 0, 0, 180), font=font_large)
        draw.text((tx, ty), title, fill="white", font=font_large)

        # Center subtitle
        bbox2 = draw.textbbox((0, 0), subtitle, font=font_small)
        tw2 = bbox2[2] - bbox2[0]
        sx = (width - tw2) / 2
        sy = ty + th + 16

        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            draw.text((sx + dx, sy + dy), subtitle, fill=(0, 0, 0, 160), font=font_small)
        draw.text((sx, sy), subtitle, fill=(220, 230, 255), font=font_small)

        return img

    def test_connection(self) -> WechatMPResult:
        """Test API connectivity by fetching an access_token."""
        try:
            token = self.client.get_access_token()
            return WechatMPResult(
                success=True,
                message=f"Connection successful. Token: {token[:8]}...{token[-4:]}",
            )
        except Exception as e:
            return WechatMPResult(success=False, message=f"Connection failed: {e}")

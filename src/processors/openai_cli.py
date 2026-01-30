"""OpenAI API wrapper for AI processing."""

from typing import Optional

from config.settings import get_settings
from src.processors.base import BaseAIProcessor, ProcessingResult


class OpenAICLI(BaseAIProcessor):
    """Wrapper for OpenAI API (GPT models)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        settings = get_settings().openai
        self.api_key = api_key or settings.api_key
        self.model = model or settings.model
        self.base_url = base_url or settings.base_url
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                client_kwargs = {"api_key": self.api_key, "timeout": self.timeout}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url

                self._client = OpenAI(**client_kwargs)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Run: pip install openai"
                )
        return self._client

    def process_content(self, title: str, content: str) -> ProcessingResult:
        """
        Process content using OpenAI API.

        Args:
            title: The title of the content.
            content: The main content text.

        Returns:
            ProcessingResult with summary, category, and importance score.
        """
        if not self.api_key:
            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message="OpenAI API key not configured. Set OPENAI_API_KEY in .env",
            )

        prompt = self._build_prompt(title, content)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个内容分析助手，请严格按照要求返回JSON格式。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content
            return self._parse_json_response(result_text)

        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower():
                error_msg = "OpenAI API key invalid or not configured"
            elif "rate_limit" in error_msg.lower():
                error_msg = "OpenAI API rate limit exceeded"

            return ProcessingResult(
                summary="",
                category="其他",
                importance_score=5,
                success=False,
                error_message=f"OpenAI API error: {error_msg}",
            )

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.api_key:
            return False
        try:
            # Try a simple API call to verify
            self.client.models.list()
            return True
        except Exception:
            return False


class OpenAICompatibleCLI(OpenAICLI):
    """
    Wrapper for OpenAI-compatible APIs (e.g., local LLMs, Azure OpenAI, etc.).

    Supports any API that implements the OpenAI chat completions interface.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__(api_key, model, base_url, timeout)
        # For compatible APIs, default to a common model name if not specified
        if not self.model:
            self.model = "gpt-3.5-turbo"

    def is_available(self) -> bool:
        """Check if the compatible API is available."""
        if not self.api_key or not self.base_url:
            return False
        try:
            # Try a simple completion to verify
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False

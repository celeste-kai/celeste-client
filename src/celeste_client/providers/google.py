from typing import Any, AsyncIterator

from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from celeste_core.config.settings import settings


class GoogleClient(BaseClient):
    def __init__(self, model: str = "gemini-2.5-flash-lite-preview-06-17", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.GOOGLE, **kwargs)
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Google provider requires optional dependency 'google-genai'.\n"
                "Install with: pip install 'celeste-client[google]'"
            ) from e
        self._genai = genai
        self.client = genai.Client(api_key=settings.google.api_key)

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        config = kwargs.pop("config", {})

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._genai.types.GenerateContentConfig(**config),
        )

        content = response.text

        return AIResponse(
            content=content,
            provider=Provider.GOOGLE,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(self, prompt: str, **kwargs: Any) -> AsyncIterator[AIResponse]:
        config = kwargs.pop("config", {})

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=self._genai.types.GenerateContentConfig(**config),
        ):
            if chunk.text:  # Only yield if there's actual content
                yield AIResponse(
                    content=chunk.text,
                    provider=Provider.GOOGLE,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )

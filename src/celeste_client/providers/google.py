from collections.abc import AsyncIterator
from typing import Any

from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from celeste_core.config.settings import settings
from google import genai
from google.genai import types


class GoogleClient(BaseClient):
    def __init__(self, model: str = "gemini-2.5-flash-lite-preview-06-17", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.GOOGLE, **kwargs)
        self.client = genai.Client(api_key=settings.google.api_key)

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        config = kwargs.pop("config", None)

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**(config or {})),
        )

        content = response.parsed if config else response.text

        return AIResponse(
            content=content,
            provider=Provider.GOOGLE,
            metadata={"model": self.model},
        )

    async def stream_generate_content(self, prompt: str, **kwargs: Any) -> AsyncIterator[AIResponse]:
        config = kwargs.pop("config", {})

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config),
        ):
            if chunk.text:  # Only yield if there's actual content
                yield AIResponse(
                    content=chunk.text,
                    provider=Provider.GOOGLE,
                    metadata={"model": self.model, "is_stream_chunk": True},
                )

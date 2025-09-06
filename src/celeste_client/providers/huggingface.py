from collections.abc import AsyncIterator
from typing import Any

from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from celeste_core.config.settings import settings
from huggingface_hub import AsyncInferenceClient


class HuggingFaceClient(BaseClient):
    def __init__(self, model: str = "google/gemma-2-2b-it", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.HUGGINGFACE, **kwargs)
        self.client = AsyncInferenceClient(
            model=self.model,
            token=settings.huggingface.access_token,
        )

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        messages = [{"role": "user", "content": prompt}]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )

        return AIResponse(
            content=response.choices[0].message.content,
            provider=Provider.HUGGINGFACE,
            metadata={"model": self.model},
        )

    async def stream_generate_content(self, prompt: str, **kwargs: Any) -> AsyncIterator[AIResponse]:
        messages = [{"role": "user", "content": prompt}]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield AIResponse(
                    content=chunk.choices[0].delta.content,
                    provider=Provider.HUGGINGFACE,
                    metadata={"model": self.model, "is_stream_chunk": True},
                )

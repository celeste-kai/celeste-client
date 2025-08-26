from typing import Any, AsyncIterator

from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from celeste_core.config.settings import settings
from mistralai import Mistral


class MistralClient(BaseClient):
    def __init__(self, model: str = "mistral-small-latest", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.MISTRAL, **kwargs)
        self.client = Mistral(api_key=settings.mistral.api_key)

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        response = await self.client.chat.complete_async(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        return AIResponse(
            content=response.choices[0].message.content,
            provider=Provider.MISTRAL,
            metadata={"model": self.model},
        )

    async def stream_generate_content(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[AIResponse]:
        response = await self.client.chat.stream_async(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        async for chunk in response:
            # Yield content chunks
            if chunk.data.choices[0].delta.content:
                yield AIResponse(
                    content=chunk.data.choices[0].delta.content,
                    provider=Provider.MISTRAL,
                    metadata={"model": self.model, "is_stream_chunk": True},
                )

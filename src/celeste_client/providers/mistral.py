from typing import Any, AsyncIterator

from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from celeste_core.config.settings import settings


class MistralClient(BaseClient):
    def __init__(self, model: str = "mistral-small-latest", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.MISTRAL, **kwargs)
        try:
            from mistralai import Mistral  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Mistral provider requires optional dependency 'mistralai'.\n"
                "Install with: pip install 'celeste-client[mistral]'"
            ) from e
        self.client = Mistral(api_key=settings.mistral.api_key)

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        response = await self.client.chat.complete_async(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        return AIResponse(
            content=response.choices[0].message.content,
            provider=Provider.MISTRAL,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(self, prompt: str, **kwargs: Any) -> AsyncIterator[AIResponse]:
        response = await self.client.chat.stream_async(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        async for chunk in response:
            # Yield content chunks
            if chunk.data.choices[0].delta.content:
                yield AIResponse(
                    content=chunk.data.choices[0].delta.content,
                    provider=Provider.MISTRAL,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )

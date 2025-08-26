from typing import Any, AsyncIterator, List

from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from celeste_core.config.settings import settings


class OpenAIClient(BaseClient):
    def __init__(self, model: str = "o4-mini-2025-04-16", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.OPENAI, **kwargs)
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires optional dependency 'openai'.\n"
                "Install with: pip install 'celeste-client[openai]'"
            ) from e
        self.client = AsyncOpenAI(api_key=settings.openai.api_key)

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        messages: List[dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]

        response = await self.client.chat.completions.create(
            messages=messages, model=self.model_name, **kwargs
        )

        content = response.choices[0].message.content or ""

        return AIResponse(
            content=content,
            provider=Provider.OPENAI,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[AIResponse]:
        messages: List[dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]

        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            stream=True,
            **kwargs,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield AIResponse(
                    content=chunk.choices[0].delta.content,
                    provider=Provider.OPENAI,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )

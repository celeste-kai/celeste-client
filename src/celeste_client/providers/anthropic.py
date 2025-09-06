from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from celeste_core.config.settings import settings

MAX_TOKENS = 1024


class AnthropicClient(BaseClient):
    def __init__(self, model: str = "claude-3-7-sonnet-20250219", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.ANTHROPIC, **kwargs)
        self.client = AsyncAnthropic(api_key=settings.anthropic.api_key)

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
        response = await self.client.messages.create(
            max_tokens=max_tokens,
            messages=[MessageParam(role="user", content=prompt)],
            model=self.model,
            **kwargs,
        )

        return AIResponse(
            content=response.content[0].text,
            provider=Provider.ANTHROPIC,
            metadata={"model": self.model},
        )

    async def stream_generate_content(self, prompt: str, **kwargs: Any) -> AsyncIterator[AIResponse]:
        max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            messages=[MessageParam(role="user", content=prompt)],
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield AIResponse(
                    content=text,
                    provider=Provider.ANTHROPIC,
                    metadata={"model": self.model, "is_stream_chunk": True},
                )

            # finalize stream
            await stream.get_final_message()

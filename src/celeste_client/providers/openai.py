from collections.abc import AsyncIterator
from typing import Any

from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from celeste_core.config.settings import settings
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionUserMessageParam,
)


class OpenAIClient(BaseClient):
    def __init__(self, model: str = "o4-mini-2025-04-16", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.OPENAI, **kwargs)
        self.client = AsyncOpenAI(api_key=settings.openai.api_key)

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        messages: list[ChatCompletionMessageParam] = [ChatCompletionUserMessageParam(role="user", content=prompt)]

        response = await self.client.chat.completions.create(messages=messages, model=self.model, **kwargs)

        content = response.choices[0].message.content or ""

        return AIResponse(
            content=content,
            provider=Provider.OPENAI,
            metadata={"model": self.model},
        )

    async def stream_generate_content(self, prompt: str, **kwargs: Any) -> AsyncIterator[AIResponse]:
        messages: list[ChatCompletionMessageParam] = [ChatCompletionUserMessageParam(role="user", content=prompt)]

        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            stream_options=ChatCompletionStreamOptionsParam(include_usage=False),
            **kwargs,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield AIResponse(
                    content=chunk.choices[0].delta.content,
                    provider=Provider.OPENAI,
                    metadata={"model": self.model, "is_stream_chunk": True},
                )

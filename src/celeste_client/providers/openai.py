from typing import Any, AsyncIterator, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionUserMessageParam,
)

from celeste_client.base import BaseClient
from celeste_client.core.config import OPENAI_API_KEY
from celeste_client.core.enums import OpenAIModel, Provider
from celeste_client.core.types import AIPrompt, AIResponse, AIUsage


class OpenAIClient(BaseClient):
    def __init__(self, model: str = OpenAIModel.GPT_O4_MINI, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model

    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert OpenAI usage data to AIUsage."""
        if not usage_data:
            return None
        return AIUsage(
            input_tokens=usage_data.prompt_tokens,
            output_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens,
        )

    async def generate_content(self, prompt: AIPrompt, **kwargs: Any) -> AIResponse:
        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content=prompt.content)
        ]
        response = await self.client.chat.completions.create(
            messages=messages, model=self.model_name, **kwargs
        )

        usage = self.format_usage(response.usage)

        return AIResponse(
            content=response.choices[0].message.content or "",
            usage=usage,
            provider=Provider.OPENAI,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: AIPrompt, **kwargs: Any
    ) -> AsyncIterator[AIResponse]:
        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content=prompt.content)
        ]
        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            stream=True,
            stream_options=ChatCompletionStreamOptionsParam(include_usage=True),
            **kwargs,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield AIResponse(
                    content=chunk.choices[0].delta.content,
                    provider=Provider.OPENAI,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )
            elif chunk.usage:
                usage = self.format_usage(chunk.usage)
                if usage:
                    yield AIResponse(
                        content="",  # Empty content for the usage-only response
                        usage=usage,
                        provider=Provider.OPENAI,
                        metadata={"model": self.model_name, "is_final_usage": True},
                    )

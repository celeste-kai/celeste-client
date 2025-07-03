from typing import Any, AsyncIterator, Optional

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from celeste_client.base import BaseClient
from celeste_client.core.config import ANTHROPIC_API_KEY
from celeste_client.core.enums import AnthropicModel, Provider
from celeste_client.core.types import AIResponse, AIUsage

MAX_TOKENS = 1024


class AnthropicClient(BaseClient):
    def __init__(
        self,
        model: str = AnthropicModel.CLAUDE_3_7_SONNET.value,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = model

    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert Anthropic usage data to AIUsage."""
        if not usage_data:
            return None
        return AIUsage(
            input_tokens=usage_data.input_tokens,
            output_tokens=usage_data.output_tokens,
            total_tokens=usage_data.input_tokens + usage_data.output_tokens,
        )

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
        response = await self.client.messages.create(
            max_tokens=max_tokens,
            messages=[MessageParam(role="user", content=prompt)],
            model=self.model_name,
            **kwargs,
        )

        return AIResponse(
            content=response.content[0].text,
            usage=self.format_usage(response.usage),
            provider=Provider.ANTHROPIC,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[AIResponse]:
        max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
        async with self.client.messages.stream(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[MessageParam(role="user", content=prompt)],
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield AIResponse(
                    content=text,
                    provider=Provider.ANTHROPIC,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )

            # Get final usage data
            final_message = await stream.get_final_message()
            usage = self.format_usage(final_message.usage if final_message else None)
            if usage:
                yield AIResponse(
                    content="",
                    usage=usage,
                    provider=Provider.ANTHROPIC,
                    metadata={
                        "model": self.model_name,
                        "is_stream_chunk": True,
                        "usage_only": True,
                    },
                )

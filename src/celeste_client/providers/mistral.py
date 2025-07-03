from typing import Any, AsyncIterator, Optional

from mistralai import Mistral

from celeste_client.base import BaseClient
from celeste_client.core.config import MISTRAL_API_KEY
from celeste_client.core.enums import MistralModel, Provider
from celeste_client.core.types import AIResponse, AIUsage


class MistralClient(BaseClient):
    def __init__(
        self, model: str = MistralModel.SMALL_LATEST.value, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model_name = model

    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert Mistral usage data to AIUsage."""
        if not usage_data:
            return None
        return AIUsage(
            input_tokens=getattr(usage_data, "prompt_tokens", 0),
            output_tokens=getattr(usage_data, "completion_tokens", 0),
            total_tokens=getattr(usage_data, "total_tokens", 0),
        )

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        response = await self.client.chat.complete_async(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        # Extract usage information if available
        usage = self.format_usage(
            response.usage if hasattr(response, "usage") else None
        )

        return AIResponse(
            content=response.choices[0].message.content,
            usage=usage,
            provider=Provider.MISTRAL,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[AIResponse]:
        response = await self.client.chat.stream_async(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        usage_data = None
        async for chunk in response:
            # Yield content chunks
            if chunk.data.choices[0].delta.content:
                yield AIResponse(
                    content=chunk.data.choices[0].delta.content,
                    provider=Provider.MISTRAL,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )

            # Check for usage data in the chunk
            if hasattr(chunk.data, "usage") and chunk.data.usage:
                usage_data = self.format_usage(chunk.data.usage)

        # Yield final usage data if available
        if usage_data:
            yield AIResponse(
                content="",
                usage=usage_data,
                provider=Provider.MISTRAL,
                metadata={"model": self.model_name, "is_final_usage": True},
            )

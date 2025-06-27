from typing import Any, AsyncIterator, Dict, Optional

from google import genai
from google.genai import types

from celeste_client.base import BaseClient
from celeste_client.core.config import GOOGLE_API_KEY
from celeste_client.core.enums import GeminiModel, Provider
from celeste_client.core.types import AIPrompt, AIResponse, AIUsage


class GeminiClient(BaseClient):
    def __init__(self, model: str = GeminiModel.FLASH_LITE, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = model

    @staticmethod
    def _get_generation_config(kwargs: Dict[str, Any]) -> types.GenerateContentConfig:
        """Get or create generation config with the default thinking budget."""
        return kwargs.pop(
            "config",
            types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1)
            ),
        )

    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert Gemini usage data to AIUsage."""
        if not usage_data:
            return None
        return AIUsage(
            input_tokens=getattr(usage_data, "prompt_token_count", 0),
            output_tokens=getattr(usage_data, "candidates_token_count", 0),
            total_tokens=getattr(usage_data, "total_token_count", 0),
        )

    async def generate_content(self, prompt: AIPrompt, **kwargs: Any) -> AIResponse:
        config = GeminiClient._get_generation_config(kwargs)

        response = await self.client.aio.models.generate_content(
            model=self.model_name, contents=prompt.content, config=config
        )

        # Extract usage information if available
        usage = None
        if hasattr(response, "usage_metadata"):
            usage = self.format_usage(response.usage_metadata)

        return AIResponse(
            content=response.text,
            usage=usage,
            provider=Provider.GEMINI,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: AIPrompt, **kwargs: Any
    ) -> AsyncIterator[AIResponse]:
        config = GeminiClient._get_generation_config(kwargs)

        last_usage_metadata = None
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model_name, contents=prompt.content, config=config
        ):
            if chunk.text:  # Only yield if there's actual content
                yield AIResponse(
                    content=chunk.text,
                    provider=Provider.GEMINI,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                last_usage_metadata = chunk.usage_metadata

        usage = self.format_usage(last_usage_metadata)
        if usage:
            yield AIResponse(
                content="",  # Empty content for the usage-only response
                usage=usage,
                provider=Provider.GEMINI,
                metadata={"model": self.model_name, "is_final_usage": True},
            )

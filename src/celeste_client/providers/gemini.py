from typing import Any, AsyncIterator, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel

from celeste_client.base import BaseClient
from celeste_client.core.config import GOOGLE_API_KEY
from celeste_client.core.enums import GeminiModel, Provider
from celeste_client.core.types import AIResponse, AIUsage


class GeminiClient(BaseClient):
    def __init__(self, model: str = GeminiModel.FLASH_LITE, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = model

    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert Gemini usage data to AIUsage."""
        if not usage_data:
            return None
        return AIUsage(
            input_tokens=getattr(usage_data, "prompt_token_count", 0),
            output_tokens=getattr(usage_data, "candidates_token_count", 0),
            total_tokens=getattr(usage_data, "total_token_count", 0),
        )

    async def generate_content(
        self, prompt: str, response_schema: Optional[BaseModel] = None, **kwargs: Any
    ) -> AIResponse:
        config = kwargs.pop("config", {})

        if response_schema is not None:
            if isinstance(config, dict):
                config["response_mime_type"] = "application/json"
                config["response_schema"] = response_schema

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config),
        )

        # Extract usage information if available
        usage = None
        if hasattr(response, "usage_metadata"):
            usage = self.format_usage(response.usage_metadata)

        # Return parsed content if using response_schema, otherwise return text
        content = (
            response.parsed
            if response_schema is not None and hasattr(response, "parsed")
            else response.text
        )

        return AIResponse(
            content=content,
            usage=usage,
            provider=Provider.GEMINI,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: str, response_schema: Optional[BaseModel] = None, **kwargs: Any
    ) -> AsyncIterator[AIResponse]:
        config = kwargs.pop("config", {})

        if response_schema is not None:
            if isinstance(config, dict):
                config["response_mime_type"] = "application/json"
                config["response_schema"] = response_schema

        last_usage_metadata = None
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config),
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

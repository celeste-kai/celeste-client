from typing import Any, AsyncIterator, Optional

from ollama import AsyncClient

from celeste_client.base import BaseClient
from celeste_client.core.config import OLLAMA_HOST
from celeste_client.core.enums import OllamaModel, Provider
from celeste_client.core.types import AIPrompt, AIResponse, AIUsage


class OllamaClient(BaseClient):
    def __init__(self, model: str = OllamaModel.LLAMA3_2.value, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.client = AsyncClient(host=OLLAMA_HOST)
        self.model_name = model

    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert Ollama usage data to AIUsage."""

        prompt_tokens = usage_data.get("prompt_eval_count", 0)
        completion_tokens = usage_data.get("eval_count", 0)
        return AIUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    async def generate_content(self, prompt: AIPrompt, **kwargs: Any) -> AIResponse:
        message = {"role": prompt.role.value, "content": prompt.content}
        # Ensure we get usage data by setting stream=False explicitly
        if "stream" not in kwargs:
            kwargs["stream"] = False

        response = await self.client.chat(
            model=self.model_name, messages=[message], **kwargs
        )

        # Extract usage information if available
        usage = self.format_usage(response)

        return AIResponse(
            content=response["message"]["content"],
            usage=usage,
            provider=Provider.OLLAMA,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: AIPrompt, **kwargs: Any
    ) -> AsyncIterator[AIResponse]:
        message = {"role": prompt.role.value, "content": prompt.content}
        stream = await self.client.chat(
            model=self.model_name, messages=[message], stream=True, **kwargs
        )
        async for chunk in stream:
            if not chunk.get("done"):
                yield AIResponse(
                    content=chunk["message"]["content"],
                    provider=Provider.OLLAMA,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )
            else:
                usage = self.format_usage(chunk)
                if usage:
                    yield AIResponse(
                        content="",
                        usage=usage,
                        provider=Provider.OLLAMA,
                        metadata={"model": self.model_name, "is_final_usage": True},
                    )

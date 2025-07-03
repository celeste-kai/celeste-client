from typing import Any, AsyncIterator, Optional

from huggingface_hub import InferenceClient

from celeste_client.base import BaseClient
from celeste_client.core.config import HUGGINGFACE_TOKEN
from celeste_client.core.enums import HuggingFaceModel, Provider
from celeste_client.core.types import AIResponse, AIUsage


class HuggingFaceClient(BaseClient):
    def __init__(
        self, model: str = HuggingFaceModel.GEMMA_2_2B.value, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.model_name = model
        # For direct model usage, we pass the model name to InferenceClient
        self.client = InferenceClient(
            model=self.model_name,
            token=HUGGINGFACE_TOKEN,
        )

    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert HuggingFace usage data to AIUsage."""
        if not usage_data:
            return None
        return AIUsage(
            input_tokens=getattr(usage_data, "prompt_tokens", 0),
            output_tokens=getattr(usage_data, "completion_tokens", 0),
            total_tokens=getattr(usage_data, "total_tokens", 0),
        )

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat_completion(messages=messages, **kwargs)

        return AIResponse(
            content=response.choices[0].message.content,
            usage=self.format_usage(response.usage),
            provider=Provider.HUGGINGFACE,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[AIResponse]:
        messages = [{"role": "user", "content": prompt}]

        # Try to enable usage information like OpenAI does
        if "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        # HuggingFace InferenceClient returns a generator for streaming
        stream = self.client.chat_completion(messages=messages, stream=True, **kwargs)

        usage_data = None
        for chunk in stream:
            # Yield content chunks
            if chunk.choices and chunk.choices[0].delta.content:
                yield AIResponse(
                    content=chunk.choices[0].delta.content,
                    provider=Provider.HUGGINGFACE,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )

            # Check for usage data in the chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = self.format_usage(chunk.usage)

        # Yield final usage data if available
        if usage_data:
            yield AIResponse(
                content="",
                usage=usage_data,
                provider=Provider.HUGGINGFACE,
                metadata={"model": self.model_name, "is_final_usage": True},
            )

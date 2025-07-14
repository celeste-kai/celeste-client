from typing import Any, AsyncIterator, List, Optional, get_origin

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, create_model

from celeste_client.base import BaseClient
from celeste_client.core.config import OPENAI_API_KEY
from celeste_client.core.enums import OpenAIModel, Provider
from celeste_client.core.types import AIResponse, AIUsage


class OpenAIClient(BaseClient):
    def __init__(
        self, model: str = OpenAIModel.GPT_O4_MINI.value, **kwargs: Any
    ) -> None:
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

    async def generate_content(
        self, prompt: str, response_schema: Optional[BaseModel] = None, **kwargs: Any
    ) -> AIResponse:
        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content=prompt)
        ]

        if response_schema is not None:
            # Handle list types by wrapping them in a Pydantic model
            if get_origin(response_schema) is list:
                # Create a dynamic model with the list type
                ListWrapper = create_model("ListWrapper", data=(response_schema, ...))

                response = await self.client.chat.completions.parse(
                    messages=messages,
                    model=self.model_name,
                    response_format=ListWrapper,
                    **kwargs,
                )
            else:
                response = await self.client.chat.completions.parse(
                    messages=messages,
                    model=self.model_name,
                    response_format=response_schema,
                    **kwargs,
                )
        else:
            response = await self.client.chat.completions.create(
                messages=messages, model=self.model_name, **kwargs
            )

        usage = self.format_usage(response.usage)

        # Return parsed content if using response_schema, otherwise return text
        if response_schema is not None and hasattr(
            response.choices[0].message, "parsed"
        ):
            parsed = response.choices[0].message.parsed
            # If we wrapped a list, extract the data field
            if get_origin(response_schema) is list and hasattr(parsed, "data"):
                content = parsed.data
            else:
                content = parsed
        else:
            content = response.choices[0].message.content or ""

        return AIResponse(
            content=content,
            usage=usage,
            provider=Provider.OPENAI,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: str, response_schema: Optional[BaseModel] = None, **kwargs: Any
    ) -> AsyncIterator[AIResponse]:
        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content=prompt)
        ]

        # For structured output in streaming, use the beta streaming API
        if response_schema is not None:
            # Handle list types by wrapping them in a Pydantic model
            actual_schema = response_schema
            is_list = get_origin(response_schema) is list

            if is_list:
                # Create a dynamic model with the list type
                ListWrapper = create_model("ListWrapper", data=(response_schema, ...))
                actual_schema = ListWrapper

            async with self.client.beta.chat.completions.stream(
                messages=messages,
                model=self.model_name,
                response_format=actual_schema,
                **kwargs,
            ) as stream:
                async for event in stream:
                    if event.type == "content.delta" and event.parsed is not None:
                        # Extract content from parsed data
                        content = (
                            event.parsed.data
                            if is_list and hasattr(event.parsed, "data")
                            else event.parsed
                        )
                        yield AIResponse(
                            content=content,
                            provider=Provider.OPENAI,
                            metadata={
                                "model": self.model_name,
                                "is_stream_chunk": True,
                            },
                        )

                # Get final completion for usage data
                final_completion = await stream.get_final_completion()
                usage = self.format_usage(final_completion.usage)
                if usage:
                    yield AIResponse(
                        content="",
                        usage=usage,
                        provider=Provider.OPENAI,
                        metadata={"model": self.model_name, "is_final_usage": True},
                    )
            return

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

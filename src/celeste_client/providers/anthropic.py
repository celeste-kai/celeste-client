import os
from typing import AsyncIterator

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from celeste_client.base import BaseClient
from celeste_client.core.enums import AnthropicModel

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class AnthropicClient(BaseClient):
    def __init__(self, model: str = AnthropicModel.CLAUDE_3_7_SONNET.value, **kwargs):
        super().__init__(**kwargs)

        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = model

    async def generate_content(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.pop("max_tokens", 1024)
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.content[0].text

    async def stream_generate_content(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[str]:
        max_tokens = kwargs.pop("max_tokens", 1024)
        async with self.client.messages.stream(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text

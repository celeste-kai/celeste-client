import os
from typing import AsyncIterator

from google import genai
from google.genai import types
from dotenv import load_dotenv

from celeste_client.base import BaseClient
from celeste_client.core.enums import GeminiModel

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class GeminiClient(BaseClient):
    def __init__(self, model: str = GeminiModel.FLASH_LITE.value, **kwargs):
        super().__init__(**kwargs)

        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = model

    async def generate_content(self, prompt: str, **kwargs) -> str:
        thinking_config = types.ThinkingConfig(thinking_budget=-1)
        config = kwargs.pop(
            "config", types.GenerateContentConfig(thinking_config=thinking_config)
        )

        response = await self.client.aio.models.generate_content(
            model=self.model_name, contents=prompt, config=config
        )
        return response.text

    async def stream_generate_content(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[str]:
        thinking_config = types.ThinkingConfig(thinking_budget=-1)
        config = kwargs.pop(
            "config", types.GenerateContentConfig(thinking_config=thinking_config)
        )

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model_name, contents=prompt, config=config
        ):
            yield chunk.text

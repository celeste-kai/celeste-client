import os
from typing import AsyncIterator

from mistralai import Mistral
from dotenv import load_dotenv

from celeste_client.base import BaseClient
from celeste_client.core.enums import MistralModel

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


class MistralClient(BaseClient):
    def __init__(self, model: str = MistralModel.SMALL_LATEST.value, **kwargs):
        super().__init__(**kwargs)

        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model_name = model

    async def generate_content(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.complete_async(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content

    async def stream_generate_content(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[str]:
        response = await self.client.chat.stream_async(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        async for chunk in response:
            if chunk.data.choices[0].delta.content:
                yield chunk.data.choices[0].delta.content

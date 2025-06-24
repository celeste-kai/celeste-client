import os
from typing import AsyncIterator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from dotenv import load_dotenv

from celeste_client.base import BaseClient
from celeste_client.core.enums import OpenAIModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class OpenAIClient(BaseClient):
    def __init__(self, model: str = OpenAIModel.GPT_O4_MINI, **kwargs):
        super().__init__(**kwargs)

        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model

    async def generate_content(self, prompt: str, **kwargs) -> str:
        message: ChatCompletionUserMessageParam = {"role": "user", "content": prompt}
        response = await self.client.chat.completions.create(
            messages=[message], model=self.model_name, **kwargs
        )
        return response.choices[0].message.content

    async def stream_generate_content(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[str]:
        message: ChatCompletionUserMessageParam = {"role": "user", "content": prompt}
        response = await self.client.chat.completions.create(
            messages=[message], model=self.model_name, stream=True, **kwargs
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

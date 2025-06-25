import os
from typing import AsyncIterator

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from celeste_client.base import BaseClient
from celeste_client.core.enums import HuggingFaceModel

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")


class HuggingFaceClient(BaseClient):
    def __init__(self, model: str = HuggingFaceModel.GEMMA_2_2B.value, **kwargs):
        super().__init__(**kwargs)

        self.model_name = model
        # For direct model usage, we pass the model name to InferenceClient
        self.client = InferenceClient(
            model=self.model_name,
            token=HUGGINGFACE_TOKEN,
        )

    async def generate_content(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat_completion(messages=messages, **kwargs)

        return response.choices[0].message.content

    async def stream_generate_content(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]

        # HuggingFace InferenceClient returns a generator for streaming
        stream = self.client.chat_completion(messages=messages, stream=True, **kwargs)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

import os
from typing import AsyncIterator

from ollama import AsyncClient
from dotenv import load_dotenv

from celeste_client.base import BaseClient
from celeste_client.core.enums import OllamaModel

load_dotenv()

# Ollama runs locally, no API key needed
# Default host is http://localhost:11434
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class OllamaClient(BaseClient):
    def __init__(self, model: str = OllamaModel.LLAMA3_2.value, **kwargs):
        super().__init__(**kwargs)

        # Use provided host or fall back to environment variable or default
        self.host = OLLAMA_HOST
        self.client = AsyncClient(host=self.host)
        self.model_name = model

    async def generate_content(self, prompt: str, **kwargs) -> str:
        message = {"role": "user", "content": prompt}
        response = await self.client.chat(
            model=self.model_name, messages=[message], **kwargs
        )
        return response["message"]["content"]

    async def stream_generate_content(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[str]:
        message = {"role": "user", "content": prompt}
        stream = await self.client.chat(
            model=self.model_name, messages=[message], stream=True, **kwargs
        )
        async for chunk in stream:
            if chunk["message"]["content"]:
                yield chunk["message"]["content"]

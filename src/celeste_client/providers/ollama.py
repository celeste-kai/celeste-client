from typing import Any, AsyncIterator

from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from celeste_core.config.settings import settings


class OllamaClient(BaseClient):
    def __init__(self, model: str = "llama3.2:latest", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.OLLAMA, **kwargs)
        try:
            from ollama import AsyncClient  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Ollama provider requires optional dependency 'ollama'.\n"
                "Install with: pip install 'celeste-client[ollama]'"
            ) from e
        self.client = AsyncClient(host=settings.ollama.host)

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        message = {"role": "user", "content": prompt}

        response = await self.client.chat(
            model=self.model_name, messages=[message], **kwargs
        )

        return AIResponse(
            content=response["message"]["content"],
            provider=Provider.OLLAMA,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(self, prompt: str, **kwargs: Any) -> AsyncIterator[AIResponse]:
        message = {"role": "user", "content": prompt}
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
                break

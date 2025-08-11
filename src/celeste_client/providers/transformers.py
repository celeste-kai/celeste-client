import asyncio
from threading import Thread
from typing import Any, AsyncIterator

import torch
from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from transformers import AsyncTextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer


class TransformersClient(BaseClient):
    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def __init__(self, model: str = "sshleifer/tiny-gpt2", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.TRANSFORMERS, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = self._resolve_device()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        ).to(self.device)

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        input_kwargs: dict[str, Any] = dict(
            self.tokenizer(prompt, return_tensors="pt").to(self.device)
        )
        max_new_tokens: int = int(kwargs.pop("max_new_tokens", 256))
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, **kwargs}
        out = await asyncio.to_thread(self.model.generate, **input_kwargs, **gen_kwargs)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return AIResponse(
            content=text,
            provider=Provider.TRANSFORMERS,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[AIResponse]:
        input_kwargs: dict[str, Any] = dict(
            self.tokenizer([prompt], return_tensors="pt").to(self.device)
        )
        streamer = AsyncTextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        max_new_tokens: int = int(kwargs.pop("max_new_tokens", 256))
        gen_kwargs: dict[str, Any] = {
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            **kwargs,
            **input_kwargs,
        }
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        async for tok in streamer:
            if tok:
                yield AIResponse(
                    content=tok,
                    provider=Provider.TRANSFORMERS,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )
        thread.join()

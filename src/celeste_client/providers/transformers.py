import asyncio
from collections.abc import AsyncIterator
from threading import Thread
from typing import Any

import torch
from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from transformers import AsyncTextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer


class TransformersClient(BaseClient):
    # Keep class minimal; rely on HF/Accelerate device_map for placement

    def __init__(self, model: str = "sshleifer/tiny-gpt2", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.TRANSFORMERS, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }

        self.pretrained_model = AutoModelForCausalLM.from_pretrained(self.model, **load_kwargs)
        # Determine an input device compatible with the loaded model
        try:
            self.input_device = next(self.pretrained_model.parameters()).device
        except StopIteration:
            self.input_device = torch.device("cpu")
        self.pretrained_model.eval()

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        input_kwargs: dict[str, Any] = dict(self.tokenizer(prompt, return_tensors="pt").to(self.input_device))
        max_new_tokens: int = int(kwargs.pop("max_new_tokens", 256))
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, **kwargs}
        out = await asyncio.to_thread(self.pretrained_model.generate, **input_kwargs, **gen_kwargs)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return AIResponse(
            content=text,
            provider=Provider.TRANSFORMERS,
            metadata={"model": self.model},
        )

    async def stream_generate_content(self, prompt: str, **kwargs: Any) -> AsyncIterator[AIResponse]:
        input_kwargs: dict[str, Any] = dict(self.tokenizer([prompt], return_tensors="pt").to(self.input_device))
        streamer = AsyncTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        max_new_tokens: int = int(kwargs.pop("max_new_tokens", 256))
        gen_kwargs: dict[str, Any] = {
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            **kwargs,
            **input_kwargs,
        }
        thread = Thread(target=self.pretrained_model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        async for tok in streamer:
            if tok:
                yield AIResponse(
                    content=tok,
                    provider=Provider.TRANSFORMERS,
                    metadata={"model": self.model, "is_stream_chunk": True},
                )
        thread.join()

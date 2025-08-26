import asyncio
from threading import Thread
from typing import Any, AsyncIterator

from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient


class TransformersClient(BaseClient):
    # Keep class minimal; rely on HF/Accelerate device_map for placement

    def __init__(self, model: str = "sshleifer/tiny-gpt2", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.TRANSFORMERS, **kwargs)
        self.tokenizer = None  # type: ignore[assignment]
        self.model = None  # type: ignore[assignment]
        self.input_device = None  # type: ignore[assignment]

    def _ensure_model_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None and self.input_device is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import (
                AsyncTextIteratorStreamer,
                AutoModelForCausalLM,
                AutoTokenizer,
            )  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Transformers provider requires optional dependencies 'transformers' and 'torch'.\n"
                "Install with: pip install 'celeste-client[transformers]'"
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, local_files_only=True
        )

        load_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        # Determine an input device compatible with the loaded model
        try:
            self.input_device = next(self.model.parameters()).device  # type: ignore[attr-defined]
        except StopIteration:
            self.input_device = torch.device("cpu")  # type: ignore[name-defined]
        self.model.eval()  # type: ignore[union-attr]
        self._AsyncTextIteratorStreamer = AsyncTextIteratorStreamer

    async def generate_content(self, prompt: str, **kwargs: Any) -> AIResponse:
        self._ensure_model_loaded()
        input_kwargs: dict[str, Any] = dict(
            self.tokenizer(prompt, return_tensors="pt").to(self.input_device)  # type: ignore[union-attr]
        )
        max_new_tokens: int = int(kwargs.pop("max_new_tokens", 256))
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, **kwargs}
        out = await asyncio.to_thread(self.model.generate, **input_kwargs, **gen_kwargs)  # type: ignore[union-attr]
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore[union-attr]
        return AIResponse(
            content=text,
            provider=Provider.TRANSFORMERS,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(self, prompt: str, **kwargs: Any) -> AsyncIterator[AIResponse]:
        self._ensure_model_loaded()
        input_kwargs: dict[str, Any] = dict(
            self.tokenizer([prompt], return_tensors="pt").to(self.input_device)  # type: ignore[union-attr]
        )
        streamer = self._AsyncTextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True  # type: ignore[union-attr]
        )
        max_new_tokens: int = int(kwargs.pop("max_new_tokens", 256))
        gen_kwargs: dict[str, Any] = {
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            **kwargs,
            **input_kwargs,
        }
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)  # type: ignore[union-attr]
        thread.start()
        async for tok in streamer:
            if tok:
                yield AIResponse(
                    content=tok,
                    provider=Provider.TRANSFORMERS,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )
        thread.join()

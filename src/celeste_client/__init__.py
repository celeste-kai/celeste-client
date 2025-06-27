"""
Celeste AI Client - Minimal predefinition AI communication for Alita agents.
"""

from typing import Any

from .base import BaseClient
from .core import AIPrompt, AIResponse, LogLevel, MessageRole, Provider

__version__ = "0.1.0"

SUPPORTED_PROVIDERS = [
    "gemini",
    "openai",
    "mistral",
    "anthropic",
    "huggingface",
    "ollama",
]


def create_client(provider: str, **kwargs: Any) -> BaseClient:
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")

    if provider == "gemini":
        from .providers.gemini import GeminiClient

        return GeminiClient(**kwargs)
    if provider == "openai":
        from .providers.openai import OpenAIClient

        return OpenAIClient(**kwargs)
    if provider == "mistral":
        from .providers.mistral import MistralClient

        return MistralClient(**kwargs)
    if provider == "anthropic":
        from .providers.anthropic import AnthropicClient

        return AnthropicClient(**kwargs)
    if provider == "huggingface":
        from .providers.huggingface import HuggingFaceClient

        return HuggingFaceClient(**kwargs)
    if provider == "ollama":
        from .providers.ollama import OllamaClient

        return OllamaClient(**kwargs)

    raise ValueError(f"Provider {provider} not implemented")


__all__ = [
    "create_client",
    "BaseClient",
    "Provider",
    "MessageRole",
    "LogLevel",
    "AIPrompt",
    "AIResponse",
]

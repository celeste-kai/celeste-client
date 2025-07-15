"""
Celeste AI Client - Minimal predefinition AI communication for Alita agents.
"""

from typing import Any, Union

from .base import BaseAIClient
from .core import AIPrompt, AIProvider, AIResponse, LogLevel, MessageRole

__version__ = "0.1.0"


def create_client(provider: Union[AIProvider, str], **kwargs: Any) -> BaseAIClient:
    if isinstance(provider, str):
        provider = AIProvider(provider)

    if provider == AIProvider.GOOGLE:
        from .providers.google import GoogleClient

        return GoogleClient(**kwargs)

    if provider == AIProvider.OPENAI:
        from .providers.openai import OpenAIClient

        return OpenAIClient(**kwargs)

    if provider == AIProvider.MISTRAL:
        from .providers.mistral import MistralClient

        return MistralClient(**kwargs)

    if provider == AIProvider.ANTHROPIC:
        from .providers.anthropic import AnthropicClient

        return AnthropicClient(**kwargs)

    if provider == AIProvider.HUGGINGFACE:
        from .providers.huggingface import HuggingFaceClient

        return HuggingFaceClient(**kwargs)

    if provider == AIProvider.OLLAMA:
        from .providers.ollama import OllamaClient

        return OllamaClient(**kwargs)

    raise ValueError(f"AIProvider {provider} not implemented")


__all__ = [
    "create_client",
    "BaseAIClient",
    "AIProvider",
    "MessageRole",
    "LogLevel",
    "AIPrompt",
    "AIResponse",
]

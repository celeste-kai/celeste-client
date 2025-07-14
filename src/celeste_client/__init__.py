"""
Celeste AI Client - Minimal predefinition AI communication for Alita agents.
"""

from typing import Any

from .base import BaseClient
from .core import AIPrompt, AIResponse, LogLevel, MessageRole, Provider

__version__ = "0.1.0"


def create_client(provider: Provider, **kwargs: Any) -> BaseClient:
    if provider == Provider.GEMINI:
        from .providers.gemini import GeminiClient

        return GeminiClient(**kwargs)

    if provider == Provider.OPENAI:
        from .providers.openai import OpenAIClient

        return OpenAIClient(**kwargs)

    if provider == Provider.MISTRAL:
        from .providers.mistral import MistralClient

        return MistralClient(**kwargs)

    if provider == Provider.ANTHROPIC:
        from .providers.anthropic import AnthropicClient

        return AnthropicClient(**kwargs)

    if provider == Provider.HUGGINGFACE:
        from .providers.huggingface import HuggingFaceClient

        return HuggingFaceClient(**kwargs)

    if provider == Provider.OLLAMA:
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

"""
Celeste AI Client - Minimal predefinition AI communication for Alita agents.
"""

from .base import BaseClient
from .core import Provider, MessageRole, LogLevel, AIPrompt, AIResponse

__version__ = "0.1.0"

SUPPORTED_PROVIDERS = ["gemini"]


def create_client(provider: str, **kwargs) -> BaseClient:
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")

    if provider == "gemini":
        from .providers.gemini import GeminiClient

        return GeminiClient(**kwargs)

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

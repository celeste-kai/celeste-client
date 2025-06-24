"""
Core enumerations for Celeste AI Client.
"""

from enum import Enum


class Provider(Enum):
    """AI provider enumeration for multi-provider agent support."""

    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"


class MessageRole(Enum):
    """Message role definitions for structured agent communication."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class LogLevel(Enum):
    """Logging level enumeration for agent operations."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class GeminiModel(Enum):
    """Gemini 2.5 model enumeration for provider-specific model selection."""

    FLASH_LITE = "gemini-2.5-flash-lite-preview-06-17"
    FLASH = "gemini-2.5-flash"
    PRO = "gemini-2.5-pro"

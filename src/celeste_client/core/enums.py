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


class OpenAIModel(Enum):
    """OpenAI model enumeration for provider-specific model selection."""

    GPT_O3 = "o3-2025-04-16"
    GPT_O4_MINI = "o4-mini-2025-04-16"
    GPT_4_1 = "gpt-4.1-2025-04-14"


class MistralModel(Enum):
    """Mistral AI model enumeration for provider-specific model selection."""

    SMALL_LATEST = "mistral-small-latest"
    MEDIUM_LATEST = "mistral-medium-latest"
    LARGE_LATEST = "mistral-large-latest"
    CODESTRAL_LATEST = "codestral-latest"


class AnthropicModel(Enum):
    """Anthropic Claude model enumeration for provider-specific model selection."""

    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_4_OPUS = "claude-opus-4-20250514"

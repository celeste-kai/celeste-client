from celeste_core.enums.capability import Capability
from celeste_core.enums.providers import Provider

# Capability for this domain package
CAPABILITY: Capability = Capability.TEXT_GENERATION

# Provider wiring for text generation clients
PROVIDER_MAPPING: dict[Provider, tuple[str, str]] = {
    Provider.GOOGLE: (".providers.google", "GoogleClient"),
    Provider.OPENAI: (".providers.openai", "OpenAIClient"),
    Provider.MISTRAL: (".providers.mistral", "MistralClient"),
    Provider.ANTHROPIC: (".providers.anthropic", "AnthropicClient"),
    Provider.HUGGINGFACE: (".providers.huggingface", "HuggingFaceClient"),
    Provider.OLLAMA: (".providers.ollama", "OllamaClient"),
    Provider.TRANSFORMERS: (".providers.transformers", "TransformersClient"),
}

__all__ = ["CAPABILITY", "PROVIDER_MAPPING"]

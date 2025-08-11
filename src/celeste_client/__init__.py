from typing import Any, Union

from celeste_core import AIResponse, Provider
from celeste_core.base.client import BaseClient
from celeste_core.config.settings import settings

__version__ = "0.1.0"

SUPPORTED_PROVIDERS: set[Provider] = {
    Provider.GOOGLE,
    Provider.OPENAI,
    Provider.MISTRAL,
    Provider.ANTHROPIC,
    Provider.HUGGINGFACE,
    Provider.OLLAMA,
    Provider.TRANSFORMERS,
}


def create_client(provider: Union[Provider, str], **kwargs: Any) -> BaseClient:
    if isinstance(provider, str):
        provider = Provider(provider)

    if provider not in SUPPORTED_PROVIDERS:
        supported = [p.value for p in SUPPORTED_PROVIDERS]
        raise ValueError(
            f"Unsupported provider: {provider.value}. Supported: {supported}"
        )

    # Validate environment for the chosen provider
    settings.validate_for_provider(provider.value)

    provider_mapping = {
        Provider.GOOGLE: (".providers.google", "GoogleClient"),
        Provider.OPENAI: (".providers.openai", "OpenAIClient"),
        Provider.MISTRAL: (".providers.mistral", "MistralClient"),
        Provider.ANTHROPIC: (".providers.anthropic", "AnthropicClient"),
        Provider.HUGGINGFACE: (".providers.huggingface", "HuggingFaceClient"),
        Provider.OLLAMA: (".providers.ollama", "OllamaClient"),
        Provider.TRANSFORMERS: (".providers.transformers", "TransformersClient"),
    }

    module_path, class_name = provider_mapping[provider]
    module = __import__(f"celeste_client{module_path}", fromlist=[class_name])
    client_class = getattr(module, class_name)
    return client_class(**kwargs)


__all__ = [
    "create_client",
    "BaseClient",
    "Provider",
    "AIResponse",
]

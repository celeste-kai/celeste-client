import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import types
import pytest

from celeste_client import create_client, Provider


class DummyClient:
    async def generate_content(self, *args, **kwargs):
        return None

    async def stream_generate_content(self, *args, **kwargs):
        if False:
            yield None

    def format_usage(self, usage_data):
        return None


@pytest.fixture(autouse=True)
def stub_providers(monkeypatch):
    provider_map = {
        Provider.GOOGLE: ("celeste_client.providers.google", "GoogleClient"),
        Provider.OPENAI: ("celeste_client.providers.openai", "OpenAIClient"),
        Provider.MISTRAL: ("celeste_client.providers.mistral", "MistralClient"),
        Provider.ANTHROPIC: (
            "celeste_client.providers.anthropic",
            "AnthropicClient",
        ),
        Provider.HUGGINGFACE: (
            "celeste_client.providers.huggingface",
            "HuggingFaceClient",
        ),
        Provider.OLLAMA: ("celeste_client.providers.ollama", "OllamaClient"),
    }
    classes = {}

    async def async_noop(*args, **kwargs):
        return None

    async def async_gen_noop(*args, **kwargs):
        if False:
            yield None

    for provider, (module_path, class_name) in provider_map.items():
        module = types.ModuleType(module_path)
        attrs = {
            "__init__": lambda self, **kwargs: None,
            "generate_content": async_noop,
            "stream_generate_content": async_gen_noop,
            "format_usage": lambda self, usage: None,
        }
        Dummy = type(class_name, (DummyClient,), attrs)
        module.__dict__[class_name] = Dummy
        monkeypatch.setitem(sys.modules, module_path, module)
        classes[provider] = Dummy

    return classes


@pytest.mark.parametrize("provider", list(Provider))
def test_create_client_returns_correct_type(provider, stub_providers):
    expected_cls = stub_providers[provider]
    client = create_client(provider)
    assert type(client) is expected_cls

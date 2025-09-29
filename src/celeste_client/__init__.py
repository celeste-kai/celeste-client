from importlib import import_module
from typing import Any

from celeste_core import Provider
from celeste_core.base.client import BaseClient
from celeste_core.config.settings import settings

from .mapping import PROVIDER_MAPPING

OPTIONAL_PROVIDER_EXTRAS: dict[Provider, tuple[str, tuple[str, ...]]] = {
    Provider.TRANSFORMERS: (
        "local",
        ("accelerate", "torch", "transformers"),
    ),
}

__version__ = "0.1.0"


def create_client(provider: Provider | str, **kwargs: Any) -> BaseClient:
    prov = Provider(provider) if isinstance(provider, str) else provider
    if prov not in PROVIDER_MAPPING:
        raise ValueError(f"Provider '{prov.value}' is not wired for text generation.")

    settings.validate_for_provider(prov.value)
    module_path, class_name = PROVIDER_MAPPING[prov]
    try:
        module = import_module(f"celeste_client{module_path}")
    except ModuleNotFoundError as exc:
        extra_info = OPTIONAL_PROVIDER_EXTRAS.get(prov)
        if extra_info is not None:
            extra_name, modules = extra_info
            if exc.name in modules:
                raise ModuleNotFoundError(
                    (
                        f"Provider '{prov.value}' requires the optional dependency "
                        f"'{exc.name}'. Install it with 'pip install celeste-client[{extra_name}]'."
                    )
                ) from exc
        raise
    return getattr(module, class_name)(**kwargs)


__all__ = ["BaseClient", "create_client"]

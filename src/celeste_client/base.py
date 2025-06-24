from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseClient(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the client, loading credentials from the environment.
        Provider-specific arguments can be passed via kwargs.
        """
        pass

    @abstractmethod
    async def generate_content(self, prompt: str, **kwargs) -> str:
        """Generates a single response."""
        pass

    @abstractmethod
    async def stream_generate_content(
        self, prompt: str, **kwargs
    ) -> AsyncIterator[str]:
        """Streams the response chunk by chunk."""
        pass

"""
Core data types for agent communication.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .enums import MessageRole, Provider


@dataclass(frozen=True)
class AIPrompt:
    """Prompt for agent-AI communication."""

    role: MessageRole
    content: str
    mcp_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AIResponse:
    """Response from AI providers."""

    content: str
    usage: Optional[Dict[str, int]] = None
    provider: Optional[Provider] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

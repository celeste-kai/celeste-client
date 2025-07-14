"""
Core data types for agent communication.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from .enums import MessageRole, Provider


class AIPrompt(BaseModel):
    """Prompt for agent-AI communication."""

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str
    mcp_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}


class AIUsage(BaseModel):
    """Token usage metrics for AI responses."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    total_tokens: int


class AIResponse(BaseModel):
    """Response from AI providers."""

    model_config = ConfigDict(frozen=True)

    content: Union[str, BaseModel, List[BaseModel]]
    usage: Optional[AIUsage] = None
    provider: Optional[Provider] = None
    metadata: Dict[str, Any] = {}

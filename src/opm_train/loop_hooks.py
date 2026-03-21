"""Hook contracts for customizing the agent loop without forking core logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opm_train.models import AgentNode


@dataclass(slots=True)
class LoopContext:
    """Immutable per-step context shared with loop hooks."""

    session_id: str
    agent_id: str
    step: int


class LoopHooks:
    """Extensible policy hooks for overriding loop behavior safely."""

    async def before_step(self, *, agent: AgentNode, context: LoopContext) -> None:
        """Run before each model step starts."""
        pass

    async def after_model_response(
        self,
        *,
        agent: AgentNode,
        context: LoopContext,
        actions: list[dict[str, Any]],
    ) -> None:
        """Run after model output is parsed into actions."""
        pass

    async def before_action(
        self,
        *,
        agent: AgentNode,
        context: LoopContext,
        action: dict[str, Any],
    ) -> None:
        """Run before one action execution."""
        pass

    async def after_action(
        self,
        *,
        agent: AgentNode,
        context: LoopContext,
        action: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Run after one action execution."""
        pass

    async def on_step_error(
        self,
        *,
        agent: AgentNode,
        context: LoopContext,
        error: Exception,
    ) -> None:
        """Run when a loop step raises an error."""
        pass


class DefaultLoopHooks(LoopHooks):
    """Default no-op hooks implementation."""

    pass

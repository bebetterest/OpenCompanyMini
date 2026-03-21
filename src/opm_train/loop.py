"""Role-agnostic think-act loop runner used by root and worker agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from opm_train.loop_hooks import LoopContext, LoopHooks
from opm_train.models import AgentNode


AskAgentFn = Callable[[AgentNode], Awaitable[list[dict[str, Any]]]]
ExecuteActionsFn = Callable[[AgentNode, list[dict[str, Any]], LoopContext], Awaitable["ActionBatchResult"]]
RequestForcedFinishFn = Callable[[AgentNode], Awaitable[dict[str, Any] | None]]
InterruptedFn = Callable[[], bool]


@dataclass(slots=True)
class ActionBatchResult:
    """Result container for one parsed action batch."""

    finish_payload: dict[str, Any] | None = None


@dataclass(slots=True)
class AgentLoopResult:
    """Terminal loop outcome used by orchestrator control flow."""

    finish_payload: dict[str, Any] | None
    interrupted: bool = False
    step_limit_reached: bool = False


class AgentLoopRunner:
    """Runs one reusable think-act-feedback loop for any agent role."""

    def __init__(self, *, max_steps: int, hooks: LoopHooks) -> None:
        """Create a runner with explicit step budget and hook policy."""
        self.max_steps = max(1, int(max_steps))
        self.hooks = hooks

    async def run(
        self,
        *,
        agent: AgentNode,
        ask_agent: AskAgentFn,
        execute_actions: ExecuteActionsFn,
        request_forced_finish: RequestForcedFinishFn,
        interrupted: InterruptedFn,
    ) -> AgentLoopResult:
        """Execute loop steps until finish payload, interruption, or budget exhaustion."""
        for _ in range(self.max_steps):
            next_step = max(0, int(agent.step_count)) + 1
            agent.step_count = next_step
            context = LoopContext(
                session_id=agent.session_id,
                agent_id=agent.id,
                step=next_step,
            )
            await self.hooks.before_step(agent=agent, context=context)
            if interrupted():
                return AgentLoopResult(finish_payload=None, interrupted=True)
            try:
                actions = await ask_agent(agent)
                await self.hooks.after_model_response(
                    agent=agent,
                    context=context,
                    actions=actions,
                )
                if interrupted():
                    return AgentLoopResult(finish_payload=None, interrupted=True)
                result = await execute_actions(agent, actions, context)
            except Exception as exc:
                await self.hooks.on_step_error(agent=agent, context=context, error=exc)
                raise
            if result.finish_payload is not None:
                return AgentLoopResult(finish_payload=result.finish_payload)
            await asyncio.sleep(0)
        forced_finish = await request_forced_finish(agent)
        return AgentLoopResult(finish_payload=forced_finish, step_limit_reached=True)

from __future__ import annotations

import asyncio

from opm_train.loop import ActionBatchResult, AgentLoopRunner
from opm_train.loop_hooks import LoopContext, LoopHooks
from opm_train.models import AgentNode, AgentRole


class RecordingHooks(LoopHooks):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def before_step(self, *, agent: AgentNode, context: LoopContext) -> None:
        self.calls.append(f"before_step:{context.step}")

    async def after_model_response(self, *, agent: AgentNode, context: LoopContext, actions):
        self.calls.append(f"after_model:{len(actions)}")

    async def before_action(self, *, agent: AgentNode, context: LoopContext, action):
        self.calls.append(f"before_action:{action['type']}")

    async def after_action(self, *, agent: AgentNode, context: LoopContext, action, result):
        self.calls.append(f"after_action:{action['type']}")


async def _run_loop() -> RecordingHooks:
    hooks = RecordingHooks()
    runner = AgentLoopRunner(max_steps=3, hooks=hooks)
    agent = AgentNode(
        id="agent-1",
        session_id="session-1",
        name="root",
        role=AgentRole.ROOT,
        instruction="test",
        workspace_path=__import__("pathlib").Path("."),
    )

    async def ask(_agent: AgentNode):
        return [{"type": "finish", "status": "completed", "summary": "done"}]

    async def execute(_agent: AgentNode, actions, _context):
        for action in actions:
            await hooks.before_action(agent=_agent, context=_context, action=action)
            await hooks.after_action(agent=_agent, context=_context, action=action, result={})
        return ActionBatchResult(finish_payload={"status": "completed", "summary": "done"})

    async def forced(_agent: AgentNode):
        return {"status": "partial", "summary": "forced"}

    result = await runner.run(
        agent=agent,
        ask_agent=ask,
        execute_actions=execute,
        request_forced_finish=forced,
        interrupted=lambda: False,
    )
    assert result.finish_payload is not None
    return hooks


def test_loop_hooks_are_invoked() -> None:
    hooks = asyncio.run(_run_loop())
    assert hooks.calls[0] == "before_step:1"
    assert "after_model:1" in hooks.calls
    assert "before_action:finish" in hooks.calls
    assert "after_action:finish" in hooks.calls

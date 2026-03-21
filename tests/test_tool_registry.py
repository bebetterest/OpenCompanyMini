from __future__ import annotations

from opm_train.orchestrator_tools.registry import SELF_COMPLETING_TOOL_ACTIONS, TOOL_REGISTRY


def test_tool_registry_contains_canonical_runtime_tools() -> None:
    expected = {
        "shell",
        "spawn_agent",
        "steer_agent",
        "cancel_agent",
        "list_agent_runs",
        "get_agent_run",
        "list_tool_runs",
        "get_tool_run",
        "wait_run",
        "cancel_tool_run",
        "wait_time",
        "compress_context",
    }
    assert set(TOOL_REGISTRY.keys()) == expected


def test_tool_registry_default_blocking_and_self_completing_contract() -> None:
    assert TOOL_REGISTRY["spawn_agent"].default_blocking is False
    assert TOOL_REGISTRY["shell"].default_blocking is True
    assert SELF_COMPLETING_TOOL_ACTIONS == {"spawn_agent", "shell"}


def test_tool_registry_executor_argument_binding() -> None:
    run = object()
    agent = object()
    action = {"type": "test"}

    calls: list[tuple[str, tuple[object, ...]]] = []

    class RuntimeStub:
        def _tool_shell(self, run_obj: object, action_obj: object) -> dict[str, str]:
            calls.append(("shell", (run_obj, action_obj)))
            return {"kind": "shell"}

        def _tool_spawn_agent(self, run_obj: object, agent_obj: object, action_obj: object) -> dict[str, str]:
            calls.append(("spawn_agent", (run_obj, agent_obj, action_obj)))
            return {"kind": "spawn_agent"}

        def _tool_wait_time(self, action_obj: object) -> dict[str, str]:
            calls.append(("wait_time", (action_obj,)))
            return {"kind": "wait_time"}

        def _tool_compress_context(self, agent_obj: object) -> dict[str, str]:
            calls.append(("compress_context", (agent_obj,)))
            return {"kind": "compress_context"}

    runtime = RuntimeStub()

    assert TOOL_REGISTRY["shell"].executor(runtime, run, agent, action) == {"kind": "shell"}
    assert TOOL_REGISTRY["spawn_agent"].executor(runtime, run, agent, action) == {"kind": "spawn_agent"}
    assert TOOL_REGISTRY["wait_time"].executor(runtime, run, agent, action) == {"kind": "wait_time"}
    assert TOOL_REGISTRY["compress_context"].executor(runtime, run, agent, action) == {"kind": "compress_context"}
    assert calls == [
        ("shell", (run, action)),
        ("spawn_agent", (run, agent, action)),
        ("wait_time", (action,)),
        ("compress_context", (agent,)),
    ]

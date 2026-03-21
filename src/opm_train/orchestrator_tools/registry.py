"""Central tool registration for runtime orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

ToolExecutor = Callable[[Any, Any, Any, dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One registered runtime tool with dispatch and execution semantics."""

    name: str
    executor: ToolExecutor
    default_blocking: bool = True
    self_completing: bool = False


def _method_executor(
    method_name: str,
    *,
    include_run: bool = False,
    include_agent: bool = False,
    include_action: bool = False,
) -> ToolExecutor:
    """Build a runtime tool executor from a target method name and arg mask."""

    def execute(runtime: Any, run_obj: Any, agent_obj: Any, action_obj: dict[str, Any]) -> Any:
        method = getattr(runtime, method_name)
        args: list[Any] = []
        if include_run:
            args.append(run_obj)
        if include_agent:
            args.append(agent_obj)
        if include_action:
            args.append(action_obj)
        return method(*args)

    return execute


def _default_specs() -> list[ToolSpec]:
    """Return canonical runtime tool registrations."""
    return [
        ToolSpec(
            name="shell",
            executor=_method_executor("_tool_shell", include_run=True, include_action=True),
            default_blocking=True,
            self_completing=True,
        ),
        ToolSpec(
            name="spawn_agent",
            executor=_method_executor(
                "_tool_spawn_agent",
                include_run=True,
                include_agent=True,
                include_action=True,
            ),
            default_blocking=False,
            self_completing=True,
        ),
        ToolSpec(
            name="steer_agent",
            executor=_method_executor("_tool_steer_agent", include_run=True, include_action=True),
        ),
        ToolSpec(
            name="cancel_agent",
            executor=_method_executor("_tool_cancel_agent", include_run=True, include_action=True),
        ),
        ToolSpec(
            name="list_agent_runs",
            executor=_method_executor("_tool_list_agent_runs", include_action=True),
        ),
        ToolSpec(
            name="get_agent_run",
            executor=_method_executor("_tool_get_agent_run", include_action=True),
        ),
        ToolSpec(
            name="list_tool_runs",
            executor=_method_executor("_tool_list_tool_runs", include_action=True),
        ),
        ToolSpec(
            name="get_tool_run",
            executor=_method_executor("_tool_get_tool_run", include_action=True),
        ),
        ToolSpec(
            name="wait_run",
            executor=_method_executor("_tool_wait_run", include_action=True),
        ),
        ToolSpec(
            name="cancel_tool_run",
            executor=_method_executor("_tool_cancel_tool_run", include_action=True),
        ),
        ToolSpec(
            name="wait_time",
            executor=_method_executor("_tool_wait_time", include_action=True),
        ),
        ToolSpec(
            name="compress_context",
            executor=_method_executor("_tool_compress_context", include_agent=True),
        ),
    ]


def build_tool_registry(specs: list[ToolSpec]) -> dict[str, ToolSpec]:
    """Build validated registry from tool specs."""
    registry: dict[str, ToolSpec] = {}
    for spec in specs:
        if spec.name in registry:
            raise ValueError(f"duplicate tool registration: {spec.name}")
        registry[spec.name] = spec
    return registry


TOOL_REGISTRY: dict[str, ToolSpec] = build_tool_registry(_default_specs())
SELF_COMPLETING_TOOL_ACTIONS: frozenset[str] = frozenset(
    spec.name for spec in TOOL_REGISTRY.values() if spec.self_completing
)

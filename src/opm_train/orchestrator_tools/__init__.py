"""Tool execution mixin for runtime orchestrator."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from opm_train.models import AgentNode, ToolRun, ToolRunStatus
from opm_train.utils import json_ready, utc_now

from .agent_ops import AgentToolMixin
from .query_ops import QueryToolMixin
from .registry import SELF_COMPLETING_TOOL_ACTIONS, TOOL_REGISTRY
from .shell import ShellToolMixin


class OrchestratorToolingMixin(AgentToolMixin, QueryToolMixin, ShellToolMixin):
    """Attach tool-run lifecycle and tool handler implementations."""

    async def _dispatch_tool_action(
        self,
        *,
        action_type: str,
        agent: AgentNode,
        run: ToolRun,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch one non-finish action through central tool registry."""
        spec = TOOL_REGISTRY.get(action_type)
        if spec is None:
            raise ValueError(f"unsupported action type: {action_type}")
        value = spec.executor(self, run, agent, action)
        if inspect.isawaitable(value):
            return await value
        return value

    def _create_tool_run(self, *, agent: AgentNode, tool_name: str, arguments: dict[str, Any], blocking: bool) -> ToolRun:
        """Create tool-run record and initialize waiter bookkeeping."""
        run = ToolRun(
            id=self._new_id("tool"),
            session_id=agent.session_id,
            agent_id=agent.id,
            tool_name=tool_name,
            arguments=dict(arguments),
            status=ToolRunStatus.QUEUED,
            blocking=blocking,
            created_at=utc_now(),
        )
        self.tool_runs[run.id] = run
        self.tool_run_events.setdefault(run.id, asyncio.Event())
        self._log_event(agent, "tool_run_submitted", {"tool_run_id": run.id, "tool_name": tool_name})
        return run

    def _mark_tool_run_running(self, run: ToolRun) -> None:
        """Mark tool run as running and stamp start time."""
        run.status = ToolRunStatus.RUNNING
        run.started_at = utc_now()

    def _complete_tool_run(self, run: ToolRun, *, result: dict[str, Any]) -> None:
        """Mark tool run completed and publish result to waiters."""
        self._set_tool_run_terminal(
            run,
            status=ToolRunStatus.COMPLETED,
            result=result,
        )

    def _fail_tool_run(self, run: ToolRun, *, error: str) -> None:
        """Mark tool run failed and notify waiters."""
        self._set_tool_run_terminal(
            run,
            status=ToolRunStatus.FAILED,
            error=error,
        )

    def _cancel_tool_run_obj(self, run: ToolRun, *, reason: str) -> None:
        """Mark tool run cancelled and notify waiters."""
        self._set_tool_run_terminal(
            run,
            status=ToolRunStatus.CANCELLED,
            error=reason,
        )

    def _set_tool_run_terminal(
        self,
        run: ToolRun,
        *,
        status: ToolRunStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Apply terminal tool-run state and set completion event."""
        run.status = status
        if result is not None:
            run.result = dict(result)
        if error is not None:
            run.error = str(error)
        run.completed_at = utc_now()
        event = self.tool_run_events.setdefault(run.id, asyncio.Event())
        event.set()

    def _resolve_action_blocking(self, *, action_type: str, action: dict[str, Any]) -> bool:
        """Resolve action blocking mode from registry defaults plus action override."""
        spec = TOOL_REGISTRY.get(action_type)
        default = spec.default_blocking if spec is not None else True
        return _as_bool(action.get("blocking"), default=default)

    async def _execute_tool_action(self, *, agent: AgentNode, action: dict[str, Any]) -> dict[str, Any]:
        """Execute one non-finish action and persist tool-run lifecycle."""
        action_type = str(action.get("type", "")).strip()
        blocking = self._resolve_action_blocking(action_type=action_type, action=action)
        run = self._create_tool_run(agent=agent, tool_name=action_type, arguments=action, blocking=blocking)
        self._mark_tool_run_running(run)
        try:
            with self._timer_scope(
                "tool_action",
                agent=agent,
                payload={"tool_name": action_type, "tool_run_id": run.id, "blocking": blocking},
            ):
                result = await self._dispatch_tool_action(
                    action_type=action_type,
                    agent=agent,
                    run=run,
                    action=action,
                )
            if run.status == ToolRunStatus.RUNNING and action_type not in SELF_COMPLETING_TOOL_ACTIONS:
                self._complete_tool_run(run, result=result)
            self._log_event(agent, "tool_call", {"action": json_ready(action), "result": json_ready(result), "tool_run_id": run.id})
            return result
        except Exception as exc:
            if run.status not in {ToolRunStatus.CANCELLED, ToolRunStatus.COMPLETED, ToolRunStatus.FAILED}:
                self._fail_tool_run(run, error=str(exc))
            self._record_exception(
                stage="tool_action",
                exc=exc,
                agent=agent,
                payload={
                    "tool_name": action_type,
                    "tool_run_id": run.id,
                    "action": json_ready(action),
                },
            )
            self._log_event(agent, "tool_call_failed", {"action": json_ready(action), "error": str(exc), "tool_run_id": run.id})
            raise
        finally:
            self._persist_snapshot()


def _as_bool(value: Any, *, default: bool) -> bool:
    """Parse permissive boolean-like values with fallback default."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off", ""}:
        return False
    return bool(value)

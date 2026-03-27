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
            id=self._new_id("toolrun"),
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

    def _resolve_action_blocking(self, *, action_type: str) -> bool:
        """Resolve action blocking mode from registry defaults."""
        spec = TOOL_REGISTRY.get(action_type)
        default = spec.default_blocking if spec is not None else True
        return bool(default)

    def _reject_tool_action(
        self,
        *,
        agent: AgentNode,
        action: dict[str, Any],
        tool_name: str,
        error_code: str,
        error_message: str,
    ) -> dict[str, Any]:
        """Create one failed tool run and return a structured non-throwing error payload."""
        run = self._create_tool_run(
            agent=agent,
            tool_name=tool_name,
            arguments=action,
            blocking=True,
        )
        self._mark_tool_run_running(run)
        self._fail_tool_run(run, error=error_message)
        exc = ValueError(error_message)
        result = self._tool_error_result(
            agent=agent,
            run=run,
            action=action,
            exc=exc,
            error_code=error_code,
        )
        self._record_exception(
            stage="tool_action_validation",
            exc=exc,
            agent=agent,
            payload={
                "tool_name": tool_name,
                "tool_run_id": run.id,
                "action": json_ready(action),
            },
        )
        self._log_event(
            agent,
            "tool_call_failed",
            {
                "action": json_ready(action),
                "error": json_ready(result.get("error")),
                "tool_run_id": run.id,
            },
        )
        self._persist_snapshot()
        return result

    def _tool_error_result(
        self,
        *,
        agent: AgentNode,
        run: ToolRun,
        action: dict[str, Any],
        exc: BaseException,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        """Build one canonical runtime payload for tool execution failure."""
        message = str(exc).strip() or "<empty>"
        code = error_code or _infer_tool_error_code(tool_name=run.tool_name, message=message)
        return {
            "error": message,
            "error_code": code,
            "action": json_ready(action),
            "available_tools": list(self.config.runtime.tools.tool_names_for_role(agent.role.value)),
        }

    async def _execute_tool_action(self, *, agent: AgentNode, action: dict[str, Any]) -> dict[str, Any]:
        """Execute one non-finish action and persist tool-run lifecycle."""
        action_type = str(action.get("type", "")).strip()
        blocking = self._resolve_action_blocking(action_type=action_type)
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
                error_text = str(result.get("error", "")).strip() if isinstance(result, dict) else ""
                if error_text:
                    self._fail_tool_run(run, error=error_text)
                else:
                    self._complete_tool_run(run, result=result)
            self._log_event(agent, "tool_call", {"action": json_ready(action), "result": json_ready(result), "tool_run_id": run.id})
            return result
        except Exception as exc:
            if run.status not in {
                ToolRunStatus.CANCELLED,
                ToolRunStatus.COMPLETED,
                ToolRunStatus.FAILED,
                ToolRunStatus.ABANDONED,
            }:
                self._fail_tool_run(run, error=str(exc))
            result = self._tool_error_result(
                agent=agent,
                run=run,
                action=action,
                exc=exc,
            )
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
            self._log_event(
                agent,
                "tool_call_failed",
                {
                    "action": json_ready(action),
                    "error": json_ready(result.get("error")),
                    "tool_run_id": run.id,
                },
            )
            return result
        finally:
            self._persist_snapshot()

def _infer_tool_error_code(*, tool_name: str, message: str) -> str:
    """Map common tool failure messages to stable machine-readable error codes."""
    text = str(message).strip().lower()
    name = str(tool_name).strip().lower()
    if "action type is required" in text:
        return "missing_tool_name"
    if "not enabled for role" in text:
        return "tool_not_enabled_for_role"
    if "unsupported action type" in text:
        return "unsupported_tool_name"
    if "unknown agent_id" in text:
        return "unknown_agent_id"
    if "unknown tool run" in text:
        return "unknown_tool_run_id"
    if "requires agent_id" in text:
        return "missing_agent_id"
    if "requires tool_run_id" in text:
        return "missing_tool_run_id"
    if "requires exactly one of 'tool_run_id' or 'agent_id'" in text:
        return "invalid_wait_run_target"
    if "requires non-empty command" in text:
        return "missing_command"
    if name == "shell" and "timed out" in text:
        return "shell_timeout"
    if "cancelled" in text:
        return "tool_cancelled"
    return "tool_execution_error"

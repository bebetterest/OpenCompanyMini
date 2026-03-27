"""Inspection and utility tool handler mixin."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any, Callable

from opm_train.context import compress_context
from opm_train.models import AgentNode, ToolRun
from opm_train.tools import TERMINAL_TOOL_RUN_STATUSES, parse_list_limit, validate_wait_time_action


class QueryToolMixin:
    """Attach list/get/wait/cancel/compress utility tool handlers."""

    agents: dict[str, AgentNode]
    tool_runs: dict[str, ToolRun]
    tool_tasks: dict[str, asyncio.Task[Any]]
    tool_run_events: dict[str, asyncio.Event]

    def _require_agent(self, agent_id: str) -> AgentNode:
        """Resolve agent by id or raise standardized unknown-id error."""
        agent = self.agents.get(agent_id)
        if agent is None:
            raise ValueError(f"unknown agent_id: {agent_id}")
        return agent

    def _require_tool_run(self, run_id: str) -> ToolRun:
        """Resolve tool run by id or raise standardized unknown-id error."""
        run = self.tool_runs.get(run_id)
        if run is None:
            raise ValueError(f"unknown tool run: {run_id}")
        return run

    @staticmethod
    def _status_filter(status: Any) -> str:
        """Normalize optional status filter values."""
        return str(status or "").strip().lower()

    def _tool_list_agent_runs(self, action: dict[str, Any]) -> dict[str, Any]:
        """List agent runs with status filter and cursor pagination."""
        ordered = sorted(
            self.agents.values(),
            key=lambda item: int(item.metadata.get("created_index", 0) or 0),
            reverse=True,
        )
        return self._list_page(
            action=action,
            ordered_items=ordered,
            status_of=lambda agent: agent.status.value,
            serialize=_agent_run_payload,
        )

    def _tool_get_agent_run(self, action: dict[str, Any]) -> dict[str, Any]:
        """Return detailed state and recent conversation for one agent."""
        target_id = str(action.get("agent_id", "")).strip()
        agent = self._require_agent(target_id)
        messages = agent.conversation[-6:]
        return {
            "id": agent.id,
            "name": agent.name,
            "role": agent.role.value,
            "status": agent.status.value,
            "status_reason": agent.status_reason,
            "summary": agent.summary,
            "next_recommendation": agent.next_recommendation,
            "last_error": agent.metadata.get("last_error") if isinstance(agent.metadata, dict) else None,
            "messages": messages,
        }

    def _tool_list_tool_runs(self, action: dict[str, Any]) -> dict[str, Any]:
        """List tool runs with status filter and cursor pagination."""
        ordered = sorted(self.tool_runs.values(), key=lambda run: run.created_at, reverse=True)
        return self._list_page(
            action=action,
            ordered_items=ordered,
            status_of=lambda run: run.status.value,
            serialize=_tool_run_payload,
        )

    def _tool_get_tool_run(self, action: dict[str, Any]) -> dict[str, Any]:
        """Return full metadata for one tool run."""
        run_id = str(action.get("run_id", "")).strip()
        run = self._require_tool_run(run_id)
        return {
            "id": run.id,
            "agent_id": run.agent_id,
            "tool_name": run.tool_name,
            "status": run.status.value,
            "result": run.result,
            "error": run.error,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
        }

    async def _tool_wait_run(self, action: dict[str, Any]) -> dict[str, Any]:
        """Wait for a tool run to finish with action/config timeout fallback."""
        run_id = str(action.get("run_id", "")).strip()
        if not run_id:
            raise ValueError("wait_run requires run_id")
        timeout_seconds = _resolve_wait_run_timeout_seconds(
            action.get("timeout_seconds"),
            default=float(self.config.runtime.tools.wait_run_timeout_seconds),
        )
        return await self._wait_for_tool_run(run_id, timeout_seconds=timeout_seconds)

    async def _wait_for_tool_run(self, run_id: str, *, timeout_seconds: float | None) -> dict[str, Any]:
        """Block until run finishes or timeout elapses."""
        run = self._require_tool_run(run_id)
        event = self.tool_run_events.setdefault(run_id, asyncio.Event())
        if run.status.value in TERMINAL_TOOL_RUN_STATUSES:
            event.set()
        timed_out = False
        if timeout_seconds is None:
            await event.wait()
        else:
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                timed_out = True
        return {
            "run_id": run_id,
            "timed_out": timed_out,
            "timeout_seconds": timeout_seconds,
            "status": run.status.value,
            "result": run.result,
            "error": run.error,
        }

    async def _tool_cancel_tool_run(self, action: dict[str, Any]) -> dict[str, Any]:
        """Cancel running tool task and mark run cancelled when needed."""
        run_id = str(action.get("run_id", "")).strip()
        run = self._require_tool_run(run_id)
        task = self.tool_tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if run.status.value not in TERMINAL_TOOL_RUN_STATUSES:
            self._cancel_tool_run_obj(run, reason="cancel_tool_run")
        return {"run_id": run.id, "status": run.status.value}

    async def _tool_wait_time(self, action: dict[str, Any]) -> dict[str, Any]:
        """Sleep for validated seconds and return elapsed duration."""
        error = validate_wait_time_action(action)
        if error:
            raise ValueError(error)
        seconds = float(action.get("seconds") or 0.0)
        await asyncio.sleep(seconds)
        return {"slept_seconds": seconds}

    def _tool_compress_context(self, agent: AgentNode) -> dict[str, Any]:
        """Trigger manual context compression for current agent."""
        result = compress_context(agent=agent, reason="manual_tool")
        self._record_context_compression(agent=agent, reason="manual_tool", result=result)
        self._log_event(agent, "context_compressed", result)
        return result

    def _list_page(
        self,
        *,
        action: dict[str, Any],
        ordered_items: list[Any],
        status_of: Callable[[Any], str],
        serialize: Callable[[Any], dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply status filtering and cursor pagination to ordered runtime records."""
        limit = parse_list_limit(action.get("limit"), config=self.config)
        offset = _decode_offset(action.get("cursor"))
        status_filter = self._status_filter(action.get("status"))
        filtered = ordered_items
        if status_filter:
            filtered = [item for item in ordered_items if status_of(item) == status_filter]
        page, next_cursor = _paginate(filtered, offset=offset, limit=limit)
        return {"items": [serialize(item) for item in page], "next_cursor": next_cursor}


def _encode_offset(offset: int) -> str:
    """Encode pagination offset as non-negative cursor string."""
    return str(max(0, int(offset)))


def _decode_offset(cursor: Any) -> int:
    """Decode pagination cursor into non-negative integer offset."""
    try:
        return max(0, int(cursor))
    except (TypeError, ValueError):
        return 0


def _paginate(items: list[Any], *, offset: int, limit: int) -> tuple[list[Any], str | None]:
    """Return page slice and next cursor from ordered items."""
    page = items[offset : offset + limit]
    next_cursor = _encode_offset(offset + len(page)) if offset + len(page) < len(items) else None
    return page, next_cursor


def _parse_timeout_seconds(value: Any) -> float:
    """Parse wait timeout value as non-negative float."""
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("wait_run timeout_seconds must be numeric") from exc


def _resolve_wait_run_timeout_seconds(value: Any, *, default: float) -> float | None:
    """Resolve effective wait_run timeout using action override then config fallback."""
    seconds = _parse_timeout_seconds(default if value is None else value)
    if seconds <= 0.0:
        return None
    return seconds


def _agent_run_payload(agent: AgentNode) -> dict[str, Any]:
    """Serialize one agent summary row for list views."""
    return {
        "id": agent.id,
        "name": agent.name,
        "role": agent.role.value,
        "status": agent.status.value,
        "status_reason": agent.status_reason,
        "parent_agent_id": agent.parent_agent_id,
        "summary": agent.summary,
    }


def _tool_run_payload(run: ToolRun) -> dict[str, Any]:
    """Serialize one tool-run summary row for list views."""
    return {
        "id": run.id,
        "agent_id": run.agent_id,
        "tool_name": run.tool_name,
        "status": run.status.value,
        "error": run.error,
        "created_at": run.created_at,
    }

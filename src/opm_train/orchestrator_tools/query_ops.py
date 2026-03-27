"""Inspection and utility tool handler mixin."""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from opm_train.context import compress_context
from opm_train.models import AgentNode, AgentStatus, ToolRun, ToolRunStatus
from opm_train.tools import (
    TERMINAL_AGENT_STATUSES,
    TERMINAL_TOOL_RUN_STATUSES,
    parse_list_limit,
    validate_wait_run_action,
    validate_wait_time_action,
)


_KNOWN_AGENT_STATUSES = frozenset(
    {
        AgentStatus.PENDING.value,
        AgentStatus.RUNNING.value,
        AgentStatus.COMPLETED.value,
        AgentStatus.FAILED.value,
        AgentStatus.CANCELLED.value,
    }
)

_KNOWN_TOOL_STATUSES = frozenset(
    {
        ToolRunStatus.QUEUED.value,
        ToolRunStatus.RUNNING.value,
        ToolRunStatus.COMPLETED.value,
        ToolRunStatus.FAILED.value,
        ToolRunStatus.CANCELLED.value,
        ToolRunStatus.ABANDONED.value,
    }
)

_GET_AGENT_RUN_MAX_MESSAGES = 5


class QueryToolMixin:
    """Attach list/get/wait/cancel/compress utility tool handlers."""

    agents: dict[str, AgentNode]
    tool_runs: dict[str, ToolRun]
    tool_tasks: dict[str, asyncio.Task[Any]]
    tool_run_events: dict[str, asyncio.Event]

    @staticmethod
    def _id_kind(identifier: str) -> str:
        """Classify identifier prefixes for explicit wait/cancel/get validation."""
        normalized = str(identifier or "").strip().lower()
        if normalized.startswith("toolrun-"):
            return "tool_run_id"
        if normalized.startswith("agent-"):
            return "agent_id"
        return "unknown"

    @staticmethod
    def _normalize_status_filter(raw: Any) -> list[str] | None:
        """Normalize status filter from string or array."""
        if isinstance(raw, str):
            item = raw.strip().lower()
            return [item] if item else None
        if isinstance(raw, (list, tuple, set)):
            values = [str(item).strip().lower() for item in raw if str(item).strip()]
            return values or None
        return None

    def _tool_list_agent_runs(self, action: dict[str, Any]) -> dict[str, Any]:
        """List agent runs with status filter and cursor pagination."""
        limit = parse_list_limit(action.get("limit"), config=self.config)
        offset = _decode_offset_cursor(action.get("cursor"))
        if offset is None:
            return {"error": "list_agent_runs received an invalid 'cursor'."}

        statuses = self._normalize_status_filter(action.get("status"))
        if statuses is not None:
            invalid = sorted(item for item in statuses if item not in _KNOWN_AGENT_STATUSES)
            if invalid:
                allowed = ", ".join(sorted(_KNOWN_AGENT_STATUSES))
                joined = ", ".join(f"'{item}'" for item in invalid)
                return {
                    "error": (
                        f"list_agent_runs received invalid status filter(s): {joined}. "
                        f"Allowed: {allowed}."
                    )
                }

        ordered = sorted(
            self.agents.values(),
            key=lambda item: int(item.metadata.get("created_index", 0) or 0),
            reverse=True,
        )
        rows: list[dict[str, Any]] = []
        for agent in ordered:
            status = str(agent.status.value).strip().lower()
            if statuses and status not in statuses:
                continue
            rows.append(
                {
                    "id": agent.id,
                    "name": agent.name,
                    "role": agent.role.value,
                    "status": agent.status.value,
                    "created_at": str(agent.metadata.get("created_at", "") or ""),
                    "summary_short": str(agent.summary or "")[:160],
                    "messages_count": len(agent.conversation),
                }
            )

        page, next_cursor, has_more = _paginate_offset(rows, offset=offset, limit=limit)
        return {
            "agent_runs_count": len(page),
            "agent_runs": page,
            "next_cursor": next_cursor,
            "has_more": has_more,
        }

    def _tool_get_agent_run(self, action: dict[str, Any]) -> dict[str, Any]:
        """Return detailed state and optional sliced conversation for one agent."""
        target_id = str(action.get("agent_id", "")).strip()
        if not target_id:
            return {"error": "get_agent_run requires 'agent_id'."}
        if self._id_kind(target_id) == "tool_run_id":
            return {
                "error": (
                    "get_agent_run expects 'agent_id' (prefix 'agent-'), "
                    f"but received tool_run_id '{target_id}'."
                )
            }
        agent = self.agents.get(target_id)
        if agent is None:
            return {"error": f"Agent {target_id} was not found."}

        messages = [dict(item) for item in agent.conversation if isinstance(item, dict)]
        messages_count = len(messages)

        start_raw = action.get("messages_start")
        end_raw = action.get("messages_end")

        if start_raw is None and end_raw is None:
            start = max(0, messages_count - 1)
            end = messages_count
        else:
            start_value = _safe_int(start_raw, default=0)
            end_value = _safe_int(end_raw, default=messages_count)
            if start_value is None:
                return {"error": "get_agent_run field 'messages_start' must be an integer."}
            if end_value is None:
                return {"error": "get_agent_run field 'messages_end' must be an integer."}
            start = _resolve_relative_index(start_value, size=messages_count)
            end = _resolve_relative_index(end_value, size=messages_count)
            start = max(0, min(start, messages_count))
            end = max(0, min(end, messages_count))
            if end < start:
                return {
                    "error": (
                        "get_agent_run received invalid [messages_start,messages_end) "
                        f"indices after normalization: start={start}, end={end}."
                    )
                }

        requested_end = end
        end = min(end, start + _GET_AGENT_RUN_MAX_MESSAGES)
        sliced = [_project_agent_run_message(item) for item in messages[start:end]]
        result: dict[str, Any] = {
            "agent_run": {
                "id": agent.id,
                "name": agent.name,
                "role": agent.role.value,
                "status": agent.status.value,
                "created_at": str(agent.metadata.get("created_at", "") or ""),
                "parent_agent_id": agent.parent_agent_id,
                "children_count": len(agent.children),
                "step_count": int(agent.step_count),
            },
            "messages": sliced,
        }
        if requested_end > end:
            result["warning"] = (
                "get_agent_run returned only the first 5 messages for the requested "
                "slice; call again with messages_start=next_messages_start to continue."
            )
            result["next_messages_start"] = end
        return result

    def _tool_list_tool_runs(self, action: dict[str, Any]) -> dict[str, Any]:
        """List tool runs with status filter and cursor pagination."""
        limit = parse_list_limit(action.get("limit"), config=self.config)
        statuses = self._normalize_status_filter(action.get("status"))
        if statuses is not None:
            invalid = sorted(item for item in statuses if item not in _KNOWN_TOOL_STATUSES)
            if invalid:
                allowed = ", ".join(sorted(_KNOWN_TOOL_STATUSES))
                joined = ", ".join(f"'{item}'" for item in invalid)
                return {
                    "error": (
                        f"list_tool_runs received invalid status filter(s): {joined}. "
                        f"Allowed: {allowed}."
                    ),
                    "invalid_statuses": invalid,
                    "allowed_statuses": sorted(_KNOWN_TOOL_STATUSES),
                }

        cursor_value = action.get("cursor")
        cursor_text = str(cursor_value).strip() if cursor_value is not None else None
        cursor = _decode_tool_run_cursor(cursor_text)
        if cursor_text is not None and cursor is None:
            return {"error": "list_tool_runs received an invalid 'cursor'."}

        ordered = sorted(
            self.tool_runs.values(),
            key=lambda run: (str(run.created_at), str(run.id)),
            reverse=True,
        )
        if statuses:
            ordered = [run for run in ordered if str(run.status.value).lower() in statuses]

        page, has_more = _paginate_tool_runs(ordered, cursor=cursor, limit=limit)
        next_cursor = _next_tool_run_cursor(page, limit=limit) if has_more else None
        return {
            "tool_runs_count": len(page),
            "tool_runs": [self._tool_run_overview(run, include_result=False) for run in page],
            "next_cursor": next_cursor,
            "has_more": has_more,
        }

    def _tool_get_tool_run(self, action: dict[str, Any]) -> dict[str, Any]:
        """Return tool run details by id."""
        tool_run_id = str(action.get("tool_run_id", "")).strip()
        if not tool_run_id:
            return {"error": "get_tool_run requires 'tool_run_id'."}
        if self._id_kind(tool_run_id) == "agent_id":
            return {
                "error": (
                    f"Expected tool_run_id (prefix 'toolrun-'), but received agent_id '{tool_run_id}'."
                )
            }
        run = self.tool_runs.get(tool_run_id)
        if run is None:
            return {"error": f"Tool run {tool_run_id} was not found."}
        include_result = _coerce_bool(action.get("include_result"), default=False)
        return {"tool_run": self._tool_run_overview(run, include_result=include_result)}

    async def _tool_wait_run(self, action: dict[str, Any]) -> dict[str, Any]:
        """Wait for tool or agent completion with timeout budget."""
        error = validate_wait_run_action(action)
        if error:
            return {"wait_run_status": False, "error": error}

        timeout_seconds = max(0.0, float(self.config.runtime.tools.wait_run_timeout_seconds))
        started = asyncio.get_running_loop().time()

        tool_run_id = str(action.get("tool_run_id", "")).strip()
        agent_id = str(action.get("agent_id", "")).strip()

        while True:
            if tool_run_id:
                run = self.tool_runs.get(tool_run_id)
                if run is None:
                    return {"wait_run_status": False, "error": f"Tool run {tool_run_id} was not found."}
                if run.status.value in TERMINAL_TOOL_RUN_STATUSES:
                    return {"wait_run_status": True}
                waiter = self.tool_run_events.setdefault(tool_run_id, asyncio.Event())
                try:
                    await asyncio.wait_for(waiter.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
                finally:
                    waiter.clear()
            else:
                if self._id_kind(agent_id) == "tool_run_id":
                    return {
                        "wait_run_status": False,
                        "error": (
                            f"Expected agent_id (prefix 'agent-'), but received tool_run_id '{agent_id}'."
                        ),
                    }
                agent = self.agents.get(agent_id)
                if agent is None:
                    return {"wait_run_status": False, "error": f"Agent {agent_id} was not found."}
                if agent.status.value in TERMINAL_AGENT_STATUSES:
                    return {"wait_run_status": True}
                await asyncio.sleep(0.2)

            if timeout_seconds > 0 and (asyncio.get_running_loop().time() - started) >= timeout_seconds:
                return {
                    "wait_run_status": False,
                    "timed_out": True,
                    "timeout_seconds": timeout_seconds,
                }

    async def _tool_cancel_tool_run(self, action: dict[str, Any]) -> dict[str, Any]:
        """Cancel a running tool and return final status."""
        tool_run_id = str(action.get("tool_run_id", "")).strip()
        if not tool_run_id:
            return {"error": "cancel_tool_run requires 'tool_run_id'."}
        if self._id_kind(tool_run_id) == "agent_id":
            return {
                "error": (
                    f"Expected tool_run_id (prefix 'toolrun-'), but received agent_id '{tool_run_id}'."
                )
            }
        run = self.tool_runs.get(tool_run_id)
        if run is None:
            return {"error": f"Tool run {tool_run_id} was not found."}

        task = self.tool_tasks.get(tool_run_id)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        if run.status.value not in TERMINAL_TOOL_RUN_STATUSES:
            self._cancel_tool_run_obj(run, reason=f"Tool run cancelled by agent {run.agent_id}.")

        cancelled_agents_count = 0
        if run.tool_name == "spawn_agent":
            child_id = str(run.arguments.get("child_agent_id", "")).strip()
            if child_id:
                cancelled = self._cancel_agent_tree(
                    child_id,
                    reason=f"Cancelled because spawn tool run {run.id} was cancelled.",
                )
                cancelled_agents_count = len(cancelled)

        return {
            "final_status": run.status.value,
            "cancelled_agents_count": cancelled_agents_count,
        }

    async def _tool_wait_time(self, action: dict[str, Any]) -> dict[str, Any]:
        """Sleep for validated seconds and return boolean completion."""
        error = validate_wait_time_action(action, config=self.config)
        if error:
            return {"wait_time_status": False, "error": error}
        seconds = float(action.get("seconds") or 0.0)
        await asyncio.sleep(seconds)
        return {"wait_time_status": True}

    async def _tool_list_mcp_servers(self, action: dict[str, Any]) -> dict[str, Any]:
        """Soft-disabled MCP helper endpoint for parity with OpenCompany."""
        del action
        return self._mcp_soft_disabled("list_mcp_servers")

    async def _tool_list_mcp_resources(self, action: dict[str, Any]) -> dict[str, Any]:
        """Soft-disabled MCP helper endpoint for parity with OpenCompany."""
        del action
        return self._mcp_soft_disabled("list_mcp_resources")

    async def _tool_read_mcp_resource(self, action: dict[str, Any]) -> dict[str, Any]:
        """Soft-disabled MCP helper endpoint for parity with OpenCompany."""
        del action
        return self._mcp_soft_disabled("read_mcp_resource")

    def _mcp_soft_disabled(self, tool_name: str) -> dict[str, Any]:
        if bool(self.config.extensions.mcp_enabled):
            return {
                "error": (
                    "MCP extension interface is not implemented in this runtime yet. "
                    "Set extensions.mcp_enabled=false or integrate an MCP backend."
                )
            }
        return {
            "error": (
                f"{tool_name} is unavailable because MCP is disabled "
                "(extensions.mcp_enabled=false)."
            )
        }

    async def _tool_compress_context(self, agent: AgentNode) -> dict[str, Any]:
        """Trigger manual context compression for current agent."""
        result = await compress_context(
            agent=agent,
            reason="manual_tool",
            config=self.config,
            prompt_library=self.prompt_library,
            llm_client=self.llm_client,
        )
        self._record_context_compression(agent=agent, reason="manual_tool", result=result)
        self._log_event(agent, "context_compressed", result)
        return result

    def _tool_run_overview(self, run: ToolRun, *, include_result: bool) -> dict[str, Any]:
        """Build tool run overview payload for list/get APIs."""
        overview: dict[str, Any] = {
            "id": run.id,
            "tool_name": run.tool_name,
            "status": run.status.value,
            "agent_id": run.agent_id,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "duration_ms": _tool_run_duration_ms(run),
            "error_summary": (str(run.error or "")[:240] or None),
        }
        if run.tool_name == "shell" and isinstance(run.result, dict):
            overview["stdout"] = str(run.result.get("stdout", ""))
            overview["stderr"] = str(run.result.get("stderr", ""))
        if include_result and isinstance(run.result, dict):
            overview["result"] = dict(run.result)
        return overview


def _tool_run_duration_ms(run: ToolRun) -> int | None:
    """Compute duration from timestamps when possible."""
    started = _parse_iso8601(run.started_at) or _parse_iso8601(run.created_at)
    completed = _parse_iso8601(run.completed_at)
    if started is None or completed is None:
        return None
    delta_ms = int((completed - started).total_seconds() * 1000)
    if delta_ms < 0:
        return None
    return delta_ms


def _parse_iso8601(raw: str | None) -> datetime | None:
    value = str(raw or "").strip()
    if not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed


def _project_agent_run_message(message: dict[str, Any]) -> dict[str, Any]:
    visible_fields = ("content", "reasoning", "role", "tool_calls", "tool_call_id")
    return {key: message[key] for key in visible_fields if key in message}


def _safe_int(value: Any, *, default: int) -> int | None:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_relative_index(index: int, *, size: int) -> int:
    if index < 0:
        return size + index
    return index


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _encode_offset_cursor(offset: int) -> str:
    """Encode pagination offset into OpenCompany-style opaque cursor."""
    normalized = max(0, int(offset))
    payload = json.dumps(
        {"offset": normalized},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii")


def _decode_offset_cursor(cursor: Any) -> int | None:
    """Decode OpenCompany-style opaque cursor into non-negative offset."""
    if cursor is None:
        return 0
    normalized = str(cursor).strip()
    if not normalized:
        return 0
    try:
        payload = base64.urlsafe_b64decode(normalized.encode("ascii"))
        parsed = json.loads(payload.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        offset = int(parsed.get("offset", 0))
    except (TypeError, ValueError):
        return None
    if offset < 0:
        return None
    return offset


def _paginate_offset(items: list[Any], *, offset: int, limit: int) -> tuple[list[Any], str | None, bool]:
    """Return page slice and next cursor from ordered items."""
    page = items[offset : offset + limit]
    stop = offset + len(page)
    has_more = stop < len(items)
    next_cursor = _encode_offset_cursor(stop) if has_more else None
    return page, next_cursor, has_more


def _encode_tool_run_cursor(created_at: str, tool_run_id: str) -> str:
    """Encode tool-run cursor as opaque base64 json token."""
    payload = json.dumps(
        {"created_at": created_at, "id": tool_run_id},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii")


def _decode_tool_run_cursor(cursor: str | None) -> tuple[str, str] | None:
    """Decode tool-run cursor token into (created_at, tool_run_id)."""
    if cursor is None:
        return None
    normalized = str(cursor).strip()
    if not normalized:
        return None
    try:
        payload = base64.urlsafe_b64decode(normalized.encode("ascii"))
        parsed = json.loads(payload.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    created_at = str(parsed.get("created_at", "")).strip()
    tool_run_id = str(parsed.get("id", "")).strip()
    if not created_at or not tool_run_id:
        return None
    return created_at, tool_run_id


def _paginate_tool_runs(
    items: list[ToolRun],
    *,
    cursor: tuple[str, str] | None,
    limit: int,
) -> tuple[list[ToolRun], bool]:
    """Apply created_at/id cursor and return page plus has_more flag."""
    filtered = items
    if cursor is not None:
        cursor_created_at, cursor_id = cursor
        filtered = [
            run
            for run in items
            if (
                str(run.created_at) < cursor_created_at
                or (str(run.created_at) == cursor_created_at and str(run.id) < cursor_id)
            )
        ]
    page = filtered[:limit]
    has_more = len(filtered) > limit
    return page, has_more


def _next_tool_run_cursor(runs: list[ToolRun], *, limit: int) -> str | None:
    """Return next tool-run cursor when there are more rows after page tail."""
    if len(runs) < limit:
        return None
    tail = runs[-1]
    created_at = str(tail.created_at).strip()
    tool_run_id = str(tail.id).strip()
    if not created_at or not tool_run_id:
        return None
    return _encode_tool_run_cursor(created_at, tool_run_id)

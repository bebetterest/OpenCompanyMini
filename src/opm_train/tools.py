"""Tool schema shaping and action-level validation helpers."""

from __future__ import annotations

import copy
import math
from typing import Any

from opm_train.config import OPMTrainConfig
from opm_train.models import AgentRole, AgentStatus, ToolRunStatus
from opm_train.prompts import PromptLibrary


TERMINAL_AGENT_STATUSES = {
    AgentStatus.COMPLETED.value,
    AgentStatus.FAILED.value,
    AgentStatus.CANCELLED.value,
}

TERMINAL_TOOL_RUN_STATUSES = {
    ToolRunStatus.COMPLETED.value,
    ToolRunStatus.FAILED.value,
    ToolRunStatus.CANCELLED.value,
    ToolRunStatus.ABANDONED.value,
}

_ALLOWED_FINISH_STATUS_BY_ROLE = {
    AgentRole.ROOT: {"completed", "partial"},
    AgentRole.WORKER: {"completed", "partial", "failed"},
}

_RUNTIME_ROLES = ("root", "worker")
_REQUIRED_CORE_TOOL_NAMES = frozenset(
    {
        "shell",
        "compress_context",
        "wait_time",
        "list_mcp_servers",
        "list_mcp_resources",
        "read_mcp_resource",
        "spawn_agent",
        "cancel_agent",
        "steer_agent",
        "list_agent_runs",
        "get_agent_run",
        "list_tool_runs",
        "get_tool_run",
        "wait_run",
        "cancel_tool_run",
        "finish",
    }
)


def tool_definitions_for_role(
    role: AgentRole | str,
    *,
    prompt_library: PromptLibrary,
    config: OPMTrainConfig,
) -> list[dict[str, Any]]:
    """Resolve role-enabled tool schemas with runtime-configured constraints."""
    role_name = AgentRole(role).value
    tool_names = config.runtime.tools.tool_names_for_role(role_name)
    blueprints = prompt_library.load_tool_definitions()
    resolved: list[dict[str, Any]] = []
    for name in tool_names:
        if name not in blueprints:
            continue
        item = copy.deepcopy(blueprints[name])
        function = item.get("function")
        if isinstance(function, dict):
            params = function.get("parameters")
            if isinstance(params, dict):
                properties = params.get("properties")
                if isinstance(properties, dict) and "limit" in properties:
                    limit_schema = properties.get("limit")
                    if isinstance(limit_schema, dict):
                        limit_schema["minimum"] = 1
                        limit_schema["maximum"] = int(config.runtime.tools.list_max_limit)
                        limit_schema["default"] = int(config.runtime.tools.list_default_limit)
                if name == "wait_time" and isinstance(properties, dict):
                    seconds_schema = properties.get("seconds")
                    if isinstance(seconds_schema, dict):
                        wait_min, wait_max = config.runtime.tools.wait_time_bounds()
                        seconds_schema["minimum"] = wait_min
                        seconds_schema["maximum"] = wait_max
                        current_desc = str(seconds_schema.get("description", "")).strip()
                        if current_desc:
                            if any("\u4e00" <= ch <= "\u9fff" for ch in current_desc):
                                seconds_schema["description"] = (
                                    f"必填等待秒数，必须 >= {wait_min:g} 且 <= {wait_max:g}。"
                                )
                            else:
                                seconds_schema["description"] = (
                                    f"Required wait duration in seconds. Must be >= {wait_min:g} and <= {wait_max:g}."
                                )
                if name == "finish":
                    _apply_finish_schema(params=params, role=role_name)
        resolved.append(item)
    return resolved


def _apply_finish_schema(*, params: dict[str, Any], role: str) -> None:
    """Mutate finish schema per role-specific contract."""
    properties = params.get("properties")
    if not isinstance(properties, dict):
        return
    status_schema = properties.get("status")
    if isinstance(status_schema, dict) and isinstance(status_schema.get("enum"), list):
        enum_values = [str(item).strip() for item in status_schema.get("enum", [])]
        if role == AgentRole.WORKER.value:
            status_schema["enum"] = [value for value in enum_values if value != "interrupted"]
        elif role == AgentRole.ROOT.value:
            status_schema["enum"] = [value for value in enum_values if value in {"completed", "partial"}]
        else:
            allowed = sorted(_ALLOWED_FINISH_STATUS_BY_ROLE[AgentRole(role)])
            status_schema["enum"] = allowed
    if role == AgentRole.ROOT.value:
        properties.pop("next_recommendation", None)


def parse_list_limit(value: Any, *, config: OPMTrainConfig) -> int:
    """Normalize list-page limit with runtime min/max constraints."""
    return config.runtime.tools.normalize_list_limit(value)


def validate_finish_action(role: AgentRole, action: dict[str, Any]) -> str | None:
    """Validate finish payload against role constraints."""
    common_keys = {"type", "_tool_call_id", "status", "summary"}
    worker_only_keys = {"next_recommendation"}
    allowed_keys = common_keys if role == AgentRole.ROOT else (common_keys | worker_only_keys)
    unknown_keys = sorted(key for key in action.keys() if key not in allowed_keys)
    if unknown_keys:
        joined = ", ".join(f"'{key}'" for key in unknown_keys)
        return f"finish received unsupported field(s): {joined}."

    allowed_statuses = _ALLOWED_FINISH_STATUS_BY_ROLE[role]

    status = str(action.get("status", "")).strip().lower()
    if not status:
        return "finish requires a non-empty 'status'."
    if status not in allowed_statuses:
        joined = ", ".join(sorted(allowed_statuses))
        return f"finish status '{status}' invalid for {role.value}; allowed: {joined}."

    summary = str(action.get("summary", "")).strip()
    if not summary:
        return "finish requires a non-empty 'summary'."

    next_recommendation = str(action.get("next_recommendation", "")).strip()
    if role == AgentRole.ROOT and next_recommendation:
        return "root finish must not include 'next_recommendation'."
    if role == AgentRole.WORKER and status in {"partial", "failed"} and not next_recommendation:
        return "worker finish with partial/failed requires 'next_recommendation'."
    return None


def validate_wait_time_action(action: dict[str, Any], *, config: OPMTrainConfig) -> str | None:
    """Validate wait_time action payload bounds."""
    allowed_keys = {"type", "_tool_call_id", "seconds"}
    unknown_keys = sorted(key for key in action.keys() if key not in allowed_keys)
    if unknown_keys:
        joined = ", ".join(f"'{key}'" for key in unknown_keys)
        return f"wait_time received unsupported field(s): {joined}."
    raw_seconds = action.get("seconds")
    if raw_seconds is None:
        return "wait_time requires 'seconds'."
    try:
        seconds = float(raw_seconds)
    except (TypeError, ValueError):
        return "wait_time field 'seconds' must be a number."
    if not math.isfinite(seconds):
        return "wait_time field 'seconds' must be finite."
    wait_min, wait_max = config.runtime.tools.wait_time_bounds()
    if seconds < wait_min:
        return f"wait_time field 'seconds' must be >= {wait_min:g}."
    if seconds > wait_max:
        return f"wait_time field 'seconds' must be <= {wait_max:g}."
    return None


def validate_compress_context_action(action: dict[str, Any]) -> str | None:
    """Validate compress_context does not receive extra fields."""
    allowed_keys = {"type", "_tool_call_id"}
    unknown_keys = sorted(key for key in action.keys() if key not in allowed_keys)
    if unknown_keys:
        joined = ", ".join(f"'{key}'" for key in unknown_keys)
        return f"compress_context received unsupported field(s): {joined}."
    return None


def validate_wait_run_action(action: dict[str, Any]) -> str | None:
    """Validate wait_run target contract and unsupported fields."""
    allowed_keys = {"type", "_tool_call_id", "tool_run_id", "agent_id"}
    unknown_keys = sorted(key for key in action.keys() if key not in allowed_keys)
    if unknown_keys:
        joined = ", ".join(f"'{key}'" for key in unknown_keys)
        return f"wait_run received unsupported field(s): {joined}."
    has_tool_run_id = bool(str(action.get("tool_run_id", "")).strip())
    has_agent_id = bool(str(action.get("agent_id", "")).strip())
    if has_tool_run_id == has_agent_id:
        return "wait_run requires exactly one of 'tool_run_id' or 'agent_id'."
    return None


def runtime_tool_contract_issues(
    *,
    config: OPMTrainConfig,
    prompt_library: PromptLibrary,
    registry_tool_names: set[str] | list[str] | tuple[str, ...],
) -> list[str]:
    """Return mismatches across configured tools, prompt schemas, and runtime registry."""
    prompt_tools = set(prompt_library.load_tool_definitions().keys())
    runtime_tools = set(registry_tool_names) | {"finish"}

    issues: list[str] = []
    role_tools = {
        "root": set(config.runtime.tools.root_tools),
        "worker": set(config.runtime.tools.worker_tools),
    }
    for role_name in _RUNTIME_ROLES:
        enabled_tools = role_tools[role_name]
        missing_core_tools = sorted(_REQUIRED_CORE_TOOL_NAMES - enabled_tools)
        if missing_core_tools:
            issues.append(_issue(role=role_name, kind="missing_core_tools", tool_names=missing_core_tools))
        missing_in_prompts = sorted(enabled_tools - prompt_tools)
        if missing_in_prompts:
            issues.append(_issue(role=role_name, kind="missing_prompt_definitions", tool_names=missing_in_prompts))
        missing_in_runtime = sorted(enabled_tools - runtime_tools)
        if missing_in_runtime:
            issues.append(_issue(role=role_name, kind="missing_registry_handlers", tool_names=missing_in_runtime))
    return issues


def _issue(*, role: str, kind: str, tool_names: list[str]) -> str:
    """Format one machine-readable doctor issue line."""
    return f"{role}_enabled_{kind}:{','.join(tool_names)}"

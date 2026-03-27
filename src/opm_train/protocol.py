"""Model response protocol helpers and strict action normalization."""

from __future__ import annotations

import json
import re
from typing import Any


class ProtocolError(ValueError):
    """Raised when model output violates the orchestration protocol."""

    pass


JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from model text."""
    match = JSON_BLOCK_RE.search(text)
    candidate = match.group(1) if match else str(text).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ProtocolError("No JSON object found in model response")
        try:
            payload = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"Invalid JSON response: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Model response must decode to a JSON object")
    return payload


def normalize_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize action objects from parsed payload."""
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ProtocolError("Response must contain a non-empty actions list")
    normalized: list[dict[str, Any]] = []
    for raw in actions:
        if not isinstance(raw, dict):
            raise ProtocolError("Each action must be an object")
        action_type = str(raw.get("type", "")).strip()
        if not action_type:
            raise ProtocolError("Each action must include a non-empty type")
        normalized.append(dict(raw))
    return normalized


def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map OpenAI-compatible tool calls to runtime action shape."""
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        name, arguments_json = _tool_call_name_and_arguments(call)
        call_id = str(call.get("id", "")).strip()
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"Invalid tool arguments for '{name}': {exc}") from exc
        if not isinstance(arguments, dict):
            raise ProtocolError(f"Tool arguments for '{name}' must decode to a JSON object")
        normalized.append({"type": name, "_tool_call_id": call_id, **arguments})
    return normalized


def canonicalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return validated provider-safe tool-call objects in OpenAI format only."""
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        name, arguments_json = _tool_call_name_and_arguments(call)
        normalized.append(
            {
                "id": str(call.get("id", "")).strip() or f"tool-call-{index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments_json},
            }
        )
    return normalized


def _tool_call_name_and_arguments(call: dict[str, Any]) -> tuple[str, str]:
    """Extract strict OpenAI-compatible tool-call name and arguments JSON."""
    call_type = str(call.get("type", "")).strip()
    if call_type != "function":
        raise ProtocolError("Tool call type must be 'function'")

    function_payload = call.get("function")
    if not isinstance(function_payload, dict):
        raise ProtocolError("Tool call is missing function payload")

    name = str(function_payload.get("name", "")).strip()
    if not name:
        raise ProtocolError("Tool call is missing function name")

    raw_arguments: Any = function_payload.get("arguments")
    if isinstance(raw_arguments, (dict, list)):
        return name, json.dumps(raw_arguments, ensure_ascii=False)
    return name, str(raw_arguments if raw_arguments is not None else "{}").strip() or "{}"

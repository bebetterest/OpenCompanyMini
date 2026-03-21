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
    """Map tool-call deltas to runtime action shape."""
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        name = str(call.get("name", "")).strip()
        if not name:
            raise ProtocolError("Tool call is missing name")
        call_id = str(call.get("id", "")).strip()
        arguments_json = str(call.get("arguments_json", "{}")).strip() or "{}"
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"Invalid tool arguments for '{name}': {exc}") from exc
        if not isinstance(arguments, dict):
            raise ProtocolError(f"Tool arguments for '{name}' must decode to a JSON object")
        normalized.append({"type": name, "_tool_call_id": call_id, **arguments})
    return normalized

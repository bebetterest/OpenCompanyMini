"""Output formatters for trajectory exports."""

from __future__ import annotations

import json
from typing import Any



def format_raw(bundle: dict[str, Any]) -> dict[str, Any]:
    """Build raw export payload with filtered events and turn details."""
    return {
        "session_id": str(bundle.get("session_id", "")),
        "schema_version": int(bundle.get("schema_version", 0) or 0),
        "scope": dict(bundle.get("scope", {})),
        "session": dict(bundle.get("session", {})),
        "agents": dict(bundle.get("agents", {})),
        "tool_runs": dict(bundle.get("tool_runs", {})),
        "events": [dict(item) for item in list(bundle.get("events", [])) if isinstance(item, dict)],
        "turns": [dict(item) for item in list(bundle.get("turns", [])) if isinstance(item, dict)],
    }



def format_sft(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Build traceable OpenAI-messages style SFT rows from filtered turn data."""
    rows: list[dict[str, Any]] = []
    session_payload = dict(bundle.get("session", {})) if isinstance(bundle.get("session"), dict) else {}
    agents_payload = dict(bundle.get("agents", {})) if isinstance(bundle.get("agents"), dict) else {}
    scope_payload = dict(bundle.get("scope", {})) if isinstance(bundle.get("scope"), dict) else {}
    config_snapshot = (
        dict(session_payload.get("config_snapshot", {}))
        if isinstance(session_payload.get("config_snapshot"), dict)
        else {}
    )
    for turn in [dict(item) for item in list(bundle.get("turns", [])) if isinstance(item, dict)]:
        final_attempt_no = _optional_int(turn.get("final_attempt"))
        if final_attempt_no is None:
            continue
        attempts = [dict(item) for item in list(turn.get("attempts", [])) if isinstance(item, dict)]
        final_attempt = _find_attempt(attempts, final_attempt_no)
        if final_attempt is None:
            continue
        if not bool(final_attempt.get("ok", False)):
            continue

        request_payload = final_attempt.get("request")
        if not isinstance(request_payload, dict):
            continue
        response_payload = final_attempt.get("response")
        response_payload = response_payload if isinstance(response_payload, dict) else {}
        messages = request_payload.get("messages")
        if not isinstance(messages, list) or not messages:
            continue

        actions = [dict(item) for item in list(turn.get("actions", [])) if isinstance(item, dict)]
        if not actions:
            continue

        # Backward-compatible action target retained for existing data pipelines.
        action_target = {
            "role": "assistant",
            "content": json.dumps({"actions": actions}, ensure_ascii=False),
        }
        assistant_response = _assistant_response_payload(response_payload=response_payload)
        messages_complete = [*messages, assistant_response]
        inference_metadata = _inference_metadata(
            request_payload=request_payload,
            response_payload=response_payload,
        )
        agent_id = str(turn.get("agent_id", ""))
        agent_snapshot = dict(agents_payload.get(agent_id, {})) if isinstance(agents_payload.get(agent_id), dict) else {}
        environment_payload = {
            "project_dir": str(session_payload.get("project_dir", "")),
            "session_task": str(session_payload.get("task", "")),
            "provider_profile": _nested_str(config_snapshot, "provider", "profile"),
            "config_snapshot": config_snapshot,
        }
        traceability_payload = {
            "turn_id": str(turn.get("turn_id", "")),
            "event_seq_start": _optional_int(turn.get("event_seq_start")),
            "event_seq_end": _optional_int(turn.get("event_seq_end")),
            "started_at": _optional_str(turn.get("started_at")),
            "completed_at": _optional_str(turn.get("completed_at")),
            "status": _optional_str(turn.get("status")),
            "final_attempt": final_attempt_no,
            "llm_sequence": _optional_int(final_attempt.get("llm_sequence")),
            "request_file": _optional_str(final_attempt.get("request_file")),
            "response_file": _optional_str(final_attempt.get("response_file")),
            "response_ok": bool(final_attempt.get("ok", False)),
            "response_parse_error": _optional_str(final_attempt.get("parse_error")),
        }
        rows.append(
            {
                "example_id": str(turn.get("turn_id", "")),
                "messages": messages,
                "messages_complete": messages_complete,
                "target": action_target,
                "assistant_response": assistant_response,
                "actions": actions,
                "metadata": {
                    "session_id": str(turn.get("session_id", "")),
                    "agent_id": str(turn.get("agent_id", "")),
                    "agent_role": str(turn.get("agent_role", "")),
                    "step": _optional_int(turn.get("step")),
                    "turn_id": str(turn.get("turn_id", "")),
                    "final_attempt": final_attempt_no,
                    "scope": scope_payload,
                    **inference_metadata,
                },
                "environment": environment_payload,
                "traceability": traceability_payload,
                "agent_snapshot": agent_snapshot,
            }
        )
    return rows



def _find_attempt(attempts: list[dict[str, Any]], target_attempt: int) -> dict[str, Any] | None:
    """Resolve one protocol attempt by attempt index."""
    for item in attempts:
        if _optional_int(item.get("attempt")) == target_attempt:
            return item
    return None



def _optional_int(value: Any) -> int | None:
    """Parse integer or return ``None`` when missing/invalid."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    """Normalize optional text values into stripped string or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _assistant_response_payload(*, response_payload: dict[str, Any]) -> dict[str, Any]:
    """Build assistant response payload with full tool/reasoning details."""
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": str(response_payload.get("content", "") or ""),
    }
    reasoning = _optional_str(response_payload.get("reasoning"))
    if reasoning is not None:
        assistant["reasoning"] = reasoning
    tool_calls = response_payload.get("tool_calls")
    if isinstance(tool_calls, list):
        assistant["tool_calls"] = [dict(item) for item in tool_calls if isinstance(item, dict)]
    usage = response_payload.get("usage")
    if isinstance(usage, dict):
        assistant["usage"] = dict(usage)
    raw_events = response_payload.get("raw_events")
    if isinstance(raw_events, list):
        assistant["raw_events"] = [dict(item) for item in raw_events if isinstance(item, dict)]
    return assistant


def _inference_metadata(
    *,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> dict[str, Any]:
    """Extract inference metadata (provider/model/params) from attempt artifacts."""
    return {
        "inference_provider": _optional_str(
            response_payload.get("inference_provider", request_payload.get("inference_provider"))
        ),
        "inference_endpoint": _optional_str(
            response_payload.get("inference_endpoint", request_payload.get("inference_endpoint"))
        ),
        "inference_model": _optional_str(
            response_payload.get("inference_model", request_payload.get("inference_model", request_payload.get("model")))
        ),
        "inference_api_key_env": _optional_str(
            response_payload.get("inference_api_key_env", request_payload.get("inference_api_key_env"))
        ),
        "inference_parameters": dict(
            response_payload.get("inference_parameters", request_payload.get("inference_parameters", {}))
        )
        if isinstance(response_payload.get("inference_parameters", request_payload.get("inference_parameters", {})), dict)
        else {},
    }


def _nested_str(payload: dict[str, Any], *path: str) -> str | None:
    """Read nested dictionary key path as optional string."""
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _optional_str(current)

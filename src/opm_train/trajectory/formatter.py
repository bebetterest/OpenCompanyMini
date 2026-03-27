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
    """Build OpenAI-messages style SFT rows from filtered turn data."""
    rows: list[dict[str, Any]] = []
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
        messages = request_payload.get("messages")
        if not isinstance(messages, list) or not messages:
            continue

        actions = [dict(item) for item in list(turn.get("actions", [])) if isinstance(item, dict)]
        if not actions:
            continue

        target = {
            "role": "assistant",
            "content": json.dumps({"actions": actions}, ensure_ascii=False),
        }
        rows.append(
            {
                "example_id": str(turn.get("turn_id", "")),
                "messages": messages,
                "target": target,
                "metadata": {
                    "session_id": str(turn.get("session_id", "")),
                    "agent_id": str(turn.get("agent_id", "")),
                    "agent_role": str(turn.get("agent_role", "")),
                    "step": _optional_int(turn.get("step")),
                    "turn_id": str(turn.get("turn_id", "")),
                    "final_attempt": final_attempt_no,
                },
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

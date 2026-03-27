"""Session trajectory loading helpers for export flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opm_train.storage import SessionStorage


class ExportSchemaError(ValueError):
    """Raised when session artifacts do not match export schema requirements."""



def load_session_bundle(*, storage: SessionStorage, session_id: str) -> dict[str, Any]:
    """Load snapshot, events, and turns for one session."""
    snapshot = storage.read_snapshot(session_id)
    if int(snapshot.schema_version) < 4:
        raise ExportSchemaError("Export requires snapshot schema_version >= 4 (new schema only).")

    events = storage.load_events(session_id)
    turns = storage.load_turns(session_id)
    session_dir = storage.session_dir(session_id)
    enriched_turns = [_enrich_turn(turn, session_dir=session_dir) for turn in turns]

    return {
        "session_id": session_id,
        "schema_version": int(snapshot.schema_version),
        "session": dict(snapshot.session),
        "agents": {str(k): dict(v) for k, v in snapshot.agents.items()},
        "tool_runs": {str(k): dict(v) for k, v in snapshot.tool_runs.items()},
        "events": events,
        "turns": enriched_turns,
    }



def _enrich_turn(turn: dict[str, Any], *, session_dir: Path) -> dict[str, Any]:
    """Attach request/response payloads referenced by attempt artifact paths."""
    payload = dict(turn)
    attempts = [dict(item) for item in list(payload.get("attempts", [])) if isinstance(item, dict)]
    enriched_attempts: list[dict[str, Any]] = []
    for item in attempts:
        request_file = _optional_str(item.get("request_file"))
        response_file = _optional_str(item.get("response_file"))
        enriched = {
            **item,
            "request": _load_json_if_exists(_resolve_session_path(session_dir, request_file)),
            "response": _load_json_if_exists(_resolve_session_path(session_dir, response_file)),
        }
        enriched_attempts.append(enriched)
    payload["attempts"] = enriched_attempts
    return payload



def _resolve_session_path(session_dir: Path, candidate: str | None) -> Path | None:
    """Resolve session-relative path strings into absolute local paths."""
    if not candidate:
        return None
    value = Path(candidate)
    if value.is_absolute():
        return value
    return session_dir / value



def _load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    """Load one JSON object file if present and valid."""
    if path is None or not path.exists() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None



def _optional_str(value: Any) -> str | None:
    """Normalize optional path-like values into stripped string or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None

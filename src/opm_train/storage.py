"""JSONL event log and snapshot persistence for runtime sessions."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from opm_train.models import (
    AgentNode,
    AgentRole,
    AgentStatus,
    RunSession,
    SessionStatus,
    SnapshotState,
    ToolRun,
    ToolRunStatus,
)
from opm_train.utils import ensure_directory


SNAPSHOT_FILENAME = "state_snapshot.json"
EVENTS_FILENAME = "events.jsonl"
AGENTS_DIRNAME = "agents"
LLM_CALLS_DIRNAME = "llm_calls"
CONTEXT_COMPRESSIONS_DIRNAME = "context_compressions"
LOGS_DIRNAME = "logs"
RUNTIME_LOG_FILENAME = "runtime.log"
ERRORS_FILENAME = "errors.jsonl"
TIMERS_DIRNAME = "timers"
MODULE_TIMING_FILENAME = "module_timings.jsonl"


class SessionStorage:
    """Filesystem-backed storage for session events and snapshots."""

    def __init__(self, *, app_dir: Path, data_dir_name: str) -> None:
        """Initialize storage roots under configured application directory."""
        self.app_dir = app_dir.resolve()
        self.data_root = ensure_directory(self.app_dir / data_dir_name)
        self.sessions_root = ensure_directory(self.data_root / "sessions")
        self._sequence_cache: dict[Path, int] = {}

    def session_dir(self, session_id: str) -> Path:
        """Return per-session directory, creating it when needed."""
        return ensure_directory(self.sessions_root / str(session_id))

    def events_path(self, session_id: str) -> Path:
        """Return path to append-only JSONL event file."""
        return self.session_dir(session_id) / EVENTS_FILENAME

    def snapshot_path(self, session_id: str) -> Path:
        """Return path to structured snapshot JSON file."""
        return self.session_dir(session_id) / SNAPSHOT_FILENAME

    def agent_dir(self, session_id: str, agent_id: str) -> Path:
        """Return per-agent directory within one session."""
        return ensure_directory(self.session_dir(session_id) / AGENTS_DIRNAME / str(agent_id))

    def agent_llm_calls_dir(self, session_id: str, agent_id: str) -> Path:
        """Return per-agent LLM call directory."""
        return ensure_directory(self.agent_dir(session_id, agent_id) / LLM_CALLS_DIRNAME)

    def agent_context_compressions_dir(self, session_id: str, agent_id: str) -> Path:
        """Return per-agent context compression directory."""
        return ensure_directory(self.agent_dir(session_id, agent_id) / CONTEXT_COMPRESSIONS_DIRNAME)

    def logs_dir(self, session_id: str) -> Path:
        """Return per-session log directory."""
        return ensure_directory(self.session_dir(session_id) / LOGS_DIRNAME)

    def runtime_log_path(self, session_id: str) -> Path:
        """Return structured runtime log path."""
        return self.logs_dir(session_id) / RUNTIME_LOG_FILENAME

    def errors_path(self, session_id: str) -> Path:
        """Return structured runtime error log path."""
        return self.logs_dir(session_id) / ERRORS_FILENAME

    def timers_dir(self, session_id: str) -> Path:
        """Return per-session timer output directory."""
        return ensure_directory(self.session_dir(session_id) / TIMERS_DIRNAME)

    def module_timing_path(self, session_id: str) -> Path:
        """Return JSONL path containing per-module timing samples."""
        return self.timers_dir(session_id) / MODULE_TIMING_FILENAME

    def append_event(self, session_id: str, payload: dict[str, Any]) -> None:
        """Append one event record as compact JSON line."""
        self._append_jsonl(self.events_path(session_id), payload)

    def append_runtime_log(self, session_id: str, payload: dict[str, Any]) -> None:
        """Append one structured runtime log line."""
        self._append_jsonl(self.runtime_log_path(session_id), payload)

    def append_error_record(self, session_id: str, payload: dict[str, Any]) -> None:
        """Append one structured runtime error line."""
        self._append_jsonl(self.errors_path(session_id), payload)

    def append_timer_sample(self, session_id: str, payload: dict[str, Any]) -> None:
        """Append one per-module timing sample."""
        self._append_jsonl(self.module_timing_path(session_id), payload)

    def append_agent_llm_call_request(self, session_id: str, agent_id: str, payload: dict[str, Any]) -> int:
        """Append one per-agent LLM request payload and return sequence index."""
        llm_dir = self.agent_llm_calls_dir(session_id, agent_id)
        sequence = self._next_sequence(llm_dir)
        self._write_json(llm_dir / f"{sequence:04d}_request.json", payload)
        return sequence

    def append_agent_llm_call_response(
        self,
        session_id: str,
        agent_id: str,
        sequence: int,
        payload: dict[str, Any],
    ) -> Path:
        """Append one per-agent LLM response payload for the given sequence."""
        llm_dir = self.agent_llm_calls_dir(session_id, agent_id)
        path = llm_dir / f"{max(1, int(sequence)):04d}_response.json"
        self._write_json(path, payload)
        return path

    def append_agent_context_compression(self, session_id: str, agent_id: str, payload: dict[str, Any]) -> int:
        """Append one context compression record for an agent."""
        compression_dir = self.agent_context_compressions_dir(session_id, agent_id)
        sequence = self._next_sequence(compression_dir)
        self._write_json(compression_dir / f"{sequence:04d}.json", payload)
        return sequence

    def load_events(self, session_id: str) -> list[dict[str, Any]]:
        """Load all event records for a session in original order."""
        path = self.events_path(session_id)
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def write_snapshot(self, session_id: str, snapshot: SnapshotState) -> None:
        """Write snapshot payload as pretty JSON for inspectability."""
        path = self.snapshot_path(session_id)
        path.write_text(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read_snapshot(self, session_id: str) -> SnapshotState:
        """Read snapshot payload and normalize expected shapes."""
        path = self.snapshot_path(session_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("snapshot payload must be an object")
        return SnapshotState(
            schema_version=int(payload.get("schema_version", 3)),
            last_event_seq=int(payload.get("last_event_seq", 0)),
            session=dict(payload.get("session", {})),
            agents=_dict_of_dict(payload.get("agents", {})),
            tool_runs=_dict_of_dict(payload.get("tool_runs", {})),
        )

    def validate_snapshot_tail(self, session_id: str, *, expected_last_event_seq: int) -> bool:
        """Validate snapshot last_event_seq against strict event tail continuity."""
        expected = int(expected_last_event_seq)
        if expected < 0:
            return False
        events = self.load_events(session_id)
        if not events:
            return expected == 0
        if len(events) != expected:
            return False
        for index, event in enumerate(events, start=1):
            try:
                seq = int(event.get("seq", -1))
            except (TypeError, ValueError):
                return False
            if seq != index:
                return False
        return True

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        """Append one JSON object as a UTF-8 JSONL line."""
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        """Write one JSON object as pretty UTF-8 file."""
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _next_sequence(self, directory: Path) -> int:
        """Return next 1-based sequence index with lazy per-directory caching."""
        key = directory.resolve()
        cached = self._sequence_cache.get(key)
        if cached is None:
            max_sequence = 0
            for candidate in key.glob("*.json"):
                prefix = candidate.stem.split("_", 1)[0]
                if not prefix.isdigit():
                    continue
                max_sequence = max(max_sequence, int(prefix))
            next_value = max_sequence + 1
        else:
            next_value = cached + 1
        self._sequence_cache[key] = next_value
        return next_value


def session_to_dict(session: RunSession) -> dict[str, Any]:
    """Serialize ``RunSession`` while normalizing enum/path fields."""
    payload = asdict(session)
    payload["project_dir"] = str(session.project_dir)
    payload["status"] = session.status.value
    return payload


def session_from_dict(payload: dict[str, Any]) -> RunSession:
    """Deserialize ``RunSession`` from persisted payload."""
    return RunSession(
        id=str(payload["id"]),
        task=str(payload["task"]),
        project_dir=Path(str(payload["project_dir"])),
        root_agent_id=str(payload["root_agent_id"]),
        status=SessionStatus(str(payload.get("status", SessionStatus.RUNNING.value))),
        status_reason=_optional_str(payload.get("status_reason")),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
        final_summary=_optional_str(payload.get("final_summary")),
        config_snapshot=dict(payload.get("config_snapshot", {})),
    )


def agent_to_dict(agent: AgentNode) -> dict[str, Any]:
    """Serialize ``AgentNode`` while normalizing enum/path fields."""
    payload = asdict(agent)
    payload["role"] = agent.role.value
    payload["status"] = agent.status.value
    payload["workspace_path"] = str(agent.workspace_path)
    return payload


def agent_from_dict(payload: dict[str, Any]) -> AgentNode:
    """Deserialize ``AgentNode`` from persisted payload."""
    return AgentNode(
        id=str(payload["id"]),
        session_id=str(payload["session_id"]),
        name=str(payload["name"]),
        role=AgentRole(str(payload["role"])),
        instruction=str(payload["instruction"]),
        workspace_path=Path(str(payload["workspace_path"])),
        parent_agent_id=_optional_str(payload.get("parent_agent_id")),
        status=AgentStatus(str(payload.get("status", AgentStatus.PENDING.value))),
        status_reason=_optional_str(payload.get("status_reason")),
        children=[str(x) for x in list(payload.get("children", []))],
        step_count=int(payload.get("step_count", 0)),
        summary=_optional_str(payload.get("summary")),
        next_recommendation=_optional_str(payload.get("next_recommendation")),
        metadata=dict(payload.get("metadata", {})),
        conversation=[dict(x) for x in list(payload.get("conversation", [])) if isinstance(x, dict)],
    )


def tool_run_to_dict(run: ToolRun) -> dict[str, Any]:
    """Serialize ``ToolRun`` while normalizing enum fields."""
    payload = asdict(run)
    payload["status"] = run.status.value
    return payload


def tool_run_from_dict(payload: dict[str, Any]) -> ToolRun:
    """Deserialize ``ToolRun`` from persisted payload."""
    return ToolRun(
        id=str(payload["id"]),
        session_id=str(payload["session_id"]),
        agent_id=str(payload["agent_id"]),
        tool_name=str(payload["tool_name"]),
        arguments=dict(payload.get("arguments", {})),
        status=ToolRunStatus(str(payload.get("status", ToolRunStatus.QUEUED.value))),
        blocking=bool(payload.get("blocking", True)),
        created_at=str(payload.get("created_at", "")),
        started_at=_optional_str(payload.get("started_at")),
        completed_at=_optional_str(payload.get("completed_at")),
        result=dict(payload["result"]) if isinstance(payload.get("result"), dict) else None,
        error=_optional_str(payload.get("error")),
    )


def _optional_str(value: Any) -> str | None:
    """Return stripped string value or ``None`` when empty."""
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _dict_of_dict(value: Any) -> dict[str, dict[str, Any]]:
    """Normalize unknown mapping-like payload into ``dict[str, dict]``."""
    return {str(k): dict(v) for k, v in dict(value).items()}

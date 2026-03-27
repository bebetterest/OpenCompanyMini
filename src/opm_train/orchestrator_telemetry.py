"""Telemetry and runtime observability mixin for the orchestrator."""

from __future__ import annotations

from contextlib import contextmanager
import time
import traceback
from typing import Any

from opm_train.models import AgentNode, EventRecord
from opm_train.utils import json_ready, utc_now


class OrchestratorTelemetryMixin:
    """Attach event logging, error recording, and module timing helpers."""

    def _log_event(self, agent: AgentNode | None, event_type: str, payload: dict[str, Any]) -> None:
        """Append one canonical event record to JSONL trajectory."""
        if self.session is None:
            return
        self.event_seq += 1
        record = EventRecord(
            seq=self.event_seq,
            timestamp=utc_now(),
            session_id=self.session.id,
            agent_id=agent.id if agent else None,
            event_type=event_type,
            payload=payload,
        )
        self.storage.append_event(
            self.session.id,
            {
                "seq": record.seq,
                "timestamp": record.timestamp,
                "session_id": record.session_id,
                "agent_id": record.agent_id,
                "event_type": record.event_type,
                "payload": json_ready(record.payload),
            },
        )
        if event_type in {"llm_token", "llm_reasoning"}:
            return
        self._append_runtime_log(
            level="INFO",
            message=f"event:{event_type}",
            agent=agent,
            payload=payload,
        )

    def _append_runtime_log(
        self,
        *,
        level: str,
        message: str,
        agent: AgentNode | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append one structured runtime log record."""
        if self.session is None:
            return
        self.storage.append_runtime_log(
            self.session.id,
            {
                "timestamp": utc_now(),
                "level": str(level).strip().upper() or "INFO",
                "message": str(message),
                "agent_id": agent.id if agent else None,
                "payload": json_ready(payload or {}),
            },
        )

    def _record_exception(
        self,
        *,
        stage: str,
        exc: BaseException,
        agent: AgentNode | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Persist structured error record plus mirrored runtime log."""
        if self.session is None:
            return
        details = {
            "timestamp": utc_now(),
            "stage": str(stage),
            "agent_id": agent.id if agent else None,
            "error_type": type(exc).__name__,
            "error_message": str(exc).strip() or "<empty>",
            "payload": json_ready(payload or {}),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        self.storage.append_error_record(self.session.id, details)
        self._append_runtime_log(
            level="ERROR",
            message=f"{stage}:{type(exc).__name__}",
            agent=agent,
            payload={
                "error_message": details["error_message"],
            },
        )

    def _record_context_compression(self, *, agent: AgentNode, reason: str, result: dict[str, Any]) -> None:
        """Persist per-agent context compression record in sequence order."""
        if self.session is None:
            return
        metadata = agent.metadata if isinstance(agent.metadata, dict) else {}
        payload = {
            "timestamp": utc_now(),
            "reason": str(reason),
            "compressed": bool(result.get("compressed", False)),
            "summary_version": int(metadata.get("summary_version", 0) or 0),
            "summarized_until_message_index": int(metadata.get("summarized_until_message_index", -1) or -1),
            "context_summary": str(metadata.get("context_summary", "")),
            "result": json_ready(result),
        }
        self.storage.append_agent_context_compression(
            self.session.id,
            agent.id,
            payload,
            agent_name=agent.name,
        )
        self._append_runtime_log(
            level="INFO",
            message="context_compression_recorded",
            agent=agent,
            payload={
                "reason": reason,
                "compressed": payload["compressed"],
                "summary_version": payload["summary_version"],
            },
        )

    def _record_llm_call_request(self, *, agent: AgentNode, payload: dict[str, Any]) -> int:
        """Persist one per-agent LLM request payload and return sequence id."""
        if self.session is None:
            return 0
        sequence = self.storage.append_agent_llm_call_request(
            self.session.id,
            agent.id,
            payload,
            agent_name=agent.name,
        )
        self._log_event(
            agent,
            "llm_call_request_recorded",
            {
                "sequence": sequence,
                "protocol_attempt": payload.get("protocol_attempt"),
                "protocol_max_attempts": payload.get("protocol_max_attempts"),
            },
        )
        self._append_runtime_log(
            level="INFO",
            message="llm_request_recorded",
            agent=agent,
            payload={"sequence": sequence},
        )
        return sequence

    def _record_llm_call_response(self, *, agent: AgentNode, sequence: int, payload: dict[str, Any]) -> None:
        """Persist one per-agent LLM response payload for a request sequence."""
        if self.session is None:
            return
        self.storage.append_agent_llm_call_response(
            self.session.id,
            agent.id,
            sequence,
            payload,
            agent_name=agent.name,
        )
        self._log_event(
            agent,
            "llm_call_response_recorded",
            {
                "sequence": max(1, int(sequence)),
                "ok": bool(payload.get("ok", False)),
                "parse_error": payload.get("parse_error"),
            },
        )
        self._append_runtime_log(
            level="INFO",
            message="llm_response_recorded",
            agent=agent,
            payload={"sequence": max(1, int(sequence))},
        )

    @contextmanager
    def _timer_scope(
        self,
        module: str,
        *,
        agent: AgentNode | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """Measure one runtime block and append sample when timer mode is enabled."""
        if not self.timer_enabled:
            yield
            return
        started = time.perf_counter()
        status = "ok"
        error_type: str | None = None
        try:
            yield
        except BaseException as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            if self.session is None:
                return
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            sample = {
                "timestamp": utc_now(),
                "module": str(module),
                "agent_id": agent.id if agent else None,
                "elapsed_ms": elapsed_ms,
                "status": status,
                "payload": json_ready(payload or {}),
            }
            if error_type:
                sample["error_type"] = error_type
            self.storage.append_timer_sample(self.session.id, sample)

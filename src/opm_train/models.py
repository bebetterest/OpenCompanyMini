"""Canonical runtime data models for sessions, agents, tools, and snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AgentRole(str, Enum):
    """Agent role categories in the runtime topology."""

    ROOT = "root"
    WORKER = "worker"


class AgentStatus(str, Enum):
    """Agent lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionStatus(str, Enum):
    """Session lifecycle states."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ToolRunStatus(str, Enum):
    """Tool execution lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"


@dataclass(slots=True)
class RunSession:
    """Top-level session record."""

    id: str
    task: str
    project_dir: Path
    root_agent_id: str
    status: SessionStatus = SessionStatus.RUNNING
    status_reason: str | None = None
    created_at: str = ""
    updated_at: str = ""
    final_summary: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentNode:
    """Agent node with lineage, state, and conversation history."""

    id: str
    session_id: str
    name: str
    role: AgentRole
    instruction: str
    workspace_path: Path
    parent_agent_id: str | None = None
    status: AgentStatus = AgentStatus.PENDING
    status_reason: str | None = None
    children: list[str] = field(default_factory=list)
    step_count: int = 0
    summary: str | None = None
    next_recommendation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    conversation: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ToolRun:
    """Tool invocation record for observability and waiting/cancellation."""

    id: str
    session_id: str
    agent_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: ToolRunStatus = ToolRunStatus.QUEUED
    blocking: bool = True
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass(slots=True)
class EventRecord:
    """Append-only event stream record."""

    seq: int
    timestamp: str
    session_id: str
    agent_id: str | None
    event_type: str
    payload: dict[str, Any]


@dataclass(slots=True)
class SnapshotState:
    """Serializable scheduler snapshot for resume semantics."""

    schema_version: int
    last_event_seq: int
    session: dict[str, Any]
    agents: dict[str, dict[str, Any]]
    tool_runs: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot dataclass into plain dictionary payload."""
        return asdict(self)

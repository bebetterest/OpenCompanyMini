"""Session and root lifecycle mixin for runtime orchestrator."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from opm_train.models import AgentNode, AgentRole, AgentStatus, RunSession, SessionStatus, ToolRun, ToolRunStatus
from opm_train.storage import agent_from_dict, session_from_dict, tool_run_from_dict
from opm_train.tools import TERMINAL_TOOL_RUN_STATUSES
from opm_train.utils import utc_now


class OrchestratorSessionLifecycleMixin:
    """Attach session bootstrap/resume and root lifecycle helpers."""

    session: RunSession | None
    agents: dict[str, AgentNode]
    tool_runs: dict[str, ToolRun]
    agent_tasks: dict[str, asyncio.Task[Any]]

    def _root_agent(self) -> AgentNode:
        """Return current root agent from in-memory runtime state."""
        assert self.session is not None
        return self.agents[self.session.root_agent_id]

    async def _run_root_lifecycle(self) -> None:
        """Launch root task and run lifecycle finalization steps."""
        root = self._root_agent()
        with self._timer_scope("root_lifecycle", agent=root):
            self._launch_agent(root.id)
            await self._wait_root_completion()
            self._finalize_session_from_root()
            self._persist_snapshot()

    async def run_task(self, task: str) -> RunSession:
        """Create a new session and run until the root agent reaches terminal state."""
        session_id = self._new_id("session")
        root_id = self._new_id("agent")
        now = utc_now()
        self.session = RunSession(
            id=session_id,
            task=task,
            project_dir=self.project_dir,
            root_agent_id=root_id,
            status=SessionStatus.RUNNING,
            created_at=now,
            updated_at=now,
            config_snapshot=self.config.as_snapshot(),
        )
        root = AgentNode(
            id=root_id,
            session_id=session_id,
            name="Root Coordinator",
            role=AgentRole.ROOT,
            instruction=task,
            workspace_path=self.project_dir,
            status=AgentStatus.PENDING,
            metadata={"keep_pinned_messages": self.config.runtime.context.keep_pinned_messages},
            conversation=[
                {
                    "role": "user",
                    "content": self.prompt_library.render_runtime_message(
                        "root_initial_message",
                        task=task,
                        project_dir=str(self.project_dir),
                    ),
                }
            ],
        )
        self.agents = {root_id: root}
        self.tool_runs = {}
        self._reset_runtime_trackers()
        self.event_seq = 0
        self._agent_created_index = 1
        root.metadata["created_index"] = self._agent_created_index
        self._log_event(root, "session_started", {"task": task})
        self._persist_snapshot()
        await self._run_root_lifecycle()
        assert self.session is not None
        return self.session

    async def resume(self, session_id: str, instruction: str) -> RunSession:
        """Resume an existing session from snapshot plus strict tail validation."""
        snapshot = self.storage.read_snapshot(session_id)
        if not self.storage.validate_snapshot_tail(session_id, expected_last_event_seq=snapshot.last_event_seq):
            raise RuntimeError("Snapshot/event tail mismatch. Refuse to resume inconsistent session.")

        self.session = session_from_dict(snapshot.session)
        self.agents = {agent_id: agent_from_dict(payload) for agent_id, payload in snapshot.agents.items()}
        self.tool_runs = {run_id: tool_run_from_dict(payload) for run_id, payload in snapshot.tool_runs.items()}
        self._reset_runtime_trackers()
        self.event_seq = int(snapshot.last_event_seq)
        self._agent_created_index = max(
            [
                int(agent.metadata.get("created_index", 0) or 0)
                for agent in self.agents.values()
                if isinstance(agent.metadata, dict)
            ]
            or [0]
        )

        self._restore_tool_runs_after_resume()

        root = self._root_agent()
        root.conversation.append(
            {
                "role": "user",
                "content": self.prompt_library.render_runtime_message(
                    "resume_instruction_message",
                    instruction=instruction,
                ),
            }
        )
        root.status = AgentStatus.PENDING
        self.session.status = SessionStatus.RUNNING
        self.session.status_reason = None
        self.session.updated_at = utc_now()
        self._log_event(root, "session_resumed", {"instruction": instruction})
        self._persist_snapshot()

        await self._run_root_lifecycle()
        return self.session

    def _restore_tool_runs_after_resume(self) -> None:
        """Restore tool waiters and abandon non-terminal runs during resume."""
        for run in self.tool_runs.values():
            event = asyncio.Event()
            if run.status.value in TERMINAL_TOOL_RUN_STATUSES:
                event.set()
            self.tool_run_events[run.id] = event
            if run.status.value not in TERMINAL_TOOL_RUN_STATUSES:
                self._set_tool_run_terminal(
                    run,
                    status=ToolRunStatus.ABANDONED,
                    error="tool_run_abandoned_on_resume",
                )

    async def _wait_root_completion(self) -> None:
        """Wait for root task and cancel all non-root tasks during shutdown."""
        assert self.session is not None
        root_task = self.agent_tasks[self.session.root_agent_id]
        root_error: BaseException | None = None
        try:
            await root_task
        except BaseException as exc:  # pragma: no cover - guarded cleanup path
            root_error = exc
        finally:
            for agent_id, task in list(self.agent_tasks.items()):
                if agent_id == self.session.root_agent_id:
                    continue
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
        if root_error is not None:
            root = self._root_agent()
            if root.status.value not in {AgentStatus.COMPLETED.value, AgentStatus.FAILED.value, AgentStatus.CANCELLED.value}:
                root.status = AgentStatus.FAILED
                root.status_reason = f"root_task_error:{type(root_error).__name__}:{root_error}"
            self._record_exception(stage="root_wait", exc=root_error, agent=root)

    def _finalize_session_from_root(self) -> None:
        """Derive session terminal status and summary from root outcome."""
        assert self.session is not None
        root = self._root_agent()
        self.session.updated_at = utc_now()
        if root.status == AgentStatus.COMPLETED:
            self.session.status = SessionStatus.COMPLETED
        elif root.status == AgentStatus.CANCELLED:
            self.session.status = SessionStatus.INTERRUPTED
        else:
            self.session.status = SessionStatus.FAILED
        self.session.status_reason = root.status_reason
        self.session.final_summary = root.summary
        self._log_event(root, "session_finished", {"status": self.session.status.value})

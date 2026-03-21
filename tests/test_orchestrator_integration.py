from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from opm_train.llm import ChatResult
from opm_train.models import AgentNode, AgentRole, AgentStatus
from opm_train.orchestrator import RuntimeOrchestrator

APP_DIR = Path(__file__).resolve().parents[1]


def _route(messages: list[dict[str, Any]]) -> str:
    initial = str(messages[1].get("content", "")) if len(messages) > 1 else ""
    if initial.startswith("User task:"):
        return "root"
    if initial.startswith("Assigned instruction:"):
        line = initial.splitlines()[0]
        instruction = line.removeprefix("Assigned instruction:").strip()
        return f"worker:{instruction}"
    return "root"


def _extract_last_tool_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        if str(message.get("role", "")) != "tool":
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


class BlockingSpawnLLM:
    def __init__(self) -> None:
        self.root_calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        route = _route(kwargs["messages"])
        if route == "root":
            self.root_calls += 1
            if self.root_calls == 1:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "spawn_agent",
                                "name": "Inspect",
                                "instruction": "Inspect the repository",
                                "blocking": True,
                            }
                        ]
                    }
                )
            else:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "finish",
                                "status": "completed",
                                "summary": "Root done.",
                            }
                        ]
                    }
                )
            return ChatResult(content=text, raw_events=[])

        text = json.dumps(
            {
                "actions": [
                    {
                        "type": "finish",
                        "status": "completed",
                        "summary": "Worker done.",
                        "next_recommendation": "Finalize.",
                    }
                ]
            }
        )
        return ChatResult(content=text, raw_events=[])


class WaitRunLLM:
    def __init__(self) -> None:
        self.root_calls = 0
        self.worker_calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        messages = kwargs["messages"]
        route = _route(messages)
        if route == "root":
            self.root_calls += 1
            if self.root_calls == 1:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "spawn_agent",
                                "name": "Slow",
                                "instruction": "slow child",
                                "blocking": False,
                            }
                        ]
                    }
                )
            elif self.root_calls == 2:
                tool_payload = _extract_last_tool_payload(messages)
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "wait_run",
                                "run_id": str(tool_payload.get("tool_run_id", "")),
                                "timeout_seconds": 2,
                            }
                        ]
                    }
                )
            else:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "finish",
                                "status": "completed",
                                "summary": "Done after wait.",
                            }
                        ]
                    }
                )
            return ChatResult(content=text, raw_events=[])

        self.worker_calls += 1
        if self.worker_calls == 1:
            text = json.dumps({"actions": [{"type": "wait_time", "seconds": 0.02}]})
        else:
            text = json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "child done",
                            "next_recommendation": "none",
                        }
                    ]
                }
            )
        return ChatResult(content=text, raw_events=[])


class FinishRejectedThenWaitLLM:
    def __init__(self) -> None:
        self.root_calls = 0
        self.shell_run_id = ""
        self.saw_finish_rejection = False

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        messages = kwargs["messages"]
        self.root_calls += 1
        tool_payload = _extract_last_tool_payload(messages)
        if self.root_calls == 1:
            text = json.dumps(
                {
                    "actions": [
                        {
                            "type": "shell",
                            "command": "python -c \"import time;time.sleep(1);print('done')\"",
                            "blocking": False,
                        }
                    ]
                }
            )
        elif self.root_calls == 2:
            text = json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "should be rejected first",
                        }
                    ]
                }
            )
        elif self.root_calls == 3:
            self.saw_finish_rejection = bool(tool_payload.get("finish_rejected"))
            unfinished = tool_payload.get("unfinished_tool_runs")
            if isinstance(unfinished, list) and unfinished:
                self.shell_run_id = str((unfinished[0] or {}).get("run_id", "")).strip()
            text = json.dumps(
                {
                    "actions": [
                        {
                            "type": "wait_run",
                            "run_id": self.shell_run_id,
                            "timeout_seconds": 3,
                        }
                    ]
                }
            )
        else:
            text = json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "finished after waiting own tool run",
                        }
                    ]
                }
            )
        return ChatResult(content=text, raw_events=[])


class SteerAndCancelLLM:
    def __init__(self) -> None:
        self.root_calls = 0
        self.worker_calls = 0
        self.worker_received_steer = False
        self.child_agent_id = ""

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        messages = kwargs["messages"]
        route = _route(messages)
        if route == "root":
            self.root_calls += 1
            tool_payload = _extract_last_tool_payload(messages)
            if self.root_calls == 1:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "spawn_agent",
                                "name": "Steered",
                                "instruction": "react to steer",
                                "blocking": False,
                            }
                        ]
                    }
                )
            elif self.root_calls == 2:
                self.child_agent_id = str(tool_payload.get("child_agent_id", "")).strip()
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "steer_agent",
                                "agent_id": self.child_agent_id,
                                "content": "please include STEERED",
                            }
                        ]
                    }
                )
            elif self.root_calls == 3:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "cancel_agent",
                                "agent_id": self.child_agent_id,
                            }
                        ]
                    }
                )
            else:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "finish",
                                "status": "partial",
                                "summary": "cancelled child after steer",
                            }
                        ]
                    }
                )
            return ChatResult(content=text, raw_events=[])

        self.worker_calls += 1
        if "please include STEERED" in json.dumps(messages, ensure_ascii=False):
            self.worker_received_steer = True
        text = json.dumps({"actions": [{"type": "wait_time", "seconds": 0.5}]})
        return ChatResult(content=text, raw_events=[])


class ResumeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            summary = "initial summary"
        else:
            summary = "resume summary"
        text = json.dumps(
            {
                "actions": [
                    {
                        "type": "finish",
                        "status": "completed",
                        "summary": summary,
                    }
                ]
            }
        )
        return ChatResult(content=text, raw_events=[])


class SpawnDefaultNonBlockingLLM:
    def __init__(self) -> None:
        self.root_calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        messages = kwargs["messages"]
        route = _route(messages)
        if route == "root":
            self.root_calls += 1
            if self.root_calls == 1:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "spawn_agent",
                                "name": "DefaultNonBlocking",
                                "instruction": "finish quickly",
                            }
                        ]
                    }
                )
            elif self.root_calls == 2:
                tool_payload = _extract_last_tool_payload(messages)
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "wait_run",
                                "run_id": str(tool_payload.get("tool_run_id", "")),
                                "timeout_seconds": 2,
                            }
                        ]
                    }
                )
            else:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "finish",
                                "status": "completed",
                                "summary": "done",
                            }
                        ]
                    }
                )
            return ChatResult(content=text, raw_events=[])

        text = json.dumps(
            {
                "actions": [
                    {
                        "type": "finish",
                        "status": "completed",
                        "summary": "child done",
                        "next_recommendation": "none",
                    }
                ]
            }
        )
        return ChatResult(content=text, raw_events=[])


class FailingWorkerVisibleErrorLLM:
    def __init__(self) -> None:
        self.root_calls = 0
        self.child_agent_id = ""

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        messages = kwargs["messages"]
        route = _route(messages)
        if route == "root":
            self.root_calls += 1
            tool_payload = _extract_last_tool_payload(messages)
            if self.root_calls == 1:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "spawn_agent",
                                "name": "CrashWorker",
                                "instruction": "return invalid finish payload",
                                "blocking": False,
                            }
                        ]
                    }
                )
            elif self.root_calls == 2:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "wait_run",
                                "run_id": str(tool_payload.get("tool_run_id", "")),
                                "timeout_seconds": 2,
                            }
                        ]
                    }
                )
            elif self.root_calls == 3:
                self.child_agent_id = str(((tool_payload.get("result") or {}).get("child_agent_id", ""))).strip()
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "get_agent_run",
                                "agent_id": self.child_agent_id,
                            }
                        ]
                    }
                )
            else:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "finish",
                                "status": "partial",
                                "summary": "worker failure observed and recorded",
                            }
                        ]
                    }
                )
            return ChatResult(content=text, raw_events=[])

        # Missing summary on finish -> runtime validation error in worker loop.
        text = json.dumps({"actions": [{"type": "finish", "status": "completed"}]})
        return ChatResult(content=text, raw_events=[])


class RootCancelBlockedLLM:
    def __init__(self) -> None:
        self.root_calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        messages = kwargs["messages"]
        self.root_calls += 1
        tool_payload = _extract_last_tool_payload(messages)
        if self.root_calls == 1:
            text = json.dumps({"actions": [{"type": "list_agent_runs", "limit": 10}]})
        elif self.root_calls == 2:
            root_id = str((tool_payload.get("items") or [{}])[0].get("id", ""))
            text = json.dumps({"actions": [{"type": "cancel_agent", "agent_id": root_id}]})
        else:
            text = json.dumps({"actions": [{"type": "finish", "status": "partial", "summary": "root cancel blocked"}]})
        return ChatResult(content=text, raw_events=[])


class CancelUnknownAgentLLM:
    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        return ChatResult(
            content=json.dumps({"actions": [{"type": "cancel_agent", "agent_id": "agent-missing"}]}),
            raw_events=[],
        )


class BlockingSpawnFailedStatusLLM:
    def __init__(self) -> None:
        self.root_calls = 0
        self.spawn_status = ""

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        messages = kwargs["messages"]
        route = _route(messages)
        if route == "root":
            self.root_calls += 1
            if self.root_calls == 1:
                return ChatResult(
                    content=json.dumps(
                        {
                            "actions": [
                                {
                                    "type": "spawn_agent",
                                    "name": "FailingChild",
                                    "instruction": "worker must fail",
                                    "blocking": True,
                                }
                            ]
                        }
                    ),
                    raw_events=[],
                )
            tool_payload = _extract_last_tool_payload(messages)
            self.spawn_status = str(tool_payload.get("status", ""))
            return ChatResult(
                content=json.dumps(
                    {
                        "actions": [
                            {
                                "type": "finish",
                                "status": "completed",
                                "summary": "root observed child status",
                            }
                        ]
                    }
                ),
                raw_events=[],
            )

        return ChatResult(
            content=json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "failed",
                            "summary": "worker failed intentionally",
                            "next_recommendation": "retry",
                        }
                    ]
                }
            ),
            raw_events=[],
        )


class StepCountLLM:
    def __init__(self) -> None:
        self.root_calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        self.root_calls += 1
        if self.root_calls == 1:
            text = json.dumps({"actions": [{"type": "wait_time", "seconds": 0}]})
        else:
            text = json.dumps({"actions": [{"type": "finish", "status": "completed", "summary": "done"}]})
        return ChatResult(content=text, raw_events=[])


class RetryInvalidJsonThenSuccessLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            return ChatResult(content="this is not json", raw_events=[])
        return ChatResult(
            content=json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "recovered after retry",
                        }
                    ]
                }
            ),
            raw_events=[],
        )


class RetryInvalidToolCallThenSuccessLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            return ChatResult(
                content="",
                raw_events=[],
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "wait_time",
                        "arguments_json": "{",
                    }
                ],
            )
        return ChatResult(
            content=json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "tool-call parse recovered",
                        }
                    ]
                }
            ),
            raw_events=[],
        )


class RetryAlwaysInvalidLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        self.calls += 1
        return ChatResult(content="still invalid payload", raw_events=[])


class ManualCompressAndChineseLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            text = json.dumps({"actions": [{"type": "compress_context"}]})
        else:
            text = json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "中文总结完成",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return ChatResult(content=text, raw_events=[])


@pytest.mark.asyncio
async def test_root_worker_blocking_spawn_flow() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = BlockingSpawnLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("Inspect project")
        assert session.status.value == "completed"
        assert session.final_summary == "Root done."
        assert len(orchestrator.agents) == 2


@pytest.mark.asyncio
async def test_parallel_wait_run_flow() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = WaitRunLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("Test wait_run")
        assert session.status.value == "completed"
        assert llm.worker_calls >= 2


@pytest.mark.asyncio
async def test_finish_is_rejected_until_agent_own_tool_runs_are_terminal() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = FinishRejectedThenWaitLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("test finish rejection on active own tool run")
        assert session.status.value == "completed"
        assert llm.saw_finish_rejection is True
        assert llm.shell_run_id

        root = orchestrator.agents[session.root_agent_id]
        tool_payloads = [
            json.loads(str(message.get("content", "")))
            for message in root.conversation
            if str(message.get("role", "")) == "tool" and str(message.get("content", "")).strip().startswith("{")
        ]
        assert any(bool(payload.get("finish_rejected")) for payload in tool_payloads if isinstance(payload, dict))


@pytest.mark.asyncio
async def test_steer_and_cancel_flow() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = SteerAndCancelLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("Test steer and cancel")
        assert session.status.value in {"completed", "failed"}  # partial finish maps to completed
        child_agents = [agent for agent in orchestrator.agents.values() if agent.parent_agent_id]
        assert child_agents
        assert child_agents[0].status in {AgentStatus.CANCELLED, AgentStatus.FAILED, AgentStatus.COMPLETED}


@pytest.mark.asyncio
async def test_resume_uses_snapshot_and_continues_events() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = ResumeLLM()
        first = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await first.run_task("first")
        events_before = len(first.storage.load_events(session.id))
        second = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        resumed = await second.resume(session.id, "continue")
        events_after = len(second.storage.load_events(session.id))
        assert resumed.final_summary == "resume summary"
        assert events_after > events_before


@pytest.mark.asyncio
async def test_resume_marks_non_restorable_tool_runs_as_abandoned() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = ResumeLLM()
        first = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await first.run_task("first")
        snapshot_path = first.storage.snapshot_path(session.id)
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        root_id = str(snapshot["session"]["root_agent_id"])

        snapshot["tool_runs"]["tool-stale"] = {
            "id": "tool-stale",
            "session_id": session.id,
            "agent_id": root_id,
            "tool_name": "shell",
            "arguments": {"type": "shell", "command": "echo stale"},
            "status": "running",
            "blocking": False,
            "created_at": "2026-03-21T00:00:00.000+00:00",
            "started_at": "2026-03-21T00:00:00.100+00:00",
            "completed_at": None,
            "result": None,
            "error": None,
        }
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

        second = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        await second.resume(session.id, "continue")
        stale = second.tool_runs["tool-stale"]
        assert stale.status.value == "abandoned"
        assert stale.error == "tool_run_abandoned_on_resume"


@pytest.mark.asyncio
async def test_agent_artifacts_persist_llm_calls_and_context_compressions() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = ManualCompressAndChineseLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("请用中文总结当前任务")
        assert session.status.value == "completed"
        assert session.final_summary == "中文总结完成"

        root = orchestrator.agents[session.root_agent_id]
        agent_dir = orchestrator.storage.agent_dir(session.id, root.id)
        llm_dir = agent_dir / "llm_calls"
        context_dir = agent_dir / "context_compressions"

        llm_files = sorted(path.name for path in llm_dir.glob("*.json"))
        assert llm_files == [
            "0001_request.json",
            "0001_response.json",
            "0002_request.json",
            "0002_response.json",
        ]
        assert "请用中文总结当前任务" in (llm_dir / "0001_request.json").read_text(encoding="utf-8")
        assert "中文总结完成" in (llm_dir / "0002_response.json").read_text(encoding="utf-8")
        first_request = json.loads((llm_dir / "0001_request.json").read_text(encoding="utf-8"))
        first_response = json.loads((llm_dir / "0001_response.json").read_text(encoding="utf-8"))
        expected_model = orchestrator.config.provider.active_profile().model
        assert first_request["inference_provider"] == "openrouter"
        assert first_request["inference_model"] == expected_model
        assert first_request["inference_parameters"]["tool_choice"] == "auto"
        assert first_request["inference_parameters"]["parallel_tool_calls"] is True
        assert first_response["inference_provider"] == "openrouter"
        assert first_response["inference_model"] == expected_model

        context_files = sorted(path.name for path in context_dir.glob("*.json"))
        assert context_files == ["0001.json"]


@pytest.mark.asyncio
async def test_spawn_agent_default_is_non_blocking() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = SpawnDefaultNonBlockingLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("verify spawn default")
        assert session.status.value == "completed"
        spawn_runs = [run for run in orchestrator.tool_runs.values() if run.tool_name == "spawn_agent"]
        assert len(spawn_runs) == 1
        assert spawn_runs[0].blocking is False


@pytest.mark.asyncio
async def test_wait_run_rejects_non_numeric_timeout() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        with pytest.raises(ValueError, match="timeout_seconds must be numeric"):
            await orchestrator._tool_wait_run({"run_id": "tool-1", "timeout_seconds": "abc"})


@pytest.mark.asyncio
async def test_finish_rejection_checks_only_current_agent_tool_runs() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        root = AgentNode(
            id="agent-root",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="root",
            workspace_path=project,
        )
        worker = AgentNode(
            id="agent-worker",
            session_id="session-1",
            name="worker",
            role=AgentRole.WORKER,
            instruction="worker",
            workspace_path=project,
            parent_agent_id=root.id,
        )
        orchestrator.agents = {root.id: root, worker.id: worker}
        orchestrator.tool_runs = {}
        orchestrator._reset_runtime_trackers()

        root_run = orchestrator._create_tool_run(
            agent=root,
            tool_name="shell",
            arguments={"type": "shell", "command": "sleep 1"},
            blocking=False,
        )
        orchestrator._mark_tool_run_running(root_run)

        allowed = orchestrator._handle_finish_action(
            agent=worker,
            action={
                "type": "finish",
                "status": "completed",
                "summary": "worker can finish when only root has running tool",
            },
        )
        assert allowed.get("ok") is True
        assert isinstance(allowed.get("finish_payload"), dict)

        worker_run = orchestrator._create_tool_run(
            agent=worker,
            tool_name="shell",
            arguments={"type": "shell", "command": "sleep 1"},
            blocking=False,
        )
        orchestrator._mark_tool_run_running(worker_run)
        rejected = orchestrator._handle_finish_action(
            agent=worker,
            action={
                "type": "finish",
                "status": "completed",
                "summary": "should be rejected",
            },
        )
        assert rejected.get("finish_rejected") is True
        unfinished = rejected.get("unfinished_tool_runs")
        assert isinstance(unfinished, list)
        assert [item["run_id"] for item in unfinished] == [worker_run.id]


@pytest.mark.asyncio
async def test_cancel_tool_run_does_not_cancel_spawned_child_agent() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=SteerAndCancelLLM())
        parent = AgentNode(
            id="agent-parent",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="spawn child",
            workspace_path=project,
        )
        orchestrator.agents = {parent.id: parent}
        orchestrator.tool_runs = {}
        orchestrator._reset_runtime_trackers()

        run = orchestrator._create_tool_run(
            agent=parent,
            tool_name="spawn_agent",
            arguments={"type": "spawn_agent", "instruction": "react to steer"},
            blocking=False,
        )
        orchestrator._mark_tool_run_running(run)
        result = await orchestrator._tool_spawn_agent(
            run,
            parent,
            {"instruction": "react to steer", "name": "Spawned"},
        )

        child_id = str(result["child_agent_id"])
        cancel_result = await orchestrator._tool_cancel_tool_run({"run_id": run.id})

        assert cancel_result["status"] == "cancelled"
        assert orchestrator.agents[child_id].status != AgentStatus.CANCELLED
        assert run.error == "cancel_tool_run"


@pytest.mark.asyncio
async def test_shell_finishes_inline_within_shell_inline_wait_seconds() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        orchestrator.config.runtime.tools.shell_inline_wait_seconds = 0.5
        agent = AgentNode(
            id="agent-1",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="run shell",
            workspace_path=project,
        )
        run = orchestrator._create_tool_run(
            agent=agent,
            tool_name="shell",
            arguments={"type": "shell", "command": "printf inline-ok"},
            blocking=True,
        )
        orchestrator._mark_tool_run_running(run)
        result = await orchestrator._tool_shell(run, {"command": "printf inline-ok", "timeout_seconds": 5})
        assert run.status.value == "completed"
        assert result["exit_code"] == 0
        assert result["stdout"] == "inline-ok"


@pytest.mark.asyncio
async def test_shell_exceeds_inline_wait_then_get_tool_run_shows_cumulative_output() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        orchestrator.config.runtime.tools.shell_inline_wait_seconds = 0.05
        agent = AgentNode(
            id="agent-1",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="run shell",
            workspace_path=project,
        )
        command = (
            "python -c \"import sys,time;"
            "sys.stdout.write('start\\\\n');sys.stdout.flush();"
            "time.sleep(0.2);"
            "sys.stdout.write('end\\\\n');sys.stdout.flush()\""
        )
        run = orchestrator._create_tool_run(
            agent=agent,
            tool_name="shell",
            arguments={"type": "shell", "command": command},
            blocking=True,
        )
        orchestrator._mark_tool_run_running(run)
        response = await orchestrator._tool_shell(run, {"command": command, "timeout_seconds": 5})
        assert response["status"] == "running"
        assert run.status.value == "running"

        stdout_seen = ""
        for _ in range(100):
            detail = orchestrator._tool_get_tool_run({"run_id": run.id})
            stdout_seen = str((detail.get("result") or {}).get("stdout", ""))
            if "start" in stdout_seen:
                break
            await asyncio.sleep(0.01)
        assert "start" in stdout_seen

        wait_result = await orchestrator._tool_wait_run({"run_id": run.id, "timeout_seconds": 2})
        assert wait_result["timed_out"] is False
        assert wait_result["status"] == "completed"
        assert "end" in str((wait_result.get("result") or {}).get("stdout", ""))


@pytest.mark.asyncio
async def test_worker_exception_recorded_and_visible_via_get_agent_run() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = FailingWorkerVisibleErrorLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("observe worker exception")
        assert session.status.value == "completed"
        assert llm.child_agent_id
        detail = orchestrator._tool_get_agent_run({"agent_id": llm.child_agent_id})
        assert detail["status"] == "failed"
        assert "agent_loop_error:ValueError" in str(detail.get("status_reason", ""))
        assert isinstance(detail.get("last_error"), dict)
        assert detail["last_error"]["type"] == "ValueError"
        errors_path = orchestrator.storage.errors_path(session.id)
        assert errors_path.exists()
        error_lines = [json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert error_lines
        assert any(str(item.get("error_type", "")) == "ValueError" for item in error_lines)


@pytest.mark.asyncio
async def test_cancel_agent_cannot_cancel_root() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RootCancelBlockedLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("attempt cancel root")
        assert session.status.value == "completed"
        cancel_runs = [run for run in orchestrator.tool_runs.values() if run.tool_name == "cancel_agent"]
        assert len(cancel_runs) == 1
        assert cancel_runs[0].result is not None
        assert cancel_runs[0].result["blocked"] is True
        assert cancel_runs[0].result["reason"] == "root_agent_not_cancellable"


@pytest.mark.asyncio
async def test_cancel_agent_unknown_agent_id_fails_session() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = CancelUnknownAgentLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("attempt cancel unknown")
        assert session.status.value == "failed"
        root = orchestrator.agents[session.root_agent_id]
        assert "unknown agent_id: agent-missing" in str(root.status_reason)


@pytest.mark.asyncio
async def test_step_count_is_incremented_each_loop_turn() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = StepCountLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("step count")
        root = orchestrator.agents[session.root_agent_id]
        assert root.step_count >= 2


@pytest.mark.asyncio
async def test_blocking_spawn_status_reflects_failed_child() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = BlockingSpawnFailedStatusLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("verify blocking spawn status")
        assert session.status.value == "completed"
        assert llm.spawn_status == "failed"


@pytest.mark.asyncio
async def test_protocol_retry_recovers_from_invalid_json_payload() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RetryInvalidJsonThenSuccessLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        orchestrator.config.runtime.limits.max_protocol_retries = 1
        session = await orchestrator.run_task("retry invalid json")
        assert session.status.value == "completed"
        assert session.final_summary == "recovered after retry"
        assert llm.calls == 2


@pytest.mark.asyncio
async def test_protocol_retry_recovers_from_invalid_tool_call_arguments() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RetryInvalidToolCallThenSuccessLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        orchestrator.config.runtime.limits.max_protocol_retries = 1
        session = await orchestrator.run_task("retry invalid tool call")
        assert session.status.value == "completed"
        assert session.final_summary == "tool-call parse recovered"
        assert llm.calls == 2


@pytest.mark.asyncio
async def test_protocol_retry_exhaustion_falls_back_to_invalid_payload_finish() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RetryAlwaysInvalidLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        orchestrator.config.runtime.limits.max_protocol_retries = 2
        session = await orchestrator.run_task("retry exhausted")
        assert session.status.value == "completed"
        assert session.final_summary == "Model returned invalid action payload."
        assert llm.calls == 3

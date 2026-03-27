from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from opm_train.llm import ChatResult
from opm_train.orchestrator import RuntimeOrchestrator
from opm_train.storage import SessionStorage
from opm_train.trajectory import ExportSchemaError, export_trajectory

APP_DIR = Path(__file__).resolve().parents[1]


class RetryThenFinishLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        _ = kwargs
        self.calls += 1
        if self.calls == 1:
            return ChatResult(
                content="",
                raw_events=[],
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "wait_time",
                            "arguments": "{",
                        },
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
                            "summary": "export-ready",
                        }
                    ]
                }
            ),
            raw_events=[],
        )


class SpawnThenFinishLLM:
    def __init__(self) -> None:
        self.root_calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        messages = kwargs["messages"]
        first_user = str(messages[1].get("content", "")) if len(messages) > 1 else ""
        if first_user.startswith("Assigned instruction:"):
            return ChatResult(
                content=json.dumps(
                    {
                        "actions": [
                            {
                                "type": "finish",
                                "status": "completed",
                                "summary": "worker done",
                                "next_recommendation": "none",
                            }
                        ]
                    }
                ),
                raw_events=[],
            )

        self.root_calls += 1
        if self.root_calls == 1:
            return ChatResult(
                content=json.dumps(
                    {
                        "actions": [
                            {
                                "type": "spawn_agent",
                                "name": "worker",
                                "instruction": "do work",
                                "blocking": True,
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
                            "status": "completed",
                            "summary": "root done",
                        }
                    ]
                }
            ),
            raw_events=[],
        )


@pytest.mark.asyncio
async def test_export_trajectory_raw_and_sft_scope_filters() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RetryThenFinishLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        orchestrator.config.runtime.limits.max_protocol_retries = 1
        session = await orchestrator.run_task("trajectory export")
        assert session.status.value == "completed"

        raw = export_trajectory(
            storage=orchestrator.storage,
            session_id=session.id,
            mode="raw",
        )
        assert isinstance(raw, dict)
        assert raw["session_id"] == session.id
        assert len(raw["turns"]) == 1
        attempts = raw["turns"][0]["attempts"]
        assert len(attempts) == 2
        assert isinstance(attempts[0]["request"], dict)
        assert isinstance(attempts[0]["response"], dict)

        scoped = export_trajectory(
            storage=orchestrator.storage,
            session_id=session.id,
            mode="raw",
            agent_id=session.root_agent_id,
            step=1,
        )
        assert isinstance(scoped, dict)
        assert len(scoped["turns"]) == 1
        turn = scoped["turns"][0]
        assert int(turn["step"]) == 1
        assert all(
            int(turn["event_seq_start"]) <= int(event["seq"]) <= int(turn["event_seq_end"])
            for event in scoped["events"]
        )

        sft_rows = export_trajectory(
            storage=orchestrator.storage,
            session_id=session.id,
            mode="sft",
        )
        assert isinstance(sft_rows, list)
        assert len(sft_rows) == 1
        target = sft_rows[0]["target"]
        payload = json.loads(target["content"])
        assert payload["actions"][0]["type"] == "finish"


@pytest.mark.asyncio
async def test_export_step_scope_filters_agents_tool_runs_and_events() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = SpawnThenFinishLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("scope filter")
        assert session.status.value == "completed"

        scoped = export_trajectory(
            storage=orchestrator.storage,
            session_id=session.id,
            mode="raw",
            agent_id=session.root_agent_id,
            step=1,
        )
        assert isinstance(scoped, dict)
        assert list(scoped["agents"].keys()) == [session.root_agent_id]
        assert all(str(item.get("agent_id", "")) == session.root_agent_id for item in scoped["events"])
        assert all(
            str(run.get("agent_id", "")) == session.root_agent_id
            for run in dict(scoped["tool_runs"]).values()
            if isinstance(run, dict)
        )


def test_export_trajectory_rejects_old_snapshot_schema() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = SessionStorage(app_dir=Path(temp_dir), data_dir_name=".opm_train")
        session_id = "session-old"
        snapshot_path = storage.snapshot_path(session_id)
        snapshot_path.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "last_event_seq": 0,
                    "session": {},
                    "agents": {},
                    "tool_runs": {},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ExportSchemaError, match="schema_version >= 4"):
            export_trajectory(storage=storage, session_id=session_id, mode="raw")

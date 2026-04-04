from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from opm_train.llm import ChatResult
from opm_train.models import AgentNode, AgentRole, AgentStatus, ToolRun
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


def _find_root_agent_id_from_runs(payload: dict[str, Any]) -> str:
    runs = payload.get("agent_runs")
    if not isinstance(runs, list):
        return ""
    for row in runs:
        if not isinstance(row, dict):
            continue
        if str(row.get("role", "")).strip() == "root":
            return str(row.get("id", "")).strip()
    return ""


def _new_wait_request_run(orchestrator: RuntimeOrchestrator, agent: AgentNode, *, tool_name: str) -> ToolRun:
    request_run = orchestrator._create_tool_run(
        agent=agent,
        tool_name=tool_name,
        arguments={"type": tool_name},
        blocking=True,
    )
    orchestrator._mark_tool_run_running(request_run)
    return request_run


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
                                "tool_run_id": str(tool_payload.get("tool_run_id", "")),
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
            text = json.dumps({"actions": [{"type": "list_agent_runs", "limit": 1}]})
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
                        }
                    ]
                }
            )
        elif self.root_calls == 2:
            self.shell_run_id = str(tool_payload.get("tool_run_id", "")).strip()
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
            self.saw_finish_rejection = bool(
                tool_payload.get("accepted") is False and str(tool_payload.get("error", "")).strip()
            )
            text = json.dumps(
                {
                    "actions": [
                        {
                            "type": "wait_run",
                            "tool_run_id": self.shell_run_id,
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
        if self.worker_received_steer:
            text = json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "STEERED",
                            "next_recommendation": "none",
                        }
                    ]
                }
            )
        else:
            text = json.dumps({"actions": [{"type": "list_agent_runs", "limit": 1}]})
        return ChatResult(content=text, raw_events=[])


class WaitInterruptedBySteerLLM:
    def __init__(self) -> None:
        self.root_calls = 0
        self.worker_calls = 0
        self.spawn_run_id = ""
        self.wait_time_interrupted = False
        self.root_received_steer = False

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        messages = kwargs["messages"]
        route = _route(messages)
        if route == "root":
            self.root_calls += 1
            tool_payload = _extract_last_tool_payload(messages)
            if "interrupt root wait" in json.dumps(messages, ensure_ascii=False):
                self.root_received_steer = True
            if self.root_calls == 1:
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "spawn_agent",
                                "name": "SteerRoot",
                                "instruction": "steer root while it waits",
                            }
                        ]
                    }
                )
            elif self.root_calls == 2:
                self.spawn_run_id = str(tool_payload.get("tool_run_id", "")).strip()
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "wait_time",
                                "seconds": 10,
                            }
                        ]
                    }
                )
            elif self.root_calls == 3:
                self.wait_time_interrupted = bool(tool_payload.get("interrupted_by_steer", False)) and str(
                    tool_payload.get("end_reason", "")
                ) == "received_steer_message"
                text = json.dumps(
                    {
                        "actions": [
                            {
                                "type": "wait_run",
                                "tool_run_id": self.spawn_run_id,
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
                                "summary": "wait interrupted and steering consumed",
                            }
                        ]
                    }
                )
            return ChatResult(content=text, raw_events=[])

        self.worker_calls += 1
        tool_payload = _extract_last_tool_payload(messages)
        if self.worker_calls == 1:
            text = json.dumps({"actions": [{"type": "list_agent_runs", "limit": 5}]})
        elif self.worker_calls == 2:
            root_id = _find_root_agent_id_from_runs(tool_payload)
            text = json.dumps(
                {
                    "actions": [
                        {
                            "type": "steer_agent",
                            "agent_id": root_id,
                            "content": "interrupt root wait",
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
                            "summary": "steer sent",
                            "next_recommendation": "none",
                        }
                    ]
                }
            )
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
                                "tool_run_id": str(tool_payload.get("tool_run_id", "")),
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
                                "type": "wait_run",
                                "tool_run_id": str(tool_payload.get("tool_run_id", "")),
                            }
                        ]
                    }
                )
            elif self.root_calls == 3:
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

        text = json.dumps(
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
        )
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
            root_id = str((tool_payload.get("agent_runs") or [{}])[0].get("id", ""))
            text = json.dumps({"actions": [{"type": "cancel_agent", "agent_id": root_id}]})
        else:
            text = json.dumps({"actions": [{"type": "finish", "status": "partial", "summary": "root cancel blocked"}]})
        return ChatResult(content=text, raw_events=[])


class CancelUnknownAgentLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.saw_unknown_agent_error = False

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            return ChatResult(
                content=json.dumps({"actions": [{"type": "cancel_agent", "agent_id": "agent-missing"}]}),
                raw_events=[],
            )
        payload = _extract_last_tool_payload(kwargs["messages"])
        self.saw_unknown_agent_error = "Cannot cancel agent agent-missing." in str(payload.get("error", ""))
        return ChatResult(
            content=json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "partial",
                            "summary": "unknown agent handled as tool error",
                        }
                    ]
                }
            ),
            raw_events=[],
        )


class BlockingSpawnFailedStatusLLM:
    def __init__(self) -> None:
        self.root_calls = 0
        self.child_agent_id = ""
        self.spawn_run_id = ""
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
                                }
                            ]
                        }
                    ),
                    raw_events=[],
                )
            tool_payload = _extract_last_tool_payload(messages)
            if self.root_calls == 2:
                self.child_agent_id = str(tool_payload.get("child_agent_id", ""))
                self.spawn_run_id = str(tool_payload.get("tool_run_id", ""))
                return ChatResult(
                    content=json.dumps({"actions": [{"type": "wait_run", "tool_run_id": self.spawn_run_id}]}),
                    raw_events=[],
                )
            if self.root_calls == 3:
                return ChatResult(
                    content=json.dumps({"actions": [{"type": "get_agent_run", "agent_id": self.child_agent_id}]}),
                    raw_events=[],
                )
            self.spawn_status = str(((tool_payload.get("agent_run") or {}).get("status") or ""))
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
            text = json.dumps({"actions": [{"type": "list_agent_runs", "limit": 1}]})
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
                            "summary": "recovered on next step",
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
                            "summary": "tool-call parse recovered on next step",
                        }
                    ]
                }
            ),
            raw_events=[],
        )


class RetryInvalidToolCallWithExecutableCandidateThenSuccessLLM:
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
                        "type": "function",
                        "function": {
                            "name": "wait_time",
                            "arguments": "{",
                        },
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {
                            "name": "list_agent_runs",
                            "arguments": "{\"limit\":1}",
                        },
                    },
                ],
            )
        return ChatResult(
            content=json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "tool-call parse recovered by protocol retry",
                        }
                    ]
                }
            ),
            raw_events=[],
        )


class ToolCallReplaySchemaLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.saw_openai_tool_call_schema = False

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        messages = kwargs["messages"]
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
                            "arguments": "{\"seconds\":10}",
                        },
                    }
                ],
            )

        replayed_assistant = next(
            (
                message
                for message in reversed(messages)
                if str(message.get("role", "")) == "assistant" and isinstance(message.get("tool_calls"), list)
            ),
            None,
        )
        tool_calls = replayed_assistant.get("tool_calls") if isinstance(replayed_assistant, dict) else []
        first_call = tool_calls[0] if isinstance(tool_calls, list) and tool_calls else {}
        function_payload = first_call.get("function") if isinstance(first_call, dict) else {}
        self.saw_openai_tool_call_schema = bool(
            isinstance(first_call, dict)
            and first_call.get("type") == "function"
            and isinstance(function_payload, dict)
            and function_payload.get("name") == "wait_time"
            and function_payload.get("arguments") == "{\"seconds\":10}"
        )

        return ChatResult(
            content=json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "replay schema ok",
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


class RetryInvalidJsonCapturePromptLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.retry_prompt = ""

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        self.calls += 1
        messages = kwargs["messages"]
        if self.calls == 1:
            return ChatResult(content="this is not json", raw_events=[])
        self.retry_prompt = str(messages[-1].get("content", "")) if messages else ""
        return ChatResult(
            content=json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "retry prompt captured",
                        }
                    ]
                }
            ),
            raw_events=[],
        )


def test_protocol_retry_exhausted_message_requires_next_step_decision() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        message = orchestrator._protocol_retry_exhausted_message(
            error="No JSON object found in model response",
            max_attempts=2,
        )
        assert "No JSON object found in model response" in message
        assert "Fix guidance:" in message
        assert "No tool was executed in this step." in message
        assert "do not call finish automatically" in message.lower()


class RetryExhaustedThenFinishLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.exhausted_prompt = ""

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        self.calls += 1
        messages = kwargs["messages"]
        if self.calls == 1:
            return ChatResult(content="still invalid payload", raw_events=[])
        self.exhausted_prompt = str(messages[-1].get("content", "")) if messages else ""
        return ChatResult(
            content=json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "finished after deciding next step",
                        }
                    ]
                }
            ),
            raw_events=[],
        )


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
        orchestrator.config.runtime.tools.shell_inline_wait_seconds = 0.05
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
        assert any(
            payload.get("accepted") is False and str(payload.get("error", "")).strip()
            for payload in tool_payloads
            if isinstance(payload, dict)
        )


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
async def test_wait_time_interrupted_by_steer_then_continue_generation() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = WaitInterruptedBySteerLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("wait interrupt test")
        assert session.status.value == "completed"
        assert llm.wait_time_interrupted is True
        assert llm.root_received_steer is True


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

        snapshot["tool_runs"]["toolrun-stale"] = {
            "id": "toolrun-stale",
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
        stale = second.tool_runs["toolrun-stale"]
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
        agent_dir = orchestrator.storage.agent_dir(session.id, root.id, agent_name=root.name)
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
async def test_spawn_agent_capacity_limit_returns_structured_rejected_result() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        orchestrator.config.runtime.limits.max_active_agents = 1
        root = AgentNode(
            id="agent-root",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="spawn child",
            workspace_path=project,
            status=AgentStatus.RUNNING,
        )
        orchestrator.agents = {root.id: root}
        orchestrator.tool_runs = {}
        orchestrator._reset_runtime_trackers()

        result = await orchestrator._execute_action(
            agent=root,
            action={
                "type": "spawn_agent",
                "name": "OverflowChild",
                "instruction": "should not be created",
            },
        )

        assert result["status"] == "rejected"
        assert result["child_agent_id"] is None
        assert result["error_code"] == "max_active_agents_limit_reached"
        assert result["capacity"]["active_agents"] >= 1
        assert len(root.children) == 0
        assert len(orchestrator.agents) == 1

        spawn_runs = [run for run in orchestrator.tool_runs.values() if run.tool_name == "spawn_agent"]
        assert len(spawn_runs) == 1
        assert spawn_runs[0].status.value == "completed"
        assert spawn_runs[0].error is None
        assert isinstance(spawn_runs[0].result, dict)
        assert spawn_runs[0].result["status"] == "rejected"


@pytest.mark.asyncio
async def test_wait_run_rejects_timeout_field() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        requester = AgentNode(
            id="agent-1",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="wait run",
            workspace_path=project,
        )
        request_run = _new_wait_request_run(orchestrator, requester, tool_name="wait_run")
        result = await orchestrator._tool_wait_run(request_run, {"tool_run_id": "toolrun-1", "timeout_seconds": "abc"})
        assert result["wait_run_status"] is False
        assert "unsupported field(s)" in str(result.get("error", ""))


@pytest.mark.asyncio
async def test_wait_run_returns_steer_interrupt_reason_when_requester_has_pending_steer() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        agent = AgentNode(
            id="agent-1",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="wait run",
            workspace_path=project,
        )
        run = orchestrator._create_tool_run(
            agent=agent,
            tool_name="shell",
            arguments={"type": "shell", "command": "sleep 1"},
            blocking=False,
        )
        orchestrator._mark_tool_run_running(run)
        orchestrator.pending_steers[agent.id] = ["interrupt now"]
        request_run = _new_wait_request_run(orchestrator, agent, tool_name="wait_run")

        result = await orchestrator._tool_wait_run(request_run, {"tool_run_id": run.id})
        assert result["wait_run_status"] is True
        assert result["interrupted_by_steer"] is True
        assert result["end_reason"] == "received_steer_message"


@pytest.mark.asyncio
async def test_wait_time_returns_steer_interrupt_reason_when_requester_has_pending_steer() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        agent = AgentNode(
            id="agent-1",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="wait time",
            workspace_path=project,
        )
        orchestrator.pending_steers[agent.id] = ["interrupt now"]
        request_run = _new_wait_request_run(orchestrator, agent, tool_name="wait_time")
        result = await orchestrator._tool_wait_time(request_run, {"seconds": 10})
        assert result["wait_time_status"] is True
        assert result["interrupted_by_steer"] is True
        assert result["end_reason"] == "received_steer_message"


@pytest.mark.asyncio
async def test_wait_run_timeout_returns_feedback_and_marks_tool_run_failed() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        orchestrator.config.runtime.tools.wait_run_timeout_seconds = 0.05
        requester = AgentNode(
            id="agent-1",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="wait run",
            workspace_path=project,
        )
        target = AgentNode(
            id="agent-2",
            session_id="session-1",
            name="worker",
            role=AgentRole.WORKER,
            instruction="still running",
            workspace_path=project,
            parent_agent_id=requester.id,
            status=AgentStatus.RUNNING,
        )
        orchestrator.agents = {requester.id: requester, target.id: target}
        result = await orchestrator._execute_tool_action(
            agent=requester,
            action={"type": "wait_run", "agent_id": target.id},
        )
        assert result["wait_run_status"] is False
        assert result["timed_out"] is True
        assert result["end_reason"] == "timeout"
        assert "timed out" in str(result.get("error", ""))
        wait_runs = [run for run in orchestrator.tool_runs.values() if run.tool_name == "wait_run"]
        assert len(wait_runs) == 1
        assert wait_runs[0].status.value == "failed"
        assert wait_runs[0].error == result["error"]
        assert isinstance(wait_runs[0].result, dict)
        assert wait_runs[0].result["timed_out"] is True


@pytest.mark.asyncio
async def test_wait_time_timeout_returns_feedback_and_marks_tool_run_failed() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        orchestrator.config.runtime.tools.wait_run_timeout_seconds = 0.05
        agent = AgentNode(
            id="agent-1",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="wait time",
            workspace_path=project,
        )
        result = await orchestrator._execute_tool_action(
            agent=agent,
            action={"type": "wait_time", "seconds": 10},
        )
        assert result["wait_time_status"] is False
        assert result["timed_out"] is True
        assert result["end_reason"] == "timeout"
        assert "timed out" in str(result.get("error", ""))
        wait_runs = [run for run in orchestrator.tool_runs.values() if run.tool_name == "wait_time"]
        assert len(wait_runs) == 1
        assert wait_runs[0].status.value == "failed"
        assert wait_runs[0].error == result["error"]
        assert isinstance(wait_runs[0].result, dict)
        assert wait_runs[0].result["timed_out"] is True


@pytest.mark.asyncio
async def test_wait_run_requires_exactly_one_target_and_returns_terminal_status() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        agent = AgentNode(
            id="agent-1",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="wait run",
            workspace_path=project,
        )
        run = orchestrator._create_tool_run(
            agent=agent,
            tool_name="shell",
            arguments={"type": "shell", "command": "sleep 1"},
            blocking=False,
        )
        orchestrator._mark_tool_run_running(run)
        orchestrator._complete_tool_run(run, result={"exit_code": 0})
        request_run = _new_wait_request_run(orchestrator, agent, tool_name="wait_run")
        invalid = await orchestrator._tool_wait_run(request_run, {})
        assert invalid["wait_run_status"] is False
        assert "requires exactly one of 'tool_run_id' or 'agent_id'" in str(invalid.get("error", ""))
        result = await orchestrator._tool_wait_run(request_run, {"tool_run_id": run.id})
        assert result["wait_run_status"] is True


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
        assert allowed.get("accepted") is True
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
        assert rejected.get("accepted") is False
        unfinished = rejected.get("unfinished_tool_runs")
        assert isinstance(unfinished, list)
        assert [item["tool_run_id"] for item in unfinished] == [worker_run.id]


@pytest.mark.asyncio
async def test_cancel_tool_run_cancels_spawned_child_agent_subtree() -> None:
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
        cancel_result = await orchestrator._tool_cancel_tool_run({"tool_run_id": run.id})

        assert cancel_result["final_status"] == "cancelled"
        assert int(cancel_result["cancelled_agents_count"]) >= 1
        assert orchestrator.agents[child_id].status == AgentStatus.CANCELLED
        assert run.error == f"Tool run cancelled by agent {run.agent_id}."


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
        result = await orchestrator._tool_shell(run, {"command": "printf inline-ok"})
        assert run.status.value == "completed"
        assert result["exit_code"] == 0
        assert result["stdout"] == "inline-ok"


@pytest.mark.asyncio
async def test_shell_uses_config_default_timeout_and_reports_timeout_details() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        orchestrator.config.runtime.tools.shell_inline_wait_seconds = 3.0
        orchestrator.config.runtime.tools.shell_timeout_seconds = 1.0
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
            arguments={"type": "shell", "command": "python -c \"import time;time.sleep(2)\""},
            blocking=True,
        )
        orchestrator._mark_tool_run_running(run)
        result = await orchestrator._tool_shell(run, {"command": "python -c \"import time;time.sleep(2)\""})
        assert run.status.value == "failed"
        assert run.error == "shell command timed out after 1s"
        assert isinstance(run.result, dict)
        assert run.result["timed_out"] is True
        assert run.result["timeout_seconds"] == 1.0
        assert result["timed_out"] is True


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
        response = await orchestrator._tool_shell(run, {"command": command})
        assert response["status"] == "running"
        assert run.status.value == "running"

        stdout_seen = ""
        for _ in range(100):
            detail = orchestrator._tool_get_tool_run({"tool_run_id": run.id})
            stdout_seen = str(((detail.get("tool_run") or {}).get("stdout") or ""))
            if "start" in stdout_seen:
                break
            await asyncio.sleep(0.01)
        assert "start" in stdout_seen

        wait_request_run = _new_wait_request_run(orchestrator, agent, tool_name="wait_run")
        wait_result = await orchestrator._tool_wait_run(wait_request_run, {"tool_run_id": run.id})
        assert wait_result["wait_run_status"] is True
        final_detail = orchestrator._tool_get_tool_run({"tool_run_id": run.id, "include_result": True})
        assert "end" in str((((final_detail.get("tool_run") or {}).get("result") or {}).get("stdout", "")))


@pytest.mark.asyncio
async def test_shell_background_internal_error_marks_run_failed() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=ResumeLLM())
        agent = AgentNode(
            id="agent-1",
            session_id="session-1",
            name="root",
            role=AgentRole.ROOT,
            instruction="run shell",
            workspace_path=project,
        )
        result = await orchestrator._execute_tool_action(
            agent=agent,
            action={"type": "shell", "command": "echo hi", "cwd": "missing-dir"},
        )
        assert str(result.get("error", "")).strip()
        shell_runs = [run for run in orchestrator.tool_runs.values() if run.tool_name == "shell"]
        assert len(shell_runs) == 1
        assert shell_runs[0].status.value == "failed"
        assert str(shell_runs[0].error or "").strip()


@pytest.mark.asyncio
async def test_worker_failure_visible_via_get_agent_run() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = FailingWorkerVisibleErrorLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("observe worker exception")
        assert session.status.value == "completed"
        assert llm.child_agent_id
        detail = orchestrator._tool_get_agent_run({"agent_id": llm.child_agent_id})
        assert str(((detail.get("agent_run") or {}).get("status") or "")) == "failed"


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
        assert cancel_runs[0].result["cancel_agent_status"] is False
        assert "cannot target the current agent itself" in str(cancel_runs[0].result["error"])


@pytest.mark.asyncio
async def test_cancel_agent_unknown_agent_id_returns_structured_error_without_failing_session() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = CancelUnknownAgentLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("attempt cancel unknown")
        assert session.status.value == "completed"
        assert llm.saw_unknown_agent_error is True
        cancel_runs = [run for run in orchestrator.tool_runs.values() if run.tool_name == "cancel_agent"]
        assert len(cancel_runs) == 1
        assert cancel_runs[0].status.value == "completed"
        assert cancel_runs[0].result is not None
        assert cancel_runs[0].result["cancel_agent_status"] is False
        assert cancel_runs[0].result["error"] == "Cannot cancel agent agent-missing."


@pytest.mark.asyncio
async def test_missing_tool_name_returns_failed_tool_run_payload() -> None:
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
        orchestrator.agents = {root.id: root}
        orchestrator.tool_runs = {}
        orchestrator._reset_runtime_trackers()

        result = await orchestrator._execute_action(agent=root, action={})
        assert result["error_code"] == "missing_tool_name"

        failed_runs = [run for run in orchestrator.tool_runs.values() if run.tool_name == "<missing>"]
        assert len(failed_runs) == 1
        assert failed_runs[0].status.value == "failed"
        assert failed_runs[0].error == "action type is required"


@pytest.mark.asyncio
async def test_disabled_tool_name_returns_failed_tool_run_payload() -> None:
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
        orchestrator.agents = {root.id: root}
        orchestrator.tool_runs = {}
        orchestrator._reset_runtime_trackers()

        result = await orchestrator._execute_action(agent=root, action={"type": "not_enabled_tool"})
        assert result["error_code"] == "tool_not_enabled_for_role"

        failed_runs = [run for run in orchestrator.tool_runs.values() if run.tool_name == "not_enabled_tool"]
        assert len(failed_runs) == 1
        assert failed_runs[0].status.value == "failed"
        assert "not enabled for role root" in str(failed_runs[0].error)


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
async def test_no_protocol_retry_when_step_has_no_executable_tool_call_from_invalid_json() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RetryInvalidJsonThenSuccessLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        orchestrator.config.runtime.limits.max_protocol_retries = 3
        session = await orchestrator.run_task("retry invalid json")
        assert session.status.value == "completed"
        assert session.final_summary == "recovered on next step"
        assert llm.calls == 2
        turns = orchestrator.storage.load_turns(session.id)
        assert len(turns) == 2
        first_turn = turns[0]
        assert int(first_turn.get("final_attempt", 0) or 0) == 1
        assert first_turn["actions"] == []
        assert first_turn["finish_payload"] is None
        events = orchestrator.storage.load_events(session.id)
        event_types = {str(item.get("event_type", "")) for item in events}
        assert "protocol_retry_scheduled" not in event_types
        assert "protocol_retry_skipped_no_executable_tool_call" in event_types


@pytest.mark.asyncio
async def test_no_protocol_retry_when_step_has_no_executable_tool_call_from_invalid_tool_args() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RetryInvalidToolCallThenSuccessLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        orchestrator.config.runtime.limits.max_protocol_retries = 3
        session = await orchestrator.run_task("retry invalid tool call")
        assert session.status.value == "completed"
        assert session.final_summary == "tool-call parse recovered on next step"
        assert llm.calls == 2
        turns = orchestrator.storage.load_turns(session.id)
        assert len(turns) == 2
        first_turn = turns[0]
        assert int(first_turn.get("final_attempt", 0) or 0) == 1
        assert first_turn["actions"] == []
        assert first_turn["finish_payload"] is None
        event_types = {str(item.get("event_type", "")) for item in orchestrator.storage.load_events(session.id)}
        assert "protocol_retry_scheduled" not in event_types
        assert "protocol_retry_skipped_no_executable_tool_call" in event_types


@pytest.mark.asyncio
async def test_replayed_assistant_tool_call_uses_openai_compatible_shape() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = ToolCallReplaySchemaLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("verify assistant tool-call replay shape")
        assert session.status.value == "completed"
        assert session.final_summary == "replay schema ok"
        assert llm.calls == 2
        assert llm.saw_openai_tool_call_schema is True


@pytest.mark.asyncio
async def test_protocol_retry_exhaustion_does_not_auto_finish() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RetryAlwaysInvalidLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        orchestrator.config.runtime.limits.max_protocol_retries = 2
        orchestrator.config.runtime.limits.max_root_steps = 1
        session = await orchestrator.run_task("retry exhausted")
        assert session.status.value == "completed"
        assert session.final_summary == "Root hit max step budget and produced forced partial finish."
        assert llm.calls == 1
        turns = orchestrator.storage.load_turns(session.id)
        assert len(turns) == 1
        assert turns[0]["actions"] == []
        assert turns[0]["finish_payload"] is None


@pytest.mark.asyncio
async def test_protocol_retry_exhaustion_feedback_allows_next_step_decision() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RetryExhaustedThenFinishLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        orchestrator.config.runtime.limits.max_protocol_retries = 0
        session = await orchestrator.run_task("retry exhausted next-step decision")
        assert session.status.value == "completed"
        assert session.final_summary == "finished after deciding next step"
        assert llm.calls == 2
        assert "did not include an executable tool call" in llm.exhausted_prompt
        event_types = {str(item.get("event_type", "")) for item in orchestrator.storage.load_events(session.id)}
        assert "protocol_retry_skipped_no_executable_tool_call" in event_types


@pytest.mark.asyncio
async def test_no_executable_tool_call_message_uses_invalid_response_with_error_details() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RetryInvalidJsonCapturePromptLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        orchestrator.config.runtime.limits.max_protocol_retries = 1
        session = await orchestrator.run_task("retry invalid json prompt")
        assert session.status.value == "completed"
        assert session.final_summary == "retry prompt captured"
        assert llm.calls == 2
        assert "Error: No JSON object found in model response" in llm.retry_prompt
        assert "Fix guidance:" in llm.retry_prompt
        assert "No protocol retry will be attempted for this step." in llm.retry_prompt


@pytest.mark.asyncio
async def test_turn_index_records_step_attempts_and_llm_events() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RetryInvalidToolCallWithExecutableCandidateThenSuccessLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        orchestrator.config.runtime.limits.max_protocol_retries = 1
        session = await orchestrator.run_task("turn index retry")
        assert session.status.value == "completed"

        turns = orchestrator.storage.load_turns(session.id)
        assert len(turns) == 1
        turn = turns[0]
        assert turn["agent_id"] == session.root_agent_id
        assert int(turn["step"]) == 1
        assert turn["status"] == "completed"
        assert int(turn["final_attempt"]) == 2
        attempts = [item for item in turn["attempts"] if isinstance(item, dict)]
        assert len(attempts) == 2
        assert attempts[0]["request_file"].endswith("0001_request.json")
        assert attempts[0]["response_file"].endswith("0001_response.json")
        assert attempts[0]["ok"] is False
        assert attempts[1]["ok"] is True
        assert turn["actions"][0]["type"] == "finish"
        assert len([item for item in turn["action_results"] if isinstance(item, dict)]) >= 1
        assert int(turn["event_seq_start"]) > 0
        assert int(turn["event_seq_end"]) >= int(turn["event_seq_start"])

        event_types = {str(item.get("event_type", "")) for item in orchestrator.storage.load_events(session.id)}
        assert "agent_step_started" in event_types
        assert "agent_step_finished" in event_types
        assert "llm_call_request_recorded" in event_types
        assert "llm_call_response_recorded" in event_types


@pytest.mark.asyncio
async def test_turn_retry_metrics_records_protocol_retry_counters() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = RetryInvalidToolCallWithExecutableCandidateThenSuccessLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        orchestrator.config.runtime.limits.max_protocol_retries = 1
        session = await orchestrator.run_task("turn retry metrics")
        assert session.status.value == "completed"

        turns = orchestrator.storage.load_turns(session.id)
        assert len(turns) == 1
        metrics = turns[0].get("retry_metrics")
        assert isinstance(metrics, dict)
        assert int(metrics.get("overall_retries", 0) or 0) == 1
        assert int(metrics.get("parse_retries", 0) or 0) == 1
        assert int(metrics.get("parse_empty_retries", 0) or 0) == 0
        assert int(metrics.get("context_overflow_retries", 0) or 0) == 0


@pytest.mark.asyncio
async def test_turn_index_keeps_root_and_worker_steps_separate() -> None:
    with TemporaryDirectory() as temp_dir:
        project = Path(temp_dir)
        llm = BlockingSpawnLLM()
        orchestrator = RuntimeOrchestrator(project_dir=project, app_dir=APP_DIR, llm_client=llm)
        session = await orchestrator.run_task("turn index root worker")
        assert session.status.value == "completed"

        turns = orchestrator.storage.load_turns(session.id)
        assert len(turns) >= 2
        agent_ids = {str(item.get("agent_id", "")) for item in turns}
        assert session.root_agent_id in agent_ids
        assert any(agent_id != session.root_agent_id for agent_id in agent_ids)
        assert all(int(item.get("step", 0) or 0) >= 1 for item in turns)

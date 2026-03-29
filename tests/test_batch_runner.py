from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

import opm_train.batch_runner as batch_runner_module
from opm_train.batch_runner import BatchRunConfig, run_batch
from opm_train.llm import ChatResult
from opm_train.models import RunSession, SessionStatus


class _FakeOrchestrator:
    def __init__(self, *, final_summary: str, status: SessionStatus = SessionStatus.COMPLETED) -> None:
        self.final_summary = final_summary
        self.status = status

    async def run_task(self, task: str) -> RunSession:
        return RunSession(
            id=f"session-{abs(hash(task)) % 100000}",
            task=task,
            project_dir=Path("."),
            root_agent_id="agent-root",
            status=self.status,
            final_summary=self.final_summary,
        )


class _FakeToolOutput:
    def __init__(self, *, text: str, reward: float, finished: bool) -> None:
        self.blocks = [SimpleNamespace(text=text)]
        self.reward = reward
        self.finished = finished


class _FakeOpenRewardSession:
    def __init__(self, *, task: dict[str, object], never_finish: bool = False) -> None:
        self.task = task
        self.never_finish = never_finish
        self.call_count = 0

    async def __aenter__(self) -> "_FakeOpenRewardSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)
        return None

    async def get_prompt(self) -> list[object]:
        return [SimpleNamespace(text=f"task:{self.task['task_id']}")]

    async def call_tool(self, name: str, arguments: dict[str, object]) -> _FakeToolOutput:
        self.call_count += 1
        if name != "answer":
            return _FakeToolOutput(text="unknown", reward=0.0, finished=False)
        if self.never_finish:
            return _FakeToolOutput(text="loop", reward=0.2, finished=False)
        expected = str(self.task.get("answer", ""))
        given = str(arguments.get("answer", ""))
        is_correct = expected == given
        return _FakeToolOutput(
            text="ok" if is_correct else "bad",
            reward=1.0 if is_correct else 0.0,
            finished=True,
        )


class _FakeOpenRewardEnvironment:
    def __init__(self, tasks: list[dict[str, object]], *, never_finish: bool = False) -> None:
        self.tasks = tasks
        self.never_finish = never_finish
        self.last_tool_format: str | None = None
        self.session_count = 0

    async def list_tasks(self, *, split: str) -> list[dict[str, object]]:
        _ = split
        return list(self.tasks)

    async def get_task(self, *, split: str, index: int) -> dict[str, object]:
        _ = split
        return self.tasks[index]

    async def get_task_range(self, *, split: str, start: int | None = None, stop: int | None = None) -> list[dict[str, object]]:
        _ = split
        return self.tasks[slice(start, stop)]

    async def list_tools(self, *, format: str) -> list[dict[str, object]]:
        self.last_tool_format = format
        return [{"type": "function", "function": {"name": "answer", "parameters": {"type": "object"}}}]

    def session(self, *, task: dict[str, object]) -> _FakeOpenRewardSession:
        self.session_count += 1
        return _FakeOpenRewardSession(task=task, never_finish=self.never_finish)


class _FallbackSignatureOpenRewardEnvironment(_FakeOpenRewardEnvironment):
    async def list_tools(self, *, tool_format: str) -> list[dict[str, object]]:
        self.last_tool_format = tool_format
        return [{"type": "function", "function": {"name": "answer", "parameters": {"type": "object"}}}]


class _SplitOpenRewardEnvironment:
    def __init__(self, tasks_by_split: dict[str, list[dict[str, object]]]) -> None:
        self.tasks_by_split = tasks_by_split
        self.last_tool_format: str | None = None
        self.session_count = 0

    async def list_tasks(self, *, split: str) -> list[dict[str, object]]:
        return list(self.tasks_by_split.get(split, []))

    async def get_task(self, *, split: str, index: int) -> dict[str, object]:
        return self.tasks_by_split[split][index]

    async def get_task_range(self, *, split: str, start: int | None = None, stop: int | None = None) -> list[dict[str, object]]:
        return self.tasks_by_split[split][slice(start, stop)]

    async def list_tools(self, *, format: str) -> list[dict[str, object]]:
        self.last_tool_format = format
        return [{"type": "function", "function": {"name": "answer", "parameters": {"type": "object"}}}]

    def session(self, *, task: dict[str, object]) -> _FakeOpenRewardSession:
        self.session_count += 1
        return _FakeOpenRewardSession(task=task, never_finish=False)


def _patch_openreward_client(monkeypatch: pytest.MonkeyPatch, environment: object) -> None:
    class _ClientFactory:
        def __init__(self) -> None:
            self.environments = SimpleNamespace(get=lambda **kwargs: environment)

    monkeypatch.setattr(batch_runner_module, "_load_async_openreward_client_cls", lambda: _ClientFactory)


class _FakeOpenRewardLLM:
    def __init__(self, *, bad_arguments: bool = False, always_tool_call: bool = False) -> None:
        self.bad_arguments = bad_arguments
        self.always_tool_call = always_tool_call
        self.calls = 0

    async def stream_chat(self, **kwargs: object) -> ChatResult:
        self.calls += 1
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        if not self.always_tool_call:
            has_tool = any(isinstance(item, dict) and item.get("role") == "tool" for item in messages)
            if has_tool:
                return ChatResult(content="done", raw_events=[], tool_calls=[])
        first = messages[0]
        assert isinstance(first, dict)
        prompt = str(first.get("content", ""))
        answer_map = {
            "task:t1": "42",
            "task:t2": "7",
            "task:t3": "11",
        }
        answer = answer_map.get(prompt, "0")
        arguments = "{" if self.bad_arguments else json.dumps({"answer": answer}, ensure_ascii=False)
        return ChatResult(
            content="",
            raw_events=[],
            tool_calls=[
                {
                    "id": f"call-{self.calls}",
                    "type": "function",
                    "function": {
                        "name": "answer",
                        "arguments": arguments,
                    },
                }
            ],
        )


@pytest.mark.asyncio
async def test_run_batch_writes_results_and_summary() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        project_dir = app_dir
        input_path = app_dir / "gsm8k.jsonl"
        input_path.write_text(
            '\n'.join(
                [
                    '{"id":"a","question":"2+3=?","answer":"#### 5"}',
                    '{"id":"b","question":"4+6=?","answer":"#### 10"}',
                ]
            ),
            encoding="utf-8",
        )

        answers = iter(["FINAL_ANSWER: 5", "FINAL_ANSWER: 999"])

        def factory() -> _FakeOrchestrator:
            return _FakeOrchestrator(final_summary=next(answers))

        output = await run_batch(
            BatchRunConfig(
                dataset="gsm8k",
                input_path=input_path,
                project_dir=project_dir,
                app_dir=app_dir,
                concurrency=2,
            ),
            orchestrator_factory=factory,
        )

        assert output.results_path.exists()
        assert output.summary_path.exists()
        assert output.summary.total == 2
        assert output.summary.validated == 2
        assert output.summary.correct == 1
        assert output.summary.accuracy == 0.5
        rows = [json.loads(line) for line in output.results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2
        assert set(rows[0].keys()) == {
            "adapter_name",
            "sample_id",
            "task_prompt",
            "reference_answer",
            "reference_answer_raw",
            "predicted_answer",
            "is_correct",
            "session_id",
            "session_status",
            "final_summary",
            "error",
        }
        assert {row["adapter_name"] for row in rows} == {"gsm8k"}
        assert rows[0]["reference_answer_raw"] is not None
        assert {row["reference_answer"] for row in rows} == {"5", "10"}


@pytest.mark.asyncio
async def test_run_batch_tolerates_per_sample_run_errors() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "gsm8k.jsonl"
        input_path.write_text(
            '\n'.join(
                [
                    '{"id":"a","question":"2+3=?","answer":"#### 5"}',
                    '{"id":"b","question":"4+6=?","answer":"#### 10"}',
                ]
            ),
            encoding="utf-8",
        )

        calls = {"count": 0}

        class _FlakyOrchestrator:
            async def run_task(self, task: str) -> RunSession:
                calls["count"] += 1
                if calls["count"] == 2:
                    raise RuntimeError("simulated failure")
                return RunSession(
                    id="session-ok",
                    task=task,
                    project_dir=Path("."),
                    root_agent_id="agent-root",
                    status=SessionStatus.COMPLETED,
                    final_summary="FINAL_ANSWER: 5",
                )

        output = await run_batch(
            BatchRunConfig(
                dataset="gsm8k",
                input_path=input_path,
                project_dir=app_dir,
                app_dir=app_dir,
                concurrency=1,
            ),
            orchestrator_factory=_FlakyOrchestrator,
        )

        rows = [json.loads(line) for line in output.results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        failed_rows = [row for row in rows if row["session_status"] == "failed"]
        assert len(failed_rows) == 1
        assert "run_error:RuntimeError:simulated failure" in str(failed_rows[0]["error"])
        assert output.summary.failed_sessions == 1
        assert output.summary.total == 2


@pytest.mark.asyncio
async def test_run_batch_executes_with_concurrency() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "gsm8k.jsonl"
        input_path.write_text(
            '\n'.join(
                [
                    '{"id":"a","question":"1+1=?","answer":"#### 2"}',
                    '{"id":"b","question":"2+2=?","answer":"#### 4"}',
                    '{"id":"c","question":"3+3=?","answer":"#### 6"}',
                    '{"id":"d","question":"4+4=?","answer":"#### 8"}',
                ]
            ),
            encoding="utf-8",
        )

        state = {"active": 0, "max_active": 0}

        class _ConcurrentOrchestrator:
            async def run_task(self, task: str) -> RunSession:
                _ = task
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                try:
                    await asyncio.sleep(0.05)
                finally:
                    state["active"] -= 1
                return RunSession(
                    id="session-concurrent",
                    task=task,
                    project_dir=Path("."),
                    root_agent_id="agent-root",
                    status=SessionStatus.COMPLETED,
                    final_summary="FINAL_ANSWER: 2",
                )

        await run_batch(
            BatchRunConfig(
                dataset="gsm8k",
                input_path=input_path,
                project_dir=app_dir,
                app_dir=app_dir,
                concurrency=3,
            ),
            orchestrator_factory=_ConcurrentOrchestrator,
        )
        assert state["max_active"] >= 2


@pytest.mark.asyncio
async def test_run_batch_realtime_writes_results_before_all_tasks_finish() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "gsm8k.jsonl"
        input_path.write_text(
            '\n'.join(
                [
                    '{"id":"a","question":"2+3=?","answer":"#### 5"}',
                    '{"id":"b","question":"4+6=?","answer":"#### 10"}',
                ]
            ),
            encoding="utf-8",
        )

        class _DelayedOrchestrator:
            async def run_task(self, task: str) -> RunSession:
                if "2+3=?" in task:
                    await asyncio.sleep(0.05)
                    summary = "FINAL_ANSWER: 5"
                else:
                    await asyncio.sleep(0.35)
                    summary = "FINAL_ANSWER: 10"
                return RunSession(
                    id=f"session-{abs(hash(task)) % 100000}",
                    task=task,
                    project_dir=Path("."),
                    root_agent_id="agent-root",
                    status=SessionStatus.COMPLETED,
                    final_summary=summary,
                )

        batch_id = "realtime-batch"
        task = asyncio.create_task(
            run_batch(
                BatchRunConfig(
                    dataset="gsm8k",
                    input_path=input_path,
                    project_dir=app_dir,
                    app_dir=app_dir,
                    concurrency=2,
                    batch_id=batch_id,
                ),
                orchestrator_factory=_DelayedOrchestrator,
            )
        )

        results_path = app_dir / ".opm_train" / "batches" / batch_id / "results.jsonl"
        saw_partial_write = False
        for _ in range(50):
            if results_path.exists():
                lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(lines) == 1:
                    saw_partial_write = True
                    break
            await asyncio.sleep(0.02)
        output = await task
        final_lines = [line for line in output.results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert saw_partial_write is True
        assert len(final_lines) == 2


@pytest.mark.asyncio
async def test_run_batch_resume_skips_completed_samples() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "gsm8k.jsonl"
        input_path.write_text(
            '\n'.join(
                [
                    '{"id":"a","question":"2+3=?","answer":"#### 5"}',
                    '{"id":"b","question":"4+6=?","answer":"#### 10"}',
                ]
            ),
            encoding="utf-8",
        )

        calls = {"count": 0}

        class _ResumeOrchestrator:
            async def run_task(self, task: str) -> RunSession:
                calls["count"] += 1
                summary = "FINAL_ANSWER: 5" if "2+3=?" in task else "FINAL_ANSWER: 10"
                return RunSession(
                    id=f"session-{calls['count']}",
                    task=task,
                    project_dir=Path("."),
                    root_agent_id="agent-root",
                    status=SessionStatus.COMPLETED,
                    final_summary=summary,
                )

        batch_id = "resume-batch"
        first = await run_batch(
            BatchRunConfig(
                dataset="gsm8k",
                input_path=input_path,
                project_dir=app_dir,
                app_dir=app_dir,
                limit=1,
                batch_id=batch_id,
            ),
            orchestrator_factory=_ResumeOrchestrator,
        )
        assert first.summary.total == 1

        resumed = await run_batch(
            BatchRunConfig(
                dataset="gsm8k",
                input_path=input_path,
                project_dir=app_dir,
                app_dir=app_dir,
                batch_id=batch_id,
                resume=True,
            ),
            orchestrator_factory=_ResumeOrchestrator,
        )
        assert calls["count"] == 2
        assert resumed.summary.total == 2
        lines = [line for line in resumed.results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 2


@pytest.mark.asyncio
async def test_run_batch_rejects_invalid_batch_id() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "gsm8k.jsonl"
        input_path.write_text('{"id":"a","question":"2+3=?","answer":"#### 5"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="batch_id may only contain"):
            await run_batch(
                BatchRunConfig(
                    dataset="gsm8k",
                    input_path=input_path,
                    project_dir=app_dir,
                    app_dir=app_dir,
                    batch_id="../bad",
                ),
            )


@pytest.mark.asyncio
async def test_run_batch_supports_mixed_adapter_rows() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "mixed.jsonl"
        input_path.write_text(
            '\n'.join(
                [
                    '{"adapter":"gsm8k","id":"m1","question":"2+3=?","answer":"#### 5"}',
                    '{"adapter":"gsm8k","id":"m2","question":"4+6=?","answer":"#### 10"}',
                ]
            ),
            encoding="utf-8",
        )

        class _MixedOrchestrator:
            async def run_task(self, task: str) -> RunSession:
                summary = "FINAL_ANSWER: 5" if "2+3=?" in task else "FINAL_ANSWER: 10"
                return RunSession(
                    id=f"session-{abs(hash(task)) % 100000}",
                    task=task,
                    project_dir=Path("."),
                    root_agent_id="agent-root",
                    status=SessionStatus.COMPLETED,
                    final_summary=summary,
                )

        output = await run_batch(
            BatchRunConfig(
                dataset="mixed",
                input_path=input_path,
                project_dir=app_dir,
                app_dir=app_dir,
                adapter_key="adapter",
                concurrency=2,
            ),
            orchestrator_factory=_MixedOrchestrator,
        )
        rows = [json.loads(line) for line in output.results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2
        assert {row["adapter_name"] for row in rows} == {"gsm8k"}


@pytest.mark.asyncio
async def test_run_batch_mixed_rejects_duplicate_adapter_sample_key() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "mixed_dup.jsonl"
        input_path.write_text(
            '\n'.join(
                [
                    '{"adapter":"gsm8k","id":"dup","question":"2+3=?","answer":"#### 5"}',
                    '{"adapter":"gsm8k","id":"dup","question":"4+6=?","answer":"#### 10"}',
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate routed sample key"):
            await run_batch(
                BatchRunConfig(
                    dataset="mixed",
                    input_path=input_path,
                    project_dir=app_dir,
                    app_dir=app_dir,
                    adapter_key="adapter",
                )
            )


@pytest.mark.asyncio
async def test_run_batch_openreward_writes_openreward_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        tasks = [
            {"task_id": "t1", "answer": "42"},
            {"task_id": "t2", "answer": "7"},
            {"task_id": "t3", "answer": "11"},
        ]
        environment = _FakeOpenRewardEnvironment(tasks)
        _patch_openreward_client(monkeypatch, environment)
        fake_llm = _FakeOpenRewardLLM()
        monkeypatch.setattr(batch_runner_module, "_build_openreward_llm_client", lambda profile: fake_llm)

        output = await run_batch(
            BatchRunConfig(
                dataset="openreward",
                input_path=None,
                project_dir=app_dir,
                app_dir=app_dir,
                environment="GeneralReasoning/OfficeQA",
                split="train",
                start=0,
                stop=2,
                provider_profile="openrouter",
            )
        )

        assert output.results_path.name == "openreward_results.jsonl"
        assert output.summary_path.name == "openreward_summary.json"
        assert isinstance(output.summary, batch_runner_module.OpenRewardBatchSummary)
        assert output.summary.total == 2
        assert output.summary.completed == 2
        assert output.summary.finished == 2
        assert output.summary.failed == 0
        assert output.summary.total_reward == 2.0
        assert output.summary.avg_reward == 1.0
        assert environment.last_tool_format == "openrouter"

        rows = [json.loads(line) for line in output.results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2
        assert {row["task_key"] for row in rows} == {"t1", "t2"}
        assert {row["session_status"] for row in rows} == {"completed"}


@pytest.mark.asyncio
async def test_run_batch_openreward_resume_skips_completed_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        tasks = [
            {"task_id": "t1", "answer": "42"},
            {"task_id": "t2", "answer": "7"},
        ]
        environment = _FakeOpenRewardEnvironment(tasks)
        _patch_openreward_client(monkeypatch, environment)
        fake_llm = _FakeOpenRewardLLM()
        monkeypatch.setattr(batch_runner_module, "_build_openreward_llm_client", lambda profile: fake_llm)

        batch_id = "openreward-resume"
        first = await run_batch(
            BatchRunConfig(
                dataset="openreward",
                input_path=None,
                project_dir=app_dir,
                app_dir=app_dir,
                environment="GeneralReasoning/OfficeQA",
                split="train",
                limit=1,
                batch_id=batch_id,
            )
        )
        assert isinstance(first.summary, batch_runner_module.OpenRewardBatchSummary)
        assert first.summary.total == 1

        resumed = await run_batch(
            BatchRunConfig(
                dataset="openreward",
                input_path=None,
                project_dir=app_dir,
                app_dir=app_dir,
                environment="GeneralReasoning/OfficeQA",
                split="train",
                batch_id=batch_id,
                resume=True,
            )
        )
        assert isinstance(resumed.summary, batch_runner_module.OpenRewardBatchSummary)
        assert resumed.summary.total == 2
        assert environment.session_count == 2


@pytest.mark.asyncio
async def test_run_batch_openreward_marks_invalid_tool_arguments_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        tasks = [{"task_id": "t1", "answer": "42"}]
        environment = _FakeOpenRewardEnvironment(tasks)
        _patch_openreward_client(monkeypatch, environment)
        fake_llm = _FakeOpenRewardLLM(bad_arguments=True)
        monkeypatch.setattr(batch_runner_module, "_build_openreward_llm_client", lambda profile: fake_llm)

        output = await run_batch(
            BatchRunConfig(
                dataset="openreward",
                input_path=None,
                project_dir=app_dir,
                app_dir=app_dir,
                environment="GeneralReasoning/OfficeQA",
                split="train",
                task_index=0,
            )
        )

        assert isinstance(output.summary, batch_runner_module.OpenRewardBatchSummary)
        assert output.summary.failed == 1
        rows = [json.loads(line) for line in output.results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows[0]["session_status"] == "failed"
        assert "JSONDecodeError" in str(rows[0]["error"])


@pytest.mark.asyncio
async def test_run_batch_openreward_marks_max_steps_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        tasks = [{"task_id": "t1", "answer": "42"}]
        environment = _FakeOpenRewardEnvironment(tasks, never_finish=True)
        _patch_openreward_client(monkeypatch, environment)
        fake_llm = _FakeOpenRewardLLM(always_tool_call=True)
        monkeypatch.setattr(batch_runner_module, "_build_openreward_llm_client", lambda profile: fake_llm)

        output = await run_batch(
            BatchRunConfig(
                dataset="openreward",
                input_path=None,
                project_dir=app_dir,
                app_dir=app_dir,
                environment="GeneralReasoning/OfficeQA",
                split="train",
                task_index=0,
                max_steps=2,
            )
        )

        assert isinstance(output.summary, batch_runner_module.OpenRewardBatchSummary)
        rows = [json.loads(line) for line in output.results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows[0]["session_status"] == "completed"
        assert rows[0]["finished"] is False
        assert rows[0]["turns"] == 2
        assert rows[0]["error"] == "max_steps_reached"


@pytest.mark.asyncio
async def test_run_batch_openreward_rejects_non_positive_max_steps() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        with pytest.raises(ValueError, match="--max-steps must be a positive integer"):
            await run_batch(
                BatchRunConfig(
                    dataset="openreward",
                    input_path=None,
                    project_dir=app_dir,
                    app_dir=app_dir,
                    environment="GeneralReasoning/OfficeQA",
                    split="train",
                    task_index=0,
                    max_steps=0,
                )
            )


@pytest.mark.asyncio
async def test_run_batch_openreward_supports_signature_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        tasks = [{"task_id": "t1", "answer": "42"}]
        environment = _FallbackSignatureOpenRewardEnvironment(tasks)
        fake_llm = _FakeOpenRewardLLM()
        state: dict[str, object] = {}

        class _FallbackClientFactory:
            def __init__(self) -> None:
                state["constructed"] = int(state.get("constructed", 0) or 0) + 1

                def get_environment(name: str) -> _FallbackSignatureOpenRewardEnvironment:
                    state["environment_name"] = name
                    return environment

                self.environments = SimpleNamespace(get=get_environment)

        monkeypatch.setenv("OPENREWARD_API_KEY", "dummy-openreward-key")
        monkeypatch.setattr(batch_runner_module, "_load_async_openreward_client_cls", lambda: _FallbackClientFactory)
        monkeypatch.setattr(batch_runner_module, "_build_openreward_llm_client", lambda profile: fake_llm)

        output = await run_batch(
            BatchRunConfig(
                dataset="openreward",
                input_path=None,
                project_dir=app_dir,
                app_dir=app_dir,
                environment="GeneralReasoning/OfficeQA",
                split="train",
                task_index=0,
                provider_profile="openrouter",
                base_url="http://localhost:9999",
            )
        )

        assert isinstance(output.summary, batch_runner_module.OpenRewardBatchSummary)
        assert output.summary.total == 1
        assert state["constructed"] == 1
        assert state["environment_name"] == "GeneralReasoning/OfficeQA"
        assert environment.last_tool_format == "openrouter"


@pytest.mark.asyncio
async def test_run_batch_openreward_supports_mixed_task_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        environment = _SplitOpenRewardEnvironment(
            {
                "train": [
                    {"task_id": "t1", "answer": "42"},
                    {"task_id": "t2", "answer": "7"},
                ],
                "validation": [
                    {"task_id": "t1", "answer": "42"},
                    {"task_id": "t3", "answer": "11"},
                ],
            }
        )
        _patch_openreward_client(monkeypatch, environment)
        fake_llm = _FakeOpenRewardLLM()
        monkeypatch.setattr(batch_runner_module, "_build_openreward_llm_client", lambda profile: fake_llm)

        output = await run_batch(
            BatchRunConfig(
                dataset="openreward",
                input_path=None,
                project_dir=app_dir,
                app_dir=app_dir,
                environment="GeneralReasoning/OfficeQA",
                task_specs=("train:0:2", "validation"),
            )
        )

        assert isinstance(output.summary, batch_runner_module.OpenRewardBatchSummary)
        assert output.summary.total == 4
        rows = [json.loads(line) for line in output.results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 4
        assert {row["split"] for row in rows} == {"train", "validation"}
        assert {row["task_key"] for row in rows} == {"train::t1", "train::t2", "validation::t1", "validation::t3"}


@pytest.mark.asyncio
async def test_run_batch_openreward_prefers_variant_environment_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        default_environment = _FakeOpenRewardEnvironment([{"task_id": "d1", "answer": "0"}])
        variant_environment = _FakeOpenRewardEnvironment([{"task_id": "t1", "answer": "42"}])

        class _VariantClientFactory:
            def __init__(self) -> None:
                def get_environment(name: str, variant: str | None = None, **kwargs: object):
                    _ = (name, kwargs)
                    return variant_environment if variant == "v2" else default_environment

                self.environments = SimpleNamespace(get=get_environment)

        fake_llm = _FakeOpenRewardLLM()
        monkeypatch.setattr(batch_runner_module, "_load_async_openreward_client_cls", lambda: _VariantClientFactory)
        monkeypatch.setattr(batch_runner_module, "_build_openreward_llm_client", lambda profile: fake_llm)

        output = await run_batch(
            BatchRunConfig(
                dataset="openreward",
                input_path=None,
                project_dir=app_dir,
                app_dir=app_dir,
                environment="GeneralReasoning/OfficeQA",
                variant="v2",
                task_index=0,
            )
        )

        assert isinstance(output.summary, batch_runner_module.OpenRewardBatchSummary)
        assert output.summary.total == 1
        assert variant_environment.session_count == 1
        assert default_environment.session_count == 0


@pytest.mark.asyncio
async def test_run_batch_openreward_client_init_fallback_keeps_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        environment = _FakeOpenRewardEnvironment([{"task_id": "t1", "answer": "42"}])
        state: dict[str, object] = {}

        class _ApiKeyOnlyClientFactory:
            def __init__(self, api_key: str) -> None:
                state["api_key"] = api_key
                self.environments = SimpleNamespace(get=lambda name: environment)

        fake_llm = _FakeOpenRewardLLM()
        monkeypatch.setenv("OPENREWARD_API_KEY", "dummy-openreward-key")
        monkeypatch.setattr(batch_runner_module, "_load_async_openreward_client_cls", lambda: _ApiKeyOnlyClientFactory)
        monkeypatch.setattr(batch_runner_module, "_build_openreward_llm_client", lambda profile: fake_llm)

        output = await run_batch(
            BatchRunConfig(
                dataset="openreward",
                input_path=None,
                project_dir=app_dir,
                app_dir=app_dir,
                environment="GeneralReasoning/OfficeQA",
                task_index=0,
                base_url="http://localhost:9999",
            )
        )

        assert isinstance(output.summary, batch_runner_module.OpenRewardBatchSummary)
        assert output.summary.total == 1
        assert state["api_key"] == "dummy-openreward-key"

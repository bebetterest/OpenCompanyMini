from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from opm_train.batch_runner import BatchRunConfig, run_batch
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

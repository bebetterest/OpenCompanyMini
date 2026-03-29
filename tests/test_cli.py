from __future__ import annotations

import json
from pathlib import Path
import runpy
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

import opm_train.batch_runner as batch_runner_module
import opm_train.cli as cli_module
from opm_train.cli import main
from opm_train.models import RunSession, SessionStatus
from opm_train.sft.contracts import SFTBackendResult
from opm_train.sft.runner import SFTRunOutput
from opm_train.storage import SessionStorage

APP_DIR = Path(__file__).resolve().parents[1]


def test_doctor_command_reports_setup(capsys) -> None:
    with TemporaryDirectory() as temp_dir:
        rc = main([
            "doctor",
            "--app-dir",
            str(APP_DIR),
            "--project-dir",
            temp_dir,
        ])
        assert rc == 0
        output = capsys.readouterr().out.strip()
        payload = json.loads(output)
        assert payload["prompts_exists"] is True
        assert payload["ready_for_smoke_run"] is True
        assert payload["tool_contract_ok"] is True
        assert payload["tool_contract_issues"] == []


def test_smoke_command_runs_without_provider_key(capsys) -> None:
    with TemporaryDirectory() as temp_dir:
        rc = main([
            "smoke",
            "--app-dir",
            str(APP_DIR),
            "--project-dir",
            temp_dir,
            "--task",
            "smoke-test",
        ])
        assert rc == 0
        output = capsys.readouterr().out.strip()
        payload = json.loads(output)
        assert payload["status"] == "completed"
        assert "session_id" in payload


def test_doctor_with_provider_override_does_not_require_api_key(capsys) -> None:
    with TemporaryDirectory() as temp_dir:
        rc = main([
            "doctor",
            "--app-dir",
            str(APP_DIR),
            "--project-dir",
            temp_dir,
            "--provider-profile",
            "custom",
        ])
        assert rc == 0
        output = capsys.readouterr().out.strip()
        payload = json.loads(output)
        assert payload["provider_profile"] == "custom"
        assert payload["provider_api_key_env"] == "OPENAI_API_KEY"


def test_doctor_reports_tool_contract_mismatch_without_failing_exit_code(capsys, monkeypatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        prompts_dir = app_dir / "prompts"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "root_coordinator.md").write_text("root", encoding="utf-8")
        (prompts_dir / "worker.md").write_text("worker", encoding="utf-8")
        (prompts_dir / "runtime_messages.json").write_text(
            json.dumps(
                {
                    "root_initial_message": "task={task}",
                    "worker_initial_message": "instruction={instruction}",
                    "resume_instruction_message": "resume={instruction}",
                    "steer_message": "steer={content}",
                    "context_latest_summary": "summary={summary}",
                }
            ),
            encoding="utf-8",
        )
        (prompts_dir / "tool_definitions.json").write_text(
            json.dumps(
                {
                    "shell": {"type": "function", "function": {"name": "shell", "parameters": {"type": "object"}}},
                    "finish": {"type": "function", "function": {"name": "finish", "parameters": {"type": "object"}}},
                }
            ),
            encoding="utf-8",
        )
        (app_dir / "opm_train.toml").write_text(
            "\n".join(
                [
                    "[runtime.tools]",
                    'root_tools = ["shell", "finish", "missing_tool"]',
                    'worker_tools = ["shell", "finish"]',
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        rc = main(
            [
                "doctor",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(app_dir),
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["tool_contract_ok"] is False
        assert payload["ready_for_real_run"] is False
        assert any("missing_tool" in item for item in payload["tool_contract_issues"])


def test_smoke_with_timer_writes_module_timings(capsys) -> None:
    with TemporaryDirectory() as temp_dir:
        rc = main(
            [
                "smoke",
                "--app-dir",
                str(APP_DIR),
                "--project-dir",
                temp_dir,
                "--task",
                "timer-smoke",
                "--timer",
            ]
        )
        assert rc == 0
        output = capsys.readouterr().out.strip()
        payload = json.loads(output)
        timing_path = Path(payload["data_dir"]) / "timers" / "module_timings.jsonl"
        assert timing_path.exists()
        rows = [json.loads(line) for line in timing_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows
        assert any(str(item.get("module", "")) in {"ask_agent", "llm_call", "persist_snapshot"} for item in rows)


def test_cli_module_entrypoint_executes_main(capsys, monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "opm_train.cli", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opm_train.cli",
            "doctor",
            "--app-dir",
            str(APP_DIR),
            "--project-dir",
            str(APP_DIR),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("opm_train.cli", run_name="__main__")
    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["prompts_exists"] is True


def test_batch_run_command_outputs_metrics_and_artifacts(capsys, monkeypatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "gsm8k.jsonl"
        input_path.write_text(
            '{"id":"q1","question":"2+3=?","answer":"#### 5"}\n',
            encoding="utf-8",
        )

        class StubOrchestrator:
            def __init__(
                self,
                *,
                project_dir: Path,
                app_dir: Path,
                model_override: str | None = None,
                timer_enabled: bool = False,
            ) -> None:
                self.project_dir = project_dir
                self.app_dir = app_dir
                self.model_override = model_override
                self.timer_enabled = timer_enabled
                self.config = SimpleNamespace(provider=SimpleNamespace(profile="openrouter"))

            def set_provider_profile(self, profile: str) -> None:
                self.config.provider.profile = profile

            async def run_task(self, task: str) -> RunSession:
                return RunSession(
                    id="session-stub-1",
                    task=task,
                    project_dir=self.project_dir,
                    root_agent_id="agent-root",
                    status=SessionStatus.COMPLETED,
                    final_summary="FINAL_ANSWER: 5",
                )

        monkeypatch.setattr(batch_runner_module, "RuntimeOrchestrator", StubOrchestrator)

        rc = main(
            [
                "batch-run",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(app_dir),
                "--dataset",
                "gsm8k",
                "--input",
                str(input_path),
                "--concurrency",
                "2",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["total"] == 1
        assert payload["validated"] == 1
        assert payload["correct"] == 1
        assert payload["accuracy"] == 1.0
        assert Path(payload["results_path"]).exists()
        assert Path(payload["summary_path"]).exists()


def test_batch_run_command_supports_simple_math_adapter(capsys, monkeypatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "simple_math.jsonl"
        input_path.write_text(
            '{"id":"m1","question":"1+2=?","answer":"3"}\n',
            encoding="utf-8",
        )

        class StubOrchestrator:
            def __init__(
                self,
                *,
                project_dir: Path,
                app_dir: Path,
                model_override: str | None = None,
                timer_enabled: bool = False,
            ) -> None:
                self.project_dir = project_dir
                self.app_dir = app_dir
                self.model_override = model_override
                self.timer_enabled = timer_enabled
                self.config = SimpleNamespace(provider=SimpleNamespace(profile="openrouter"))

            def set_provider_profile(self, profile: str) -> None:
                self.config.provider.profile = profile

            async def run_task(self, task: str) -> RunSession:
                return RunSession(
                    id="session-simple-math",
                    task=task,
                    project_dir=self.project_dir,
                    root_agent_id="agent-root",
                    status=SessionStatus.COMPLETED,
                    final_summary="FINAL_ANSWER: 3",
                )

        monkeypatch.setattr(batch_runner_module, "RuntimeOrchestrator", StubOrchestrator)

        rc = main(
            [
                "batch-run",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(app_dir),
                "--dataset",
                "simple_math",
                "--input",
                str(input_path),
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["total"] == 1
        assert payload["validated"] == 1
        assert payload["correct"] == 1
        assert payload["accuracy"] == 1.0


def test_batch_run_command_smoke_mode_runs_without_provider_key(capsys) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "gsm8k.jsonl"
        input_path.write_text(
            '{"id":"q1","question":"2+3=?","answer":"#### 5"}\n',
            encoding="utf-8",
        )
        rc = main(
            [
                "batch-run",
                "--app-dir",
                str(APP_DIR),
                "--project-dir",
                str(app_dir),
                "--dataset",
                "gsm8k",
                "--input",
                str(input_path),
                "--limit",
                "1",
                "--smoke",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["total"] == 1


def test_batch_run_command_supports_resume(capsys, monkeypatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "gsm8k.jsonl"
        input_path.write_text(
            '\n'.join(
                [
                    '{"id":"q1","question":"2+3=?","answer":"#### 5"}',
                    '{"id":"q2","question":"4+6=?","answer":"#### 10"}',
                ]
            ),
            encoding="utf-8",
        )

        class StubOrchestrator:
            def __init__(
                self,
                *,
                project_dir: Path,
                app_dir: Path,
                model_override: str | None = None,
                timer_enabled: bool = False,
            ) -> None:
                self.project_dir = project_dir
                self.app_dir = app_dir
                self.model_override = model_override
                self.timer_enabled = timer_enabled
                self.config = SimpleNamespace(provider=SimpleNamespace(profile="openrouter"))

            def set_provider_profile(self, profile: str) -> None:
                self.config.provider.profile = profile

            async def run_task(self, task: str) -> RunSession:
                summary = "FINAL_ANSWER: 5" if "2+3=?" in task else "FINAL_ANSWER: 10"
                return RunSession(
                    id=f"session-{abs(hash(task)) % 100000}",
                    task=task,
                    project_dir=self.project_dir,
                    root_agent_id="agent-root",
                    status=SessionStatus.COMPLETED,
                    final_summary=summary,
                )

        monkeypatch.setattr(batch_runner_module, "RuntimeOrchestrator", StubOrchestrator)

        batch_id = "cli-resume-batch"
        rc_first = main(
            [
                "batch-run",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(app_dir),
                "--dataset",
                "gsm8k",
                "--input",
                str(input_path),
                "--batch-id",
                batch_id,
                "--limit",
                "1",
            ]
        )
        assert rc_first == 0
        first_payload = json.loads(capsys.readouterr().out.strip())
        assert first_payload["total"] == 1

        rc_resume = main(
            [
                "batch-run",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(app_dir),
                "--dataset",
                "gsm8k",
                "--input",
                str(input_path),
                "--batch-id",
                batch_id,
                "--resume",
            ]
        )
        assert rc_resume == 0
        resumed_payload = json.loads(capsys.readouterr().out.strip())
        assert resumed_payload["total"] == 2


def test_batch_run_command_supports_mixed_dataset_mode(capsys, monkeypatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "mixed.jsonl"
        input_path.write_text(
            '\n'.join(
                [
                    '{"adapter":"gsm8k","id":"q1","question":"2+3=?","answer":"#### 5"}',
                    '{"adapter":"gsm8k","id":"q2","question":"4+6=?","answer":"#### 10"}',
                ]
            ),
            encoding="utf-8",
        )

        class StubOrchestrator:
            def __init__(
                self,
                *,
                project_dir: Path,
                app_dir: Path,
                model_override: str | None = None,
                timer_enabled: bool = False,
            ) -> None:
                self.project_dir = project_dir
                self.app_dir = app_dir
                self.model_override = model_override
                self.timer_enabled = timer_enabled
                self.config = SimpleNamespace(provider=SimpleNamespace(profile="openrouter"))

            def set_provider_profile(self, profile: str) -> None:
                self.config.provider.profile = profile

            async def run_task(self, task: str) -> RunSession:
                summary = "FINAL_ANSWER: 5" if "2+3=?" in task else "FINAL_ANSWER: 10"
                return RunSession(
                    id=f"session-{abs(hash(task)) % 100000}",
                    task=task,
                    project_dir=self.project_dir,
                    root_agent_id="agent-root",
                    status=SessionStatus.COMPLETED,
                    final_summary=summary,
                )

        monkeypatch.setattr(batch_runner_module, "RuntimeOrchestrator", StubOrchestrator)

        rc = main(
            [
                "batch-run",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(app_dir),
                "--dataset",
                "mixed",
                "--input",
                str(input_path),
                "--adapter-key",
                "adapter",
                "--concurrency",
                "2",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["total"] == 2


def test_batch_run_command_openreward_requires_environment() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        with pytest.raises(ValueError, match="--environment is required"):
            main(
                [
                    "batch-run",
                    "--app-dir",
                    str(app_dir),
                    "--project-dir",
                    str(app_dir),
                    "--dataset",
                    "openreward",
                ]
            )


def test_batch_run_command_rejects_task_index_with_range() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        with pytest.raises(ValueError, match="--task-index cannot be used with --start/--stop"):
            main(
                [
                    "batch-run",
                    "--app-dir",
                    str(app_dir),
                    "--project-dir",
                    str(app_dir),
                    "--dataset",
                    "openreward",
                    "--environment",
                    "GeneralReasoning/OfficeQA",
                    "--task-index",
                    "0",
                    "--start",
                    "0",
                    "--stop",
                    "10",
                ]
            )


def test_batch_run_command_rejects_task_spec_with_task_index() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        with pytest.raises(ValueError, match="--task-spec cannot be used with --task-index/--start/--stop"):
            main(
                [
                    "batch-run",
                    "--app-dir",
                    str(app_dir),
                    "--project-dir",
                    str(app_dir),
                    "--dataset",
                    "openreward",
                    "--environment",
                    "GeneralReasoning/OfficeQA",
                    "--task-spec",
                    "train:0:10",
                    "--task-index",
                    "0",
                ]
            )


def test_batch_run_command_non_openreward_still_requires_input() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        with pytest.raises(ValueError, match="--input is required unless --dataset openreward"):
            main(
                [
                    "batch-run",
                    "--app-dir",
                    str(app_dir),
                    "--project-dir",
                    str(app_dir),
                    "--dataset",
                    "gsm8k",
                ]
            )


def test_batch_run_command_openreward_rejects_non_positive_max_steps() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        with pytest.raises(ValueError, match="--max-steps must be a positive integer"):
            main(
                [
                    "batch-run",
                    "--app-dir",
                    str(app_dir),
                    "--project-dir",
                    str(app_dir),
                    "--dataset",
                    "openreward",
                    "--environment",
                    "GeneralReasoning/OfficeQA",
                    "--max-steps",
                    "0",
                ]
            )


def test_batch_run_command_openreward_outputs_openreward_metrics(capsys, monkeypatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        batch_dir = app_dir / ".opm_train" / "batches" / "or-batch-1"
        results_path = batch_dir / "openreward_results.jsonl"
        summary_path = batch_dir / "openreward_summary.json"
        batch_dir.mkdir(parents=True, exist_ok=True)
        results_path.write_text("", encoding="utf-8")
        summary_path.write_text("{}", encoding="utf-8")

        async def _fake_run_batch(config, orchestrator_factory=None):
            _ = orchestrator_factory
            assert config.dataset == "openreward"
            assert config.environment == "GeneralReasoning/OfficeQA"
            assert config.task_index == 0
            assert config.input_path is None
            assert config.task_specs == ()
            return batch_runner_module.BatchRunOutput(
                batch_id="or-batch-1",
                batch_dir=batch_dir,
                results_path=results_path,
                summary_path=summary_path,
                summary=batch_runner_module.OpenRewardBatchSummary(
                    total=1,
                    completed=1,
                    finished=1,
                    failed=0,
                    total_reward=1.0,
                    avg_reward=1.0,
                    output_paths={
                        "openreward_results_jsonl": str(results_path),
                        "openreward_summary_json": str(summary_path),
                    },
                ),
            )

        monkeypatch.setattr(cli_module, "run_batch", _fake_run_batch)

        rc = main(
            [
                "batch-run",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(app_dir),
                "--dataset",
                "openreward",
                "--environment",
                "GeneralReasoning/OfficeQA",
                "--task-index",
                "0",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["batch_id"] == "or-batch-1"
        assert payload["total"] == 1
        assert payload["completed"] == 1
        assert payload["finished"] == 1
        assert payload["failed"] == 0
        assert payload["total_reward"] == 1.0
        assert payload["avg_reward"] == 1.0
        assert payload["results_path"] == str(results_path)
        assert payload["summary_path"] == str(summary_path)


def test_batch_run_command_openreward_passes_task_specs(capsys, monkeypatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        batch_dir = app_dir / ".opm_train" / "batches" / "or-batch-spec"
        results_path = batch_dir / "openreward_results.jsonl"
        summary_path = batch_dir / "openreward_summary.json"
        batch_dir.mkdir(parents=True, exist_ok=True)
        results_path.write_text("", encoding="utf-8")
        summary_path.write_text("{}", encoding="utf-8")

        async def _fake_run_batch(config, orchestrator_factory=None):
            _ = orchestrator_factory
            assert config.dataset == "openreward"
            assert config.environment == "GeneralReasoning/OfficeQA"
            assert config.task_specs == ("train:0:2", "validation")
            assert config.task_index is None
            assert config.start is None
            assert config.stop is None
            return batch_runner_module.BatchRunOutput(
                batch_id="or-batch-spec",
                batch_dir=batch_dir,
                results_path=results_path,
                summary_path=summary_path,
                summary=batch_runner_module.OpenRewardBatchSummary(
                    total=2,
                    completed=2,
                    finished=2,
                    failed=0,
                    total_reward=2.0,
                    avg_reward=1.0,
                    output_paths={
                        "openreward_results_jsonl": str(results_path),
                        "openreward_summary_json": str(summary_path),
                    },
                ),
            )

        monkeypatch.setattr(cli_module, "run_batch", _fake_run_batch)

        rc = main(
            [
                "batch-run",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(app_dir),
                "--dataset",
                "openreward",
                "--environment",
                "GeneralReasoning/OfficeQA",
                "--task-spec",
                "train:0:2",
                "--task-spec",
                "validation",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["batch_id"] == "or-batch-spec"
        assert payload["total"] == 2


def test_sft_command_outputs_structured_payload(capsys, monkeypatch) -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "sft.jsonl"
        input_path.write_text('{"id":"x1","prompt":"hello","completion":"world"}\n', encoding="utf-8")
        artifact_dir = app_dir / ".opm_train" / "sft_runs" / "sft-cli-run"
        artifact_dir.mkdir(parents=True)
        result_path = artifact_dir / "result.json"
        metrics_path = artifact_dir / "metrics.jsonl"
        result_path.write_text("{}", encoding="utf-8")
        metrics_path.write_text("", encoding="utf-8")

        def _fake_run_sft(config):
            assert config.backend == "tinker"
            assert config.base_model == "Qwen/Qwen3"
            return SFTRunOutput(
                run_id="sft-cli-run",
                artifact_dir=artifact_dir,
                result_path=result_path,
                metrics_path=metrics_path,
                total_examples=1,
                result=SFTBackendResult(
                    backend="tinker",
                    base_model="Qwen/Qwen3",
                    output_model="demo-model",
                    losses=[0.7, 0.3],
                    checkpoint_path="tinker://demo/checkpoint",
                    sample_output="ok",
                ),
            )

        monkeypatch.setattr(cli_module, "run_sft", _fake_run_sft)

        rc = main(
            [
                "sft",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(app_dir),
                "--backend",
                "tinker",
                "--input",
                str(input_path),
                "--base-model",
                "Qwen/Qwen3",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["run_id"] == "sft-cli-run"
        assert payload["backend"] == "tinker"
        assert payload["total_examples"] == 1
        assert payload["steps"] == 2
        assert payload["checkpoint_path"] == "tinker://demo/checkpoint"


def test_export_command_writes_raw_and_sft_outputs(capsys) -> None:
    with TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        app_dir = workspace / "app"
        project_dir = workspace / "project"
        project_dir.mkdir(parents=True)
        shutil.copytree(APP_DIR / "prompts", app_dir / "prompts")
        shutil.copy2(APP_DIR / "opm_train.toml", app_dir / "opm_train.toml")

        rc = main(
            [
                "smoke",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(project_dir),
                "--task",
                "export-smoke",
            ]
        )
        assert rc == 0
        smoke_payload = json.loads(capsys.readouterr().out.strip())
        session_id = smoke_payload["session_id"]

        raw_output = workspace / "raw.json"
        rc = main(
            [
                "export",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(project_dir),
                "--session-id",
                session_id,
                "--mode",
                "raw",
                "--output",
                str(raw_output),
            ]
        )
        assert rc == 0
        raw_meta = json.loads(capsys.readouterr().out.strip())
        assert raw_meta["count"] >= 1
        raw_payload = json.loads(raw_output.read_text(encoding="utf-8"))
        assert raw_payload["session_id"] == session_id
        assert len(raw_payload["turns"]) >= 1
        root_agent_id = str(raw_payload["session"]["root_agent_id"])

        scoped_output = workspace / "scoped.json"
        rc = main(
            [
                "export",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(project_dir),
                "--session-id",
                session_id,
                "--agent-id",
                root_agent_id,
                "--step",
                "1",
                "--mode",
                "raw",
                "--output",
                str(scoped_output),
            ]
        )
        assert rc == 0
        scoped_meta = json.loads(capsys.readouterr().out.strip())
        assert scoped_meta["count"] == 1
        scoped_payload = json.loads(scoped_output.read_text(encoding="utf-8"))
        assert len(scoped_payload["turns"]) == 1
        assert int(scoped_payload["turns"][0]["step"]) == 1

        sft_output = workspace / "sft.jsonl"
        rc = main(
            [
                "export",
                "--app-dir",
                str(app_dir),
                "--project-dir",
                str(project_dir),
                "--session-id",
                session_id,
                "--mode",
                "sft",
                "--output",
                str(sft_output),
            ]
        )
        assert rc == 0
        sft_meta = json.loads(capsys.readouterr().out.strip())
        assert sft_meta["count"] >= 1
        rows = [json.loads(line) for line in sft_output.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows
        target = rows[0]["target"]
        assert target["role"] == "assistant"


def test_export_command_rejects_old_snapshot_schema() -> None:
    with TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        app_dir = workspace / "app"
        project_dir = workspace / "project"
        project_dir.mkdir(parents=True)
        shutil.copytree(APP_DIR / "prompts", app_dir / "prompts")
        shutil.copy2(APP_DIR / "opm_train.toml", app_dir / "opm_train.toml")

        storage = SessionStorage(app_dir=app_dir, data_dir_name=".opm_train")
        session_id = "session-old-schema"
        storage.snapshot_path(session_id).write_text(
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

        with pytest.raises(ValueError, match="schema_version >= 4"):
            main(
                [
                    "export",
                    "--app-dir",
                    str(app_dir),
                    "--project-dir",
                    str(project_dir),
                    "--session-id",
                    session_id,
                    "--mode",
                    "raw",
                ]
            )


def test_export_command_rejects_step_without_agent_id() -> None:
    with TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        app_dir = workspace / "app"
        project_dir = workspace / "project"
        project_dir.mkdir(parents=True)
        shutil.copytree(APP_DIR / "prompts", app_dir / "prompts")
        shutil.copy2(APP_DIR / "opm_train.toml", app_dir / "opm_train.toml")

        with pytest.raises(ValueError, match="--step requires --agent-id"):
            main(
                [
                    "export",
                    "--app-dir",
                    str(app_dir),
                    "--project-dir",
                    str(project_dir),
                    "--session-id",
                    "session-any",
                    "--step",
                    "1",
                    "--mode",
                    "raw",
                ]
            )

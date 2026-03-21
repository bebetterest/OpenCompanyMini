from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

import opm_train.batch_runner as batch_runner_module
from opm_train.cli import main
from opm_train.models import RunSession, SessionStatus

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

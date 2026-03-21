from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory

import pytest

from opm_train.cli import main

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

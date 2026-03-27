from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from opm_train.diagnostics import build_doctor_payload
from opm_train.orchestrator import RuntimeOrchestrator

APP_DIR = Path(__file__).resolve().parents[1]


def test_build_doctor_payload_reports_contract_mismatch() -> None:
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
        orchestrator = RuntimeOrchestrator(project_dir=APP_DIR, app_dir=app_dir, llm_client=object())
        payload = build_doctor_payload(orchestrator=orchestrator, app_dir=app_dir, project_dir=APP_DIR)
        assert payload["tool_contract_ok"] is False
        assert payload["ready_for_real_run"] is False
        assert any("missing_tool" in item for item in payload["tool_contract_issues"])


def test_build_doctor_payload_reports_missing_core_tools() -> None:
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
                    "finish": {"type": "function", "function": {"name": "finish", "parameters": {"type": "object"}}},
                }
            ),
            encoding="utf-8",
        )
        (app_dir / "opm_train.toml").write_text(
            "\n".join(
                [
                    "[runtime.tools]",
                    'root_tools = ["finish"]',
                    'worker_tools = ["finish"]',
                ]
            ),
            encoding="utf-8",
        )
        orchestrator = RuntimeOrchestrator(project_dir=APP_DIR, app_dir=app_dir, llm_client=object())
        payload = build_doctor_payload(orchestrator=orchestrator, app_dir=app_dir, project_dir=APP_DIR)
        assert payload["tool_contract_ok"] is False
        issues = payload["tool_contract_issues"]
        assert any(str(item).startswith("root_enabled_missing_core_tools:") for item in issues)
        assert any(str(item).startswith("worker_enabled_missing_core_tools:") for item in issues)
        assert any("shell" in str(item) for item in issues)

"""Command-line interface for running, resuming, and diagnosing sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from opm_train.diagnostics import build_doctor_payload
from opm_train.llm import ChatResult
from opm_train.orchestrator import RuntimeOrchestrator, default_app_dir

_PROVIDER_CHOICES = ["openrouter", "tinker", "custom"]


def build_parser() -> argparse.ArgumentParser:
    """Build ``opm-train`` argument parser."""
    parser = argparse.ArgumentParser(prog="opm-train")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Create a new session and run task")
    run_parser.add_argument("task", help="Task for root coordinator")
    _add_common_path_args(run_parser)
    _add_timer_arg(run_parser)
    _add_provider_profile_arg(run_parser, help_text="Override provider profile")
    run_parser.add_argument("--model", default=None, help="Override model for this run")

    resume_parser = subparsers.add_parser("resume", help="Resume an existing session")
    resume_parser.add_argument("session_id", help="Session id")
    resume_parser.add_argument("instruction", help="Follow-up instruction")
    _add_common_path_args(resume_parser)
    _add_timer_arg(resume_parser)
    _add_provider_profile_arg(resume_parser, help_text="Override provider profile")
    resume_parser.add_argument("--model", default=None, help="Override model for resume")

    smoke_parser = subparsers.add_parser("smoke", help="Run local smoke test without external LLM API")
    _add_common_path_args(smoke_parser)
    _add_timer_arg(smoke_parser)
    smoke_parser.add_argument("--task", default="Run a local smoke test and verify core runtime flow", help="Smoke task")

    doctor_parser = subparsers.add_parser("doctor", help="Check runtime setup and configuration")
    _add_common_path_args(doctor_parser)
    _add_provider_profile_arg(doctor_parser, help_text="Override provider profile for check")

    return parser


class SmokeLLM:
    """Deterministic local LLM stub for smoke tests."""

    def __init__(self) -> None:
        """Initialize deterministic call counters."""
        self.root_calls = 0

    async def stream_chat(self, **kwargs: Any) -> ChatResult:
        """Return deterministic tool/action payloads for smoke coverage."""
        messages = kwargs["messages"]
        first_user = str(messages[1].get("content", "")) if len(messages) > 1 else ""
        route = "worker" if first_user.startswith("Assigned instruction:") else "root"
        if route == "root":
            self.root_calls += 1
            if self.root_calls == 1:
                payload = {
                    "actions": [
                        {
                            "type": "spawn_agent",
                            "name": "Smoke Worker",
                            "instruction": "Run a minimal execution and report.",
                            "blocking": True,
                        }
                    ]
                }
            else:
                payload = {
                    "actions": [
                        {
                            "type": "finish",
                            "status": "completed",
                            "summary": "Smoke run completed successfully.",
                        }
                    ]
                }
            return ChatResult(content=json.dumps(payload), raw_events=[])

        payload = {
            "actions": [
                {
                    "type": "finish",
                    "status": "completed",
                    "summary": "Worker executed smoke flow.",
                    "next_recommendation": "No follow-up required.",
                }
            ]
        }
        return ChatResult(content=json.dumps(payload), raw_events=[])


def _add_common_path_args(parser: argparse.ArgumentParser) -> None:
    """Add shared path options to one subcommand parser."""
    parser.add_argument("--project-dir", default=".", help="Target project directory")
    parser.add_argument("--app-dir", default=None, help="Application directory containing opm_train.toml")


def _add_provider_profile_arg(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    """Add shared provider-profile option to one subcommand parser."""
    parser.add_argument(
        "--provider-profile",
        default=None,
        choices=_PROVIDER_CHOICES,
        help=help_text,
    )


def _add_timer_arg(parser: argparse.ArgumentParser) -> None:
    """Add optional runtime timer flag for per-module profiling output."""
    parser.add_argument(
        "--timer",
        action="store_true",
        help="Enable per-module runtime timing logs under session/timers/",
    )


def _build_orchestrator(*, args: argparse.Namespace, app_dir: Path, project_dir: Path) -> RuntimeOrchestrator:
    """Construct orchestrator according to command mode."""
    if args.command in {"smoke", "doctor"}:
        orchestrator = RuntimeOrchestrator(
            project_dir=project_dir,
            app_dir=app_dir,
            llm_client=SmokeLLM(),
            model_override=getattr(args, "model", None),
            timer_enabled=bool(getattr(args, "timer", False)),
        )
    else:
        orchestrator = RuntimeOrchestrator(
            project_dir=project_dir,
            app_dir=app_dir,
            model_override=getattr(args, "model", None),
            timer_enabled=bool(getattr(args, "timer", False)),
        )
    profile_override = getattr(args, "provider_profile", None)
    if profile_override:
        if args.command in {"smoke", "doctor"}:
            orchestrator.config.provider.profile = str(profile_override)
        else:
            orchestrator.set_provider_profile(str(profile_override))
    return orchestrator


def _print_session_payload(*, session: Any, data_dir: Path) -> None:
    """Print compact JSON output for successful run-like commands."""
    print(
        json.dumps(
            {
                "session_id": session.id,
                "status": session.status.value,
                "final_summary": session.final_summary,
                "data_dir": str(data_dir),
            },
            ensure_ascii=False,
        )
    )


async def _run_async(args: argparse.Namespace) -> int:
    """Dispatch CLI command and print compact JSON result payloads."""
    app_dir = Path(args.app_dir).resolve() if args.app_dir else default_app_dir()
    project_dir = Path(args.project_dir).resolve()
    orchestrator = _build_orchestrator(args=args, app_dir=app_dir, project_dir=project_dir)

    if args.command == "resume":
        session = await orchestrator.resume(args.session_id, args.instruction)
    elif args.command in {"run", "smoke"}:
        session = await orchestrator.run_task(args.task)
    elif args.command == "doctor":
        print(
            json.dumps(
                build_doctor_payload(orchestrator=orchestrator, app_dir=app_dir, project_dir=project_dir),
                ensure_ascii=False,
            )
        )
        return 0
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command: {args.command}")

    _print_session_payload(session=session, data_dir=orchestrator.storage.session_dir(session.id))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint returning process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

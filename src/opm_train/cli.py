"""Command-line interface for running, batching, resuming, and diagnosing."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from opm_train.batch_runner import BatchRunConfig, OpenRewardBatchSummary, OrchestratorFactory, run_batch
from opm_train.config import OPMTrainConfig, PROVIDER_PROFILE_NAMES
from opm_train.data import list_dataset_adapters
from opm_train.diagnostics import build_doctor_payload
from opm_train.llm import ChatResult
from opm_train.orchestrator import RuntimeOrchestrator, default_app_dir
from opm_train.sft import SFTRunConfig, list_sft_backends, run_sft
from opm_train.storage import SessionStorage
from opm_train.trajectory import ExportSchemaError, export_trajectory


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

    batch_parser = subparsers.add_parser("batch-run", help="Run dataset prompts in batch and validate outputs")
    _add_common_path_args(batch_parser)
    _add_timer_arg(batch_parser)
    _add_provider_profile_arg(batch_parser, help_text="Override provider profile for batch run")
    batch_parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset adapter name (available: {', '.join([*list_dataset_adapters(), 'mixed', 'openreward'])})",
    )
    batch_parser.add_argument("--input", required=False, help="Path to local dataset JSONL input")
    batch_parser.add_argument(
        "--adapter-key",
        default="adapter",
        help="Per-row adapter key used when --dataset mixed (default: adapter)",
    )
    batch_parser.add_argument("--limit", type=int, default=None, help="Optional max sample count")
    batch_parser.add_argument("--concurrency", type=int, default=4, help="Batch concurrency (default: 4)")
    batch_parser.add_argument("--batch-id", default=None, help="Optional stable batch id")
    batch_parser.add_argument("--resume", action="store_true", help="Resume from existing batch directory")
    batch_parser.add_argument("--smoke", action="store_true", help="Run batch with local smoke LLM (no API key)")
    batch_parser.add_argument("--model", default=None, help="Override model for this batch run")
    batch_parser.add_argument(
        "--environment",
        default=None,
        help="OpenReward environment name (required when --dataset openreward)",
    )
    batch_parser.add_argument(
        "--split",
        default="train",
        help="OpenReward task split (default: train)",
    )
    batch_parser.add_argument(
        "--task-spec",
        action="append",
        default=None,
        help="Repeatable OpenReward selector: '<split>' or '<split>:<start>:<stop>'",
    )
    batch_parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="OpenReward single task index selector (mutually exclusive with --start/--stop)",
    )
    batch_parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="OpenReward range start index (inclusive)",
    )
    batch_parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="OpenReward range stop index (exclusive)",
    )
    batch_parser.add_argument(
        "--variant",
        default=None,
        help="Optional OpenReward environment variant",
    )
    batch_parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenReward base URL (for self-hosted/local environments)",
    )
    batch_parser.add_argument(
        "--openreward-tool-format",
        default=None,
        help="Override OpenReward tool schema format (for example: openai/openrouter/anthropic/google)",
    )
    batch_parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
        help="Max model turns per OpenReward task loop (default: 64)",
    )

    sft_parser = subparsers.add_parser("sft", help="Run supervised fine-tuning with pluggable backend")
    _add_common_path_args(sft_parser)
    sft_parser.add_argument("--backend", default="tinker", choices=list_sft_backends(), help="SFT backend name")
    sft_parser.add_argument("--input", required=True, help="Path to local SFT JSONL input")
    sft_parser.add_argument("--base-model", required=True, help="Base model id/path for selected SFT backend")
    sft_parser.add_argument("--output-model", default=None, help="Optional output model name")
    sft_parser.add_argument("--steps", type=int, default=6, help="Training steps (default: 6)")
    sft_parser.add_argument("--batch-size", type=int, default=8, help="Training batch size (default: 8)")
    sft_parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate (default: 1e-4)")
    sft_parser.add_argument("--rank", type=int, default=32, help="LoRA rank for backends that support it (default: 32)")
    sft_parser.add_argument("--limit", type=int, default=None, help="Optional max training sample count")
    sft_parser.add_argument("--prompt-key", default=None, help="Override prompt field key in JSONL")
    sft_parser.add_argument("--completion-key", default=None, help="Override completion field key in JSONL")
    sft_parser.add_argument("--sample-prompt", default=None, help="Optional prompt to sample after training")
    sft_parser.add_argument("--sample-max-tokens", type=int, default=64, help="Max tokens for optional post-train sample")
    sft_parser.add_argument("--sample-temperature", type=float, default=0.0, help="Temperature for optional post-train sample")
    sft_parser.add_argument("--run-id", default=None, help="Optional stable SFT run id")

    export_parser = subparsers.add_parser("export", help="Export session trajectory as raw or SFT payload")
    _add_common_path_args(export_parser)
    export_parser.add_argument("--session-id", required=True, help="Session id to export")
    export_parser.add_argument("--agent-id", default=None, help="Optional target agent id for scoped export")
    export_parser.add_argument("--step", type=int, default=None, help="Optional step number (requires --agent-id)")
    export_parser.add_argument("--mode", required=True, choices=["raw", "sft"], help="Export mode")
    export_parser.add_argument("--output", default=None, help="Optional output path (json for raw, jsonl for sft)")

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
        choices=list(PROVIDER_PROFILE_NAMES),
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
    if args.command == "export":
        if args.step is not None and not str(args.agent_id or "").strip():
            raise ValueError("--step requires --agent-id")
        if args.step is not None and int(args.step) <= 0:
            raise ValueError("--step must be a positive integer")
        runtime_config = OPMTrainConfig.load(app_dir)
        storage = SessionStorage(app_dir=app_dir, data_dir_name=runtime_config.project.data_dir)
        try:
            payload = export_trajectory(
                storage=storage,
                session_id=str(args.session_id),
                mode=str(args.mode),
                agent_id=getattr(args, "agent_id", None),
                step=getattr(args, "step", None),
            )
        except ExportSchemaError as exc:
            raise ValueError(str(exc)) from exc

        output_path = Path(str(args.output)).resolve() if getattr(args, "output", None) else None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if args.mode == "raw":
                output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                exported_count = len((payload or {}).get("turns", [])) if isinstance(payload, dict) else 0
            else:
                rows = payload if isinstance(payload, list) else []
                with output_path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False))
                        handle.write("\n")
                exported_count = len(rows)
            print(
                json.dumps(
                    {
                        "session_id": str(args.session_id),
                        "mode": str(args.mode),
                        "agent_id": getattr(args, "agent_id", None),
                        "step": getattr(args, "step", None),
                        "output_path": str(output_path),
                        "count": int(exported_count),
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.command == "sft":
        sft_output = run_sft(
            SFTRunConfig(
                backend=str(args.backend),
                input_path=Path(str(args.input)),
                project_dir=project_dir,
                app_dir=app_dir,
                base_model=str(args.base_model),
                output_model=getattr(args, "output_model", None),
                steps=int(getattr(args, "steps", 6)),
                batch_size=int(getattr(args, "batch_size", 8)),
                learning_rate=float(getattr(args, "learning_rate", 1e-4)),
                rank=int(getattr(args, "rank", 32)),
                limit=getattr(args, "limit", None),
                prompt_key=getattr(args, "prompt_key", None),
                completion_key=getattr(args, "completion_key", None),
                sample_prompt=getattr(args, "sample_prompt", None),
                sample_max_tokens=int(getattr(args, "sample_max_tokens", 64)),
                sample_temperature=float(getattr(args, "sample_temperature", 0.0)),
                run_id=getattr(args, "run_id", None),
            )
        )
        print(
            json.dumps(
                {
                    "run_id": sft_output.run_id,
                    "backend": sft_output.result.backend,
                    "base_model": sft_output.result.base_model,
                    "output_model": sft_output.result.output_model,
                    "total_examples": sft_output.total_examples,
                    "steps": len(sft_output.result.losses),
                    "losses": sft_output.result.losses,
                    "checkpoint_path": sft_output.result.checkpoint_path,
                    "sample_output": sft_output.result.sample_output,
                    "artifact_dir": str(sft_output.artifact_dir),
                    "result_path": str(sft_output.result_path),
                    "metrics_path": str(sft_output.metrics_path),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "batch-run":
        _validate_batch_run_args(args)
        input_arg = str(getattr(args, "input", "") or "").strip()
        input_path = Path(input_arg) if input_arg else None
        batch_orchestrator_factory = _build_batch_orchestrator_factory(
            args=args,
            app_dir=app_dir,
            project_dir=project_dir,
        )
        batch_output = await run_batch(
            BatchRunConfig(
                dataset=str(args.dataset),
                input_path=input_path,
                project_dir=project_dir,
                app_dir=app_dir,
                provider_profile=getattr(args, "provider_profile", None),
                model=getattr(args, "model", None),
                timer=bool(getattr(args, "timer", False)),
                limit=getattr(args, "limit", None),
                concurrency=int(getattr(args, "concurrency", 4)),
                batch_id=getattr(args, "batch_id", None),
                resume=bool(getattr(args, "resume", False)),
                adapter_key=str(getattr(args, "adapter_key", "adapter")),
                environment=getattr(args, "environment", None),
                split=str(getattr(args, "split", "train")),
                task_index=getattr(args, "task_index", None),
                start=getattr(args, "start", None),
                stop=getattr(args, "stop", None),
                variant=getattr(args, "variant", None),
                base_url=getattr(args, "base_url", None),
                openreward_tool_format=getattr(args, "openreward_tool_format", None),
                max_steps=int(getattr(args, "max_steps", 64)),
                task_specs=tuple(getattr(args, "task_spec", []) or ()),
            ),
            orchestrator_factory=batch_orchestrator_factory,
        )
        if isinstance(batch_output.summary, OpenRewardBatchSummary):
            print(
                json.dumps(
                    {
                        "batch_id": batch_output.batch_id,
                        "total": batch_output.summary.total,
                        "completed": batch_output.summary.completed,
                        "finished": batch_output.summary.finished,
                        "failed": batch_output.summary.failed,
                        "total_reward": batch_output.summary.total_reward,
                        "avg_reward": batch_output.summary.avg_reward,
                        "results_path": str(batch_output.results_path),
                        "summary_path": str(batch_output.summary_path),
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        print(
            json.dumps(
                {
                    "batch_id": batch_output.batch_id,
                    "total": batch_output.summary.total,
                    "validated": batch_output.summary.validated,
                    "correct": batch_output.summary.correct,
                    "accuracy": batch_output.summary.accuracy,
                    "failed_sessions": batch_output.summary.failed_sessions,
                    "results_path": str(batch_output.results_path),
                    "summary_path": str(batch_output.summary_path),
                },
                ensure_ascii=False,
            )
        )
        return 0

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


def _build_batch_orchestrator_factory(
    *,
    args: argparse.Namespace,
    app_dir: Path,
    project_dir: Path,
) -> OrchestratorFactory | None:
    """Return optional batch orchestrator factory for smoke mode."""
    if not bool(getattr(args, "smoke", False)):
        return None

    def create() -> RuntimeOrchestrator:
        orchestrator = RuntimeOrchestrator(
            project_dir=project_dir,
            app_dir=app_dir,
            llm_client=SmokeLLM(),
            model_override=getattr(args, "model", None),
            timer_enabled=bool(getattr(args, "timer", False)),
        )
        profile_override = getattr(args, "provider_profile", None)
        if profile_override:
            orchestrator.config.provider.profile = str(profile_override)
        return orchestrator

    return create


def _validate_batch_run_args(args: argparse.Namespace) -> None:
    """Validate batch-run argument combinations before dispatch."""
    dataset = str(getattr(args, "dataset", "")).strip().lower()
    input_value = str(getattr(args, "input", "") or "").strip()
    is_openreward = dataset == "openreward"

    if not is_openreward and not input_value:
        raise ValueError("--input is required unless --dataset openreward")

    if not is_openreward:
        return

    environment = str(getattr(args, "environment", "") or "").strip()
    if not environment:
        raise ValueError("--environment is required when --dataset openreward")

    task_index = getattr(args, "task_index", None)
    start = getattr(args, "start", None)
    stop = getattr(args, "stop", None)
    task_specs = [str(item).strip() for item in (getattr(args, "task_spec", None) or []) if str(item).strip()]
    if task_index is not None and (start is not None or stop is not None):
        raise ValueError("--task-index cannot be used with --start/--stop")
    if task_specs and (task_index is not None or start is not None or stop is not None):
        raise ValueError("--task-spec cannot be used with --task-index/--start/--stop")

    max_steps = int(getattr(args, "max_steps", 64))
    if max_steps <= 0:
        raise ValueError("--max-steps must be a positive integer")


if __name__ == "__main__":
    raise SystemExit(main())

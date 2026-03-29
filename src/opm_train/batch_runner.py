"""Dataset batch execution runner."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, TypeVar

from opm_train.config import OPMTrainConfig, ProviderProfileConfig
from opm_train.data import BatchItemResult, BatchSummary, DatasetAdapter, DatasetSample, get_dataset_adapter
from opm_train.data.jsonl import iter_json_objects
from opm_train.llm import OpenAICompatibleClient
from opm_train.models import RunSession, SessionStatus
from opm_train.orchestrator import RuntimeOrchestrator
from opm_train.storage import SessionStorage
from opm_train.utils import ensure_directory

_MIXED_DATASET_NAME = "mixed"
_OPENREWARD_DATASET_NAME = "openreward"
_OPENREWARD_RESULTS_FILENAME = "openreward_results.jsonl"
_OPENREWARD_SUMMARY_FILENAME = "openreward_summary.json"
_RowT = TypeVar("_RowT")


@dataclass(slots=True, frozen=True)
class BatchRunConfig:
    """Input configuration for one dataset-driven batch run."""

    dataset: str
    input_path: Path | None
    project_dir: Path
    app_dir: Path
    provider_profile: str | None = None
    model: str | None = None
    timer: bool = False
    limit: int | None = None
    concurrency: int = 4
    batch_id: str | None = None
    resume: bool = False
    adapter_key: str = "adapter"
    environment: str | None = None
    split: str = "train"
    task_index: int | None = None
    start: int | None = None
    stop: int | None = None
    variant: str | None = None
    base_url: str | None = None
    openreward_tool_format: str | None = None
    max_steps: int = 64
    task_specs: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class OpenRewardBatchItemResult:
    """Per-task output row for OpenReward batch execution."""

    environment: str
    split: str
    variant: str | None
    task_key: str
    task_index: int | None
    reward_total: float
    finished: bool
    tool_calls: int
    turns: int
    session_status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert row into JSON-serializable dictionary."""
        return asdict(self)


@dataclass(slots=True, frozen=True)
class OpenRewardBatchSummary:
    """Aggregated summary for OpenReward batch execution."""

    total: int
    completed: int
    finished: int
    failed: int
    total_reward: float
    avg_reward: float
    output_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Convert summary into JSON-serializable dictionary."""
        return asdict(self)


@dataclass(slots=True, frozen=True)
class BatchRunOutput:
    """Batch run artifacts and aggregated metrics."""

    batch_id: str
    batch_dir: Path
    results_path: Path
    summary_path: Path
    summary: BatchSummary | OpenRewardBatchSummary


@dataclass(slots=True, frozen=True)
class RoutedSample:
    """One sample routed to a concrete adapter."""

    adapter_name: str
    adapter: DatasetAdapter
    sample: DatasetSample

    @property
    def sample_key(self) -> str:
        """Return stable resume key for one routed sample."""
        return _row_key(self.adapter_name, self.sample.sample_id)


@dataclass(slots=True, frozen=True)
class OpenRewardTaskRef:
    """One OpenReward task with resolved stable key metadata."""

    task: Any
    split: str
    task_key: str
    task_index: int | None


@dataclass(slots=True, frozen=True)
class OpenRewardLoopOutcome:
    """Execution telemetry for one OpenReward session loop."""

    reward_total: float
    finished: bool
    tool_calls: int
    turns: int
    error: str | None = None


class OpenRewardLoopError(RuntimeError):
    """Raised when OpenReward loop fails after partial progress."""

    def __init__(self, *, original_error: Exception, outcome: OpenRewardLoopOutcome) -> None:
        super().__init__(str(original_error))
        self.original_error = original_error
        self.outcome = outcome


@dataclass(slots=True, frozen=True)
class OpenRewardTaskSpec:
    """One OpenReward mixed selector: full split or split range."""

    split: str
    start: int | None = None
    stop: int | None = None


class TaskRunner(Protocol):
    """Minimal task-runner contract used by the batch scheduler."""

    async def run_task(self, task: str) -> RunSession:
        """Run one task and return terminal run session payload."""


OrchestratorFactory = Callable[[], TaskRunner]


async def run_batch(
    config: BatchRunConfig,
    *,
    orchestrator_factory: OrchestratorFactory | None = None,
) -> BatchRunOutput:
    """Execute dataset samples in parallel with realtime writes and resume support."""
    app_dir = config.app_dir.resolve()
    project_dir = config.project_dir.resolve()

    runtime_config = OPMTrainConfig.load(app_dir)
    storage = SessionStorage(app_dir=app_dir, data_dir_name=runtime_config.project.data_dir)

    if _is_openreward_dataset(config.dataset):
        return await _run_openreward_batch(
            config=config,
            runtime_config=runtime_config,
            storage=storage,
        )

    return await _run_dataset_batch(
        config=config,
        app_dir=app_dir,
        project_dir=project_dir,
        storage=storage,
        orchestrator_factory=orchestrator_factory,
    )


async def _run_dataset_batch(
    *,
    config: BatchRunConfig,
    app_dir: Path,
    project_dir: Path,
    storage: SessionStorage,
    orchestrator_factory: OrchestratorFactory | None,
) -> BatchRunOutput:
    """Run legacy dataset adapters (gsm8k/simple_math/mixed) via orchestrator."""
    routed_samples = _load_routed_samples(config)

    batch_id = _resolve_batch_id(config.batch_id)
    batch_dir = storage.data_root / "batches" / batch_id
    results_path = batch_dir / "results.jsonl"
    summary_path = batch_dir / "summary.json"

    existing_rows = _prepare_batch_output(
        batch_dir=batch_dir,
        results_path=results_path,
        resume=config.resume,
        default_adapter_name=_default_adapter_name(config.dataset),
    )
    completed_keys = {_row_key(row.adapter_name, row.sample_id) for row in existing_rows}
    pending_samples = [item for item in routed_samples if item.sample_key not in completed_keys]

    factory = orchestrator_factory or _default_orchestrator_factory(
        project_dir=project_dir,
        app_dir=app_dir,
        model_override=config.model,
        timer_enabled=config.timer,
        provider_profile=config.provider_profile,
    )

    append_lock = asyncio.Lock()
    new_rows = await _run_pending_samples(
        routed_samples=pending_samples,
        orchestrator_factory=factory,
        concurrency=max(1, int(config.concurrency)),
        results_path=results_path,
        append_lock=append_lock,
    )

    all_rows = [*existing_rows, *new_rows]
    summary = _build_summary(rows=all_rows, results_path=results_path, summary_path=summary_path)
    summary_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return BatchRunOutput(
        batch_id=batch_id,
        batch_dir=batch_dir,
        results_path=results_path,
        summary_path=summary_path,
        summary=summary,
    )


async def _run_openreward_batch(
    *,
    config: BatchRunConfig,
    runtime_config: OPMTrainConfig,
    storage: SessionStorage,
) -> BatchRunOutput:
    """Run one OpenReward environment split using OpenAI-compatible model loop."""
    _validate_openreward_config(config)

    profile_name, provider_profile = _resolve_provider_profile(
        runtime_config=runtime_config,
        provider_profile=config.provider_profile,
    )
    llm_client = _build_openreward_llm_client(profile=provider_profile)
    model_name = _resolve_openreward_model(config=config, profile=provider_profile)
    tool_format = _resolve_openreward_tool_format(
        profile_name=profile_name,
        override=config.openreward_tool_format,
    )

    async_openreward_cls = _load_async_openreward_client_cls()
    or_client = _instantiate_openreward_client(async_openreward_cls=async_openreward_cls, config=config)
    environment = await _resolve_openreward_environment(or_client=or_client, config=config)

    task_refs = await _load_openreward_tasks(environment=environment, config=config)
    _ensure_unique_openreward_task_keys(task_refs)

    tools = await _list_openreward_tools(environment=environment, tool_format=tool_format)

    batch_id = _resolve_batch_id(config.batch_id)
    batch_dir = storage.data_root / "batches" / batch_id
    results_path = batch_dir / _OPENREWARD_RESULTS_FILENAME
    summary_path = batch_dir / _OPENREWARD_SUMMARY_FILENAME

    existing_rows = _prepare_openreward_output(
        batch_dir=batch_dir,
        results_path=results_path,
        resume=config.resume,
    )
    completed_keys = _resume_openreward_completed_keys(existing_rows)
    pending_tasks = [item for item in task_refs if item.task_key not in completed_keys]

    append_lock = asyncio.Lock()
    new_rows = await _run_pending_openreward_tasks(
        task_refs=pending_tasks,
        environment=environment,
        tools=tools,
        llm_client=llm_client,
        model_name=model_name,
        temperature=float(provider_profile.temperature),
        max_tokens=int(provider_profile.max_tokens),
        max_steps=int(config.max_steps),
        concurrency=max(1, int(config.concurrency)),
        results_path=results_path,
        append_lock=append_lock,
        environment_name=str(config.environment or ""),
        variant_name=_optional_str(config.variant),
    )

    all_rows = [*existing_rows, *new_rows]
    summary = _build_openreward_summary(rows=all_rows, results_path=results_path, summary_path=summary_path)
    summary_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    return BatchRunOutput(
        batch_id=batch_id,
        batch_dir=batch_dir,
        results_path=results_path,
        summary_path=summary_path,
        summary=summary,
    )


async def _run_pending_samples(
    *,
    routed_samples: list[RoutedSample],
    orchestrator_factory: OrchestratorFactory,
    concurrency: int,
    results_path: Path,
    append_lock: asyncio.Lock,
) -> list[BatchItemResult]:
    """Run pending samples and append each result row immediately."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task[BatchItemResult]] = [
        asyncio.create_task(
            _run_one_routed_sample(
                routed_sample=item,
                semaphore=semaphore,
                orchestrator_factory=orchestrator_factory,
            )
        )
        for item in routed_samples
    ]
    
    async def append(row: BatchItemResult) -> None:
        await _append_result_row(results_path=results_path, row=row, append_lock=append_lock)

    return await _collect_task_results(tasks=tasks, on_result=append)


async def _run_pending_openreward_tasks(
    *,
    task_refs: list[OpenRewardTaskRef],
    environment: Any,
    tools: list[dict[str, Any]],
    llm_client: OpenAICompatibleClient,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_steps: int,
    concurrency: int,
    results_path: Path,
    append_lock: asyncio.Lock,
    environment_name: str,
    variant_name: str | None,
) -> list[OpenRewardBatchItemResult]:
    """Run pending OpenReward tasks and append each row immediately."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task[OpenRewardBatchItemResult]] = [
        asyncio.create_task(
            _run_one_openreward_task(
                task_ref=item,
                semaphore=semaphore,
                environment=environment,
                tools=tools,
                llm_client=llm_client,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                max_steps=max_steps,
                environment_name=environment_name,
                variant_name=variant_name,
            )
        )
        for item in task_refs
    ]
    
    async def append(row: OpenRewardBatchItemResult) -> None:
        await _append_openreward_result_row(results_path=results_path, row=row, append_lock=append_lock)

    return await _collect_task_results(tasks=tasks, on_result=append)


async def _collect_task_results(
    *,
    tasks: list[asyncio.Task[_RowT]],
    on_result: Callable[[_RowT], Awaitable[None]],
) -> list[_RowT]:
    """Collect rows as tasks finish, append eagerly, and cancel siblings on errors."""
    rows: list[_RowT] = []
    try:
        for completed in asyncio.as_completed(tasks):
            row = await completed
            rows.append(row)
            await on_result(row)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return rows


async def _run_one_routed_sample(
    *,
    routed_sample: RoutedSample,
    semaphore: asyncio.Semaphore,
    orchestrator_factory: OrchestratorFactory,
) -> BatchItemResult:
    """Run one routed sample end-to-end and return one canonical result row."""
    adapter = routed_sample.adapter
    sample = routed_sample.sample
    adapter_name = routed_sample.adapter_name

    try:
        prepared = adapter.build_task_prompt(sample)
    except Exception as exc:
        return _failed_item_result(
            adapter_name=adapter_name,
            sample_id=sample.sample_id,
            task_prompt="",
            reference_answer=sample.answer,
            reference_answer_raw=sample.answer_raw,
            error=f"prompt_build_error:{type(exc).__name__}:{exc}",
        )

    async with semaphore:
        try:
            orchestrator = orchestrator_factory()
            session = await orchestrator.run_task(prepared.task_prompt)
        except Exception as exc:
            return _failed_item_result(
                adapter_name=adapter_name,
                sample_id=prepared.sample_id,
                task_prompt=prepared.task_prompt,
                reference_answer=prepared.reference_answer,
                reference_answer_raw=prepared.reference_answer_raw,
                error=f"run_error:{type(exc).__name__}:{exc}",
            )

        if session.status != SessionStatus.COMPLETED:
            return BatchItemResult(
                adapter_name=adapter_name,
                sample_id=prepared.sample_id,
                task_prompt=prepared.task_prompt,
                reference_answer=prepared.reference_answer,
                reference_answer_raw=prepared.reference_answer_raw,
                predicted_answer=None,
                is_correct=False,
                session_id=session.id,
                session_status=session.status.value,
                final_summary=session.final_summary,
                error=f"session_not_completed:{session.status.value}",
            )

        validation_error: str | None
        predicted_answer: str | None
        is_correct: bool

        try:
            validation = adapter.validate_result(sample=sample, session=session)
        except Exception as exc:
            validation_error = f"validation_error:{type(exc).__name__}:{exc}"
            predicted_answer = None
            is_correct = False
        else:
            validation_error = validation.error
            predicted_answer = validation.predicted_answer
            is_correct = bool(validation.is_correct)

        return BatchItemResult(
            adapter_name=adapter_name,
            sample_id=prepared.sample_id,
            task_prompt=prepared.task_prompt,
            reference_answer=prepared.reference_answer,
            reference_answer_raw=prepared.reference_answer_raw,
            predicted_answer=predicted_answer,
            is_correct=is_correct,
            session_id=session.id,
            session_status=session.status.value,
            final_summary=session.final_summary,
            error=validation_error,
        )


async def _run_one_openreward_task(
    *,
    task_ref: OpenRewardTaskRef,
    semaphore: asyncio.Semaphore,
    environment: Any,
    tools: list[dict[str, Any]],
    llm_client: OpenAICompatibleClient,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_steps: int,
    environment_name: str,
    variant_name: str | None,
) -> OpenRewardBatchItemResult:
    """Run one OpenReward task by looping model/tool calls until termination."""
    async with semaphore:
        try:
            outcome = await _run_openreward_session_loop(
                task=task_ref.task,
                environment=environment,
                tools=tools,
                llm_client=llm_client,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                max_steps=max_steps,
            )
        except OpenRewardLoopError as exc:
            return OpenRewardBatchItemResult(
                environment=environment_name,
                split=task_ref.split,
                variant=variant_name,
                task_key=task_ref.task_key,
                task_index=task_ref.task_index,
                reward_total=exc.outcome.reward_total,
                finished=False,
                tool_calls=exc.outcome.tool_calls,
                turns=exc.outcome.turns,
                session_status="failed",
                error=f"run_error:{type(exc.original_error).__name__}:{exc.original_error}",
            )
        except Exception as exc:
            return OpenRewardBatchItemResult(
                environment=environment_name,
                split=task_ref.split,
                variant=variant_name,
                task_key=task_ref.task_key,
                task_index=task_ref.task_index,
                reward_total=0.0,
                finished=False,
                tool_calls=0,
                turns=0,
                session_status="failed",
                error=f"run_error:{type(exc).__name__}:{exc}",
            )

        return OpenRewardBatchItemResult(
            environment=environment_name,
            split=task_ref.split,
            variant=variant_name,
            task_key=task_ref.task_key,
            task_index=task_ref.task_index,
            reward_total=outcome.reward_total,
            finished=outcome.finished,
            tool_calls=outcome.tool_calls,
            turns=outcome.turns,
            session_status="completed",
            error=outcome.error,
        )


async def _run_openreward_session_loop(
    *,
    task: Any,
    environment: Any,
    tools: list[dict[str, Any]],
    llm_client: OpenAICompatibleClient,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_steps: int,
) -> OpenRewardLoopOutcome:
    """Execute one OpenReward task loop and return loop-level telemetry."""
    reward_total = 0.0
    finished = False
    tool_calls = 0
    turns = 0

    try:
        session_cm = environment.session(task=task)
        async with _as_async_context_manager(session_cm) as session:
            prompt_blocks = await _maybe_await(session.get_prompt())
            prompt_text = _render_prompt_blocks(prompt_blocks)
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt_text}]

            while not finished and turns < max_steps:
                turns += 1
                chat_result = await llm_client.stream_chat(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                )
                normalized_tool_calls = _normalize_tool_calls(chat_result.tool_calls)
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": str(chat_result.content or ""),
                }
                if normalized_tool_calls:
                    assistant_message["tool_calls"] = normalized_tool_calls
                messages.append(assistant_message)

                if not normalized_tool_calls:
                    break

                for call in normalized_tool_calls:
                    tool_calls += 1
                    call_id = str(call.get("id", f"tool-call-{turns}-{tool_calls}"))
                    function_payload = call.get("function")
                    if not isinstance(function_payload, dict):
                        raise ValueError("tool_call missing function payload")
                    tool_name = str(function_payload.get("name", "")).strip()
                    if not tool_name:
                        raise ValueError("tool_call missing function name")
                    tool_arguments = _parse_tool_arguments(function_payload.get("arguments"))

                    tool_output = await _maybe_await(session.call_tool(tool_name, tool_arguments))
                    reward_total += _coerce_reward(_object_get(tool_output, "reward"))
                    finished = bool(_object_get(tool_output, "finished", False))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": _render_tool_output(tool_output),
                        }
                    )
                    if finished:
                        break
    except Exception as exc:
        raise OpenRewardLoopError(
            original_error=exc,
            outcome=OpenRewardLoopOutcome(
                reward_total=reward_total,
                finished=False,
                tool_calls=tool_calls,
                turns=turns,
            ),
        ) from exc

    loop_error = "max_steps_reached" if turns >= max_steps and not finished else None
    return OpenRewardLoopOutcome(
        reward_total=reward_total,
        finished=finished,
        tool_calls=tool_calls,
        turns=turns,
        error=loop_error,
    )


def _load_routed_samples(config: BatchRunConfig) -> list[RoutedSample]:
    """Load routed samples for single-adapter or mixed-adapter modes."""
    input_path = _require_input_path(config)
    if _is_mixed_dataset(config.dataset):
        routed = _load_mixed_routed_samples(
            input_path=input_path,
            adapter_key=config.adapter_key,
            limit=config.limit,
        )
    else:
        adapter = get_dataset_adapter(config.dataset)
        routed = [
            RoutedSample(adapter_name=adapter.name, adapter=adapter, sample=sample)
            for sample in adapter.load_samples(input_path=input_path, limit=config.limit)
        ]
    _ensure_unique_routed_sample_keys(routed)
    return routed


async def _load_openreward_tasks(*, environment: Any, config: BatchRunConfig) -> list[OpenRewardTaskRef]:
    """Resolve OpenReward tasks from one split using index/range/full selectors."""
    refs: list[OpenRewardTaskRef] = []
    global_order = 0
    task_specs = _parse_openreward_task_specs(config.task_specs)
    if task_specs:
        include_split_in_key = len({item.split for item in task_specs}) > 1
        for spec in task_specs:
            tasks, indices = await _fetch_openreward_tasks_for_selector(
                environment=environment,
                split=spec.split,
                task_index=None,
                start=spec.start,
                stop=spec.stop,
            )
            for local_order, task in enumerate(tasks):
                task_index = indices[local_order] if local_order < len(indices) else None
                refs.append(
                    OpenRewardTaskRef(
                        task=task,
                        split=spec.split,
                        task_key=_build_openreward_task_key(
                            task=task,
                            split=spec.split,
                            task_index=task_index,
                            order=global_order,
                            include_split_in_key=include_split_in_key,
                        ),
                        task_index=task_index,
                    )
                )
                global_order += 1
        refs = _dedupe_openreward_task_refs(refs)
        return _limit_openreward_task_refs(refs, config.limit)

    split = str(config.split or "train")
    tasks, indices = await _fetch_openreward_tasks_for_selector(
        environment=environment,
        split=split,
        task_index=config.task_index,
        start=config.start,
        stop=config.stop,
    )
    for local_order, task in enumerate(tasks):
        task_index = indices[local_order] if local_order < len(indices) else None
        refs.append(
            OpenRewardTaskRef(
                task=task,
                split=split,
                task_key=_build_openreward_task_key(
                    task=task,
                    split=split,
                    task_index=task_index,
                    order=local_order,
                    include_split_in_key=False,
                ),
                task_index=task_index,
            )
        )
    return _limit_openreward_task_refs(refs, config.limit)


async def _fetch_openreward_tasks_for_selector(
    *,
    environment: Any,
    split: str,
    task_index: int | None,
    start: int | None,
    stop: int | None,
) -> tuple[list[Any], list[int | None]]:
    """Load tasks and aligned indices for one OpenReward selector."""
    if task_index is not None:
        task = await _maybe_await(environment.get_task(split=split, index=int(task_index)))
        return [task], [int(task_index)]

    if start is not None or stop is not None:
        tasks_payload = await _maybe_await(
            environment.get_task_range(
                split=split,
                start=start,
                stop=stop,
            )
        )
        if not isinstance(tasks_payload, list):
            raise ValueError("OpenReward environment.get_task_range(...) must return a list")
        tasks = list(tasks_payload)
        if start is None:
            return tasks, [None for _ in tasks]
        start_value = int(start)
        if start_value < 0:
            return tasks, [None for _ in tasks]
        return tasks, [start_value + offset for offset in range(len(tasks))]

    tasks_payload = await _maybe_await(environment.list_tasks(split=split))
    if not isinstance(tasks_payload, list):
        raise ValueError("OpenReward environment.list_tasks(...) must return a list")
    tasks = list(tasks_payload)
    return tasks, [offset for offset in range(len(tasks))]


def _limit_openreward_task_refs(
    refs: list[OpenRewardTaskRef],
    limit: int | None,
) -> list[OpenRewardTaskRef]:
    """Apply optional global task limit after selector expansion/dedup."""
    if limit is None:
        return refs
    return refs[: max(0, int(limit))]


def _dedupe_openreward_task_refs(task_refs: list[OpenRewardTaskRef]) -> list[OpenRewardTaskRef]:
    """Drop duplicated OpenReward task refs while preserving order."""
    seen: set[str] = set()
    deduped: list[OpenRewardTaskRef] = []
    for item in task_refs:
        if item.task_key in seen:
            continue
        deduped.append(item)
        seen.add(item.task_key)
    return deduped


def _parse_openreward_task_specs(raw_specs: tuple[str, ...]) -> list[OpenRewardTaskSpec]:
    """Parse `--task-spec` selectors: `<split>` or `<split>:<start>:<stop>`."""
    specs: list[OpenRewardTaskSpec] = []
    for raw in raw_specs:
        text = str(raw).strip()
        if not text:
            continue
        parts = text.split(":")
        if len(parts) == 1:
            split = parts[0].strip()
            if not split:
                raise ValueError("OpenReward task spec split must be non-empty")
            specs.append(OpenRewardTaskSpec(split=split))
            continue
        if len(parts) != 3:
            raise ValueError(
                "Invalid OpenReward task spec format. Use `<split>` or `<split>:<start>:<stop>`."
            )
        split = parts[0].strip()
        if not split:
            raise ValueError("OpenReward task spec split must be non-empty")
        start = _parse_optional_spec_int(parts[1], field_name="start")
        stop = _parse_optional_spec_int(parts[2], field_name="stop")
        specs.append(OpenRewardTaskSpec(split=split, start=start, stop=stop))
    return specs


def _parse_optional_spec_int(raw: str, *, field_name: str) -> int | None:
    """Parse optional integer token from task spec segment."""
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"OpenReward task spec {field_name} must be integer or empty") from exc


def _load_mixed_routed_samples(
    *,
    input_path: Path,
    adapter_key: str,
    limit: int | None,
) -> list[RoutedSample]:
    """Load mixed JSONL rows where each row selects its own adapter."""
    key_name = str(adapter_key or "adapter").strip() or "adapter"
    routed: list[RoutedSample] = []
    for line_no, payload in iter_json_objects(input_path):
        adapter_name = str(payload.get(key_name, "")).strip().lower()
        if not adapter_name:
            raise ValueError(f"line {line_no} missing adapter selector key '{key_name}'")
        adapter = get_dataset_adapter(adapter_name)
        sample = adapter.sample_from_payload(payload, line_no=line_no)
        routed.append(RoutedSample(adapter_name=adapter_name, adapter=adapter, sample=sample))
        if limit is not None and len(routed) >= max(0, int(limit)):
            break
    return routed


def _prepare_batch_output(
    *,
    batch_dir: Path,
    results_path: Path,
    resume: bool,
    default_adapter_name: str,
) -> list[BatchItemResult]:
    """Prepare output directory/files and load prior rows for resume mode."""
    return _prepare_results_output(
        batch_dir=batch_dir,
        results_path=results_path,
        resume=resume,
        load_existing=lambda path: _load_result_rows(path, default_adapter_name=default_adapter_name),
    )


def _prepare_openreward_output(
    *,
    batch_dir: Path,
    results_path: Path,
    resume: bool,
) -> list[OpenRewardBatchItemResult]:
    """Prepare OpenReward batch output artifacts and optionally load existing rows."""
    return _prepare_results_output(
        batch_dir=batch_dir,
        results_path=results_path,
        resume=resume,
        load_existing=_load_openreward_result_rows,
    )


def _resume_openreward_completed_keys(rows: list[OpenRewardBatchItemResult]) -> set[str]:
    """Build resume key set, including legacy and split-prefixed compatibility keys."""
    completed: set[str] = set()
    for row in rows:
        key = str(row.task_key)
        completed.add(key)
        split = str(row.split or "").strip()
        if split and "::" not in key:
            completed.add(f"{split}::{key}")
    return completed


def _prepare_results_output(
    *,
    batch_dir: Path,
    results_path: Path,
    resume: bool,
    load_existing: Callable[[Path], list[_RowT]],
) -> list[_RowT]:
    """Prepare batch directory and optionally load existing rows for resume mode."""
    if resume:
        if not batch_dir.is_dir():
            raise ValueError(f"resume requested but batch directory does not exist: {batch_dir}")
        return load_existing(results_path)

    if batch_dir.exists():
        raise ValueError(
            f"batch directory already exists: {batch_dir}. Use --resume to continue this batch."
        )
    ensure_directory(batch_dir)
    results_path.write_text("", encoding="utf-8")
    return []


async def _append_result_row(*, results_path: Path, row: BatchItemResult, append_lock: asyncio.Lock) -> None:
    """Append one row to JSONL with coroutine-safe file writes."""
    await _append_jsonl_object(results_path=results_path, payload=row.to_dict(), append_lock=append_lock)


async def _append_openreward_result_row(
    *,
    results_path: Path,
    row: OpenRewardBatchItemResult,
    append_lock: asyncio.Lock,
) -> None:
    """Append one OpenReward result row to JSONL."""
    await _append_jsonl_object(results_path=results_path, payload=row.to_dict(), append_lock=append_lock)


async def _append_jsonl_object(*, results_path: Path, payload: dict[str, Any], append_lock: asyncio.Lock) -> None:
    """Append one JSON object row to JSONL with coroutine-safe writes."""
    async with append_lock:
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def _load_result_rows(results_path: Path, *, default_adapter_name: str) -> list[BatchItemResult]:
    """Load existing rows from JSONL for resume."""
    return [
        _row_from_dict(payload, default_adapter_name=default_adapter_name)
        for payload in _load_jsonl_dict_rows(results_path)
    ]


def _load_openreward_result_rows(results_path: Path) -> list[OpenRewardBatchItemResult]:
    """Load existing OpenReward result rows from JSONL for resume."""
    return [_openreward_row_from_dict(payload) for payload in _load_jsonl_dict_rows(results_path)]


def _load_jsonl_dict_rows(results_path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON object rows from JSONL path."""
    if not results_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _row_from_dict(payload: dict[str, Any], *, default_adapter_name: str) -> BatchItemResult:
    """Normalize one dictionary row into canonical result dataclass."""
    adapter_name = str(payload.get("adapter_name", default_adapter_name)).strip() or default_adapter_name or "unknown"
    return BatchItemResult(
        adapter_name=adapter_name,
        sample_id=str(payload.get("sample_id", "")),
        task_prompt=str(payload.get("task_prompt", "")),
        reference_answer=str(payload.get("reference_answer", "")),
        reference_answer_raw=_optional_str(payload.get("reference_answer_raw")),
        predicted_answer=_optional_str(payload.get("predicted_answer")),
        is_correct=bool(payload.get("is_correct", False)),
        session_id=_optional_str(payload.get("session_id")),
        session_status=str(payload.get("session_status", "failed")),
        final_summary=_optional_str(payload.get("final_summary")),
        error=_optional_str(payload.get("error")),
    )


def _openreward_row_from_dict(payload: dict[str, Any]) -> OpenRewardBatchItemResult:
    """Normalize one dictionary row into OpenReward result dataclass."""
    return OpenRewardBatchItemResult(
        environment=str(payload.get("environment", "")),
        split=str(payload.get("split", "")),
        variant=_optional_str(payload.get("variant")),
        task_key=str(payload.get("task_key", "")),
        task_index=_optional_int(payload.get("task_index")),
        reward_total=float(payload.get("reward_total", 0.0) or 0.0),
        finished=bool(payload.get("finished", False)),
        tool_calls=int(payload.get("tool_calls", 0) or 0),
        turns=int(payload.get("turns", 0) or 0),
        session_status=str(payload.get("session_status", "failed")),
        error=_optional_str(payload.get("error")),
    )


def _default_orchestrator_factory(
    *,
    project_dir: Path,
    app_dir: Path,
    model_override: str | None,
    timer_enabled: bool,
    provider_profile: str | None,
) -> OrchestratorFactory:
    """Build default orchestrator factory used by production batch runs."""

    def create() -> RuntimeOrchestrator:
        orchestrator = RuntimeOrchestrator(
            project_dir=project_dir,
            app_dir=app_dir,
            model_override=model_override,
            timer_enabled=timer_enabled,
        )
        profile = str(provider_profile or "").strip()
        if profile:
            orchestrator.set_provider_profile(profile)
        return orchestrator

    return create


def _failed_item_result(
    *,
    adapter_name: str,
    sample_id: str,
    task_prompt: str,
    reference_answer: str,
    reference_answer_raw: str | None,
    error: str,
) -> BatchItemResult:
    """Build standardized failed item row payload."""
    return BatchItemResult(
        adapter_name=adapter_name,
        sample_id=sample_id,
        task_prompt=task_prompt,
        reference_answer=reference_answer,
        reference_answer_raw=reference_answer_raw,
        predicted_answer=None,
        is_correct=False,
        session_id=None,
        session_status="failed",
        final_summary=None,
        error=error,
    )


def _build_summary(*, rows: list[BatchItemResult], results_path: Path, summary_path: Path) -> BatchSummary:
    """Build aggregate metrics from per-sample output rows."""
    total = len(rows)
    validated = sum(1 for row in rows if row.predicted_answer is not None)
    correct = sum(1 for row in rows if row.is_correct)
    failed_sessions = sum(1 for row in rows if row.session_status == "failed")
    accuracy = float(correct / validated) if validated else 0.0
    return BatchSummary(
        total=total,
        validated=validated,
        correct=correct,
        accuracy=accuracy,
        failed_sessions=failed_sessions,
        output_paths={
            "results_jsonl": str(results_path),
            "summary_json": str(summary_path),
        },
    )


def _build_openreward_summary(
    *,
    rows: list[OpenRewardBatchItemResult],
    results_path: Path,
    summary_path: Path,
) -> OpenRewardBatchSummary:
    """Build OpenReward aggregate metrics from per-task rows."""
    total = len(rows)
    completed = sum(1 for row in rows if row.session_status == "completed")
    finished = sum(1 for row in rows if row.finished)
    failed = sum(1 for row in rows if row.session_status == "failed")
    total_reward = float(sum(row.reward_total for row in rows))
    avg_reward = float(total_reward / total) if total else 0.0
    return OpenRewardBatchSummary(
        total=total,
        completed=completed,
        finished=finished,
        failed=failed,
        total_reward=total_reward,
        avg_reward=avg_reward,
        output_paths={
            "openreward_results_jsonl": str(results_path),
            "openreward_summary_json": str(summary_path),
        },
    )


def _resolve_batch_id(batch_id: str | None) -> str:
    """Resolve batch id from user input or random generator."""
    raw = str(batch_id or "").strip()
    if not raw:
        return f"batch-{uuid.uuid4().hex[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", raw):
        raise ValueError("batch_id may only contain letters, numbers, '.', '_' and '-'")
    return raw


def _optional_str(value: Any) -> str | None:
    """Normalize optional string-like values."""
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    """Normalize optional integer-like values."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_mixed_dataset(dataset: str) -> bool:
    """Return whether dataset selector is mixed mode."""
    return str(dataset).strip().lower() == _MIXED_DATASET_NAME


def _is_openreward_dataset(dataset: str) -> bool:
    """Return whether dataset selector is OpenReward mode."""
    return str(dataset).strip().lower() == _OPENREWARD_DATASET_NAME


def _default_adapter_name(dataset: str) -> str:
    """Resolve default adapter name used when reading legacy rows without adapter_name."""
    if _is_mixed_dataset(dataset):
        return "unknown"
    return str(dataset).strip().lower()


def _row_key(adapter_name: str, sample_id: str) -> str:
    """Build stable key for resume de-duplication."""
    return f"{adapter_name}::{sample_id}"


def _ensure_unique_routed_sample_keys(routed_samples: list[RoutedSample]) -> None:
    """Guard against duplicate routed sample keys."""
    seen: set[str] = set()
    for item in routed_samples:
        key = item.sample_key
        if key in seen:
            raise ValueError(f"duplicate routed sample key found in input: {key}")
        seen.add(key)


def _ensure_unique_openreward_task_keys(task_refs: list[OpenRewardTaskRef]) -> None:
    """Guard against duplicate OpenReward task keys."""
    seen: set[str] = set()
    for item in task_refs:
        if item.task_key in seen:
            raise ValueError(f"duplicate OpenReward task key found in selection: {item.task_key}")
        seen.add(item.task_key)


def _require_input_path(config: BatchRunConfig) -> Path:
    """Require input path for non-OpenReward datasets."""
    input_path = config.input_path
    if input_path is None:
        raise ValueError("input_path is required for non-openreward datasets")
    return Path(input_path)


def _validate_openreward_config(config: BatchRunConfig) -> None:
    """Validate OpenReward-specific config constraints."""
    environment = str(config.environment or "").strip()
    if not environment:
        raise ValueError("--environment is required when --dataset openreward")
    if config.task_index is not None and (config.start is not None or config.stop is not None):
        raise ValueError("--task-index cannot be used with --start/--stop")
    if int(config.max_steps) <= 0:
        raise ValueError("--max-steps must be a positive integer")
    if config.task_specs and (config.task_index is not None or config.start is not None or config.stop is not None):
        raise ValueError("--task-spec cannot be used with --task-index/--start/--stop")


def _resolve_provider_profile(
    *,
    runtime_config: OPMTrainConfig,
    provider_profile: str | None,
) -> tuple[str, ProviderProfileConfig]:
    """Resolve provider profile override into profile name + config payload."""
    selected_name = str(provider_profile or runtime_config.provider.profile).strip().lower()
    profiles: dict[str, ProviderProfileConfig] = {
        "openrouter": runtime_config.provider.openrouter,
        "tinker": runtime_config.provider.tinker,
        "custom": runtime_config.provider.custom,
    }
    if selected_name not in profiles:
        selected_name = str(runtime_config.provider.profile).strip().lower()
    profile = profiles.get(selected_name, runtime_config.provider.openrouter)
    return selected_name, profile


def _build_openreward_llm_client(*, profile: ProviderProfileConfig) -> OpenAICompatibleClient:
    """Build OpenAI-compatible client for OpenReward rollout model loop."""
    if not str(profile.base_url or "").strip():
        raise RuntimeError("Provider base_url is required")
    if not profile.api_key:
        raise RuntimeError(
            f"Missing API key in env '{profile.api_key_env}'. "
            "Set the environment variable before running opm-train."
        )
    return OpenAICompatibleClient(
        base_url=profile.base_url,
        api_key=profile.api_key,
        timeout_seconds=profile.timeout_seconds,
        max_retries=profile.max_retries,
        retry_backoff_seconds=profile.retry_backoff_seconds,
        headers=profile.headers,
    )


def _resolve_openreward_model(*, config: BatchRunConfig, profile: ProviderProfileConfig) -> str:
    """Resolve model id for OpenReward rollout (override > provider profile)."""
    resolved = str(config.model or "").strip() or str(profile.model or "").strip()
    if not resolved:
        raise ValueError("A non-empty model is required for --dataset openreward")
    return resolved


def _resolve_openreward_tool_format(*, profile_name: str, override: str | None) -> str:
    """Resolve OpenReward tool schema format for current provider profile."""
    override_value = str(override or "").strip().lower()
    if override_value:
        return override_value
    return "openrouter" if profile_name == "openrouter" else "openai"


def _load_async_openreward_client_cls() -> Any:
    """Load AsyncOpenReward client class lazily so dependency remains optional."""
    try:
        from openreward import AsyncOpenReward  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
        raise RuntimeError(
            "OpenReward SDK is not installed. Install optional dependency with: "
            "pip install -e '.[openreward]'"
        ) from exc
    return AsyncOpenReward


def _instantiate_openreward_client(*, async_openreward_cls: Any, config: BatchRunConfig) -> Any:
    """Instantiate AsyncOpenReward client with conservative kwargs fallback."""
    api_key = str(os.environ.get("OPENREWARD_API_KEY", "")).strip()
    base_url = _optional_str(config.base_url)
    candidates: list[dict[str, Any]] = []
    if api_key and base_url:
        candidates.append({"api_key": api_key, "base_url": base_url})
        candidates.append({"api_key": api_key})
        candidates.append({"base_url": base_url})
    elif api_key:
        candidates.append({"api_key": api_key})
    elif base_url:
        candidates.append({"base_url": base_url})
    candidates.append({})

    last_shape_error: TypeError | None = None
    for kwargs in candidates:
        try:
            return async_openreward_cls(**kwargs)
        except TypeError as exc:
            if _is_argument_shape_error(exc):
                last_shape_error = exc
                continue
            raise
    if last_shape_error is not None:
        raise last_shape_error
    raise RuntimeError("Unable to instantiate AsyncOpenReward client")


async def _resolve_openreward_environment(*, or_client: Any, config: BatchRunConfig) -> Any:
    """Resolve OpenReward environment handle with signature-compatible fallbacks."""
    environments = _object_get(or_client, "environments")
    get_environment = _object_get(environments, "get")
    if not callable(get_environment):
        raise TypeError("OpenReward client must expose environments.get(...)")

    name = str(config.environment or "").strip()
    variant = _optional_str(config.variant)
    base_url = _optional_str(config.base_url)
    candidate_calls = _openreward_environment_candidates(name=name, variant=variant, base_url=base_url)

    last_shape_error: TypeError | None = None
    for args, kwargs in candidate_calls:
        try:
            environment = get_environment(*args, **kwargs)
        except TypeError as exc:
            if _is_argument_shape_error(exc):
                last_shape_error = exc
                continue
            raise
        resolved = await _maybe_await(environment)
        if resolved is not None:
            return resolved

    if last_shape_error is not None:
        raise last_shape_error
    raise RuntimeError("Unable to resolve OpenReward environment")


def _openreward_environment_candidates(
    *,
    name: str,
    variant: str | None,
    base_url: str | None,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Build environment.get(...) candidate calls, prioritizing explicit selectors."""
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    seen: set[tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]]] = set()

    def add(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        normalized = (args, tuple(sorted(kwargs.items())))
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append((args, kwargs))

    if variant and base_url:
        add((name,), {"variant": variant, "base_url": base_url})
        add((), {"name": name, "variant": variant, "base_url": base_url})
        add((), {"environment": name, "variant": variant, "base_url": base_url})
    if variant:
        add((name,), {"variant": variant})
        add((), {"name": name, "variant": variant})
        add((), {"environment": name, "variant": variant})
        add((), {"id": name, "variant": variant})
    if base_url:
        add((name,), {"base_url": base_url})
        add((), {"name": name, "base_url": base_url})
        add((), {"environment": name, "base_url": base_url})
        add((), {"id": name, "base_url": base_url})

    add((name,), {})
    add((), {"name": name})
    add((), {"environment": name})
    add((), {"id": name})
    return candidates


async def _list_openreward_tools(*, environment: Any, tool_format: str) -> list[dict[str, Any]]:
    """Load OpenReward tool schema with argument-name fallbacks."""
    list_tools = _object_get(environment, "list_tools")
    if not callable(list_tools):
        raise TypeError("OpenReward environment must expose list_tools(...)")
    call_kwargs: list[dict[str, Any]] = [
        {"format": tool_format},
        {"tool_format": tool_format},
        {},
    ]
    last_shape_error: TypeError | None = None
    for kwargs in call_kwargs:
        try:
            payload = list_tools(**kwargs)
        except TypeError as exc:
            if _is_argument_shape_error(exc):
                last_shape_error = exc
                continue
            raise
        tools = await _maybe_await(payload)
        if not isinstance(tools, list):
            raise ValueError("OpenReward environment.list_tools(...) must return a list")
        if not all(isinstance(item, dict) for item in tools):
            raise ValueError("OpenReward environment.list_tools(...) items must be dict payloads")
        return tools

    if last_shape_error is not None:
        raise last_shape_error
    raise RuntimeError("Unable to list OpenReward tools")


def _build_openreward_task_key(
    *,
    task: Any,
    split: str,
    task_index: int | None,
    order: int,
    include_split_in_key: bool,
) -> str:
    """Build stable task key: prefer task_id/id, fallback to task index/order."""
    prefix = f"{split}::" if include_split_in_key else ""
    task_id = _extract_task_id(task)
    if task_id:
        return f"{prefix}{task_id}"
    if task_index is not None:
        return f"{prefix}task-index-{task_index}"
    return f"{prefix}task-order-{order}"


def _extract_task_id(task: Any) -> str | None:
    """Extract stable task identifier from dict/object payload."""
    for key in ("task_id", "id"):
        value = _object_get(task, key)
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return None


def _normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize tool calls into OpenAI-compatible message payloads."""
    normalized: list[dict[str, Any]] = []
    for index, raw_call in enumerate(tool_calls):
        if not isinstance(raw_call, dict):
            continue
        function_payload = raw_call.get("function")
        if not isinstance(function_payload, dict):
            continue
        function_name = str(function_payload.get("name", "")).strip()
        if not function_name:
            continue
        arguments = function_payload.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        normalized.append(
            {
                "id": str(raw_call.get("id", f"tool-call-{index + 1}")),
                "type": str(raw_call.get("type", "function") or "function"),
                "function": {
                    "name": function_name,
                    "arguments": arguments,
                },
            }
        )
    return normalized


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Parse model-emitted function arguments into dictionary payload."""
    if isinstance(arguments, dict):
        return dict(arguments)
    text = str(arguments or "").strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to a JSON object")
    return parsed


def _render_prompt_blocks(prompt_blocks: Any) -> str:
    """Render OpenReward prompt blocks into one text payload."""
    if prompt_blocks is None:
        return ""
    if isinstance(prompt_blocks, str):
        return prompt_blocks
    blocks = prompt_blocks if isinstance(prompt_blocks, list) else [prompt_blocks]
    parts: list[str] = []
    for block in blocks:
        text_value = _object_get(block, "text")
        if text_value is not None:
            text = str(text_value)
            if text:
                parts.append(text)
                continue
        if isinstance(block, str):
            if block:
                parts.append(block)
            continue
        block_str = str(block)
        if block_str:
            parts.append(block_str)
    return "\n".join(parts)


def _render_tool_output(tool_output: Any) -> str:
    """Render one OpenReward tool output object into text for model feedback."""
    blocks = _object_get(tool_output, "blocks")
    if isinstance(blocks, list):
        text_parts: list[str] = []
        for block in blocks:
            text = _object_get(block, "text")
            if text is not None:
                text_parts.append(str(text))
            else:
                text_parts.append(str(block))
        joined = "".join(text_parts).strip()
        if joined:
            return joined

    data = _object_get(tool_output, "data")
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False)
    if data is not None:
        return str(data)
    return str(tool_output)


def _coerce_reward(value: Any) -> float:
    """Normalize reward scalar into float value."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _object_get(value: Any, key: str, default: Any = None) -> Any:
    """Get one field from dict/object payloads with default fallback."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _is_argument_shape_error(exc: TypeError) -> bool:
    """Detect TypeError patterns caused by argument-name/signature mismatch."""
    message = str(exc)
    patterns = (
        "unexpected keyword argument",
        "missing 1 required positional argument",
        "required positional argument",
        "positional arguments but",
        "takes",
    )
    return any(pattern in message for pattern in patterns)


class _SyncContextAdapter:
    """Wrap synchronous context manager into async-compatible adapter."""

    def __init__(self, context_manager: Any) -> None:
        self._context_manager = context_manager

    async def __aenter__(self) -> Any:
        return self._context_manager.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._context_manager.__exit__(exc_type, exc, tb)


def _as_async_context_manager(value: Any) -> Any:
    """Return async context manager for session payloads."""
    if hasattr(value, "__aenter__") and hasattr(value, "__aexit__"):
        return value
    if hasattr(value, "__enter__") and hasattr(value, "__exit__"):
        return _SyncContextAdapter(value)
    raise TypeError("environment.session(...) must return a context manager")


async def _maybe_await(value: Any) -> Any:
    """Await value only when it is awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value

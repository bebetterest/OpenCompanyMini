"""Dataset batch execution runner."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from opm_train.config import OPMTrainConfig
from opm_train.data import BatchItemResult, BatchSummary, DatasetAdapter, DatasetSample, get_dataset_adapter
from opm_train.data.jsonl import iter_json_objects
from opm_train.models import RunSession, SessionStatus
from opm_train.orchestrator import RuntimeOrchestrator
from opm_train.storage import SessionStorage
from opm_train.utils import ensure_directory

_MIXED_DATASET_NAME = "mixed"


@dataclass(slots=True, frozen=True)
class BatchRunConfig:
    """Input configuration for one dataset-driven batch run."""

    dataset: str
    input_path: Path
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


@dataclass(slots=True, frozen=True)
class BatchRunOutput:
    """Batch run artifacts and aggregated metrics."""

    batch_id: str
    batch_dir: Path
    results_path: Path
    summary_path: Path
    summary: BatchSummary


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
    tasks = [
        asyncio.create_task(
            _run_one_routed_sample(
                routed_sample=item,
                semaphore=semaphore,
                orchestrator_factory=orchestrator_factory,
            )
        )
        for item in routed_samples
    ]

    rows: list[BatchItemResult] = []
    try:
        for completed in asyncio.as_completed(tasks):
            row = await completed
            rows.append(row)
            await _append_result_row(results_path=results_path, row=row, append_lock=append_lock)
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


def _load_routed_samples(config: BatchRunConfig) -> list[RoutedSample]:
    """Load routed samples for single-adapter or mixed-adapter modes."""
    if _is_mixed_dataset(config.dataset):
        routed = _load_mixed_routed_samples(
            input_path=config.input_path,
            adapter_key=config.adapter_key,
            limit=config.limit,
        )
    else:
        adapter = get_dataset_adapter(config.dataset)
        routed = [
            RoutedSample(adapter_name=adapter.name, adapter=adapter, sample=sample)
            for sample in adapter.load_samples(input_path=config.input_path, limit=config.limit)
        ]
    _ensure_unique_routed_sample_keys(routed)
    return routed


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
    if resume:
        if not batch_dir.is_dir():
            raise ValueError(f"resume requested but batch directory does not exist: {batch_dir}")
        return _load_result_rows(results_path, default_adapter_name=default_adapter_name)

    if batch_dir.exists():
        raise ValueError(
            f"batch directory already exists: {batch_dir}. Use --resume to continue this batch."
        )
    ensure_directory(batch_dir)
    results_path.write_text("", encoding="utf-8")
    return []


async def _append_result_row(*, results_path: Path, row: BatchItemResult, append_lock: asyncio.Lock) -> None:
    """Append one row to JSONL with coroutine-safe file writes."""
    async with append_lock:
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False))
            handle.write("\n")


def _load_result_rows(results_path: Path, *, default_adapter_name: str) -> list[BatchItemResult]:
    """Load existing rows from JSONL for resume."""
    if not results_path.exists():
        return []
    rows: list[BatchItemResult] = []
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(_row_from_dict(payload, default_adapter_name=default_adapter_name))
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


def _is_mixed_dataset(dataset: str) -> bool:
    """Return whether dataset selector is mixed mode."""
    return str(dataset).strip().lower() == _MIXED_DATASET_NAME


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

"""SFT run orchestration and artifact persistence."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from opm_train.config import OPMTrainConfig
from opm_train.sft.contracts import SFTBackendConfig, SFTBackendResult
from opm_train.sft.jsonl import load_sft_examples
from opm_train.sft.registry import get_sft_backend
from opm_train.storage import SessionStorage
from opm_train.utils import ensure_directory, utc_now


@dataclass(slots=True, frozen=True)
class SFTRunConfig:
    """Input configuration for one supervised fine-tuning run."""

    backend: str
    input_path: Path
    project_dir: Path
    app_dir: Path
    base_model: str
    output_model: str | None = None
    steps: int = 6
    batch_size: int = 8
    learning_rate: float = 1e-4
    rank: int = 32
    limit: int | None = None
    prompt_key: str | None = None
    completion_key: str | None = None
    sample_prompt: str | None = None
    sample_max_tokens: int = 64
    sample_temperature: float = 0.0
    run_id: str | None = None


@dataclass(slots=True, frozen=True)
class SFTRunOutput:
    """SFT run output metadata and produced artifacts."""

    run_id: str
    artifact_dir: Path
    result_path: Path
    metrics_path: Path
    total_examples: int
    result: SFTBackendResult


def run_sft(config: SFTRunConfig) -> SFTRunOutput:
    """Run one SFT job through selected backend and persist artifacts."""
    app_dir = config.app_dir.resolve()
    runtime_config = OPMTrainConfig.load(app_dir)
    storage = SessionStorage(app_dir=app_dir, data_dir_name=runtime_config.project.data_dir)

    run_id = _resolve_run_id(config.run_id)
    artifact_dir = ensure_directory(storage.data_root / "sft_runs" / run_id)
    result_path = artifact_dir / "result.json"
    metrics_path = artifact_dir / "metrics.jsonl"

    examples = load_sft_examples(
        input_path=config.input_path,
        limit=config.limit,
        prompt_key=config.prompt_key,
        completion_key=config.completion_key,
    )

    backend = get_sft_backend(config.backend)
    output_model = str(config.output_model or f"{run_id}-model").strip() or f"{run_id}-model"
    backend_config = _build_backend_config(config=config, output_model=output_model)

    config_payload = {
        "timestamp": utc_now(),
        "run_id": run_id,
        "project_dir": str(config.project_dir.resolve()),
        "input_path": str(config.input_path.resolve()),
        "backend": backend.name,
        "settings": asdict(backend_config),
        "dataset": {
            "total_examples": len(examples),
            "prompt_key": config.prompt_key,
            "completion_key": config.completion_key,
        },
    }
    (artifact_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_metric(payload: dict[str, object]) -> None:
        row = {"timestamp": utc_now(), **payload}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = backend.train(config=backend_config, examples=examples, on_metric=on_metric)

    result_payload = {
        "timestamp": utc_now(),
        "run_id": run_id,
        "backend": result.backend,
        "base_model": result.base_model,
        "output_model": result.output_model,
        "total_examples": len(examples),
        "losses": result.losses,
        "checkpoint_path": result.checkpoint_path,
        "sample_output": result.sample_output,
        "artifact_dir": str(artifact_dir),
        "metrics_path": str(metrics_path),
    }
    result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return SFTRunOutput(
        run_id=run_id,
        artifact_dir=artifact_dir,
        result_path=result_path,
        metrics_path=metrics_path,
        total_examples=len(examples),
        result=result,
    )


def _build_backend_config(*, config: SFTRunConfig, output_model: str) -> SFTBackendConfig:
    """Build normalized backend config with strict numeric bounds."""
    return SFTBackendConfig(
        base_model=str(config.base_model).strip(),
        output_model=output_model,
        steps=max(1, int(config.steps)),
        batch_size=max(1, int(config.batch_size)),
        learning_rate=max(0.0, float(config.learning_rate)),
        rank=max(1, int(config.rank)),
        sample_prompt=str(config.sample_prompt).strip() if config.sample_prompt else None,
        sample_max_tokens=max(1, int(config.sample_max_tokens)),
        sample_temperature=float(config.sample_temperature),
    )


def _resolve_run_id(candidate: str | None) -> str:
    """Resolve stable run id and enforce safe file-name shape."""
    value = str(candidate or "").strip()
    if not value:
        return f"sft-{uuid.uuid4().hex[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("run_id may only contain letters, digits, dot, underscore, or hyphen")
    return value

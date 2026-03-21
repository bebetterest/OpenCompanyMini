"""Dataset contracts for prompt construction and result validation."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from opm_train.models import RunSession


@dataclass(slots=True, frozen=True)
class DatasetSample:
    """One raw dataset sample normalized for runtime consumption."""

    sample_id: str
    question: str
    # Canonical extracted answer used for validation.
    answer: str
    # Optional original raw answer payload from the source dataset.
    answer_raw: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PreparedTask:
    """Prompt payload generated from one dataset sample."""

    sample_id: str
    task_prompt: str
    # Canonical extracted answer used for scoring.
    reference_answer: str
    # Optional original raw answer payload kept for debugging/auditing.
    reference_answer_raw: str | None = None


@dataclass(slots=True, frozen=True)
class ValidationResult:
    """Validation outcome derived from one completed runtime session."""

    predicted_answer: str | None
    is_correct: bool
    error: str | None = None


@dataclass(slots=True, frozen=True)
class BatchItemResult:
    """Canonical per-sample output row persisted into batch results JSONL."""

    adapter_name: str
    sample_id: str
    task_prompt: str
    reference_answer: str
    reference_answer_raw: str | None
    predicted_answer: str | None
    is_correct: bool
    session_id: str | None
    session_status: str
    final_summary: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result row into JSON-serializable dictionary."""
        return asdict(self)


@dataclass(slots=True, frozen=True)
class BatchSummary:
    """Aggregated batch-level metrics and output locations."""

    total: int
    validated: int
    correct: int
    accuracy: float
    failed_sessions: int
    output_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Convert summary into JSON-serializable dictionary."""
        return asdict(self)


@runtime_checkable
class DatasetAdapter(Protocol):
    """Extensible adapter protocol for new datasets."""

    name: str

    def sample_from_payload(self, payload: dict[str, Any], *, line_no: int) -> DatasetSample:
        """Parse one JSON object payload into one normalized dataset sample."""

    def load_samples(self, *, input_path: Path, limit: int | None = None) -> list[DatasetSample]:
        """Load and validate raw samples from one input source."""

    def build_task_prompt(self, sample: DatasetSample) -> PreparedTask:
        """Build one runtime task prompt from one normalized sample."""

    def validate_result(self, *, sample: DatasetSample, session: RunSession) -> ValidationResult:
        """Validate one completed runtime session against the sample reference."""

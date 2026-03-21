from __future__ import annotations

from pathlib import Path

from opm_train.data.contracts import DatasetSample, PreparedTask, ValidationResult
from opm_train.data.registry import get_dataset_adapter, list_dataset_adapters, register_dataset_adapter
from opm_train.models import RunSession, SessionStatus


class DummyDatasetAdapter:
    name = "dummy"

    def sample_from_payload(self, payload: dict[str, object], *, line_no: int) -> DatasetSample:
        _ = (payload, line_no)
        return DatasetSample(sample_id="d-1", question="q", answer="a")

    def load_samples(self, *, input_path: Path, limit: int | None = None) -> list[DatasetSample]:
        _ = (input_path, limit)
        return [DatasetSample(sample_id="d-1", question="q", answer="a")]

    def build_task_prompt(self, sample: DatasetSample) -> PreparedTask:
        return PreparedTask(sample_id=sample.sample_id, task_prompt="task", reference_answer=sample.answer)

    def validate_result(self, *, sample: DatasetSample, session: RunSession) -> ValidationResult:
        _ = sample
        return ValidationResult(predicted_answer=str(session.final_summary or ""), is_correct=True, error=None)


def test_dataset_registry_supports_custom_adapter_registration() -> None:
    adapter = DummyDatasetAdapter()
    register_dataset_adapter(adapter, replace=True)

    resolved = get_dataset_adapter("dummy")
    assert resolved is adapter
    assert "dummy" in list_dataset_adapters()

    session = RunSession(
        id="s-1",
        task="task",
        project_dir=Path("."),
        root_agent_id="root",
        status=SessionStatus.COMPLETED,
        final_summary="ok",
    )
    validation = resolved.validate_result(sample=DatasetSample(sample_id="d", question="q", answer="a"), session=session)
    assert validation.is_correct is True


def test_dataset_registry_contains_builtin_math_adapters() -> None:
    names = set(list_dataset_adapters())
    assert "gsm8k" in names
    assert "simple_math" in names

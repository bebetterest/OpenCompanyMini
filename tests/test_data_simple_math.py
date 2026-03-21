from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from opm_train.data import DatasetSample
from opm_train.data.simple_math import SimpleMathDatasetAdapter
from opm_train.models import RunSession, SessionStatus


def _session_with_summary(summary: str) -> RunSession:
    return RunSession(
        id="session-1",
        task="task",
        project_dir=Path("."),
        root_agent_id="agent-root",
        status=SessionStatus.COMPLETED,
        final_summary=summary,
    )


def test_simple_math_load_samples_parses_numeric_answers() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "simple_math.jsonl"
        path.write_text(
            '\n'.join(
                [
                    '{"id":"m1","question":"1+2=?","answer":"3"}',
                    '{"id":"m2","question":"2+4=?","answer":"6.0"}',
                ]
            ),
            encoding="utf-8",
        )
        adapter = SimpleMathDatasetAdapter()
        samples = adapter.load_samples(input_path=path)
        assert [item.sample_id for item in samples] == ["m1", "m2"]
        assert [item.answer for item in samples] == ["3", "6.0"]
        assert [item.answer_raw for item in samples] == ["3", "6.0"]


def test_simple_math_load_samples_rejects_non_numeric_answer() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "invalid.jsonl"
        path.write_text('{"id":"m1","question":"1+2=?","answer":"unknown"}\n', encoding="utf-8")
        adapter = SimpleMathDatasetAdapter()
        with pytest.raises(ValueError, match="answer could not be parsed"):
            adapter.load_samples(input_path=path)


def test_simple_math_validate_result_numeric_equivalence() -> None:
    adapter = SimpleMathDatasetAdapter()
    sample = DatasetSample(sample_id="m1", question="1+2=?", answer="3", answer_raw="3")
    result = adapter.validate_result(sample=sample, session=_session_with_summary("FINAL_ANSWER: 3.0"))
    assert result.is_correct is True
    assert result.predicted_answer == "3.0"

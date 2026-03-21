from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from opm_train.data import DatasetSample
from opm_train.data.gsm8k import (
    GSM8KDatasetAdapter,
    parse_predicted_answer,
    parse_reference_answer,
    render_predicted_answer,
)
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


def test_gsm8k_load_samples_reads_local_jsonl_and_limit() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "gsm8k.jsonl"
        path.write_text(
            '\n'.join(
                [
                    '{"id":"q1","question":"If A has 2 apples and gets 3 more, how many?","answer":"5 #### 5"}',
                    '{"question":"Tom has 7 pens and loses 2. How many left?","answer":"Work... #### 5"}',
                ]
            ),
            encoding="utf-8",
        )
        adapter = GSM8KDatasetAdapter()
        samples = adapter.load_samples(input_path=path, limit=1)
        assert len(samples) == 1
        assert samples[0].sample_id == "q1"
        assert samples[0].question.startswith("If A has 2 apples")
        assert samples[0].answer == "5"
        assert samples[0].answer_raw == "5 #### 5"


def test_gsm8k_load_samples_requires_question_and_answer() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "broken.jsonl"
        path.write_text('{"question":"x"}\n', encoding="utf-8")
        adapter = GSM8KDatasetAdapter()
        with pytest.raises(ValueError, match="missing non-empty 'answer'"):
            adapter.load_samples(input_path=path)


def test_gsm8k_load_samples_requires_parseable_numeric_answer() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "broken_answer.jsonl"
        path.write_text('{"question":"x","answer":"N/A"}\n', encoding="utf-8")
        adapter = GSM8KDatasetAdapter()
        with pytest.raises(ValueError, match="answer could not be parsed"):
            adapter.load_samples(input_path=path)


def test_gsm8k_sample_from_payload_parses_id_question_answer() -> None:
    adapter = GSM8KDatasetAdapter()
    sample = adapter.sample_from_payload(
        {
            "id": "gsm-custom-1",
            "question": "What is 1 + 2?",
            "answer": "#### 3",
        },
        line_no=7,
    )
    assert sample.sample_id == "gsm-custom-1"
    assert sample.question == "What is 1 + 2?"
    assert sample.answer == "3"
    assert sample.answer_raw == "#### 3"


def test_gsm8k_load_samples_accepts_unicode_line_separator_in_field() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "unicode.jsonl"
        payload = {
            "id": "u1",
            "question": "Line A\u2028Line B",
            "answer": "#### 5",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

        adapter = GSM8KDatasetAdapter()
        samples = adapter.load_samples(input_path=path)
        assert len(samples) == 1
        assert samples[0].sample_id == "u1"
        assert samples[0].question == "Line A\u2028Line B"
        assert samples[0].answer == "5"


def test_parse_reference_answer_with_math_verify_extracts_expression() -> None:
    parsed = parse_reference_answer("Some steps here. #### 1,234.50")
    assert parsed
    assert render_predicted_answer(parsed) == "1234.50"


def test_parse_predicted_answer_with_math_verify_extracts_expression() -> None:
    parsed = parse_predicted_answer("analysis text\nFINAL_ANSWER: 42\nmore text")
    assert parsed
    assert render_predicted_answer(parsed) == "42"


@pytest.mark.parametrize(
    ("reference", "summary", "expected"),
    [
        ("1234", "Done.\nFINAL_ANSWER: 1234", True),
        ("3.50", "Done.\nFINAL_ANSWER: 3.5", True),
        ("20", "Done.\nFINAL_ANSWER: 19", False),
    ],
)
def test_gsm8k_validate_result_numeric_equivalence(reference: str, summary: str, expected: bool) -> None:
    adapter = GSM8KDatasetAdapter()
    sample = DatasetSample(sample_id="s1", question="q", answer=reference, answer_raw=f"steps #### {reference}")
    result = adapter.validate_result(sample=sample, session=_session_with_summary(summary))
    assert result.is_correct is expected


def test_gsm8k_validate_result_marks_missing_prediction_as_error() -> None:
    adapter = GSM8KDatasetAdapter()
    sample = DatasetSample(sample_id="s1", question="q", answer="12", answer_raw="x #### 12")
    result = adapter.validate_result(sample=sample, session=_session_with_summary("No number present"))
    assert result.is_correct is False
    assert result.predicted_answer is None
    assert result.error == "predicted_answer_not_parsed"

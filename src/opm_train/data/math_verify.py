"""Shared Math-Verify parsing/validation helpers and reusable math dataset adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from math_verify import ExprExtractionConfig, parse, verify  # type: ignore[import-untyped]
from opm_train.data.contracts import DatasetAdapter, DatasetSample, PreparedTask, ValidationResult
from opm_train.data.jsonl import iter_json_objects
from opm_train.models import RunSession

_FINAL_ANSWER_PATTERN = re.compile(r"FINAL_ANSWER\s*:\s*(.+)", flags=re.IGNORECASE)
_REFERENCE_ANCHOR = "####"


def parse_math_text(text: str) -> list[Any]:
    """Parse one text payload into Math-Verify candidates."""
    value = str(text or "")
    if not value.strip():
        return []
    try:
        return list(
            parse(
                value,
                extraction_config=[ExprExtractionConfig(try_extract_without_anchor=True)],
                fallback_mode="first_match",
                extraction_mode="any_match",
            )
        )
    except Exception:
        return []


def render_math_answer(parsed: list[Any]) -> str | None:
    """Render one stable display answer from parsed Math-Verify candidates."""
    for item in parsed:
        if isinstance(item, str):
            text = item.strip()
            if text:
                return text
    if not parsed:
        return None
    text = str(parsed[0]).strip()
    return text or None


def parse_reference_answer(raw_answer: str) -> list[Any]:
    """Parse reference answer text, preferring GSM-style `####` suffix when present."""
    text = str(raw_answer or "").strip()
    if not text:
        return []

    candidates: list[str] = []
    if _REFERENCE_ANCHOR in text:
        anchored = text.rsplit(_REFERENCE_ANCHOR, maxsplit=1)[1].strip()
        if anchored:
            candidates.append(anchored)
    candidates.append(text)

    for candidate in candidates:
        parsed = parse_math_text(candidate)
        if parsed:
            return parsed
    return []


def extract_reference_answer(raw_answer: str) -> str | None:
    """Extract canonical reference answer text from raw dataset answer payload."""
    return render_math_answer(parse_reference_answer(raw_answer))


def parse_predicted_answer(summary: str) -> list[Any]:
    """Parse predicted answer text, preferring explicit `FINAL_ANSWER:` lines."""
    text = str(summary or "")
    for match in reversed(list(_FINAL_ANSWER_PATTERN.finditer(text))):
        parsed = parse_math_text(match.group(1).strip())
        if parsed:
            return parsed
    return parse_math_text(text)


class MathVerifyDatasetAdapter(DatasetAdapter):
    """Reusable dataset adapter for numeric-answer math tasks validated by Math-Verify."""

    name = "math_verify"
    prompt_template = """Solve the following math problem.
Show your reasoning, then finish with exactly one line:
FINAL_ANSWER: <number>

Requirements:
- Keep FINAL_ANSWER as a plain numeric value only.
- Do not include units in FINAL_ANSWER.

Question:
{question}
"""

    def sample_from_payload(self, payload: dict[str, Any], *, line_no: int) -> DatasetSample:
        """Parse one math dataset row into normalized sample payload."""
        question = str(payload.get("question", "")).strip()
        raw_answer = str(payload.get("answer", "")).strip()
        if not question:
            raise ValueError(f"line {line_no} missing non-empty 'question'")
        if not raw_answer:
            raise ValueError(f"line {line_no} missing non-empty 'answer'")

        canonical_answer = extract_reference_answer(raw_answer)
        if canonical_answer is None:
            raise ValueError(f"line {line_no} answer could not be parsed into numeric reference")

        raw_id = str(payload.get("id", "")).strip()
        sample_id = raw_id or f"{self.name}-{line_no:06d}"
        return DatasetSample(
            sample_id=sample_id,
            question=question,
            answer=canonical_answer,
            answer_raw=raw_answer,
            metadata={"line_no": line_no},
        )

    def load_samples(self, *, input_path: Path, limit: int | None = None) -> list[DatasetSample]:
        """Load local JSONL rows for one math dataset."""
        samples: list[DatasetSample] = []
        for line_no, payload in iter_json_objects(input_path):
            samples.append(self.sample_from_payload(payload, line_no=line_no))
            if limit is not None and len(samples) >= max(0, int(limit)):
                break
        return samples

    def build_task_prompt(self, sample: DatasetSample) -> PreparedTask:
        """Build one runtime prompt + canonical reference answer."""
        return PreparedTask(
            sample_id=sample.sample_id,
            task_prompt=self.prompt_template.format(question=sample.question),
            reference_answer=sample.answer,
            reference_answer_raw=sample.answer_raw,
        )

    def validate_result(self, *, sample: DatasetSample, session: RunSession) -> ValidationResult:
        """Validate session summary against canonical reference answer via Math-Verify."""
        reference = parse_math_text(sample.answer)
        if not reference:
            return ValidationResult(
                predicted_answer=None,
                is_correct=False,
                error="reference_answer_not_parsed",
            )

        predicted = parse_predicted_answer(str(session.final_summary or ""))
        if not predicted:
            return ValidationResult(
                predicted_answer=None,
                is_correct=False,
                error="predicted_answer_not_parsed",
            )

        predicted_text = render_math_answer(predicted)
        try:
            return ValidationResult(
                predicted_answer=predicted_text,
                is_correct=bool(verify(reference, predicted)),
                error=None,
            )
        except Exception as exc:
            return ValidationResult(
                predicted_answer=predicted_text,
                is_correct=False,
                error=f"math_verify_error:{type(exc).__name__}:{exc}",
            )

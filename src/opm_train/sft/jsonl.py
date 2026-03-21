"""JSONL parsing helpers for supervised fine-tuning datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opm_train.data.jsonl import iter_json_objects
from opm_train.sft.contracts import SFTExample

_DEFAULT_KEY_PAIRS: tuple[tuple[str, str], ...] = (
    ("prompt", "completion"),
    ("input", "output"),
    ("instruction", "output"),
    ("question", "answer"),
)


def load_sft_examples(
    *,
    input_path: Path,
    limit: int | None = None,
    prompt_key: str | None = None,
    completion_key: str | None = None,
) -> list[SFTExample]:
    """Load and normalize SFT examples from local JSONL."""
    if (prompt_key and not completion_key) or (completion_key and not prompt_key):
        raise ValueError("prompt_key and completion_key must be provided together")

    examples: list[SFTExample] = []
    normalized_limit = _normalized_limit(limit)

    for line_no, payload in iter_json_objects(input_path):
        resolved_prompt_key, resolved_completion_key = _resolve_prompt_completion_keys(
            payload=payload,
            prompt_key=prompt_key,
            completion_key=completion_key,
        )
        prompt = _as_required_text(payload.get(resolved_prompt_key), field=resolved_prompt_key, line_no=line_no)
        completion = _as_required_text(payload.get(resolved_completion_key), field=resolved_completion_key, line_no=line_no)
        example_id = _example_id(payload=payload, line_no=line_no)
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"id", resolved_prompt_key, resolved_completion_key}
        }
        examples.append(
            SFTExample(
                example_id=example_id,
                prompt=prompt,
                completion=completion,
                metadata=metadata,
            )
        )
        if normalized_limit is not None and len(examples) >= normalized_limit:
            break

    if not examples:
        raise ValueError(f"no SFT examples found in {input_path.resolve()}")
    return examples


def _normalized_limit(limit: int | None) -> int | None:
    """Normalize limit with validation and lower-bound clamp."""
    if limit is None:
        return None
    parsed = int(limit)
    if parsed <= 0:
        raise ValueError("limit must be greater than 0")
    return parsed


def _resolve_prompt_completion_keys(
    *,
    payload: dict[str, Any],
    prompt_key: str | None,
    completion_key: str | None,
) -> tuple[str, str]:
    """Resolve prompt/completion keys for one payload."""
    if prompt_key and completion_key:
        if prompt_key not in payload:
            raise ValueError(f"missing prompt key '{prompt_key}'")
        if completion_key not in payload:
            raise ValueError(f"missing completion key '{completion_key}'")
        return prompt_key, completion_key

    for candidate_prompt, candidate_completion in _DEFAULT_KEY_PAIRS:
        if candidate_prompt in payload and candidate_completion in payload:
            return candidate_prompt, candidate_completion

    expected = ", ".join(f"('{prompt}', '{completion}')" for prompt, completion in _DEFAULT_KEY_PAIRS)
    raise ValueError(
        "could not resolve prompt/completion keys from row payload; "
        f"expected one of {expected} or pass --prompt-key/--completion-key"
    )


def _as_required_text(value: Any, *, field: str, line_no: int) -> str:
    """Normalize one text field and enforce non-empty content."""
    text = str(value or "")
    if not text.strip():
        raise ValueError(f"line {line_no}: field '{field}' must be a non-empty string")
    return text


def _example_id(*, payload: dict[str, Any], line_no: int) -> str:
    """Resolve stable example id from payload or fallback line marker."""
    candidate = str(payload.get("id", "")).strip()
    if candidate:
        return candidate
    return f"line-{line_no}"

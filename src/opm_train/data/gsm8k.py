"""GSM8K dataset adapter."""

from __future__ import annotations

from typing import Any

from opm_train.data.math_verify import (
    MathVerifyDatasetAdapter,
    parse_predicted_answer,
    parse_reference_answer,
    render_math_answer,
)


_GSM8K_PROMPT_TEMPLATE = """Solve the following GSM8K math word problem.
Show your reasoning, then finish with exactly one line:
FINAL_ANSWER: <number>

Requirements:
- Keep FINAL_ANSWER as a plain numeric value only.
- Do not include units in FINAL_ANSWER.

Question:
{question}
"""


class GSM8KDatasetAdapter(MathVerifyDatasetAdapter):
    """GSM8K adapter using shared Math-Verify loading and validation flow."""

    name = "gsm8k"
    prompt_template = _GSM8K_PROMPT_TEMPLATE


def render_predicted_answer(parsed: list[Any]) -> str | None:
    """Backward-compatible alias used by existing tests."""
    return render_math_answer(parsed)

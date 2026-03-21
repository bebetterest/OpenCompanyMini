"""Simple math dataset adapter (local JSONL)."""

from __future__ import annotations

from opm_train.data.math_verify import MathVerifyDatasetAdapter

_SIMPLE_MATH_PROMPT_TEMPLATE = """Solve the following math problem.
Show your reasoning, then finish with exactly one line:
FINAL_ANSWER: <number>

Requirements:
- Keep FINAL_ANSWER as a plain numeric value only.
- Do not include units in FINAL_ANSWER.

Problem:
{question}
"""


class SimpleMathDatasetAdapter(MathVerifyDatasetAdapter):
    """Adapter for lightweight arithmetic/math datasets with numeric answers."""

    name = "simple_math"
    prompt_template = _SIMPLE_MATH_PROMPT_TEMPLATE

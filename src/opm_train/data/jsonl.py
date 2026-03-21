"""JSONL loading helpers for dataset adapters and batch routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_json_objects(input_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield `(line_no, payload)` objects from one JSONL file."""
    path = input_path.resolve()
    if not path.exists():
        raise ValueError(f"input JSONL does not exist: {path}")
    # Stream file lines directly instead of splitlines(); splitlines() treats
    # Unicode separators (for example U+2028) as line breaks and can corrupt
    # otherwise valid JSON strings found in public datasets.
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_no} must be a JSON object")
            yield line_no, payload

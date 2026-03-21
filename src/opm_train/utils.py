"""Shared low-level helpers used across orchestration modules."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return current UTC time in stable ISO-8601 milliseconds format."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def ensure_directory(path: Path) -> Path:
    """Create a directory recursively and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_ready(value: Any) -> Any:
    """Recursively convert values into JSON-serializable structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    return value


def estimate_text_tokens(text: str) -> int:
    """Estimate token count with a model-agnostic rough heuristic."""
    return max(1, len(text) // 4)

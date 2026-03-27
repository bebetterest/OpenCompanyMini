"""Trajectory export helpers for session/agent/step scopes."""

from __future__ import annotations

from typing import Any

from opm_train.storage import SessionStorage
from opm_train.trajectory.filter import select_scope
from opm_train.trajectory.formatter import format_raw, format_sft
from opm_train.trajectory.loader import ExportSchemaError, load_session_bundle



def export_trajectory(
    *,
    storage: SessionStorage,
    session_id: str,
    mode: str,
    agent_id: str | None = None,
    step: int | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Export one scoped session trajectory as raw or sft payload."""
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"raw", "sft"}:
        raise ValueError("mode must be one of: raw, sft")

    bundle = load_session_bundle(storage=storage, session_id=session_id)
    scoped = select_scope(bundle, agent_id=agent_id, step=step)

    if resolved_mode == "raw":
        return format_raw(scoped)
    return format_sft(scoped)


__all__ = [
    "ExportSchemaError",
    "export_trajectory",
]

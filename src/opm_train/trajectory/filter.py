"""Scope filtering helpers for trajectory exports."""

from __future__ import annotations

from typing import Any



def select_scope(
    bundle: dict[str, Any],
    *,
    agent_id: str | None,
    step: int | None,
) -> dict[str, Any]:
    """Filter loaded bundle by session/agent/agent-step scope."""
    selected_agent = str(agent_id).strip() if agent_id is not None else None
    selected_step = int(step) if step is not None else None

    if selected_step is not None and selected_agent is None:
        raise ValueError("--step requires --agent-id")

    events = [dict(item) for item in list(bundle.get("events", [])) if isinstance(item, dict)]
    turns = [dict(item) for item in list(bundle.get("turns", [])) if isinstance(item, dict)]
    agents = {str(k): dict(v) for k, v in dict(bundle.get("agents", {})).items()}
    tool_runs = {str(k): dict(v) for k, v in dict(bundle.get("tool_runs", {})).items()}

    if selected_agent is None:
        selected_turns = turns
        selected_events = events
        selected_agents = agents
        selected_tool_runs = tool_runs
    elif selected_step is None:
        selected_turns = [turn for turn in turns if str(turn.get("agent_id", "")).strip() == selected_agent]
        selected_events = [event for event in events if str(event.get("agent_id", "")).strip() == selected_agent]
        selected_agents = {key: value for key, value in agents.items() if key == selected_agent}
        selected_tool_runs = {
            key: value
            for key, value in tool_runs.items()
            if str(value.get("agent_id", "")).strip() == selected_agent
        }
    else:
        selected_turns = [
            turn
            for turn in turns
            if str(turn.get("agent_id", "")).strip() == selected_agent and _int_or_default(turn.get("step"), -1) == selected_step
        ]
        if not selected_turns:
            raise ValueError(f"No turn found for agent_id={selected_agent} step={selected_step}")
        turn = selected_turns[0]
        start_seq = _int_or_default(turn.get("event_seq_start"), -1)
        end_seq = _int_or_default(turn.get("event_seq_end"), -1)
        if start_seq <= 0 or end_seq < start_seq:
            raise ValueError(f"Turn has invalid event_seq range for agent_id={selected_agent} step={selected_step}")
        selected_events = [
            event
            for event in events
            if start_seq <= _int_or_default(event.get("seq"), -1) <= end_seq
            and str(event.get("agent_id", "")).strip() == selected_agent
        ]
        selected_agents = {key: value for key, value in agents.items() if key == selected_agent}
        selected_tool_runs = {
            key: value
            for key, value in tool_runs.items()
            if str(value.get("agent_id", "")).strip() == selected_agent
        }

    scoped = {
        **bundle,
        "agents": selected_agents,
        "tool_runs": selected_tool_runs,
        "events": selected_events,
        "turns": selected_turns,
        "scope": {
            "session_id": str(bundle.get("session_id", "")),
            "agent_id": selected_agent,
            "step": selected_step,
        },
    }
    return scoped



def _int_or_default(value: Any, default: int) -> int:
    """Parse integer with explicit fallback value."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

"""Context window assembly and compression helpers for agent prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from opm_train.config import OPMTrainConfig
from opm_train.models import AgentNode
from opm_train.prompts import PromptLibrary
from opm_train.tools import tool_definitions_for_agent
from opm_train.utils import estimate_text_tokens


@dataclass(frozen=True, slots=True)
class PromptWindowProjection:
    """Projection of visible conversation indices used to build prompt window."""

    summary: str
    summary_version: int
    pinned_message_indices: tuple[int, ...]
    tail_message_indices: tuple[int, ...]
    hidden_message_indices: tuple[int, ...]
    internal_message_indices: tuple[int, ...]

    @property
    def prompt_message_indices(self) -> tuple[int, ...]:
        """Return ordered indices included in model request."""
        return (*self.pinned_message_indices, *self.tail_message_indices)


def _metadata(agent: AgentNode) -> dict[str, Any]:
    """Ensure metadata is always a mutable dictionary."""
    if not isinstance(agent.metadata, dict):
        agent.metadata = {}
    return agent.metadata


def _indices(metadata: dict[str, Any], key: str) -> set[int]:
    """Read non-negative integer indices list from metadata."""
    raw = metadata.get(key)
    if not isinstance(raw, list):
        return set()
    resolved: set[int] = set()
    for item in raw:
        value = _to_non_negative_int(item)
        if value is not None:
            resolved.add(value)
    return resolved


def prompt_window_projection_from_metadata(
    *,
    message_count: int,
    metadata: dict[str, Any] | None,
    keep_pinned_messages: int,
) -> PromptWindowProjection:
    """Build prompt projection from metadata while preserving head/tail guarantees."""
    meta = metadata if isinstance(metadata, dict) else {}
    internal = _indices(meta, "internal_message_indices")
    visible = [i for i in range(max(0, int(message_count))) if i not in internal]
    summary = str(meta.get("context_summary", "")).strip()
    summary_version = int(meta.get("summary_version", 0) or 0)
    internal_indices = tuple(sorted(internal))

    if not summary:
        return PromptWindowProjection(
            summary="",
            summary_version=max(0, summary_version),
            pinned_message_indices=(),
            tail_message_indices=tuple(visible),
            hidden_message_indices=(),
            internal_message_indices=internal_indices,
        )

    keep_count = max(0, int(keep_pinned_messages))
    pinned = visible[:keep_count]
    pinned_set = set(pinned)
    summarized_until = int(meta.get("summarized_until_message_index", -1) or -1)
    tail = [i for i in visible if i not in pinned_set and i > summarized_until]
    hidden = [i for i in visible if i not in pinned_set and i <= summarized_until]
    return PromptWindowProjection(
        summary=summary,
        summary_version=max(1, summary_version),
        pinned_message_indices=tuple(pinned),
        tail_message_indices=tuple(tail),
        hidden_message_indices=tuple(hidden),
        internal_message_indices=internal_indices,
    )


class ContextAssembler:
    """Build role-aware system prompt, tools, and compact message window."""

    def __init__(self, *, config: OPMTrainConfig, prompt_library: PromptLibrary) -> None:
        """Create assembler bound to runtime config and prompt library."""
        self.config = config
        self.prompt_library = prompt_library

    def system_prompt(self, agent: AgentNode) -> str:
        """Load role-specific system prompt text."""
        return self.prompt_library.load_agent_prompt(agent.role.value).rstrip()

    def tools(self, agent: AgentNode) -> list[dict[str, Any]]:
        """Resolve available tool schemas for the agent role."""
        return tool_definitions_for_agent(
            agent,
            prompt_library=self.prompt_library,
            config=self.config,
        )

    def messages(self, agent: AgentNode, *, system_prompt: str) -> list[dict[str, Any]]:
        """Assemble final request messages including optional summary marker."""
        projection = prompt_window_projection_from_metadata(
            message_count=len(agent.conversation),
            metadata=_metadata(agent),
            keep_pinned_messages=self.config.runtime.context.keep_pinned_messages,
        )
        prompt_messages = _select_messages(
            conversation=agent.conversation,
            indices=projection.prompt_message_indices,
        )
        base = [{"role": "system", "content": system_prompt}]
        if not projection.summary:
            return [*base, *prompt_messages]
        pinned_count = len(projection.pinned_message_indices)
        pinned_head = prompt_messages[:pinned_count]
        tail = prompt_messages[pinned_count:]
        summary_message = {
            "role": "user",
            "content": self.prompt_library.render_runtime_message(
                "context_latest_summary",
                summary_version=projection.summary_version,
                summary=projection.summary,
            ),
        }
        return [*base, *pinned_head, summary_message, *tail]


def estimate_conversation_tokens(agent: AgentNode) -> int:
    """Estimate total tokens used by conversation content."""
    total = sum(estimate_text_tokens(str(message.get("content", ""))) for message in agent.conversation)
    return max(1, int(total))


def maybe_auto_compress(*, agent: AgentNode, config: OPMTrainConfig) -> bool:
    """Return whether auto-compress should trigger at current context usage."""
    if not config.runtime.context.enabled:
        return False
    limit = max(1, int(config.runtime.context.max_context_tokens))
    usage_ratio = estimate_conversation_tokens(agent) / limit
    return usage_ratio >= float(config.runtime.context.auto_compress_ratio)


async def compress_context(
    *,
    agent: AgentNode,
    reason: str,
    config: OPMTrainConfig,
    prompt_library: PromptLibrary,
    llm_client: Any | None,
) -> dict[str, Any]:
    """Compress older middle turns via dedicated LLM compression model."""
    metadata = _metadata(agent)
    if llm_client is None:
        return {"compressed": False, "error": "llm_client is required for context compression."}
    compression_model = str(config.runtime.context.compression_model).strip()
    if not compression_model:
        compression_model = str(config.provider.active_profile().model).strip()
    if not compression_model:
        return {"compressed": False, "error": "runtime.context.compression_model is required."}
    conversation = list(agent.conversation)
    if len(conversation) <= 2:
        return {"compressed": False, "reason": "conversation_too_short"}

    keep_pinned = max(0, int(metadata.get("keep_pinned_messages", 1) or 1))
    keep_pinned = min(keep_pinned, len(conversation) - 1)
    start = keep_pinned
    end = len(conversation) - 2
    if end < start:
        return {"compressed": False, "reason": "no_compressible_window"}

    window = conversation[start : end + 1]
    previous_summary = str(metadata.get("context_summary", "")).strip()
    system_prompt = (
        prompt_library.render_runtime_message("context_compression_system_prompt")
        if "context_compression_system_prompt" in prompt_library.load_runtime_messages()
        else (
            "Summarize conversation history for future continuation. Keep factual "
            "decisions, pending tasks, blockers, and next steps in concise English bullets."
        )
    )
    request_payload = {
        "reason": reason,
        "previous_summary": previous_summary,
        "messages": [
            {
                "index": start + index,
                "role": str(message.get("role", "unknown")).strip() or "unknown",
                "content": str(message.get("content", "")),
            }
            for index, message in enumerate(window)
        ],
    }
    request_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
    ]
    result = await llm_client.stream_chat(
        model=compression_model,
        messages=request_messages,
        temperature=0.0,
        max_tokens=max(256, int(config.provider.active_profile().max_tokens)),
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
    )
    summary = str(getattr(result, "content", "")).strip()
    if not summary:
        return {"compressed": False, "error": "compression model returned empty summary."}
    summary_version = int(metadata.get("summary_version", 0) or 0) + 1
    metadata["context_summary"] = summary
    metadata["summary_version"] = summary_version
    metadata["summarized_until_message_index"] = end
    metadata["last_compression_reason"] = reason
    metadata["compression_model"] = compression_model
    return {
        "compressed": True,
        "reason": reason,
        "model": compression_model,
        "summary_version": summary_version,
        "summarized_until_message_index": end,
        "message_range": {"start": start, "end": end},
    }


def _select_messages(
    *,
    conversation: list[dict[str, Any]],
    indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Select valid messages from conversation by index sequence."""
    size = len(conversation)
    return [conversation[index] for index in indices if 0 <= index < size]


def _to_non_negative_int(value: Any) -> int | None:
    """Parse non-negative integer or return ``None`` when invalid."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed

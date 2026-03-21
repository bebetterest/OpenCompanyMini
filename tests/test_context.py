from __future__ import annotations

from pathlib import Path

from opm_train.config import OPMTrainConfig
from opm_train.context import (
    ContextAssembler,
    compress_context,
    maybe_auto_compress,
    prompt_window_projection_from_metadata,
)
from opm_train.models import AgentNode, AgentRole
from opm_train.prompts import PromptLibrary, default_prompts_dir


def test_projection_uses_summary_and_tail() -> None:
    projection = prompt_window_projection_from_metadata(
        message_count=6,
        metadata={
            "context_summary": "summary",
            "summary_version": 2,
            "summarized_until_message_index": 3,
        },
        keep_pinned_messages=1,
    )
    assert projection.summary == "summary"
    assert projection.pinned_message_indices == (0,)
    assert projection.tail_message_indices == (4, 5)


def test_compress_context_updates_metadata() -> None:
    agent = AgentNode(
        id="a",
        session_id="s",
        name="root",
        role=AgentRole.ROOT,
        instruction="i",
        workspace_path=Path("."),
        conversation=[
            {"role": "user", "content": "head"},
            {"role": "assistant", "content": "middle one"},
            {"role": "assistant", "content": "middle two"},
            {"role": "assistant", "content": "tail"},
        ],
        metadata={"keep_pinned_messages": 1},
    )
    result = compress_context(agent=agent, reason="manual")
    assert result["compressed"] is True
    assert "context_summary" in agent.metadata


def test_context_assembler_injects_summary_message() -> None:
    config = OPMTrainConfig()
    library = PromptLibrary(default_prompts_dir())
    assembler = ContextAssembler(config=config, prompt_library=library)
    agent = AgentNode(
        id="a",
        session_id="s",
        name="root",
        role=AgentRole.ROOT,
        instruction="i",
        workspace_path=Path("."),
        conversation=[
            {"role": "user", "content": "head"},
            {"role": "assistant", "content": "middle"},
            {"role": "assistant", "content": "tail"},
        ],
        metadata={
            "keep_pinned_messages": 1,
            "context_summary": "short summary",
            "summary_version": 1,
            "summarized_until_message_index": 1,
        },
    )
    messages = assembler.messages(agent, system_prompt="SYSTEM")
    assert messages[0]["role"] == "system"
    assert "compressed as follows" in str(messages[2]["content"])


def test_auto_compress_triggers_over_threshold() -> None:
    config = OPMTrainConfig()
    config.runtime.context.max_context_tokens = 10
    config.runtime.context.auto_compress_ratio = 0.5
    agent = AgentNode(
        id="a",
        session_id="s",
        name="root",
        role=AgentRole.ROOT,
        instruction="i",
        workspace_path=Path("."),
        conversation=[
            {"role": "user", "content": "x" * 400},
            {"role": "assistant", "content": "y" * 400},
            {"role": "assistant", "content": "z" * 400},
        ],
        metadata={"keep_pinned_messages": 1},
    )
    assert maybe_auto_compress(agent=agent, config=config) is True

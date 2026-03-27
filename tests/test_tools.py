from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from opm_train.config import OPMTrainConfig
from opm_train.models import AgentRole
from opm_train.prompts import PromptLibrary, default_prompts_dir
from opm_train.tools import runtime_tool_contract_issues, tool_definitions_for_role, validate_finish_action


def _tool_by_name(tools: list[dict[str, object]], name: str) -> dict[str, object]:
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == name:
            return tool
    raise AssertionError(f"tool not found: {name}")


def test_tool_definitions_finish_schema_root_omits_next_recommendation() -> None:
    config = OPMTrainConfig()
    library = PromptLibrary(default_prompts_dir())
    tools = tool_definitions_for_role(AgentRole.ROOT, prompt_library=library, config=config)
    finish = _tool_by_name(tools, "finish")
    properties = finish["function"]["parameters"]["properties"]  # type: ignore[index]
    assert "next_recommendation" not in properties
    assert set(properties["status"]["enum"]) == {"completed", "partial"}  # type: ignore[index]


def test_tool_definitions_list_limit_uses_runtime_config() -> None:
    config = OPMTrainConfig()
    config.runtime.tools.list_default_limit = 37
    config.runtime.tools.list_max_limit = 81
    library = PromptLibrary(default_prompts_dir())
    tools = tool_definitions_for_role(AgentRole.WORKER, prompt_library=library, config=config)
    list_tool = _tool_by_name(tools, "list_tool_runs")
    limit_schema = list_tool["function"]["parameters"]["properties"]["limit"]  # type: ignore[index]
    assert limit_schema["default"] == 37
    assert limit_schema["maximum"] == 81


def test_tool_definitions_timeout_defaults_use_runtime_config() -> None:
    config = OPMTrainConfig()
    config.runtime.tools.shell_timeout_seconds = 91
    config.runtime.tools.wait_run_timeout_seconds = 2.5
    library = PromptLibrary(default_prompts_dir())
    tools = tool_definitions_for_role(AgentRole.ROOT, prompt_library=library, config=config)
    shell_tool = _tool_by_name(tools, "shell")
    shell_timeout = shell_tool["function"]["parameters"]["properties"]["timeout_seconds"]  # type: ignore[index]
    assert shell_timeout["default"] == 91.0
    assert shell_timeout["minimum"] == 1.0

    wait_tool = _tool_by_name(tools, "wait_run")
    wait_timeout = wait_tool["function"]["parameters"]["properties"]["timeout_seconds"]  # type: ignore[index]
    assert wait_timeout["default"] == 2.5
    assert wait_timeout["minimum"] == 0.0


def test_validate_finish_action_worker_requires_next_recommendation_on_failed() -> None:
    error = validate_finish_action(
        AgentRole.WORKER,
        {"type": "finish", "status": "failed", "summary": "x"},
    )
    assert error is not None


def test_runtime_tool_contract_issues_reports_missing_prompt_and_registry_items() -> None:
    config = OPMTrainConfig()
    config.runtime.tools.root_tools = ["shell", "finish", "missing_one"]
    config.runtime.tools.worker_tools = ["shell", "finish"]
    with TemporaryDirectory() as temp_dir:
        prompts_dir = Path(temp_dir)
        (prompts_dir / "root_coordinator.md").write_text("root", encoding="utf-8")
        (prompts_dir / "worker.md").write_text("worker", encoding="utf-8")
        (prompts_dir / "runtime_messages.json").write_text(json.dumps({"x": "x"}), encoding="utf-8")
        (prompts_dir / "tool_definitions.json").write_text(
            json.dumps(
                {
                    "shell": {"type": "function", "function": {"name": "shell"}},
                    "finish": {"type": "function", "function": {"name": "finish"}},
                }
            ),
            encoding="utf-8",
        )
        library = PromptLibrary(prompts_dir)
        issues = runtime_tool_contract_issues(
            config=config,
            prompt_library=library,
            registry_tool_names={"shell"},
        )
    assert "root_enabled_missing_prompt_definitions:missing_one" in issues
    assert "root_enabled_missing_registry_handlers:missing_one" in issues

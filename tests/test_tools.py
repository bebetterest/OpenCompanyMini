from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from opm_train.config import OPMTrainConfig
from opm_train.models import AgentNode, AgentRole
from opm_train.prompts import PromptLibrary, default_prompts_dir
from opm_train.tools import (
    runtime_tool_contract_issues,
    tool_definitions_for_agent,
    tool_definitions_for_role,
    validate_finish_action,
    visible_tool_names_for_agent,
)


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
    for name in ("list_agent_runs", "list_tool_runs"):
        tool = _tool_by_name(tools, name)
        limit_schema = tool["function"]["parameters"]["properties"]["limit"]  # type: ignore[index]
        assert limit_schema["default"] == 37
        assert limit_schema["maximum"] == 81
        assert limit_schema["minimum"] == 1


def test_tool_definitions_list_mcp_resources_limit_schema_stays_static() -> None:
    config = OPMTrainConfig()
    config.runtime.tools.list_default_limit = 37
    config.runtime.tools.list_max_limit = 81
    library = PromptLibrary(default_prompts_dir())
    tools = tool_definitions_for_role(AgentRole.WORKER, prompt_library=library, config=config)
    mcp_tool = _tool_by_name(tools, "list_mcp_resources")
    limit_schema = mcp_tool["function"]["parameters"]["properties"]["limit"]  # type: ignore[index]
    assert limit_schema["minimum"] == 1
    assert "default" not in limit_schema
    assert "maximum" not in limit_schema


def test_tool_definitions_shell_schema_matches_opencompany_contract() -> None:
    config = OPMTrainConfig()
    library = PromptLibrary(default_prompts_dir())
    tools = tool_definitions_for_role(AgentRole.ROOT, prompt_library=library, config=config)
    shell = _tool_by_name(tools, "shell")
    properties = shell["function"]["parameters"]["properties"]  # type: ignore[index]
    assert "command" in properties
    assert "cwd" in properties
    assert "blocking" not in properties
    assert "timeout_seconds" not in properties


def test_tool_definitions_wait_time_bounds_follow_runtime_config() -> None:
    config = OPMTrainConfig()
    config.runtime.tools.wait_time_min_seconds = 12
    config.runtime.tools.wait_time_max_seconds = 48
    library = PromptLibrary(default_prompts_dir())
    tools = tool_definitions_for_role(AgentRole.ROOT, prompt_library=library, config=config)
    wait_time = _tool_by_name(tools, "wait_time")
    seconds_schema = wait_time["function"]["parameters"]["properties"]["seconds"]  # type: ignore[index]
    assert seconds_schema["minimum"] == 12
    assert seconds_schema["maximum"] == 48


def test_tool_definitions_get_agent_and_tool_run_extended_fields() -> None:
    config = OPMTrainConfig()
    library = PromptLibrary(default_prompts_dir())
    tools = tool_definitions_for_role(AgentRole.ROOT, prompt_library=library, config=config)

    get_agent = _tool_by_name(tools, "get_agent_run")
    agent_props = get_agent["function"]["parameters"]["properties"]  # type: ignore[index]
    assert "messages_start" in agent_props
    assert "messages_end" in agent_props

    get_tool = _tool_by_name(tools, "get_tool_run")
    tool_props = get_tool["function"]["parameters"]["properties"]  # type: ignore[index]
    assert "include_result" in tool_props


def test_tool_definitions_cancel_agent_has_recursive_flag() -> None:
    config = OPMTrainConfig()
    library = PromptLibrary(default_prompts_dir())
    tools = tool_definitions_for_role(AgentRole.ROOT, prompt_library=library, config=config)
    cancel_agent = _tool_by_name(tools, "cancel_agent")
    properties = cancel_agent["function"]["parameters"]["properties"]  # type: ignore[index]
    assert "recursive" in properties


def test_tool_definitions_wait_run_schema_avoids_top_level_combinators() -> None:
    config = OPMTrainConfig()
    library = PromptLibrary(default_prompts_dir())
    tools = tool_definitions_for_role(AgentRole.ROOT, prompt_library=library, config=config)
    wait_run = _tool_by_name(tools, "wait_run")
    params = wait_run["function"]["parameters"]  # type: ignore[index]
    assert params["type"] == "object"
    assert "oneOf" not in params
    assert "anyOf" not in params
    assert "allOf" not in params
    assert "not" not in params


def test_tool_definitions_disallow_additional_properties() -> None:
    config = OPMTrainConfig()
    library = PromptLibrary(default_prompts_dir())
    tools = tool_definitions_for_role(AgentRole.ROOT, prompt_library=library, config=config)
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        params = function.get("parameters")
        if not isinstance(params, dict):
            continue
        assert params.get("additionalProperties") is False


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


def test_visible_tools_and_definitions_hide_mcp_helpers_when_mcp_disabled() -> None:
    config = OPMTrainConfig()
    config.extensions.mcp_enabled = False
    library = PromptLibrary(default_prompts_dir())
    agent = AgentNode(
        id="agent-root",
        session_id="session-1",
        name="root",
        role=AgentRole.ROOT,
        instruction="task",
        workspace_path=Path("."),
    )
    visible = visible_tool_names_for_agent(agent, config=config)
    assert "list_mcp_servers" not in visible
    assert "list_mcp_resources" not in visible
    assert "read_mcp_resource" not in visible

    tools = tool_definitions_for_agent(agent, prompt_library=library, config=config)
    function_names = {
        str(tool.get("function", {}).get("name", "")).strip()
        for tool in tools
        if isinstance(tool.get("function"), dict)
    }
    assert "list_mcp_servers" not in function_names
    assert "list_mcp_resources" not in function_names
    assert "read_mcp_resource" not in function_names

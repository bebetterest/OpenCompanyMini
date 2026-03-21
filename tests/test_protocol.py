from __future__ import annotations

import pytest

from opm_train.protocol import ProtocolError, extract_json_object, normalize_actions, normalize_tool_calls


def test_extract_json_object_parses_fenced_json() -> None:
    payload = extract_json_object("```json\n{\"actions\":[{\"type\":\"finish\"}]}\n```")
    assert payload["actions"][0]["type"] == "finish"


def test_normalize_actions_requires_non_empty_list() -> None:
    with pytest.raises(ProtocolError):
        normalize_actions({"actions": []})


def test_normalize_tool_calls_parses_arguments() -> None:
    calls = normalize_tool_calls(
        [
            {
                "id": "call-1",
                "name": "shell",
                "arguments_json": "{\"command\":\"echo hi\"}",
            }
        ]
    )
    assert calls == [
        {
            "type": "shell",
            "_tool_call_id": "call-1",
            "command": "echo hi",
        }
    ]

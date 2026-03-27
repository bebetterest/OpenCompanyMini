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
                "type": "function",
                "function": {
                    "name": "shell",
                    "arguments": "{\"command\":\"echo hi\"}",
                },
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


def test_normalize_tool_calls_accepts_openai_compatible_shape() -> None:
    calls = normalize_tool_calls(
        [
            {
                "id": "call-2",
                "type": "function",
                "function": {
                    "name": "wait_time",
                    "arguments": "{\"seconds\":0}",
                },
            }
        ]
    )
    assert calls == [
        {
            "type": "wait_time",
            "_tool_call_id": "call-2",
            "seconds": 0,
        }
    ]


def test_normalize_tool_calls_rejects_legacy_shape() -> None:
    with pytest.raises(ProtocolError, match="type must be 'function'"):
        normalize_tool_calls(
            [
                {
                    "id": "call-legacy",
                    "name": "wait_time",
                    "arguments_json": "{\"seconds\":0}",
                }
            ]
        )

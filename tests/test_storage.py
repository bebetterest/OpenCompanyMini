from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from opm_train.models import ToolRun, ToolRunStatus
from opm_train.storage import SessionStorage, tool_run_from_dict, tool_run_to_dict


def _new_storage(base_dir: Path) -> SessionStorage:
    return SessionStorage(app_dir=base_dir, data_dir_name=".opm_train")


def test_validate_snapshot_tail_accepts_strict_contiguous_events() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        storage.append_event("s-1", {"seq": 1, "event_type": "a"})
        storage.append_event("s-1", {"seq": 2, "event_type": "b"})
        assert storage.validate_snapshot_tail("s-1", expected_last_event_seq=2) is True


def test_validate_snapshot_tail_rejects_gapped_events() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        storage.append_event("s-2", {"seq": 1, "event_type": "a"})
        storage.append_event("s-2", {"seq": 3, "event_type": "b"})
        assert storage.validate_snapshot_tail("s-2", expected_last_event_seq=3) is False


def test_validate_snapshot_tail_rejects_count_mismatch() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        storage.append_event("s-3", {"seq": 1, "event_type": "a"})
        storage.append_event("s-3", {"seq": 2, "event_type": "b"})
        assert storage.validate_snapshot_tail("s-3", expected_last_event_seq=3) is False


def test_load_events_keeps_unicode_line_separator_inside_json_string() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        session_id = "s-unicode-events"
        storage.append_event(
            session_id,
            {
                "seq": 1,
                "event_type": "unicode",
                "message": "Line A\u2028Line B",
            },
        )
        loaded = storage.load_events(session_id)
        assert len(loaded) == 1
        assert loaded[0]["message"] == "Line A\u2028Line B"


def test_load_turns_keeps_unicode_line_separator_inside_json_string() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        session_id = "s-unicode-turns"
        storage.append_turn(
            session_id,
            {
                "turn_id": "turn-1",
                "agent_id": "agent-1",
                "step": 1,
                "summary": "Line A\u2028Line B",
            },
        )
        loaded = storage.load_turns(session_id)
        assert len(loaded) == 1
        assert loaded[0]["summary"] == "Line A\u2028Line B"


def test_tool_run_roundtrip_without_parent_run_id() -> None:
    run = ToolRun(
        id="toolrun-1",
        session_id="session-1",
        agent_id="agent-1",
        tool_name="shell",
        arguments={"command": "echo hi"},
        status=ToolRunStatus.RUNNING,
        blocking=False,
        created_at="2026-03-20T00:00:00.000+00:00",
    )
    payload = tool_run_to_dict(run)
    assert "parent_run_id" not in payload
    restored = tool_run_from_dict(payload)
    assert restored.id == run.id
    assert restored.status == ToolRunStatus.RUNNING
    assert restored.arguments == {"command": "echo hi"}


def test_tool_run_roundtrip_supports_abandoned_status() -> None:
    run = ToolRun(
        id="toolrun-2",
        session_id="session-1",
        agent_id="agent-1",
        tool_name="shell",
        arguments={"command": "echo hi"},
        status=ToolRunStatus.ABANDONED,
        blocking=False,
        created_at="2026-03-21T00:00:00.000+00:00",
        error="tool_run_abandoned_on_resume",
    )
    payload = tool_run_to_dict(run)
    restored = tool_run_from_dict(payload)
    assert restored.status == ToolRunStatus.ABANDONED
    assert restored.error == "tool_run_abandoned_on_resume"


def test_agent_llm_call_and_context_compression_records_are_sequential_and_utf8() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        session_id = "s-cn"
        agent_id = "agent-cn"
        agent_name = "Agent CN"

        seq1 = storage.append_agent_llm_call_request(
            session_id,
            agent_id,
            {"content": "请求：请输出中文。"},
            agent_name=agent_name,
        )
        storage.append_agent_llm_call_response(
            session_id,
            agent_id,
            seq1,
            {"content": "响应：中文结果。"},
            agent_name=agent_name,
        )
        seq2 = storage.append_agent_llm_call_request(
            session_id,
            agent_id,
            {"content": "第二次请求"},
            agent_name=agent_name,
        )
        storage.append_agent_llm_call_response(
            session_id,
            agent_id,
            seq2,
            {"content": "第二次响应"},
            agent_name=agent_name,
        )

        llm_dir = storage.agent_llm_calls_dir(session_id, agent_id, agent_name=agent_name)
        names = sorted(path.name for path in llm_dir.glob("*.json"))
        assert names == [
            "0001_request.json",
            "0001_response.json",
            "0002_request.json",
            "0002_response.json",
        ]
        assert "请求：请输出中文。" in (llm_dir / "0001_request.json").read_text(encoding="utf-8")
        assert "响应：中文结果。" in (llm_dir / "0001_response.json").read_text(encoding="utf-8")

        c1 = storage.append_agent_context_compression(
            session_id,
            agent_id,
            {"reason": "manual", "context_summary": "中文摘要"},
            agent_name=agent_name,
        )
        c2 = storage.append_agent_context_compression(
            session_id,
            agent_id,
            {"reason": "auto", "context_summary": "第二条摘要"},
            agent_name=agent_name,
        )
        assert c1 == 1
        assert c2 == 2
        compression_dir = storage.agent_context_compressions_dir(session_id, agent_id, agent_name=agent_name)
        compression_files = sorted(path.name for path in compression_dir.glob("*.json"))
        assert compression_files == ["0001.json", "0002.json"]
        assert "中文摘要" in (compression_dir / "0001.json").read_text(encoding="utf-8")


def test_runtime_log_error_and_timer_are_written_to_session() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        session_id = "s-log"
        storage.append_runtime_log(session_id, {"level": "INFO", "message": "hello"})
        storage.append_error_record(session_id, {"stage": "test", "error_type": "ValueError"})
        storage.append_timer_sample(session_id, {"module": "ask_agent", "elapsed_ms": 1.23})

        runtime_lines = storage.runtime_log_path(session_id).read_text(encoding="utf-8").splitlines()
        error_lines = storage.errors_path(session_id).read_text(encoding="utf-8").splitlines()
        timer_lines = storage.module_timing_path(session_id).read_text(encoding="utf-8").splitlines()

        assert json.loads(runtime_lines[0])["message"] == "hello"
        assert json.loads(error_lines[0])["error_type"] == "ValueError"
        assert json.loads(timer_lines[0])["module"] == "ask_agent"


def test_next_sequence_uses_existing_files_then_increments_from_cache() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        session_id = "s-seq"
        agent_id = "agent-1"
        agent_name = "Agent 1"
        llm_dir = storage.agent_llm_calls_dir(session_id, agent_id, agent_name=agent_name)
        (llm_dir / "0007_request.json").write_text("{}", encoding="utf-8")

        seq1 = storage.append_agent_llm_call_request(session_id, agent_id, {"a": 1}, agent_name=agent_name)
        seq2 = storage.append_agent_llm_call_request(session_id, agent_id, {"a": 2}, agent_name=agent_name)
        assert seq1 == 8
        assert seq2 == 9
        assert (llm_dir / "0008_request.json").exists()
        assert (llm_dir / "0009_request.json").exists()


def test_agent_dir_uses_name_between_prefix_and_id_suffix() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        path = storage.agent_dir(
            "s-agent-dir",
            "agent-08277a53f952",
            agent_name="Tester",
        )
        assert path.name == "agent-tester-08277a53f952"


def test_agent_dir_does_not_reuse_legacy_directory_name() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        session_id = "s-agent-reuse"
        agent_id = "agent-08277a53f952"
        agents_root = storage.agents_dir(session_id)

        legacy = agents_root / agent_id
        legacy.mkdir(parents=True, exist_ok=True)
        resolved = storage.agent_dir(session_id, agent_id, agent_name="Tester")
        assert resolved.name == "agent-tester-08277a53f952"
        assert resolved != legacy


def test_agent_dir_requires_name_for_agent_prefix_ids() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        with pytest.raises(ValueError, match="agent_name is required"):
            storage.agent_dir("s-agent-name-required", "agent-08277a53f952")


def test_turns_are_appended_and_loaded_in_original_order() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        session_id = "s-turns"
        storage.append_turn(
            session_id,
            {"turn_id": "turn-a", "agent_id": "agent-1", "step": 1},
        )
        storage.append_turn(
            session_id,
            {"turn_id": "turn-b", "agent_id": "agent-2", "step": 1},
        )
        storage.append_turn(
            session_id,
            {"turn_id": "turn-c", "agent_id": "agent-1", "step": 2},
        )

        loaded = storage.load_turns(session_id)
        assert [item["turn_id"] for item in loaded] == ["turn-a", "turn-b", "turn-c"]


def test_load_turns_supports_agent_and_step_filters() -> None:
    with TemporaryDirectory() as temp_dir:
        storage = _new_storage(Path(temp_dir))
        session_id = "s-turns-filter"
        storage.append_turn(session_id, {"turn_id": "turn-a", "agent_id": "agent-1", "step": 1})
        storage.append_turn(session_id, {"turn_id": "turn-b", "agent_id": "agent-2", "step": 1})
        storage.append_turn(session_id, {"turn_id": "turn-c", "agent_id": "agent-1", "step": 2})

        by_agent = storage.load_turns(session_id, agent_id="agent-1")
        assert [item["turn_id"] for item in by_agent] == ["turn-a", "turn-c"]

        by_agent_step = storage.load_turns(session_id, agent_id="agent-1", step=2)
        assert [item["turn_id"] for item in by_agent_step] == ["turn-c"]

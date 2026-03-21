from __future__ import annotations

import json
from pathlib import Path

import opm_train.prompts as prompts_module
from opm_train.prompts import PromptLibrary


def test_default_prompts_dir_falls_back_to_packaged_assets(monkeypatch, tmp_path: Path) -> None:
    fake_module = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "opm_train" / "prompts.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("# fake module file", encoding="utf-8")

    prompt_assets = fake_module.parent / "prompt_assets"
    prompt_assets.mkdir()
    (prompt_assets / "root_coordinator.md").write_text("ROOT PROMPT", encoding="utf-8")
    (prompt_assets / "worker.md").write_text("WORKER PROMPT", encoding="utf-8")
    (prompt_assets / "runtime_messages.json").write_text(
        json.dumps({"root_initial_message": "task={task}"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (prompt_assets / "tool_definitions.json").write_text(
        json.dumps({}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(prompts_module, "__file__", str(fake_module))
    resolved = prompts_module.default_prompts_dir()
    assert resolved == prompt_assets

    library = PromptLibrary(resolved)
    assert library.load_agent_prompt("root") == "ROOT PROMPT"
    assert library.load_agent_prompt("worker") == "WORKER PROMPT"

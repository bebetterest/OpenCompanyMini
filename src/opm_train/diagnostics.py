"""Runtime diagnostics helpers for doctor-like health checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opm_train.orchestrator import RuntimeOrchestrator
from opm_train.orchestrator_tools.registry import TOOL_REGISTRY
from opm_train.tools import runtime_tool_contract_issues


def build_doctor_payload(
    *,
    orchestrator: RuntimeOrchestrator,
    app_dir: Path,
    project_dir: Path,
) -> dict[str, Any]:
    """Build doctor command payload from orchestrator state and resolved paths."""
    profile = orchestrator.config.provider.active_profile()
    prompts_dir = orchestrator.prompt_library.prompts_dir
    contract_issues = runtime_tool_contract_issues(
        config=orchestrator.config,
        prompt_library=orchestrator.prompt_library,
        registry_tool_names=set(TOOL_REGISTRY.keys()),
    )
    contract_ok = not contract_issues
    return {
        "app_dir": str(app_dir),
        "project_dir": str(project_dir),
        "config_path": str(app_dir / "opm_train.toml"),
        "config_exists": (app_dir / "opm_train.toml").exists(),
        "prompts_dir": str(prompts_dir),
        "prompts_exists": prompts_dir.is_dir(),
        "provider_profile": orchestrator.config.provider.profile,
        "provider_base_url": profile.base_url,
        "provider_api_key_env": profile.api_key_env,
        "provider_api_key_set": bool(profile.api_key),
        "tool_contract_ok": contract_ok,
        "tool_contract_issues": contract_issues,
        "ready_for_real_run": bool(profile.api_key and prompts_dir.is_dir() and contract_ok),
        "ready_for_smoke_run": prompts_dir.is_dir(),
    }

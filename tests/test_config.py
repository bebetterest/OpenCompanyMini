from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from opm_train.config import OPMTrainConfig


def test_load_runtime_tools_shell_inline_wait_seconds() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        (app_dir / "opm_train.toml").write_text(
            "\n".join(
                [
                    "[runtime.tools]",
                    "list_default_limit = 33",
                    "shell_timeout_seconds = 75",
                    "wait_run_timeout_seconds = 4.5",
                    "shell_inline_wait_seconds = 5.5",
                ]
            ),
            encoding="utf-8",
        )
        config = OPMTrainConfig.load(app_dir)
        assert config.runtime.tools.list_default_limit == 33
        assert config.runtime.tools.shell_timeout_seconds == 75.0
        assert config.runtime.tools.wait_run_timeout_seconds == 4.5
        assert config.runtime.tools.shell_inline_wait_seconds == 5.5


def test_snapshot_includes_shell_inline_wait_seconds() -> None:
    config = OPMTrainConfig()
    config.runtime.tools.shell_timeout_seconds = 22.0
    config.runtime.tools.wait_run_timeout_seconds = 3.0
    config.runtime.tools.shell_inline_wait_seconds = 3.25
    tools_snapshot = config.as_snapshot()["runtime"]["tools"]
    assert tools_snapshot["shell_timeout_seconds"] == 22.0
    assert tools_snapshot["wait_run_timeout_seconds"] == 3.0
    assert tools_snapshot["shell_inline_wait_seconds"] == 3.25


def test_load_runtime_limits_protocol_retry_fields() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        (app_dir / "opm_train.toml").write_text(
            "\n".join(
                [
                    "[runtime.limits]",
                    "max_protocol_retries = 4",
                    "protocol_retry_backoff_seconds = 0.6",
                ]
            ),
            encoding="utf-8",
        )
        config = OPMTrainConfig.load(app_dir)
        assert config.runtime.limits.max_protocol_retries == 4
        assert config.runtime.limits.protocol_retry_backoff_seconds == 0.6


def test_default_tinker_profile_matches_openai_compatible_inference_endpoint() -> None:
    config = OPMTrainConfig()
    profile = config.provider.tinker
    assert profile.base_url == "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    assert profile.model.startswith("tinker://")

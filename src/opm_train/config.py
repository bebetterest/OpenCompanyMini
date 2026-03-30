"""Configuration models and TOML merge helpers for runtime startup."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROVIDER_PROFILE_NAMES: tuple[str, ...] = ("openrouter", "tinker", "custom")


_DEFAULT_RUNTIME_TOOL_NAMES = (
    "shell",
    "compress_context",
    "wait_time",
    "list_mcp_servers",
    "list_mcp_resources",
    "read_mcp_resource",
    "list_agent_runs",
    "get_agent_run",
    "spawn_agent",
    "cancel_agent",
    "steer_agent",
    "list_tool_runs",
    "get_tool_run",
    "wait_run",
    "cancel_tool_run",
    "finish",
)


def _default_runtime_tool_names() -> list[str]:
    """Return a fresh copy of default runtime tools."""
    return list(_DEFAULT_RUNTIME_TOOL_NAMES)


def _provider_profile(
    *,
    base_url: str,
    api_key_env: str,
    model: str,
    timeout_seconds: int = 120,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.0,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    headers: dict[str, str] | None = None,
) -> ProviderProfileConfig:
    """Build provider profile with shared defaults."""
    return ProviderProfileConfig(
        base_url=base_url,
        api_key_env=api_key_env,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
        headers=dict(headers or {}),
    )


@dataclass(slots=True)
class ProjectConfig:
    """Project-level config shared by all sessions."""

    name: str = "opm-train"
    data_dir: str = ".opm_train"


@dataclass(slots=True)
class ProviderProfileConfig:
    """One LLM provider profile."""

    base_url: str = ""
    api_key_env: str = ""
    model: str = ""
    timeout_seconds: int = 120
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    temperature: float = 0.2
    max_tokens: int = 4096
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        """Resolve API key from configured environment variable."""
        env_name = str(self.api_key_env or "").strip()
        if not env_name:
            return ""
        return str(os.environ.get(env_name, "")).strip()


@dataclass(slots=True)
class ProviderConfig:
    """Provider selection and per-profile settings."""

    profile: str = "openrouter"
    openrouter: ProviderProfileConfig = field(
        default_factory=lambda: _provider_profile(
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            model="openai/gpt-4o-mini",
            max_retries=8,
            headers={
                "HTTP-Referer": "https://github.com/opencompany/opm-train",
                "X-Title": "opm-train",
            },
        )
    )
    tinker: ProviderProfileConfig = field(
        default_factory=lambda: _provider_profile(
            base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
            api_key_env="TINKER_API_KEY",
            model="tinker://replace-with-sampler-checkpoint",
        )
    )
    custom: ProviderProfileConfig = field(
        default_factory=lambda: _provider_profile(
            base_url="",
            api_key_env="OPENAI_API_KEY",
            model="gpt-4o-mini",
        )
    )

    def active_profile(self) -> ProviderProfileConfig:
        """Return currently selected profile, defaulting to openrouter."""
        selected = str(self.profile or "openrouter").strip().lower()
        profiles = {
            "openrouter": self.openrouter,
            "tinker": self.tinker,
            "custom": self.custom,
        }
        return profiles.get(selected, self.openrouter)


@dataclass(slots=True)
class RuntimeLimitsConfig:
    """Orchestration fan-out, concurrency, and step budgets."""

    max_children_per_agent: int = 6
    max_active_agents: int = 8
    max_root_steps: int = 64
    max_agent_steps: int = 48
    max_protocol_retries: int = 2
    protocol_retry_backoff_seconds: float = 0.25


@dataclass(slots=True)
class RuntimeContextConfig:
    """Context window shaping and auto-compression settings."""

    enabled: bool = True
    auto_compress_ratio: float = 0.8
    keep_pinned_messages: int = 1
    max_context_tokens: int = 96_000
    compression_model: str = ""


@dataclass(slots=True)
class RuntimeToolsConfig:
    """Tool allow-lists and tool-level runtime defaults."""

    root_tools: list[str] = field(default_factory=_default_runtime_tool_names)
    worker_tools: list[str] = field(default_factory=_default_runtime_tool_names)
    list_default_limit: int = 20
    list_max_limit: int = 200
    shell_timeout_seconds: float = 300.0
    wait_run_timeout_seconds: float = 20.0
    shell_inline_wait_seconds: float = 5.0
    wait_time_min_seconds: float = 10.0
    wait_time_max_seconds: float = 60.0

    def tool_names_for_role(self, role: str) -> list[str]:
        """Return enabled tool names for role."""
        if role == "root":
            return list(self.root_tools)
        return list(self.worker_tools)

    def normalize_list_limit(self, value: Any | None) -> int:
        """Clamp list page size to configured bounds."""
        if value is None:
            parsed = int(self.list_default_limit)
        else:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = int(self.list_default_limit)
        return max(1, min(int(self.list_max_limit), parsed))

    def wait_time_bounds(self) -> tuple[float, float]:
        """Return normalized [min,max] bounds for wait_time seconds."""
        minimum = max(0.0, float(self.wait_time_min_seconds))
        maximum = max(minimum, float(self.wait_time_max_seconds))
        return minimum, maximum


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime config grouping for limits, context, and tools."""

    limits: RuntimeLimitsConfig = field(default_factory=RuntimeLimitsConfig)
    context: RuntimeContextConfig = field(default_factory=RuntimeContextConfig)
    tools: RuntimeToolsConfig = field(default_factory=RuntimeToolsConfig)


@dataclass(slots=True)
class DeferredExtensionsConfig:
    """Feature flags for deferred v0 extension points."""

    sandbox_enabled: bool = False
    mcp_enabled: bool = False
    skills_enabled: bool = False


@dataclass(slots=True)
class OPMTrainConfig:
    """Root configuration object loaded from ``opm_train.toml``."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    extensions: DeferredExtensionsConfig = field(default_factory=DeferredExtensionsConfig)

    @classmethod
    def load(cls, app_dir: Path) -> OPMTrainConfig:
        """Load config from ``app_dir/opm_train.toml`` when available."""
        config = cls()
        config_path = app_dir / "opm_train.toml"
        if not config_path.exists():
            return config
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        config._merge(payload)
        return config

    def _merge(self, payload: dict[str, Any]) -> None:
        """Merge partial TOML payload into current config."""
        self._merge_project(_as_dict(payload.get("project")))
        self._merge_provider(_as_dict(payload.get("provider")))
        self._merge_runtime(_as_dict(payload.get("runtime")))
        self._merge_extensions(_as_dict(payload.get("extensions")))

    def _merge_project(self, payload: dict[str, Any]) -> None:
        """Merge ``[project]`` section."""
        if not payload:
            return
        self.project = ProjectConfig(
            name=str(payload.get("name", self.project.name)),
            data_dir=str(payload.get("data_dir", self.project.data_dir)),
        )

    def _merge_provider(self, payload: dict[str, Any]) -> None:
        """Merge ``[provider]`` section and profile overrides."""
        if not payload:
            return
        self.provider.profile = str(payload.get("profile", self.provider.profile)).strip() or self.provider.profile
        for profile_name in PROVIDER_PROFILE_NAMES:
            profile_payload = _as_dict(payload.get(profile_name))
            if not profile_payload:
                continue
            current = getattr(self.provider, profile_name)
            setattr(self.provider, profile_name, _merge_profile(current, profile_payload))

    def _merge_runtime(self, payload: dict[str, Any]) -> None:
        """Merge ``[runtime]`` section."""
        if not payload:
            return
        self._merge_runtime_limits(_as_dict(payload.get("limits")))
        self._merge_runtime_context(_as_dict(payload.get("context")))
        self._merge_runtime_tools(_as_dict(payload.get("tools")))

    def _merge_runtime_limits(self, payload: dict[str, Any]) -> None:
        """Merge ``[runtime.limits]`` section."""
        if not payload:
            return
        current = self.runtime.limits
        self.runtime.limits = RuntimeLimitsConfig(
            max_children_per_agent=_as_int(payload.get("max_children_per_agent"), current.max_children_per_agent, minimum=1),
            max_active_agents=_as_int(payload.get("max_active_agents"), current.max_active_agents, minimum=1),
            max_root_steps=_as_int(payload.get("max_root_steps"), current.max_root_steps, minimum=1),
            max_agent_steps=_as_int(payload.get("max_agent_steps"), current.max_agent_steps, minimum=1),
            max_protocol_retries=_as_int(payload.get("max_protocol_retries"), current.max_protocol_retries, minimum=0),
            protocol_retry_backoff_seconds=_as_float(
                payload.get("protocol_retry_backoff_seconds"),
                current.protocol_retry_backoff_seconds,
                minimum=0.0,
            ),
        )

    def _merge_runtime_context(self, payload: dict[str, Any]) -> None:
        """Merge ``[runtime.context]`` section."""
        if not payload:
            return
        current = self.runtime.context
        self.runtime.context = RuntimeContextConfig(
            enabled=bool(payload.get("enabled", current.enabled)),
            auto_compress_ratio=_as_float(payload.get("auto_compress_ratio"), current.auto_compress_ratio, minimum=0.1, maximum=1.0),
            keep_pinned_messages=_as_int(payload.get("keep_pinned_messages"), current.keep_pinned_messages, minimum=0),
            max_context_tokens=_as_int(payload.get("max_context_tokens"), current.max_context_tokens, minimum=1),
            compression_model=str(payload.get("compression_model", current.compression_model)).strip(),
        )

    def _merge_runtime_tools(self, payload: dict[str, Any]) -> None:
        """Merge ``[runtime.tools]`` section."""
        if not payload:
            return
        current = self.runtime.tools
        wait_time_min_seconds = _as_float(
            payload.get("wait_time_min_seconds"),
            current.wait_time_min_seconds,
            minimum=0.0,
        )
        wait_time_max_seconds = _as_float(
            payload.get("wait_time_max_seconds"),
            current.wait_time_max_seconds,
            minimum=0.0,
        )
        if wait_time_max_seconds < wait_time_min_seconds:
            wait_time_max_seconds = wait_time_min_seconds
        self.runtime.tools = RuntimeToolsConfig(
            root_tools=_as_str_list(payload.get("root_tools"), current.root_tools),
            worker_tools=_as_str_list(payload.get("worker_tools"), current.worker_tools),
            list_default_limit=_as_int(payload.get("list_default_limit"), current.list_default_limit, minimum=1),
            list_max_limit=_as_int(payload.get("list_max_limit"), current.list_max_limit, minimum=1),
            shell_timeout_seconds=_as_float(
                payload.get("shell_timeout_seconds"),
                current.shell_timeout_seconds,
                minimum=1.0,
            ),
            wait_run_timeout_seconds=_as_float(
                payload.get("wait_run_timeout_seconds"),
                current.wait_run_timeout_seconds,
                minimum=0.0,
            ),
            shell_inline_wait_seconds=_as_float(
                payload.get("shell_inline_wait_seconds"),
                current.shell_inline_wait_seconds,
                minimum=0.0,
            ),
            wait_time_min_seconds=wait_time_min_seconds,
            wait_time_max_seconds=wait_time_max_seconds,
        )

    def _merge_extensions(self, payload: dict[str, Any]) -> None:
        """Merge ``[extensions]`` section."""
        if not payload:
            return
        self.extensions = DeferredExtensionsConfig(
            sandbox_enabled=bool(payload.get("sandbox_enabled", self.extensions.sandbox_enabled)),
            mcp_enabled=bool(payload.get("mcp_enabled", self.extensions.mcp_enabled)),
            skills_enabled=bool(payload.get("skills_enabled", self.extensions.skills_enabled)),
        )

    def as_snapshot(self) -> dict[str, Any]:
        """Convert config into snapshot-ready plain dictionary."""
        return asdict(self)


def _merge_profile(profile: ProviderProfileConfig, payload: dict[str, Any]) -> ProviderProfileConfig:
    """Merge profile payload with strict type normalization."""
    return ProviderProfileConfig(
        base_url=str(payload.get("base_url", profile.base_url)).strip() or profile.base_url,
        api_key_env=str(payload.get("api_key_env", profile.api_key_env)).strip() or profile.api_key_env,
        model=str(payload.get("model", profile.model)).strip() or profile.model,
        timeout_seconds=_as_int(payload.get("timeout_seconds"), profile.timeout_seconds, minimum=1),
        max_retries=_as_int(payload.get("max_retries"), profile.max_retries, minimum=0),
        retry_backoff_seconds=_as_float(
            payload.get("retry_backoff_seconds"),
            profile.retry_backoff_seconds,
            minimum=0.0,
        ),
        temperature=_as_float(payload.get("temperature"), profile.temperature, minimum=0.0, maximum=2.0),
        max_tokens=_as_int(payload.get("max_tokens"), profile.max_tokens, minimum=1),
        headers=_as_str_dict(payload.get("headers"), profile.headers),
    )


def _as_dict(value: Any) -> dict[str, Any]:
    """Return dictionary value or an empty dictionary fallback."""
    return value if isinstance(value, dict) else {}


def _as_int(value: Any, fallback: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Parse integer with optional clamp range and fallback."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(fallback)
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_float(value: Any, fallback: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """Parse float with optional clamp range and fallback."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_str_list(value: Any, fallback: list[str]) -> list[str]:
    """Normalize non-empty string list, or fallback copy."""
    if not isinstance(value, list):
        return list(fallback)
    normalized = [str(item).strip() for item in value if str(item).strip()]
    return normalized or list(fallback)


def _as_str_dict(value: Any, fallback: dict[str, str]) -> dict[str, str]:
    """Normalize mapping keys to non-empty strings."""
    if not isinstance(value, dict):
        return dict(fallback)
    return {
        str(key).strip(): str(entry)
        for key, entry in value.items()
        if str(key).strip()
    }

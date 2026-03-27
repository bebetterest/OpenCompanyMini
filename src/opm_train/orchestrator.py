"""Core runtime orchestration loop for sessions, agents, and tool executions."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from opm_train.config import OPMTrainConfig
from opm_train.context import ContextAssembler
from opm_train.extensions import ExtensionRegistry, build_default_extensions
from opm_train.llm import OpenAICompatibleClient
from opm_train.loop_hooks import LoopHooks
from opm_train.models import AgentNode, RunSession, SnapshotState, ToolRun
from opm_train.orchestrator_agents import OrchestratorAgentLifecycleMixin
from opm_train.orchestrator_session import OrchestratorSessionLifecycleMixin
from opm_train.orchestrator_telemetry import OrchestratorTelemetryMixin
from opm_train.orchestrator_tools import OrchestratorToolingMixin
from opm_train.prompts import PromptLibrary, default_prompts_dir
from opm_train.storage import SessionStorage, agent_to_dict, session_to_dict, tool_run_to_dict
from opm_train.utils import utc_now


def default_app_dir(start: Path | None = None) -> Path:
    """Locate app directory containing ``opm_train.toml`` and ``prompts``."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "opm_train.toml").exists() and (candidate / "prompts").is_dir():
            return candidate
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "opm_train.toml").exists() and (candidate / "prompts").is_dir():
            return candidate
    return cwd


class RuntimeOrchestrator(
    OrchestratorTelemetryMixin,
    OrchestratorSessionLifecycleMixin,
    OrchestratorAgentLifecycleMixin,
    OrchestratorToolingMixin,
):
    """Stateful runtime orchestrator coordinating root/worker agent execution."""

    def __init__(
        self,
        *,
        project_dir: Path,
        app_dir: Path | None = None,
        hooks: LoopHooks | None = None,
        llm_client: OpenAICompatibleClient | Any | None = None,
        model_override: str | None = None,
        timer_enabled: bool = False,
    ) -> None:
        """Initialize runtime services, configuration, and in-memory trackers."""
        self.app_dir = (app_dir or default_app_dir()).resolve()
        self.project_dir = project_dir.resolve()
        self.config = OPMTrainConfig.load(self.app_dir)
        self.storage = SessionStorage(app_dir=self.app_dir, data_dir_name=self.config.project.data_dir)
        self.prompt_library = PromptLibrary(self._resolve_prompts_dir())
        self.context_assembler = ContextAssembler(config=self.config, prompt_library=self.prompt_library)
        self.hooks = hooks or self._default_hooks()
        self.extensions: ExtensionRegistry = build_default_extensions()
        self.llm_client = llm_client or self._build_llm_client()
        self.model_override = str(model_override or "").strip() or None
        self.timer_enabled = bool(timer_enabled)

        self.session: RunSession | None = None
        self.agents: dict[str, AgentNode] = {}
        self.tool_runs: dict[str, ToolRun] = {}
        self.agent_tasks: dict[str, asyncio.Task[Any]] = {}
        self.tool_tasks: dict[str, asyncio.Task[Any]] = {}
        self.tool_run_events: dict[str, asyncio.Event] = {}
        self.spawn_run_by_child_agent: dict[str, str] = {}
        self.pending_steers: dict[str, list[str]] = {}
        self.active_turns: dict[str, dict[str, Any]] = {}
        self.event_seq = 0
        self._agent_created_index = 0

    @staticmethod
    def _default_hooks() -> LoopHooks:
        """Lazily construct default loop hooks to avoid import cycles in mixin modules."""
        from opm_train.loop_hooks import DefaultLoopHooks

        return DefaultLoopHooks()

    def set_provider_profile(self, profile: str) -> None:
        """Switch provider profile and rebuild LLM client."""
        self.config.provider.profile = str(profile).strip() or self.config.provider.profile
        self.llm_client = self._build_llm_client()

    def _resolve_prompts_dir(self) -> Path:
        """Resolve runtime prompt directory with local-first fallback."""
        local_dir = self.app_dir / "prompts"
        if local_dir.is_dir():
            return local_dir
        packaged_dir = default_prompts_dir()
        if packaged_dir.is_dir():
            return packaged_dir
        raise RuntimeError(
            "Could not find prompts directory. "
            "Expected <app_dir>/prompts or packaged prompts assets."
        )

    def _build_llm_client(self) -> OpenAICompatibleClient:
        """Construct OpenAI-compatible client from active provider profile."""
        profile = self.config.provider.active_profile()
        if not profile.base_url.strip():
            raise RuntimeError("Provider base_url is required")
        if not profile.api_key:
            raise RuntimeError(
                f"Missing API key in env '{profile.api_key_env}'. "
                "Set the environment variable before running opm-train."
            )
        return OpenAICompatibleClient(
            base_url=profile.base_url,
            api_key=profile.api_key,
            timeout_seconds=profile.timeout_seconds,
            max_retries=profile.max_retries,
            retry_backoff_seconds=profile.retry_backoff_seconds,
            headers=profile.headers,
        )

    def _persist_snapshot(self) -> None:
        """Persist scheduler snapshot for deterministic resume support."""
        if self.session is None:
            return
        with self._timer_scope("persist_snapshot"):
            snapshot = SnapshotState(
                schema_version=4,
                last_event_seq=self.event_seq,
                session=session_to_dict(self.session),
                agents={agent_id: agent_to_dict(agent) for agent_id, agent in self.agents.items()},
                tool_runs={run_id: tool_run_to_dict(run) for run_id, run in self.tool_runs.items()},
            )
            self.storage.write_snapshot(self.session.id, snapshot)

    def _reset_runtime_trackers(self) -> None:
        """Reset transient in-memory task and waiter maps."""
        self.agent_tasks = {}
        self.tool_tasks = {}
        self.tool_run_events = {}
        self.spawn_run_by_child_agent = {}
        self.pending_steers = {}
        self.active_turns = {}

    @staticmethod
    def _new_id(prefix: str) -> str:
        """Generate short random identifier with stable prefix."""
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

"""Agent loop lifecycle mixin for runtime orchestrator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import Any

from opm_train.context import maybe_auto_compress
from opm_train.loop import ActionBatchResult, AgentLoopRunner
from opm_train.loop_hooks import LoopContext
from opm_train.models import AgentNode, AgentRole, AgentStatus, ToolRun
from opm_train.protocol import ProtocolError, extract_json_object, normalize_actions, normalize_tool_calls
from opm_train.tools import TERMINAL_AGENT_STATUSES, TERMINAL_TOOL_RUN_STATUSES, validate_finish_action
from opm_train.utils import json_ready, utc_now

_INVALID_MODEL_PAYLOAD_SUMMARY = "Model returned invalid action payload."


@dataclass(frozen=True, slots=True)
class ProtocolRetryPolicy:
    """Runtime policy for protocol parsing retries."""

    max_retries: int
    max_attempts: int
    backoff_seconds: float


class OrchestratorAgentLifecycleMixin:
    """Attach agent-loop execution, finish handling, and lineage control."""

    def _launch_agent(self, agent_id: str) -> None:
        """Launch one agent task if not already running."""
        if agent_id in self.agent_tasks and not self.agent_tasks[agent_id].done():
            return
        task = asyncio.create_task(self._run_agent(agent_id), name=f"agent:{agent_id}")
        self.agent_tasks[agent_id] = task

    async def _run_agent(self, agent_id: str) -> None:
        """Run one agent loop to terminal completion with safety guards."""
        agent = self.agents[agent_id]
        agent.status = AgentStatus.RUNNING
        self._persist_snapshot()
        self._log_event(agent, "agent_started", {"role": agent.role.value})
        max_steps = (
            self.config.runtime.limits.max_root_steps
            if agent.role == AgentRole.ROOT
            else self.config.runtime.limits.max_agent_steps
        )
        runner = AgentLoopRunner(max_steps=max_steps, hooks=self.hooks)
        try:
            with self._timer_scope("agent_loop", agent=agent, payload={"max_steps": max_steps}):
                result = await runner.run(
                    agent=agent,
                    ask_agent=self._ask_agent,
                    execute_actions=self._execute_actions,
                    request_forced_finish=self._forced_finish,
                    interrupted=lambda: agent.status.value in TERMINAL_AGENT_STATUSES,
                )
            if result.finish_payload is None:
                return
            self._apply_finish_payload(agent=agent, payload=result.finish_payload)
        except Exception as exc:  # pragma: no cover - safety net
            self._record_agent_exception(agent=agent, exc=exc)
        finally:
            if agent.status.value not in TERMINAL_AGENT_STATUSES:
                agent.status = AgentStatus.FAILED
                agent.status_reason = "agent_finished_without_terminal_status"
            self._log_event(
                agent,
                "agent_finished",
                {
                    "status": agent.status.value,
                    "summary": agent.summary,
                },
            )
            self._persist_snapshot()
            self._complete_spawn_run_for_child(agent)

    def _record_agent_exception(self, *, agent: AgentNode, exc: Exception) -> None:
        """Persist one agent exception into lifecycle state and metadata."""
        error_type = type(exc).__name__
        error_message = str(exc).strip() or "<empty>"
        error_payload = {
            "type": error_type,
            "message": error_message,
        }
        if not isinstance(agent.metadata, dict):
            agent.metadata = {}
        agent.metadata["last_error"] = error_payload
        agent.status = AgentStatus.FAILED
        agent.status_reason = f"agent_loop_error:{error_type}:{error_message}"
        if not str(agent.summary or "").strip():
            agent.summary = f"{error_type}: {error_message}"
        self._record_exception(stage="agent_loop", exc=exc, agent=agent, payload={"error": error_payload})
        self._log_event(agent, "agent_failed", {"error": error_payload})

    async def _ask_agent(self, agent: AgentNode) -> list[dict[str, Any]]:
        """Assemble context, call model, and normalize response into actions."""
        with self._timer_scope("ask_agent", agent=agent):
            self._maybe_record_auto_compression(agent)
            self._apply_pending_steers(agent)

            profile = self.config.provider.active_profile()
            model = self.model_override or profile.model
            inference_metadata = self._inference_metadata(profile=profile, model=model)
            with self._timer_scope("prompt_primitives", agent=agent):
                system_prompt = self.context_assembler.system_prompt(agent)
                tools = self.context_assembler.tools(agent)
            policy = self._protocol_retry_policy()

            for attempt in range(policy.max_attempts):
                with self._timer_scope(
                    "prompt_assembly",
                    agent=agent,
                    payload={"protocol_attempt": attempt + 1, "protocol_max_attempts": policy.max_attempts},
                ):
                    request_messages = self.context_assembler.messages(agent, system_prompt=system_prompt)
                self._log_event(
                    agent,
                    "agent_prompt",
                    {
                        "model": model,
                        "inference_provider": inference_metadata["inference_provider"],
                        "inference_parameters": inference_metadata["inference_parameters"],
                        "message_count": len(request_messages),
                        "tool_count": len(tools),
                        "protocol_attempt": attempt + 1,
                        "protocol_max_attempts": policy.max_attempts,
                    },
                )
                llm_sequence = self._record_llm_call_request(
                    agent=agent,
                    payload={
                        "timestamp": utc_now(),
                        "protocol_attempt": attempt + 1,
                        "protocol_max_attempts": policy.max_attempts,
                        **inference_metadata,
                        "tool_count": len(tools),
                        "message_count": len(request_messages),
                        "messages": json_ready(request_messages),
                        "tools": json_ready(tools),
                    },
                )
                on_token, on_reasoning, on_retry = self._llm_callbacks(agent)

                try:
                    with self._timer_scope(
                        "llm_call",
                        agent=agent,
                        payload={"protocol_attempt": attempt + 1, "model": model},
                    ):
                        result = await self.llm_client.stream_chat(
                            model=model,
                            messages=request_messages,
                            temperature=profile.temperature,
                            max_tokens=profile.max_tokens,
                            tools=tools,
                            tool_choice="auto",
                            parallel_tool_calls=True,
                            on_token=on_token,
                            on_reasoning=on_reasoning,
                            on_retry=on_retry,
                        )
                except Exception as exc:
                    self._record_llm_call_response(
                        agent=agent,
                        sequence=llm_sequence,
                        payload={
                            "timestamp": utc_now(),
                            **inference_metadata,
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    self._record_exception(
                        stage="llm_call",
                        exc=exc,
                        agent=agent,
                        payload={
                            "protocol_attempt": attempt + 1,
                            "protocol_max_attempts": policy.max_attempts,
                            "sequence": llm_sequence,
                        },
                    )
                    raise

                assistant_message = {
                    "role": "assistant",
                    "content": result.content,
                    "reasoning": result.reasoning,
                    "tool_calls": result.tool_calls,
                }
                agent.conversation.append(assistant_message)
                response_payload = {
                    "timestamp": utc_now(),
                    "ok": True,
                    "protocol_attempt": attempt + 1,
                    "protocol_max_attempts": policy.max_attempts,
                    **inference_metadata,
                    "content": result.content,
                    "reasoning": result.reasoning,
                    "tool_calls": json_ready(result.tool_calls),
                    "usage": json_ready(result.usage),
                    "raw_events": json_ready(result.raw_events),
                }
                try:
                    with self._timer_scope(
                        "model_parse",
                        agent=agent,
                        payload={"protocol_attempt": attempt + 1, "sequence": llm_sequence},
                    ):
                        if result.tool_calls:
                            actions = normalize_tool_calls(result.tool_calls)
                        else:
                            payload = extract_json_object(result.content)
                            actions = normalize_actions(payload)
                    response_payload["actions"] = json_ready(actions)
                    self._record_llm_call_response(agent=agent, sequence=llm_sequence, payload=response_payload)
                    return actions
                except ProtocolError as exc:
                    response_payload["ok"] = False
                    response_payload["parse_error"] = str(exc)
                    self._record_llm_call_response(agent=agent, sequence=llm_sequence, payload=response_payload)
                    self._log_event(
                        agent,
                        "invalid_model_response",
                        {
                            "error": str(exc),
                            "content": result.content,
                            "protocol_attempt": attempt + 1,
                            "protocol_max_attempts": policy.max_attempts,
                        },
                    )
                    if attempt >= policy.max_retries:
                        return self._invalid_model_response_actions(agent)
                    retry_message = self._protocol_retry_message(
                        error=str(exc),
                        next_attempt=attempt + 2,
                        max_attempts=policy.max_attempts,
                    )
                    agent.conversation.append({"role": "user", "content": retry_message})
                    if policy.backoff_seconds > 0:
                        await asyncio.sleep(policy.backoff_seconds * (attempt + 1))
                    continue
            return self._invalid_model_response_actions(agent)

    def _inference_metadata(self, *, profile: Any, model: str) -> dict[str, Any]:
        """Build canonical inference metadata persisted in LLM request/response artifacts."""
        provider_name = str(self.config.provider.profile or "openrouter").strip().lower() or "openrouter"
        inference_parameters = {
            "temperature": float(profile.temperature),
            "max_tokens": int(profile.max_tokens),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        return {
            "inference_provider": provider_name,
            "inference_endpoint": str(profile.base_url),
            "inference_model": str(model),
            "inference_api_key_env": str(profile.api_key_env),
            "inference_parameters": inference_parameters,
            # Backward-compat mirror fields for old artifact readers.
            "model": str(model),
            "temperature": float(profile.temperature),
            "max_tokens": int(profile.max_tokens),
        }

    def _protocol_retry_policy(self) -> ProtocolRetryPolicy:
        """Resolve protocol retry policy from runtime limits."""
        max_retries = max(0, int(self.config.runtime.limits.max_protocol_retries))
        backoff_seconds = max(0.0, float(self.config.runtime.limits.protocol_retry_backoff_seconds))
        return ProtocolRetryPolicy(
            max_retries=max_retries,
            max_attempts=max_retries + 1,
            backoff_seconds=backoff_seconds,
        )

    def _llm_callbacks(self, agent: AgentNode) -> tuple[Any, Any, Any]:
        """Build LLM stream callbacks bound to one agent."""

        async def on_token(token: str) -> None:
            self._log_event(agent, "llm_token", {"token": token})

        async def on_reasoning(token: str) -> None:
            self._log_event(agent, "llm_reasoning", {"token": token})

        async def on_retry(payload: dict[str, Any]) -> None:
            self._log_event(agent, "llm_retry", payload)

        return on_token, on_reasoning, on_retry

    def _maybe_record_auto_compression(self, agent: AgentNode) -> None:
        """Trigger auto-compress and persist actual compression outcome."""
        metadata = agent.metadata if isinstance(agent.metadata, dict) else {}
        summary_version_before = int(metadata.get("summary_version", 0) or 0)
        triggered = maybe_auto_compress(agent=agent, config=self.config)
        if not triggered:
            return
        metadata_after = agent.metadata if isinstance(agent.metadata, dict) else {}
        summary_version_after = int(metadata_after.get("summary_version", 0) or 0)
        compression_result = {
            "compressed": summary_version_after > summary_version_before,
            "summary_version": summary_version_after,
            "summarized_until_message_index": int(metadata_after.get("summarized_until_message_index", -1) or -1),
        }
        self._record_context_compression(agent=agent, reason="auto_threshold", result=compression_result)
        self._log_event(agent, "context_compressed", {"reason": "auto_threshold", **compression_result})

    def _protocol_retry_message(self, *, error: str, next_attempt: int, max_attempts: int) -> str:
        """Render protocol retry prompt with safe fallback when key is absent."""
        try:
            return self.prompt_library.render_runtime_message(
                "protocol_retry_message",
                error=error,
                next_attempt=next_attempt,
                max_attempts=max_attempts,
            )
        except KeyError:
            return (
                "Previous response could not be parsed into valid actions JSON. "
                f"Error: {error}. Please reply with valid actions JSON only "
                f"(retry {next_attempt}/{max_attempts})."
            )

    def _apply_pending_steers(self, agent: AgentNode) -> None:
        """Inject queued steering messages into agent conversation."""
        for pending in self.pending_steers.pop(agent.id, []):
            agent.conversation.append(
                {
                    "role": "user",
                    "content": self.prompt_library.render_runtime_message(
                        "steer_message",
                        content=pending,
                    ),
                }
            )

    def _invalid_model_response_actions(self, agent: AgentNode) -> list[dict[str, Any]]:
        """Return deterministic fallback finish action for invalid model payload."""
        if agent.role == AgentRole.WORKER:
            return [
                {
                    "type": "finish",
                    "status": "failed",
                    "summary": _INVALID_MODEL_PAYLOAD_SUMMARY,
                    "next_recommendation": "Retry with a valid JSON actions object.",
                }
            ]
        return [
            {
                "type": "finish",
                "status": "partial",
                "summary": _INVALID_MODEL_PAYLOAD_SUMMARY,
            }
        ]

    async def _execute_actions(
        self,
        agent: AgentNode,
        actions: list[dict[str, Any]],
        context: LoopContext,
    ) -> ActionBatchResult:
        """Execute action batch sequentially until optional finish payload appears."""
        with self._timer_scope(
            "execute_actions_batch",
            agent=agent,
            payload={"step": context.step, "action_count": len(actions)},
        ):
            finish_payload: dict[str, Any] | None = None
            for action in actions:
                await self.hooks.before_action(agent=agent, context=context, action=action)
                result = await self._execute_action(agent=agent, action=action)
                agent.conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(action.get("_tool_call_id", "")),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                await self.hooks.after_action(agent=agent, context=context, action=action, result=result)
                if isinstance(result.get("finish_payload"), dict):
                    finish_payload = dict(result["finish_payload"])
                    break
            self._persist_snapshot()
            return ActionBatchResult(finish_payload=finish_payload)

    async def _forced_finish(self, agent: AgentNode) -> dict[str, Any] | None:
        """Provide deterministic finish payload when step budget is exhausted."""
        if agent.role == AgentRole.ROOT:
            return {
                "status": "partial",
                "summary": "Root hit max step budget and produced forced partial finish.",
            }
        return {
            "status": "failed",
            "summary": "Worker hit max step budget and stopped.",
            "next_recommendation": "Split the task into smaller chunks or increase max_agent_steps.",
        }

    async def _execute_action(self, *, agent: AgentNode, action: dict[str, Any]) -> dict[str, Any]:
        """Execute one validated action and persist tool-run lifecycle."""
        action_type = str(action.get("type", "")).strip()
        if not action_type:
            raise ValueError("action type is required")
        if action_type not in set(self.config.runtime.tools.tool_names_for_role(agent.role.value)):
            raise ValueError(f"tool '{action_type}' is not enabled for role {agent.role.value}")

        if action_type == "finish":
            return self._handle_finish_action(agent=agent, action=action)
        return await self._execute_tool_action(agent=agent, action=action)

    def _handle_finish_action(self, *, agent: AgentNode, action: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize finish action into loop return payload."""
        error = validate_finish_action(agent.role, action)
        if error:
            raise ValueError(error)
        if self._has_unfinished_children(agent):
            raise ValueError("cannot finish while child agents are still active")
        unfinished_runs = self._unfinished_tool_runs_for_agent(agent)
        if unfinished_runs:
            return self._finish_rejected_result(agent=agent, unfinished_runs=unfinished_runs)
        payload = {
            "status": str(action.get("status", "")).strip().lower(),
            "summary": str(action.get("summary", "")).strip(),
        }
        if agent.role == AgentRole.WORKER:
            payload["next_recommendation"] = str(action.get("next_recommendation", "")).strip()
        return {"ok": True, "finish_payload": payload}

    def _finish_rejected_result(self, *, agent: AgentNode, unfinished_runs: list[dict[str, Any]]) -> dict[str, Any]:
        """Build structured finish-rejected payload and corresponding event log."""
        result = {
            "ok": False,
            "finish_rejected": True,
            "error": "cannot finish while own tool runs are still active",
            "unfinished_tool_runs": unfinished_runs,
        }
        self._log_event(
            agent,
            "finish_rejected",
            {
                "reason": "unfinished_tool_runs",
                "unfinished_tool_runs": unfinished_runs,
            },
        )
        return result

    def _apply_finish_payload(self, *, agent: AgentNode, payload: dict[str, Any]) -> None:
        """Translate finish payload into canonical agent terminal fields."""
        status = str(payload.get("status", "")).strip().lower()
        summary = str(payload.get("summary", "")).strip()
        next_reco = str(payload.get("next_recommendation", "")).strip()

        if status in {"completed", "partial"}:
            agent.status = AgentStatus.COMPLETED
        elif status == "failed":
            agent.status = AgentStatus.FAILED
        else:
            agent.status = AgentStatus.FAILED
            agent.status_reason = f"invalid_finish_status:{status}"

        agent.summary = summary
        if next_reco:
            agent.next_recommendation = next_reco
        agent.status_reason = agent.status_reason or f"finish:{status}"

    def _has_unfinished_children(self, agent: AgentNode) -> bool:
        """Return whether any child agent is still active."""
        for child_id in agent.children:
            child = self.agents.get(child_id)
            if child is None:
                continue
            if child.status.value not in TERMINAL_AGENT_STATUSES:
                return True
        return False

    def _unfinished_tool_runs_for_agent(self, agent: AgentNode) -> list[dict[str, Any]]:
        """Return non-terminal tool runs owned by this agent only."""
        pending: list[dict[str, Any]] = []
        for run in self.tool_runs.values():
            if run.agent_id != agent.id:
                continue
            if run.status.value in TERMINAL_TOOL_RUN_STATUSES:
                continue
            pending.append(self._unfinished_tool_run_item(run))
        pending.sort(key=lambda item: str(item.get("created_at", "")))
        return pending

    @staticmethod
    def _unfinished_tool_run_item(run: ToolRun) -> dict[str, Any]:
        """Convert one active tool run object into finish-rejection payload shape."""
        return {
            "run_id": run.id,
            "tool_name": run.tool_name,
            "status": run.status.value,
            "blocking": run.blocking,
            "created_at": run.created_at,
        }

    def _cancel_agent_tree(self, agent_id: str, *, reason: str) -> list[str]:
        """Cancel target agent and descendants, returning cancelled IDs."""
        if agent_id not in self.agents:
            return []
        cancelled: list[str] = []

        def walk(current_id: str) -> None:
            """Depth-first cancellation traversal over agent lineage tree."""
            agent = self.agents.get(current_id)
            if agent is None:
                return
            for child_id in list(agent.children):
                walk(child_id)
            if agent.status.value not in TERMINAL_AGENT_STATUSES:
                agent.status = AgentStatus.CANCELLED
                agent.status_reason = reason
                cancelled.append(agent.id)
            task = self.agent_tasks.get(agent.id)
            if task is not None and not task.done():
                task.cancel()
            spawn_run_id = self.spawn_run_by_child_agent.get(agent.id)
            if spawn_run_id:
                run = self.tool_runs.get(spawn_run_id)
                if run and run.status.value not in TERMINAL_TOOL_RUN_STATUSES:
                    self._cancel_tool_run_obj(run, reason=f"child_cancelled:{agent.id}")

        walk(agent_id)
        return cancelled

    def _complete_spawn_run_for_child(self, child: AgentNode) -> None:
        """Resolve parent spawn tool run once child reaches terminal state."""
        run_id = self.spawn_run_by_child_agent.get(child.id)
        if not run_id:
            return
        run = self.tool_runs.get(run_id)
        if run is None or run.status.value in TERMINAL_TOOL_RUN_STATUSES:
            return
        result = {
            "child_agent_id": child.id,
            "status": child.status.value,
            "summary": child.summary,
            "next_recommendation": child.next_recommendation,
        }
        if child.status == AgentStatus.CANCELLED:
            self._cancel_tool_run_obj(run, reason=f"child_cancelled:{child.id}")
        elif child.status == AgentStatus.FAILED:
            self._fail_tool_run(run, error=f"child_failed:{child.id}")
            run.result = result
        else:
            self._complete_tool_run(run, result=result)

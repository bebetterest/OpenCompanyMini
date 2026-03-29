"""Agent-control tool handler mixin."""

from __future__ import annotations

from typing import Any

from opm_train.models import AgentNode, AgentRole, AgentStatus, ToolRun
from opm_train.tools import TERMINAL_AGENT_STATUSES


class AgentToolMixin:
    """Attach spawn/steer/cancel tool handlers."""

    agents: dict[str, AgentNode]
    spawn_run_by_child_agent: dict[str, str]

    async def _tool_spawn_agent(self, run: ToolRun, agent: AgentNode, action: dict[str, Any]) -> dict[str, Any]:
        """Create child worker agent and return child/tool run identifiers."""
        try:
            self._enforce_spawn_capacity(agent)
        except ValueError as exc:
            return self._spawn_rejected_result(run=run, agent=agent, error_message=str(exc))
        instruction, name = self._spawn_request(action)
        child = self._build_child_agent(parent=agent, instruction=instruction, name=name)
        self._register_spawned_child(parent=agent, child=child, run=run)
        self._log_event(agent, "agent_spawned", {"child_agent_id": child.id, "instruction": instruction})
        self._launch_agent(child.id)
        return self._spawn_running_result(run=run, child=child)

    async def _tool_steer_agent(self, run: ToolRun, action: dict[str, Any]) -> dict[str, Any]:
        """Queue steering message for target agent's next model turn."""
        target_id = str(action.get("agent_id", "")).strip()
        content = str(action.get("content", "")).strip()
        if not target_id or not content:
            raise ValueError("steer_agent requires agent_id and content")
        if target_id not in self.agents:
            raise ValueError(f"unknown agent_id: {target_id}")
        self.pending_steers.setdefault(target_id, []).append(content)
        result = {
            "steer_agent_status": True,
            "target_agent_id": target_id,
            "status": "waiting",
            "steer_run_id": self._new_id("steerrun"),
        }
        self._complete_tool_run(run, result=result)
        return result

    async def _tool_cancel_agent(self, run: ToolRun, action: dict[str, Any]) -> dict[str, Any]:
        """Cancel target agent subtree recursively."""
        target_id = str(action.get("agent_id", "")).strip()
        recursive = bool(action.get("recursive", True))
        if not target_id:
            raise ValueError("cancel_agent requires agent_id")
        requester = self.agents.get(run.agent_id)
        if requester is not None and target_id == requester.id:
            result = {
                "cancel_agent_status": False,
                "error": "cancel_agent cannot target the current agent itself.",
            }
            self._complete_tool_run(run, result=result)
            return result
        if requester is not None and not self._is_descendant(requester.id, target_id):
            result = {
                "cancel_agent_status": False,
                "error": f"Cannot cancel agent {target_id}.",
            }
            self._complete_tool_run(run, result=result)
            return result
        if target_id not in self.agents:
            result = {
                "cancel_agent_status": False,
                "error": f"Cannot cancel agent {target_id}.",
            }
            self._complete_tool_run(run, result=result)
            return result
        cancelled = self._cancel_agent_tree(target_id, reason="cancel_agent_tool", recursive=recursive)
        result = {"cancel_agent_status": bool(cancelled)}
        self._complete_tool_run(run, result=result)
        return result

    def _is_descendant(self, parent_id: str, target_id: str) -> bool:
        """Return whether target_id is inside parent_id's descendant tree."""
        if not parent_id or not target_id or parent_id == target_id:
            return False
        parent = self.agents.get(parent_id)
        if parent is None:
            return False
        stack = list(parent.children)
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current == target_id:
                return True
            node = self.agents.get(current)
            if node is None:
                continue
            stack.extend(node.children)
        return False

    def _enforce_spawn_capacity(self, agent: AgentNode) -> None:
        """Validate per-parent and global active-agent spawn limits."""
        if len(agent.children) >= self.config.runtime.limits.max_children_per_agent:
            raise ValueError("max_children_per_agent limit reached")
        if self._active_agent_count() >= self.config.runtime.limits.max_active_agents:
            raise ValueError("max_active_agents limit reached")

    def _spawn_rejected_result(self, *, run: ToolRun, agent: AgentNode, error_message: str) -> dict[str, Any]:
        """Return one structured non-throwing spawn rejection payload."""
        code = "spawn_capacity_limited"
        message = str(error_message).strip() or "spawn_agent rejected due to runtime capacity limits"
        if "max_children_per_agent" in message:
            code = "max_children_per_agent_limit_reached"
        elif "max_active_agents" in message:
            code = "max_active_agents_limit_reached"
        result = {
            "status": "rejected",
            "child_agent_id": None,
            "tool_run_id": run.id,
            "error": message,
            "error_code": code,
            "capacity": {
                "active_agents": self._active_agent_count(),
                "max_active_agents": int(self.config.runtime.limits.max_active_agents),
                "children_of_parent": len(agent.children),
                "max_children_per_agent": int(self.config.runtime.limits.max_children_per_agent),
            },
        }
        self._log_event(
            agent,
            "spawn_rejected",
            {
                "tool_run_id": run.id,
                "reason": {"code": code, "message": message},
                "capacity": result["capacity"],
            },
        )
        # Keep direct handler calls deterministic; _execute_tool_action will skip re-completion.
        self._complete_tool_run(run, result=result)
        return result

    def _active_agent_count(self) -> int:
        """Count agents that have not yet reached a terminal state."""
        return len([agent for agent in self.agents.values() if agent.status.value not in TERMINAL_AGENT_STATUSES])

    @staticmethod
    def _spawn_request(action: dict[str, Any]) -> tuple[str, str]:
        """Extract validated child instruction and optional display name."""
        instruction = str(action.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("spawn_agent requires non-empty instruction")
        name = str(action.get("name", "")).strip() or "Worker"
        return instruction, name

    def _build_child_agent(self, *, parent: AgentNode, instruction: str, name: str) -> AgentNode:
        """Construct one worker agent inheriting parent workspace and runtime metadata."""
        child_id = self._new_id("agent")
        self._agent_created_index += 1
        return AgentNode(
            id=child_id,
            session_id=parent.session_id,
            name=name,
            role=AgentRole.WORKER,
            instruction=instruction,
            workspace_path=parent.workspace_path,
            parent_agent_id=parent.id,
            status=AgentStatus.PENDING,
            metadata={
                "created_index": self._agent_created_index,
                "keep_pinned_messages": self.config.runtime.context.keep_pinned_messages,
            },
            conversation=[
                {
                    "role": "user",
                    "content": self.prompt_library.render_runtime_message(
                        "worker_initial_message",
                        instruction=instruction,
                        workspace_path=str(parent.workspace_path),
                        parent_agent_id=parent.id,
                    ),
                }
            ],
        )

    def _register_spawned_child(self, *, parent: AgentNode, child: AgentNode, run: ToolRun) -> None:
        """Attach child agent to lineage maps and spawn-run bookkeeping."""
        self.agents[child.id] = child
        parent.children.append(child.id)
        self.spawn_run_by_child_agent[child.id] = run.id
        run.arguments = {
            **dict(run.arguments),
            "child_agent_id": child.id,
        }

    @staticmethod
    def _spawn_running_result(*, run: ToolRun, child: AgentNode) -> dict[str, Any]:
        """Return OpenCompany-style spawn payload."""
        return {
            "child_agent_id": child.id,
            "tool_run_id": run.id,
        }

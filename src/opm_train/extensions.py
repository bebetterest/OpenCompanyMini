"""Deferred extension adapters kept as explicit v0 integration points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SandboxAdapter:
    """Placeholder sandbox adapter retained for future integration."""

    enabled: bool = False

    async def run(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """Reject sandbox calls because sandbox integration is deferred."""
        raise RuntimeError("sandbox adapter is deferred in v0")


@dataclass(slots=True)
class McpAdapter:
    """Placeholder MCP adapter retained for future integration."""

    enabled: bool = False

    async def list_servers(self) -> list[dict[str, Any]]:
        """Return no servers in v0."""
        return []


@dataclass(slots=True)
class SkillsAdapter:
    """Placeholder skills adapter retained for future integration."""

    enabled: bool = False

    def selected_skills(self) -> list[str]:
        """Return no skills in v0."""
        return []


@dataclass(slots=True)
class ExtensionRegistry:
    """Container for all deferred extension adapters."""

    sandbox: SandboxAdapter
    mcp: McpAdapter
    skills: SkillsAdapter


def build_default_extensions() -> ExtensionRegistry:
    """Construct a disabled extension registry for v0 runtime."""
    return ExtensionRegistry(
        sandbox=SandboxAdapter(),
        mcp=McpAdapter(),
        skills=SkillsAdapter(),
    )

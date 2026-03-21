"""Prompt asset loader with caching and safe copy-on-read behavior."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


_AGENT_PROMPTS = {
    "root": "root_coordinator.md",
    "worker": "worker.md",
}


def default_prompts_dir() -> Path:
    """Resolve prompts directory for both source-tree and installed layouts."""
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[2] / "prompts",  # repository layout
        module_path.parent / "prompt_assets",  # packaged assets layout
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


class PromptLibrary:
    """English runtime prompt loader with mirrored assets in repository."""

    def __init__(self, prompts_dir: Path) -> None:
        """Initialize prompt loader and in-memory caches."""
        self.prompts_dir = prompts_dir
        self._cache_text: dict[str, str] = {}
        self._cache_json: dict[str, dict[str, Any]] = {}

    def load_agent_prompt(self, role: str) -> str:
        """Load agent system prompt template for the requested role."""
        filename = _AGENT_PROMPTS[str(role)]
        return self._load_text(filename)

    def load_runtime_messages(self) -> dict[str, str]:
        """Load runtime message templates used for dynamic inserts."""
        payload = self._load_json("runtime_messages.json")
        return {str(k): str(v) for k, v in payload.items()}

    def render_runtime_message(self, key: str, **values: Any) -> str:
        """Render one runtime message template with keyword values."""
        template = self.load_runtime_messages()[key]
        return template.format(**values)

    def load_tool_definitions(self) -> dict[str, dict[str, Any]]:
        """Load tool definitions and return deep-copied mutable payload."""
        payload = self._load_json("tool_definitions.json")
        return {
            str(name): copy.deepcopy(entry)
            for name, entry in payload.items()
            if isinstance(entry, dict)
        }

    def _load_text(self, filename: str) -> str:
        """Read and cache plain-text prompt assets."""
        cached = self._cache_text.get(filename)
        if cached is not None:
            return cached
        content = (self.prompts_dir / filename).read_text(encoding="utf-8")
        self._cache_text[filename] = content
        return content

    def _load_json(self, filename: str) -> dict[str, Any]:
        """Read and cache JSON prompt assets as object payloads."""
        cached = self._cache_json.get(filename)
        if cached is not None:
            return copy.deepcopy(cached)
        payload = json.loads((self.prompts_dir / filename).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Prompt payload {filename} must be a JSON object")
        self._cache_json[filename] = payload
        return copy.deepcopy(payload)

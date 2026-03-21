# OPM-Train Agent Principles

- Build `opm-train` as a terminal-first, modular multi-agent runtime for research and training.
- Follow the Bitter Lesson and first principles: prefer simple scalable primitives over fragile handcrafted workflows.
- Use Conda environment `OpenCompany` (Python 3.12) for development and verification.
- Keep root as organizer-first and workers execution-first.
- Preserve explicit parent/child lineage and lifecycle state for every agent.
- Keep orchestration limits explicit and configurable: fan-out, active agents, and step budgets.
- Keep tool set minimal and composable: shell, spawn/steer/cancel, run inspection/wait/cancel, context compression, finish.
- Keep the core loop isolated in `AgentLoopRunner` and extensible via `LoopHooks`.
- Keep prompts centralized under `prompts/` and runtime prompt language English-only.
- Keep bilingual mirrors (`*_cn`) for prompts/docs/README/AGENTS synchronized in structure and facts.
- Persist canonical raw trajectories in JSONL and store lightweight snapshots for resume.
- Keep snapshot-to-event-tail validation strict before resuming sessions.
- Keep `sandbox`, `mcp`, and `skills` deferred in v0, but preserve extension interfaces for future integration.
- Combine module-level tests and system tests to prevent recurring failures.
- Update docs and mirrors in the same change whenever behavior changes.

# Technical Route

## Baseline

- Greenfield runtime at repository root as `opm-train`.
- Reference `OpenCompany/` is intentionally ignored from tracked implementation.
- Terminal-first architecture with modular runtime core.

## Core v0 Direction

- Multi-agent orchestration with explicit tool calls and parallel child execution.
- Tooling layer is modularized by lifecycle and capability (session/agent/tools), and tools are dispatched via a central registry.
- OpenAI-compatible provider abstraction with built-in `openrouter`, `tinker`, and `custom` profiles.
- Context management with auto-threshold compression + manual `compress_context`.
- JSONL canonical event logging and snapshot-based resume.
- Strict snapshot-to-event-tail validation before resume (continuous sequence + tail consistency).
- Slimming A removes dormant states/fields while keeping shell inline wait budget (`shell_inline_wait_seconds`) for shell run UX.

## Deferred Modules

- `sandbox`, `mcp`, and `skills` are deferred in v0.
- No-op extension interfaces are kept stable for future versions.

## Engineering Principles

- Bitter Lesson + first principles.
- Minimal policy branches and explicit runtime state.
- Keep loop behavior override-friendly (`LoopHooks`).

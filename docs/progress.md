# Progress

## 2026-03-21

- Added finish-guard semantics for active own tool runs: `finish` now returns structured rejection payload (`finish_rejected`, `unfinished_tool_runs`) instead of failing the agent loop.
- Added `ToolRunStatus.abandoned` and changed resume behavior to mark non-restorable non-terminal tool runs as `abandoned` with explicit reason.
- Enhanced `doctor` output with tool-contract validation (`tool_contract_ok`, `tool_contract_issues`) and gated `ready_for_real_run` on contract consistency.
- Added regression tests for finish rejection + wait flow, own-vs-other agent tool-run finish checks, abandoned-on-resume behavior, and doctor mismatch reporting.
- Added `mypy` to dev dependencies, introduced baseline `[tool.mypy]` config, and documented local type-check command (`mypy src`).
- Added GitHub Actions workflow `.github/workflows/ci.yml` to run `pytest -q` and `mypy src` on push/PR.
- Simplified tool registry by replacing per-tool forwarding wrappers with declarative method-binding specs, and added executor argument-binding regression tests.
- Fixed `python -m opm_train.cli` module execution by adding `cli.py` `__main__` guard, with entrypoint regression coverage.
- Replaced awaitable checks from `hasattr(__await__)` to `inspect.isawaitable` in tool dispatch and LLM callback awaiting.
- Added per-agent artifact directories under each session:
  - `agents/<agent_id>/llm_calls/` for ordered request/response payloads.
  - `agents/<agent_id>/context_compressions/` for ordered compression records.
- Added structured runtime observability files:
  - `logs/runtime.log` and `logs/errors.jsonl`.
- Added CLI `--timer` (run/resume/smoke) and persisted per-module timings to `timers/module_timings.jsonl`.
- Added tests for artifact persistence, UTF-8/Chinese payload recording, timer output, and error-log visibility.
- Refactored telemetry methods out of `RuntimeOrchestrator` into `OrchestratorTelemetryMixin` for cleaner modular boundaries.
- Optimized per-agent artifact sequencing with lazy cache in `SessionStorage` to avoid repeated directory scans.
- `cancel_agent` now rejects unknown `agent_id` with explicit `ValueError` instead of silent no-op.
- Blocking `spawn_agent` now reports child terminal state at top-level `status` (`completed` / `failed` / `cancelled`).
- `cancel_tool_run` is kept scoped to tool-run lifecycle; cancelling a `spawn_agent` run no longer terminates an already running child agent.
- Refactored spawn/list runtime helpers to keep agent/tool orchestration paths smaller and easier to extend.
- SSE parser now supports CRLF/LF mixed framing in streamed responses.
- Added packaged prompt assets under `src/opm_train/prompt_assets` with fallback path resolution + package-data declaration.
- Added regressions for unknown-agent cancel, blocking-spawn status mapping, CRLF SSE parsing, and packaged-prompt fallback.

## 2026-03-20

- Bootstrapped greenfield `opm-train` package and CLI (`run`, `resume`).
- Added `doctor` and local deterministic `smoke` commands for direct runnable verification.
- Implemented OpenAI-compatible provider client and profile-based config.
- Implemented core orchestration loop, tool runtime, context compression, JSONL events, and snapshot resume.
- Removed redundant custom suppress helper and tightened root-task cleanup behavior.
- Slimming A: removed dormant `AgentStatus.waiting`, `ToolRun.parent_run_id`, `tool_run_owner_agent`, and `stable_json_dumps`.
- Restored `runtime.tools.shell_inline_wait_seconds` and wired shell behavior: inline return if finished within budget, otherwise background run with incremental output via `get_tool_run`.
- Snapshot writer schema version bumped to `3` for the Slimming A shape.
- Added bilingual prompts/docs/README/AGENTS mirrors.
- Added unit and integration tests for v0 core behaviors.

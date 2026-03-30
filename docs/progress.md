# Progress

## 2026-03-29

- Increased OpenRouter profile retry budget for real batch runs (`provider.openrouter.max_retries`) to better absorb transient provider-side rate limits (especially free-tier models).
- Extended `opm-train batch-run` with `--dataset openreward` mode, including OpenReward selectors/options: `--environment`, `--split`, `--task-index`, `--start`, `--stop`, `--variant`, `--base-url`, `--openreward-tool-format`, and `--max-steps`.
- Added CLI validation rules for OpenReward mode (`--environment` required, `--task-index` mutually exclusive with `--start/--stop`, `--input` still required for non-OpenReward datasets, `--max-steps` must be positive).
- Added OpenReward execution backend in `batch_runner`: task selection via `list_tasks/get_task/get_task_range`, session tool loop via `session/call_tool`, OpenAI-compatible model turns, and deterministic stop conditions (`finished`, no tool calls, step budget).
- Added provider-aware default tool format selection (`openrouter` profile -> `openrouter`; others -> `openai`) with explicit override support.
- Added OpenReward-specific artifacts and metrics:
  - `.opm_train/batches/<batch_id>/openreward_results.jsonl` (`environment/split/variant/task_key/task_index/reward_total/finished/tool_calls/turns/session_status/error`).
  - `.opm_train/batches/<batch_id>/openreward_summary.json` (`total/completed/finished/failed/total_reward/avg_reward`).
- Added OpenReward resume support with stable per-task keys (prefer `task_id`, fallback `task_index`/order) while preserving existing batch-run behavior for `gsm8k/simple_math/mixed`.
- Added optional dependency extra `.[openreward]`, updated README/README_cn with OfficeQA examples and generic environment-switching commands, and expanded test coverage for OpenReward backend/CLI argument contracts and outputs.
- Extended OpenReward selection with mixed selectors via repeatable `--task-spec` (`<split>` or `<split>:<start>:<stop>`), allowing one batch to combine multiple splits and ranges.
- Added mixed-selector validation (`--task-spec` mutually exclusive with legacy `--task-index` / `--start` / `--stop`) and tests for multi-split/range expansion.
- Hardened OpenReward SDK compatibility: environment resolution now prioritizes explicit `variant` selectors, and client construction retries `api_key/base_url` argument combinations before fallback.
- Hardened OpenReward tool-call reliability with three runtime safeguards:
  - Normalize OpenReward tool schemas into OpenAI-compatible function shape (`tools[*].function.parameters`) before model requests.
  - Repair missing required `answer` arguments from assistant free text when model emits incomplete submit calls.
  - Auto-submit one final answer attempt when model returns text-only output without tool calls and a submission tool is available.
- Added configurable tool-output bounding before replay into model context via global `[runtime.context]` (`tool_output_truncate_enabled`, `tool_output_truncate_max_chars`), used by OpenReward with defaults set to no truncation and trace metadata (`content_chars`/`content_truncated`) for observability.
- Removed legacy `[runtime.openreward]` compatibility; config load now rejects that section and requires `[runtime.context]` for truncation controls.
- Added explicit OpenReward loop stop reason `no_tool_calls` when the model returns no tool call and no auto-submit path finishes the task, so unfinished exits are diagnosable instead of silent `error: null`.
- Added `.opm_train/batches/<batch_id>/openreward_trace.jsonl` to persist per-turn OpenReward request/response/tool events for debugging and audits.
- Expanded OpenReward regression coverage for tool-schema normalization, missing-answer repair, and text-only auto-submit paths.

## 2026-03-25

- Changed `spawn_agent` capacity behavior (`max_active_agents` / `max_children_per_agent`): runtime now returns a structured `status: rejected` payload to the caller agent instead of raising a tool execution exception, and records the spawn tool run as `completed` with rejection details.
- Added regression coverage for capacity-limited spawn behavior to ensure no child agent is created and the rejection remains observable through tool results.
- Restored `shell` in `opm_train.toml` runtime tool allow-lists for both root and worker roles, aligned with default runtime/tooling contract.
- Hardened JSONL readers in `SessionStorage` (`load_events`, `load_turns`) to stream physical lines instead of `splitlines()`, preventing `U+2028` content from corrupting JSON decoding.
- Hardened SSE parsing to split only on protocol LF boundaries (not `splitlines()`), preventing Unicode line-separator content inside `data:` JSON payloads from being truncated.
- Added doctor contract guard for missing core runtime tool set (`shell` + agent/tool inspection/control + `compress_context` + `finish`).
- Fixed mypy typing issue in spawn rejection event payload logging and added regression tests for shell allow-list, Unicode JSONL/SSE handling, and missing-core-tools doctor reporting.

## 2026-03-24

- Fixed OpenAI-compatible tool-call replay shape: assistant `tool_calls` now preserves required `type/function` fields across turns, preventing provider 400 errors like `messages[*].tool_calls[*].type` missing.
- Tightened tool-call protocol to OpenAI-compatible format only (`type=function`, `function.name`, `function.arguments`); legacy `name/arguments_json` shape is no longer accepted.
- Updated per-agent artifact folder naming to include agent name slug between prefix and id suffix (for example `agent-tester-08277a53f952`) with no legacy folder-name compatibility.
- Added regression coverage for replayed assistant tool-call schema and protocol parsing compatibility.
- Added per-step turn index persistence `.opm_train/sessions/<session_id>/turns.jsonl` with stable fields (`turn_id`, step scope, attempt references, action/results, finish payload, step error).
- Added step-level runtime telemetry events: `agent_step_started`, `agent_step_finished`, `llm_call_request_recorded`, and `llm_call_response_recorded`.
- Upgraded snapshot writer schema to `schema_version = 4`.
- Added modular trajectory export subsystem under `src/opm_train/trajectory` (`loader`, `filter`, `formatter`) for session/agent/agent-step scoped exports.
- Added CLI subcommand `opm-train export --session-id ... --mode raw|sft` with optional `--agent-id` and `--step`.
- Added export guard that rejects old sessions with snapshot schema `< 4`.
- Added regression coverage for turns storage/filtering, turn-index runtime persistence, trajectory raw/sft exports, CLI export happy path, and old-schema rejection.
- Updated README/README_cn with export usage, turns index description, and schema v4 notes.

## 2026-03-21

- Added `sft` dependency extra in `pyproject.toml` (`tinker`) and updated environment bootstrap to install `.[dev,sft]`.
- Extended README/README_cn with concrete SFT dataset organization, JSONL row contract, and training data examples.
- Added canonical inference metadata in per-agent LLM artifacts (`inference_provider`, `inference_endpoint`, `inference_model`, `inference_parameters`) for request/response records.
- Added regression assertions to keep inference metadata fields stable in integration artifacts.
- Added modular SFT subsystem under `src/opm_train/sft` with backend protocol/registry, JSONL loader, and artifact runner.
- Added built-in `tinker` SFT backend with lazy SDK import, cyclic batching, weighted-loss tracking, and optional post-train sample generation.
- Added CLI subcommand `opm-train sft` with backend/base-model/input/optimizer controls and structured JSON output.
- Added SFT artifacts under `.opm_train/sft_runs/<run_id>/` (`config.json`, `metrics.jsonl`, `result.json`).
- Updated default `provider.tinker` runtime profile to Tinker OpenAI-compatible inference endpoint and sampler-path model shape.
- Added regression tests for SFT parsing/backend execution/CLI output and Tinker default profile wiring.
- Added extensible dataset subsystem under `src/opm_train/data` with reusable contracts (`DatasetSample`, `PreparedTask`, `ValidationResult`, `BatchItemResult`) and adapter registry.
- Refactored math datasets onto shared `MathVerifyDatasetAdapter` and added built-in `simple_math` adapter.
- Updated `gsm8k` loading to extract canonical numeric references at ingest time while preserving raw answer text separately.
- Fixed JSONL parsing to stream physical lines (instead of `splitlines()`), avoiding failures on valid Hugging Face rows that contain Unicode separators (for example `U+2028`) inside fields.
- Added dataset batch runner `opm_train.batch_runner` with configurable concurrency, per-sample fault tolerance, realtime JSONL append, and aggregate summary output.
- Added CLI subcommand `opm-train batch-run` with `--dataset/--input/--concurrency/--limit/--batch-id/--resume` and structured JSON output for batch metrics + artifact paths.
- Added batch resume support via `--batch-id` + `--resume`, skipping already completed `sample_id`s from existing `results.jsonl`.
- Added mixed-dataset routing (`--dataset mixed` + per-row adapter key) to dispatch each row to the corresponding adapter.
- Added `--smoke` support for `batch-run` to validate end-to-end batch flow without external API keys.
- Removed static `contract_samples` fixtures to avoid hard-coded payload drift; contract validation stays runtime-driven (`doctor` + tool schema checks).
- Added tests for GSM8K/simple-math parsing/validation, dataset registry extensibility, mixed routing, realtime writes/resume behavior, and batch-run CLI integration with local runtime stubs.
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
- Tool/action execution errors (including unknown `agent_id`) now return structured tool error payloads while still recording error logs and failed tool-run state, instead of bubbling exceptions that fail the whole agent loop.
- Shell background execution now consumes late task exceptions and marks affected runs as failed (instead of leaving them stuck in `running` on internal subprocess startup errors).
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

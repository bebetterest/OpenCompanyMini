# opm-train

Language: **English** | [中文](README_cn.md)

`opm-train` is the greenfield OpenCompany mini runtime for research, iteration, and training-data collection.

## Why This Project

- Keep only the core runtime loop: tool-calling, multi-agent collaboration, context management.
- Remove heavy surfaces in v0: no TUI/Web UI, terminal-first only.
- Keep architecture modular and override-friendly (`AgentLoopRunner` + `LoopHooks`).
- Persist canonical raw trajectories for analysis/training (`events.jsonl`) with resumable snapshots.

## Directly Runnable Path

You can verify the runtime end-to-end **without any external API key**:

```bash
# 1) Setup
conda env create -f environment.yml
conda activate OpenCompany
pip install -e ".[dev]"

# 2) Check environment/config
opm-train doctor

# 3) Run local deterministic smoke flow (no provider call)
opm-train smoke --project-dir .
```

For real model execution:

```bash
export OPENROUTER_API_KEY="<your_key>"
opm-train run "Inspect this repository and propose a refactor plan"
```

## Commands

```bash
opm-train run <task>
opm-train resume <session_id> <instruction>
opm-train smoke [--task <task>]
opm-train doctor
```

Common options:

- `--project-dir`: target workspace (default `.`)
- `--app-dir`: directory containing `opm_train.toml` and `prompts/`
- `--provider-profile`: `openrouter` | `tinker` | `custom`
- `--model`: per-run model override
- `--timer`: enable per-module timing output under `sessions/<session_id>/timers/`

## Architecture Overview

- `RuntimeOrchestrator`: assembly layer for runtime services + shared persistence/event primitives.
- `OrchestratorSessionLifecycleMixin`: session bootstrap/resume and root lifecycle finalization.
- `OrchestratorAgentLifecycleMixin`: agent think-act loop, finish semantics, and lineage cancellation.
- `OrchestratorToolingMixin`: tool-run lifecycle and dispatch through a central tool registry.
- `orchestrator_tools/registry.py`: canonical tool registry (`ToolSpec`) for unified add/remove/modify.
- `AgentLoopRunner`: reusable think-act-feedback loop with step limits.
- `LoopHooks`: independent extension seam for custom loop behavior.
- `OpenAICompatibleClient`: unified provider path for OpenRouter/Tinker/custom OpenAI-style endpoints.
- `ContextAssembler`: prompt window projection and compression routing.

## Latest Slimming (A)

- Removed dormant runtime status/fields: `AgentStatus.waiting`, `ToolRun.parent_run_id`.
- Removed unused runtime tracker and helper: `tool_run_owner_agent`, `stable_json_dumps`.
- Kept `runtime.tools.shell_inline_wait_seconds` as shell inline-return threshold.
- Snapshot schema now writes `schema_version = 3` (no compatibility guarantee for older snapshots in this slimming pass).

## Runtime Behavior Updates

- `finish` is rejected (not failed) when the current agent still has non-terminal own tool runs; tool result includes `finish_rejected` and `unfinished_tool_runs`.
- `resume` marks non-restorable non-terminal tool runs as `abandoned` instead of `failed`.
- `doctor` now reports tool-contract checks via `tool_contract_ok` and `tool_contract_issues`; `ready_for_real_run` requires tool contract consistency.

## Runtime Data

By default, runtime data is written under:

- `.opm_train/sessions/<session_id>/events.jsonl`
- `.opm_train/sessions/<session_id>/state_snapshot.json`
- `.opm_train/sessions/<session_id>/agents/<agent_id>/llm_calls/<index>_request.json`
- `.opm_train/sessions/<session_id>/agents/<agent_id>/llm_calls/<index>_response.json`
- `.opm_train/sessions/<session_id>/agents/<agent_id>/context_compressions/<index>.json`
- `.opm_train/sessions/<session_id>/logs/runtime.log`
- `.opm_train/sessions/<session_id>/logs/errors.jsonl`
- `.opm_train/sessions/<session_id>/timers/module_timings.jsonl` (when `--timer` is enabled)

`events.jsonl` is the canonical raw trajectory log for analysis/training.
Before `resume`, runtime performs strict snapshot/event-tail validation (contiguous `seq` and count match).

## Configuration

Edit `opm_train.toml`:

- Provider profile routing (`openrouter`, `tinker`, `custom`) via one OpenAI-compatible client.
- Runtime limits (`max_children_per_agent`, `max_active_agents`, step budgets).
- Protocol retry controls for invalid model payloads (`max_protocol_retries`, `protocol_retry_backoff_seconds`).
- Runtime tools and context compression thresholds.
- Deferred extension flags (`sandbox`, `mcp`, `skills`) reserved for later versions.

## Tool Extension Workflow

Tool implementation files are grouped under `src/opm_train/orchestrator_tools/`.

1. Add a tool
   In one capability module (`shell.py`, `agent_ops.py`, `query_ops.py`, or a new module), implement `_tool_<name>(...)`.
2. Register the tool
   Add `ToolSpec(name=..., executor=..., default_blocking=..., self_completing=...)` in `src/opm_train/orchestrator_tools/registry.py`.
3. Expose and allow the tool
   Update `prompts/tool_definitions.json` and `prompts/tool_definitions_cn.json`, then update allow-lists in `opm_train.toml` (`[runtime.tools].root_tools/worker_tools`).
4. Delete a tool
   Remove its `ToolSpec` entry first, then remove handler code and prompt/config entries.
5. Modify a tool
   Edit handler logic in the module, then adjust `ToolSpec` metadata and prompt schema if input/output contract changes.

## Prompts and Bilingual Policy

- Runtime uses **English prompts only**.
- Chinese prompt mirrors are kept in sync (`*_cn`) for maintainability.
- `README`, `docs`, `AGENTS`, and prompt files maintain English + Chinese mirrors.

## Tests

```bash
pytest -q
mypy src
```

Current suite covers protocol parsing, OpenAI-compatible streaming parser behavior, loop hooks, context projection/compression, and core orchestration flows (spawn/wait/steer/cancel/resume).

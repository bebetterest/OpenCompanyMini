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
pip install -e ".[dev,sft]"
# Optional: OpenReward integration for batch-run --dataset openreward
pip install -e ".[openreward]"

# 2) Check environment/config
opm-train doctor

# 3) Run local deterministic smoke flow (no provider call)
opm-train smoke --project-dir .
```

For real model execution:

```bash
export OPENROUTER_API_KEY="<your_key>"
opm-train run "Inspect this repository and propose a refactor plan"

export TINKER_API_KEY="<your_key>"
opm-train run "Summarize this codebase" \
  --provider-profile tinker \
  --model "tinker://<train_id>/sampler_weights/<checkpoint_id>"
```

## Commands

```bash
opm-train run <task>
opm-train resume <session_id> <instruction>
opm-train smoke [--task <task>]
opm-train doctor
opm-train batch-run --dataset gsm8k --input <path/to/gsm8k.jsonl> [--concurrency 4] [--limit N] [--batch-id <id>] [--resume] [--smoke]
opm-train batch-run --dataset simple_math --input <path/to/simple_math.jsonl> [--concurrency 4] [--limit N] [--batch-id <id>] [--resume] [--smoke]
opm-train batch-run --dataset mixed --input <path/to/mixed.jsonl> --adapter-key adapter [--batch-id <id>] [--resume]
opm-train batch-run --dataset openreward --environment <owner/env> [--split train] [--task-index N | --start N --stop M | --task-spec <split> | --task-spec <split:start:stop>] [--variant <name>] [--base-url <url>] [--openreward-tool-format <fmt>] [--max-steps 64] [--concurrency 4] [--limit N] [--batch-id <id>] [--resume]
opm-train sft --backend tinker --input <path/to/sft.jsonl> --base-model <base_model> [--steps 6] [--batch-size 8] [--learning-rate 1e-4]
opm-train export --session-id <session_id> [--agent-id <agent_id>] [--step <n>] --mode raw|sft [--output <path>]
```

Common options:

- `--project-dir`: target workspace (default `.`)
- `--app-dir`: directory containing `opm_train.toml` and `prompts/`
- `--provider-profile`: `openrouter` | `tinker` | `custom`
- `--model`: per-run model override
- `--timer`: enable per-module timing output under `sessions/<session_id>/timers/`
- `sft` supports `--prompt-key/--completion-key` for custom JSONL schemas.

## Architecture Overview

- `RuntimeOrchestrator`: assembly layer for runtime services + shared persistence/event primitives.
- `OrchestratorSessionLifecycleMixin`: session bootstrap/resume and root lifecycle finalization.
- `OrchestratorAgentLifecycleMixin`: agent think-act loop, finish semantics, and lineage cancellation.
- `OrchestratorToolingMixin`: tool-run lifecycle and dispatch through a central tool registry.
- `batch_runner`: dataset batch scheduler (parallel sample runs + artifact writing).
- `sft/`: pluggable supervised fine-tuning runtime (backend ports + artifact persistence).
- `data/`: pluggable dataset adapters for prompt construction and result validation.
- `orchestrator_tools/registry.py`: canonical tool registry (`ToolSpec`) for unified add/remove/modify.
- `AgentLoopRunner`: reusable think-act-feedback loop with step limits.
- `LoopHooks`: independent extension seam for custom loop behavior.
- `OpenAICompatibleClient`: unified provider path for OpenRouter/Tinker/custom OpenAI-style endpoints.
- `ContextAssembler`: prompt window projection and compression routing.

## Latest Slimming (A)

- Removed dormant runtime status/fields: `AgentStatus.waiting`, `ToolRun.parent_run_id`.
- Removed unused runtime tracker and helper: `tool_run_owner_agent`, `stable_json_dumps`.
- Kept `runtime.tools.shell_inline_wait_seconds` as shell inline-return threshold.
- Snapshot schema now writes `schema_version = 4` (no compatibility guarantee for older snapshots in this slimming pass).

## Runtime Behavior Updates

- `finish` is rejected (not failed) when the current agent still has non-terminal own tool runs; tool result includes `finish_rejected` and `unfinished_tool_runs`.
- `resume` marks non-restorable non-terminal tool runs as `abandoned` instead of `failed`.
- `doctor` now reports tool-contract checks via `tool_contract_ok` and `tool_contract_issues`; `ready_for_real_run` requires tool contract consistency.
- Tool/action execution errors (for example unknown `agent_id`, missing tool `type`, or disabled tool name) no longer terminate the agent loop by bubbling exceptions; runtime records the error, marks the tool run failed, and returns a structured error payload (`error.code` / `error.type` / `error.message`).
- `shell` and `wait_run` now read default timeout settings from `runtime.tools.shell_timeout_seconds` and `runtime.tools.wait_run_timeout_seconds` (action-level `timeout_seconds` still overrides); timeout responses now include timeout context (`timed_out`, `timeout_seconds`).

## Runtime Data

By default, runtime data is written under:

- `.opm_train/sessions/<session_id>/events.jsonl`
- `.opm_train/sessions/<session_id>/turns.jsonl`
- `.opm_train/sessions/<session_id>/state_snapshot.json`
- `.opm_train/sessions/<session_id>/agents/<agent_folder>/llm_calls/<index>_request.json`
- `.opm_train/sessions/<session_id>/agents/<agent_folder>/llm_calls/<index>_response.json`
- `.opm_train/sessions/<session_id>/agents/<agent_folder>/context_compressions/<index>.json`
- `.opm_train/sessions/<session_id>/logs/runtime.log`
- `.opm_train/sessions/<session_id>/logs/errors.jsonl`
- `.opm_train/sessions/<session_id>/timers/module_timings.jsonl` (when `--timer` is enabled)
- `.opm_train/batches/<batch_id>/results.jsonl` (per-sample evaluation records)
- `.opm_train/batches/<batch_id>/summary.json` (aggregate metrics)
- `.opm_train/batches/<batch_id>/openreward_results.jsonl` (per-task OpenReward records)
- `.opm_train/batches/<batch_id>/openreward_summary.json` (OpenReward aggregate metrics)
- `.opm_train/batches/<batch_id>/openreward_trace.jsonl` (per-turn OpenReward request/response/tool trace)
- `.opm_train/sft_runs/<run_id>/config.json` (resolved run config + dataset mapping)
- `.opm_train/sft_runs/<run_id>/metrics.jsonl` (per-step training metrics)
- `.opm_train/sft_runs/<run_id>/result.json` (terminal backend summary)

Each `llm_calls/*_request.json` and `*_response.json` includes canonical inference metadata:
`inference_provider`, `inference_endpoint`, `inference_model`, and `inference_parameters` (for example `temperature`, `max_tokens`, tool-call switches).

`<agent_folder>` defaults to `agent-<name_slug>-<agent_id_suffix>` (for example `agent-tester-08277a53f952`).

`events.jsonl` is the canonical raw trajectory log for analysis/training.
Before `resume`, runtime performs strict snapshot/event-tail validation (contiguous `seq` and count match).

`turns.jsonl` is the per-agent step index. Each row captures one step-turn with:
`turn_id`, `agent_id`, `step`, `event_seq_start/end`, `attempts`, `final_attempt`, `actions`, `action_results`, `finish_payload`, and `step_error`.

`export` supports scoped extraction:

- Session scope: `--session-id <id> --mode raw|sft`
- Agent scope: `--session-id <id> --agent-id <id> --mode raw|sft`
- Agent-step scope: `--session-id <id> --agent-id <id> --step <n> --mode raw|sft`

Rules:

- `--step` requires `--agent-id`.
- `--mode sft` emits one row per turn using the final successful protocol attempt.
  - Backward-compatible `target` is preserved as action-supervision payload (`{"actions":[...]}`).
  - Additional fields are exported for traceability and richer training:
    - `messages_complete`: request `messages` plus full assistant response.
    - `assistant_response`: includes `content`, optional `reasoning`, `tool_calls`, `usage`, and `raw_events` when available.
    - `metadata.inference_*`: provider/model/endpoint/parameter metadata.
    - `environment`: session task/project/provider profile + full `config_snapshot`.
    - `traceability`: `llm_sequence`, request/response artifact paths, event-seq range, and turn timestamps.
- Export requires snapshot `schema_version >= 4` (older sessions are rejected).

## SFT Workflow

`sft` currently ships with pluggable backend ports and one built-in backend: `tinker`.

Install backend dependency:

```bash
conda run -n OpenCompany pip install tinker
```

Input is local JSONL. Supported key pairs per row:

- `prompt` + `completion`
- `input` + `output`
- `instruction` + `output`
- `question` + `answer`

If your schema differs, pass `--prompt-key` and `--completion-key`.

Recommended dataset layout:

```text
data/
  sft/
    train.jsonl
    valid.jsonl        # optional
    README.md          # optional dataset note/version
```

Row contract:

- UTF-8 JSON object per physical line.
- One prompt field + one completion field (from supported key pairs, or overridden keys).
- Optional `id` field for stable traceability (`line-<n>` is used when absent).
- Additional fields are preserved as metadata in normalized samples.

Example:

```jsonl
{"id":"demo-0001","prompt":"Summarize the following diff...","completion":"Here is the summary...","source":"internal-v1"}
{"id":"demo-0002","input":"Translate to Chinese: Hello","output":"你好","split":"train"}
```

Minimal run:

```bash
opm-train sft \
  --backend tinker \
  --input data/sft/train.jsonl \
  --base-model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --output-model demo-sampler \
  --steps 6 \
  --batch-size 8 \
  --learning-rate 1e-4
```

After training, use the checkpoint path in `result.json` (or `--model`) with `opm-train run --provider-profile tinker`.

`batch-run` currently ships with built-in `gsm8k` and `simple_math` adapters for local JSONL inputs with fields:

- `question` (string, required)
- `answer` (string, required)
- `id` (string, optional)

During loading, math adapters normalize `answer` into:

- `reference_answer`: canonical extracted numeric answer used for validation.
- `reference_answer_raw`: original dataset answer text (kept for audit/debug).

To fetch official `gsm8k` splits from Hugging Face (`openai/gsm8k:main`) into local JSONL (requires `datasets`: `conda run -n OpenCompany pip install datasets`):

```bash
conda run -n OpenCompany python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

out_dir = Path("data/gsm8k")
out_dir.mkdir(parents=True, exist_ok=True)
dataset = load_dataset("openai/gsm8k", "main")
for split in ("train", "test"):
    out_path = out_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset[split], start=1):
            payload = {
                "id": f"gsm8k-{split}-{index:06d}",
                "question": row["question"],
                "answer": row["answer"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(split, len(dataset[split]), out_path)
PY
```

Note: the JSONL loader streams physical lines (instead of using `splitlines()`), so Unicode separators inside questions (for example `U+2028`) are handled safely.

`gsm8k` and `simple_math` validation are handled by [Math-Verify](https://github.com/huggingface/Math-Verify), and expect the runtime summary to include `FINAL_ANSWER: <number>`.

`mixed` dataset mode:

- Each JSONL row must include adapter selector key (default `adapter`).
- Example row: `{"adapter":"gsm8k","id":"m1","question":"...","answer":"#### 42"}`.
- Example row: `{"adapter":"simple_math","id":"m2","question":"13*7","answer":"91"}`.
- Runtime routes each row to its adapter and writes `adapter_name` into `results.jsonl`.

`openreward` dataset mode:

- Uses OpenReward environments directly (`AsyncOpenReward`) instead of local JSONL adapters.
- Required argument: `--environment <owner/environment>` (for example `GeneralReasoning/OfficeQA`).
- Task selection:
  - Single task: `--task-index <n>`
  - Task range: `--start <n> --stop <m>`
  - Full split: omit all selectors (default split is `train`)
  - Mixed selectors (repeatable): `--task-spec <split>` and/or `--task-spec <split>:<start>:<stop>`
  - `--task-spec` cannot be combined with `--task-index` or `--start/--stop`
- Optional environment routing: `--variant <name>` and `--base-url <url>` for multi-variant/self-hosted environments.
- Tool schema format defaults by provider profile:
  - `openrouter` profile -> `openrouter`
  - `tinker` / `custom` profile -> `openai`
  - Override with `--openreward-tool-format`.
- Runtime normalizes OpenReward tool definitions into OpenAI-compatible `tools[*].function.parameters` payloads before model calls.
- Defensive fallback behavior for fragile tool-argument generations:
  - If model emits a tool call missing required `answer`, runtime extracts a candidate from assistant text and retries with repaired arguments.
  - If model emits text only (no tool call) while a submission tool is available, runtime performs one auto-submit attempt.
- Tool-output truncation is configured globally via `[runtime.context]` (used by OpenReward tool replay):
  - Default: disabled (`tool_output_truncate_enabled = false`)
  - Optional cap: `tool_output_truncate_max_chars` (effective only when truncation is enabled)
  - Trace metadata: `openreward_trace.jsonl` records `content_chars` and `content_truncated`
    (`content_truncated=true` when runtime truncates output, or when upstream tool output is already marked truncated)
  - Trace rows also carry end-to-end traceability fields, including:
    - batch/provider/model metadata (`batch_id`, `provider_profile`, `inference_*`, tool format, selector settings)
    - environment/task lineage (`openreward_environment`, `variant`, `task_key/task_index/task_id`, per-task `trace_session_id`)
    - per-turn request/response details (`llm_request`, `llm_response`, `reasoning`, `raw_events`, `trace_event_seq`)
- Results/summary use OpenReward-specific fields (reward-based), persisted to:
  - `openreward_results.jsonl`
  - `openreward_summary.json`
  - `openreward_trace.jsonl`
- `openreward_results.jsonl` rows include stable per-task `session_id` (same value format as trace `trace_session_id`: `<batch_id>:<split>:<task_key>`), plus
  `environment/split/variant/task_key/task_index/reward_total/finished/tool_calls/turns/session_status/error`.

OfficeQA example (single task):

```bash
export OPENREWARD_API_KEY="<your_openreward_key>"
export OPENROUTER_API_KEY="<your_openrouter_key>"

opm-train batch-run \
  --dataset openreward \
  --environment GeneralReasoning/OfficeQA \
  --split train \
  --task-index 0 \
  --provider-profile openrouter \
  --model "deepseek/deepseek-v3.2"
```

OpenReward range example (other environments):

```bash
opm-train batch-run \
  --dataset openreward \
  --environment GeneralReasoning/CTF \
  --split train \
  --start 0 \
  --stop 10 \
  --concurrency 4 \
  --max-steps 64
```

OpenReward mixed-selector example (multiple splits/ranges in one batch):

```bash
opm-train batch-run \
  --dataset openreward \
  --environment GeneralReasoning/OfficeQA \
  --task-spec train:0:20 \
  --task-spec validation \
  --concurrency 4
```

## Dataset Extension Workflow

1. Add adapter
   Create `src/opm_train/data/<dataset>.py` implementing and inheriting `DatasetAdapter` (`sample_from_payload`, `load_samples`, `build_task_prompt`, `validate_result`).
2. Register adapter
   Register it in `src/opm_train/data/__init__.py` via `register_dataset_adapter(...)`.
3. Run batch
   Execute `opm-train batch-run --dataset <dataset_name> --input <jsonl>`.
4. Add tests
   Add adapter parsing/validation tests and one batch CLI integration test.

Minimal adapter shape:

```python
from opm_train.data.contracts import DatasetAdapter, DatasetSample, PreparedTask, ValidationResult
from opm_train.models import RunSession

class MyDatasetAdapter(DatasetAdapter):
    name = "my_dataset"

    def sample_from_payload(self, payload: dict, *, line_no: int) -> DatasetSample: ...
    def load_samples(self, *, input_path, limit=None) -> list[DatasetSample]: ...
    def build_task_prompt(self, sample: DatasetSample) -> PreparedTask: ...
    def validate_result(self, *, sample: DatasetSample, session: RunSession) -> ValidationResult: ...
```

`batch-run` behavior:

- Concurrent sample execution (configurable by `--concurrency`).
- Realtime JSONL append (`results.jsonl` is written as each sample finishes).
- Resume support (`--batch-id <id> --resume`) skips already completed `(adapter_name, sample_id)` and continues the rest.
- Smoke mode (`--smoke`) runs batch without external LLM API keys.
- `openreward` mode writes dedicated reward-based artifacts (`openreward_results.jsonl`, `openreward_summary.json`, `openreward_trace.jsonl`) and reports `completed`/`finished`/`avg_reward` metrics.

`results.jsonl` rows include:

- `adapter_name`, `sample_id`, `task_prompt`, `reference_answer`
- `reference_answer_raw`
- `predicted_answer`, `is_correct`
- `session_id`, `session_status`, `final_summary`, `error`

## Configuration

Edit `opm_train.toml`:

- Provider profile routing (`openrouter`, `tinker`, `custom`) via one OpenAI-compatible client.
- Default `provider.tinker.base_url` is set to Tinker OpenAI-compatible inference endpoint:
  `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`.
- Default model for all provider profiles is `qwen/qwen3.6-plus-preview:free`; override per profile or via `--model` in `run/resume/batch-run` when needed.
- Runtime limits (`max_children_per_agent`, `max_active_agents`, step budgets).
- Protocol retry controls for invalid model payloads (`max_protocol_retries`, `protocol_retry_backoff_seconds`) plus independent context-overflow retry budget (`max_context_overflow_retries`).
- Per-turn retry metrics are persisted in `turns.jsonl` (`overall_retries`, API/network retries, empty-stream retries, parse retries, parse-empty retries, context-overflow retries).
- Runtime tools and context compression thresholds.
- Global tool-output replay controls under `[runtime.context]`:
  - `tool_output_truncate_enabled` (default `false`)
  - `tool_output_truncate_max_chars` (default `8000`; used when truncation is enabled)
- Legacy `[runtime.openreward]` is no longer supported.
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

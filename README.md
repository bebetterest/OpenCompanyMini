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
opm-train batch-run --dataset gsm8k --input <path/to/gsm8k.jsonl> [--concurrency 4] [--limit N] [--batch-id <id>] [--resume] [--smoke]
opm-train batch-run --dataset simple_math --input <path/to/simple_math.jsonl> [--concurrency 4] [--limit N] [--batch-id <id>] [--resume] [--smoke]
opm-train batch-run --dataset mixed --input <path/to/mixed.jsonl> --adapter-key adapter [--batch-id <id>] [--resume]
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
- `batch_runner`: dataset batch scheduler (parallel sample runs + artifact writing).
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
- `.opm_train/batches/<batch_id>/results.jsonl` (per-sample evaluation records)
- `.opm_train/batches/<batch_id>/summary.json` (aggregate metrics)

`events.jsonl` is the canonical raw trajectory log for analysis/training.
Before `resume`, runtime performs strict snapshot/event-tail validation (contiguous `seq` and count match).

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

`results.jsonl` rows include:

- `adapter_name`, `sample_id`, `task_prompt`, `reference_answer`
- `reference_answer_raw`
- `predicted_answer`, `is_correct`
- `session_id`, `session_status`, `final_summary`, `error`

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

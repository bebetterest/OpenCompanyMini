# opm-train

语言：**中文** | [English](README.md)

`opm-train` 是全新构建的 OpenCompany mini 运行时，用于研究、迭代与训练数据沉淀。

## 项目目标

- 只保留核心运行循环：工具调用、多代理协作、上下文管理。
- v0 去除重型界面：不含 TUI/Web UI，仅终端运行。
- 保持模块化与可重载（`AgentLoopRunner` + `LoopHooks`）。
- 以 `events.jsonl` 持久化规范原始轨迹，并用快照支持 resume。

## 可直接运行路径

无需任何外部 API Key，也可以先验证系统是否跑通：

```bash
# 1) 环境准备
conda env create -f environment.yml
conda activate OpenCompany
pip install -e ".[dev,sft]"
# 可选：如需使用 OpenReward 批量环境接入（--dataset openreward）
pip install -e ".[openreward]"

# 2) 检查配置与环境
opm-train doctor

# 3) 运行本地确定性 smoke 流程（不调用 provider）
opm-train smoke --project-dir .
```

若要真实模型推理：

```bash
export OPENROUTER_API_KEY="<your_key>"
opm-train run "Inspect this repository and propose a refactor plan"

export TINKER_API_KEY="<your_key>"
opm-train run "Summarize this codebase" \
  --provider-profile tinker \
  --model "tinker://<train_id>/sampler_weights/<checkpoint_id>"
```

## 命令

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

常用参数：

- `--project-dir`：目标工作目录（默认 `.`）
- `--app-dir`：包含 `opm_train.toml` 与 `prompts/` 的目录
- `--provider-profile`：`openrouter` | `tinker` | `custom`
- `--model`：本次运行模型覆盖
- `--timer`：开启后将各模块计时写入 `sessions/<session_id>/timers/`
- `sft` 支持 `--prompt-key/--completion-key` 以适配自定义 JSONL 字段。

## 架构概览

- `RuntimeOrchestrator`：运行时装配层，负责共享基础设施与事件/快照持久化。
- `OrchestratorSessionLifecycleMixin`：会话启动/恢复与 root 生命周期收敛。
- `OrchestratorAgentLifecycleMixin`：代理 think-act 循环、finish 语义与谱系取消。
- `OrchestratorToolingMixin`：工具运行生命周期与统一注册表分发。
- `batch_runner`：数据集批量调度器（并行样本执行 + 产物落盘）。
- `sft/`：可插拔监督微调运行时（后端端口 + 产物落盘）。
- `data/`：可插拔数据适配层，负责 prompt 构造与结果校验。
- `orchestrator_tools/registry.py`：规范工具注册表（`ToolSpec`），支持统一新增/删除/修改。
- `AgentLoopRunner`：可复用 think-act-feedback 循环与步数限制。
- `LoopHooks`：独立扩展点，可自定义循环行为。
- `OpenAICompatibleClient`：统一 Provider 路径（OpenRouter/Tinker/custom）。
- `ContextAssembler`：上下文窗口投影与压缩处理。

## 最新瘦身（A）

- 移除休眠状态与字段：`AgentStatus.waiting`、`ToolRun.parent_run_id`。
- 移除未使用的运行时跟踪与工具函数：`tool_run_owner_agent`、`stable_json_dumps`。
- 保留 `runtime.tools.shell_inline_wait_seconds` 作为 shell 内联返回阈值。
- 快照写入版本更新为 `schema_version = 4`（本轮瘦身不保证旧快照兼容）。

## 运行时行为更新

- 当当前代理仍有自身未终态 tool run 时，`finish` 会被拒绝（不是直接失败）；工具结果包含 `finish_rejected` 与 `unfinished_tool_runs`。
- `resume` 对不可恢复的未终态 tool run 记为 `abandoned`，不再记为 `failed`。
- `doctor` 新增工具契约检查输出 `tool_contract_ok` 与 `tool_contract_issues`；`ready_for_real_run` 需要工具契约一致。
- 工具/动作执行错误（如未知 `agent_id`、缺失工具 `type`、或工具名未启用）不再通过抛异常直接终止 agent loop；运行时会记录错误、将 tool run 标记为 failed，并返回结构化错误负载（`error.code` / `error.type` / `error.message`）。
- `shell` 与 `wait_run` 的默认超时现由 `runtime.tools.shell_timeout_seconds` 与 `runtime.tools.wait_run_timeout_seconds` 统一配置（动作内 `timeout_seconds` 仍可覆盖）；超时返回会包含超时上下文（`timed_out`、`timeout_seconds`）。

## 运行数据

默认写入：

- `.opm_train/sessions/<session_id>/events.jsonl`
- `.opm_train/sessions/<session_id>/turns.jsonl`
- `.opm_train/sessions/<session_id>/state_snapshot.json`
- `.opm_train/sessions/<session_id>/agents/<agent_folder>/llm_calls/<index>_request.json`
- `.opm_train/sessions/<session_id>/agents/<agent_folder>/llm_calls/<index>_response.json`
- `.opm_train/sessions/<session_id>/agents/<agent_folder>/context_compressions/<index>.json`
- `.opm_train/sessions/<session_id>/logs/runtime.log`
- `.opm_train/sessions/<session_id>/logs/errors.jsonl`
- `.opm_train/sessions/<session_id>/timers/module_timings.jsonl`（启用 `--timer` 时写入）
- `.opm_train/batches/<batch_id>/results.jsonl`（逐样本评测记录）
- `.opm_train/batches/<batch_id>/summary.json`（汇总指标）
- `.opm_train/batches/<batch_id>/openreward_results.jsonl`（OpenReward 逐任务记录）
- `.opm_train/batches/<batch_id>/openreward_summary.json`（OpenReward 汇总指标）
- `.opm_train/batches/<batch_id>/openreward_trace.jsonl`（OpenReward 逐轮请求/返回/工具调用追踪）
- `.opm_train/sft_runs/<run_id>/config.json`（解析后的运行配置与数据映射）
- `.opm_train/sft_runs/<run_id>/metrics.jsonl`（逐 step 训练指标）
- `.opm_train/sft_runs/<run_id>/result.json`（后端终态摘要）

`llm_calls/*_request.json` 与 `*_response.json` 中会记录统一推理元数据：
`inference_provider`、`inference_endpoint`、`inference_model`、`inference_parameters`（如 `temperature`、`max_tokens`、工具调用开关）。

`<agent_folder>` 默认采用 `agent-<name_slug>-<agent_id_suffix>`（例如 `agent-tester-08277a53f952`）。

`events.jsonl` 是用于分析/训练的规范原始轨迹日志。
在 `resume` 前，运行时会严格校验 snapshot 与 event-tail（`seq` 连续且数量一致）。

`turns.jsonl` 是按 agent step 组织的轮次索引。每行包含：
`turn_id`、`agent_id`、`step`、`event_seq_start/end`、`attempts`、`final_attempt`、`actions`、`action_results`、`finish_payload`、`step_error`。

`export` 支持按范围导出：

- session 级：`--session-id <id> --mode raw|sft`
- agent 级：`--session-id <id> --agent-id <id> --mode raw|sft`
- agent+step 级：`--session-id <id> --agent-id <id> --step <n> --mode raw|sft`

约束：

- `--step` 必须搭配 `--agent-id`。
- `--mode sft` 每个 turn 输出一条样本，使用该 turn 最终成功的协议尝试。
  - 保留向后兼容的 `target`（动作监督负载：`{"actions":[...]}`）。
  - 同时导出更完整的训练/追溯字段：
    - `messages_complete`：请求 `messages` + 完整 assistant 响应。
    - `assistant_response`：包含 `content`，以及可用时的 `reasoning`、`tool_calls`、`usage`、`raw_events`。
    - `metadata.inference_*`：provider/model/endpoint/参数等推理元数据。
    - `environment`：会话任务/项目/provider profile 与完整 `config_snapshot`。
    - `traceability`：`llm_sequence`、request/response 产物路径、event-seq 范围、turn 时间戳。
- 仅支持快照版本 `schema_version >= 4` 的会话导出（旧会话会被拒绝）。

## SFT 工作流

`sft` 当前提供可扩展后端端口，内置后端为 `tinker`。

安装后端依赖：

```bash
conda run -n OpenCompany pip install tinker
```

输入为本地 JSONL。每行支持以下字段对之一：

- `prompt` + `completion`
- `input` + `output`
- `instruction` + `output`
- `question` + `answer`

若字段名不同，可通过 `--prompt-key` 与 `--completion-key` 指定。

推荐数据组织：

```text
data/
  sft/
    train.jsonl
    valid.jsonl        # 可选
    README.md          # 可选，记录数据版本与说明
```

单行数据约束：

- UTF-8 编码，每个物理行必须是一个 JSON 对象。
- 必须有一组 prompt/completion 字段（支持默认字段对，或通过参数覆盖）。
- 可选 `id` 字段用于稳定追踪（缺省时使用 `line-<n>`）。
- 其他字段会保留到样本 metadata 中。

示例：

```jsonl
{"id":"demo-0001","prompt":"Summarize the following diff...","completion":"Here is the summary...","source":"internal-v1"}
{"id":"demo-0002","input":"Translate to Chinese: Hello","output":"你好","split":"train"}
```

最小示例：

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

训练后可将 `result.json` 里的 checkpoint 路径（或通过 `--model`）用于 `opm-train run --provider-profile tinker` 推理。

`batch-run` 当前内置 `gsm8k` 与 `simple_math` 适配器，输入为本地 JSONL，字段要求：

- `question`（字符串，必填）
- `answer`（字符串，必填）
- `id`（字符串，可选）

数学类适配器在加载时会把 `answer` 归一化为：

- `reference_answer`：用于校验的规范化数值答案。
- `reference_answer_raw`：保留原始答案文本，便于审计/排查。

可使用 Hugging Face 官方数据（`openai/gsm8k:main`）生成本地 `train/test` JSONL（需先安装 `datasets`：`conda run -n OpenCompany pip install datasets`）：

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

说明：JSONL 读取器按“物理行”流式读取（不使用 `splitlines()`），可正确处理题干中出现的 Unicode 分隔符（如 `U+2028`）。

`gsm8k` 与 `simple_math` 校验均使用 [Math-Verify](https://github.com/huggingface/Math-Verify)，并期望运行时总结包含 `FINAL_ANSWER: <number>`。

`mixed` 混合数据模式：

- 每行 JSONL 需包含适配器选择键（默认 `adapter`）。
- 示例：`{"adapter":"gsm8k","id":"m1","question":"...","answer":"#### 42"}`。
- 示例：`{"adapter":"simple_math","id":"m2","question":"13*7","answer":"91"}`。
- 运行时会按行路由到对应适配器，并在 `results.jsonl` 写入 `adapter_name`。

`openreward` 数据模式：

- 直接对接 OpenReward 环境（`AsyncOpenReward`），不依赖本地 JSONL 适配器。
- 必填参数：`--environment <owner/environment>`（例如 `GeneralReasoning/OfficeQA`）。
- 任务选择方式：
  - 单任务：`--task-index <n>`
  - 区间任务：`--start <n> --stop <m>`
  - 全 split：不传选择器（默认 `train`）
  - 混合选择（可重复）：`--task-spec <split>` 与/或 `--task-spec <split>:<start>:<stop>`
  - `--task-spec` 不能和 `--task-index` 或 `--start/--stop` 同时使用
- 可选路由参数：`--variant <name>` 与 `--base-url <url>`，用于多变体或自托管环境。
- 工具 schema 格式按 provider profile 默认：
  - `openrouter` -> `openrouter`
  - `tinker` / `custom` -> `openai`
  - 可用 `--openreward-tool-format` 覆盖。
- 运行时会先把 OpenReward 工具定义归一化为 OpenAI 兼容的 `tools[*].function.parameters` 结构，再发给模型。
- 对工具参数脆弱场景提供兜底：
  - 若模型发出缺少必填 `answer` 的工具调用，会从 assistant 文本提取候选答案并修复参数后重试。
  - 若模型只输出文本（无工具调用）且存在提交工具，会自动尝试一次提交。
- 工具输出截断是全局配置，位于 `[runtime.context]`（OpenReward 工具回填会使用该配置）：
  - 默认关闭（`tool_output_truncate_enabled = false`）
  - 可设置长度上限 `tool_output_truncate_max_chars`（仅在开启截断时生效）
  - `openreward_trace.jsonl` 会记录 `content_chars` 与 `content_truncated`
    （当运行时执行截断，或上游工具输出已带截断标记时，`content_truncated=true`）
  - trace 行还会记录端到端可追溯字段，包括：
    - batch/provider/model 元数据（`batch_id`、`provider_profile`、`inference_*`、工具格式、任务选择器）
    - 环境与任务谱系（`openreward_environment`、`variant`、`task_key/task_index/task_id`、每任务 `trace_session_id`）
    - 逐轮请求/返回细节（`llm_request`、`llm_response`、`reasoning`、`raw_events`、`trace_event_seq`）
- 结果输出采用 OpenReward 专用奖励语义，产物文件：
  - `openreward_results.jsonl`
  - `openreward_summary.json`
  - `openreward_trace.jsonl`
- `openreward_results.jsonl` 每行包含稳定的任务级 `session_id`（与 trace 中 `trace_session_id` 同格式：`<batch_id>:<split>:<task_key>`），以及
  `environment/split/variant/task_key/task_index/reward_total/finished/tool_calls/turns/session_status/error`。

OfficeQA 单任务示例：

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

其他环境区间任务示例：

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

OpenReward 混合选择示例（一次批跑包含多个 split/区间）：

```bash
opm-train batch-run \
  --dataset openreward \
  --environment GeneralReasoning/OfficeQA \
  --task-spec train:0:20 \
  --task-spec validation \
  --concurrency 4
```

## 数据集扩展流程

1. 新增适配器
   在 `src/opm_train/data/<dataset>.py` 继承并实现 `DatasetAdapter`（`sample_from_payload`、`load_samples`、`build_task_prompt`、`validate_result`）。
2. 注册适配器
   在 `src/opm_train/data/__init__.py` 通过 `register_dataset_adapter(...)` 注册。
3. 批量运行
   执行 `opm-train batch-run --dataset <dataset_name> --input <jsonl>`。
4. 补充测试
   增加适配器解析/校验测试，以及一条 batch CLI 集成测试。

最小适配器形态：

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

`batch-run` 行为：

- 并发执行样本（`--concurrency` 可配置）。
- 实时写入 JSONL（每个样本完成即追加到 `results.jsonl`）。
- 支持断点续跑（`--batch-id <id> --resume`），按 `(adapter_name, sample_id)` 跳过已完成样本并继续未完成部分。
- 支持 `--smoke` 模式，无需外部 LLM API Key 即可跑通批量链路。
- `openreward` 模式会写入奖励语义专用产物（`openreward_results.jsonl`、`openreward_summary.json`、`openreward_trace.jsonl`），并输出 `completed`/`finished`/`avg_reward` 等指标。

`results.jsonl` 单行字段包括：

- `adapter_name`、`sample_id`、`task_prompt`、`reference_answer`
- `reference_answer_raw`
- `predicted_answer`、`is_correct`
- `session_id`、`session_status`、`final_summary`、`error`

## 配置

通过 `opm_train.toml` 配置：

- Provider 选择（`openrouter`、`tinker`、`custom`）统一走 OpenAI 兼容客户端。
- `provider.tinker.base_url` 默认已设为 Tinker OpenAI 兼容推理端点：
  `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`。
- 所有 provider profile 的默认模型统一为 `qwen/qwen3.6-plus-preview:free`；如有需要，可在 profile 中或通过 `run/resume/batch-run` 的 `--model` 覆盖。
- 运行限制（子代理数量、并发代理数、步数预算）。
- 模型协议解析失败重试控制（`max_protocol_retries`、`protocol_retry_backoff_seconds`），以及独立的上下文超窗重试预算（`max_context_overflow_retries`）。
- 当某个 step 没有可执行工具调用时，运行时会跳过该 step 的协议重试，回填反馈后进入下一步继续自主决策。
- 当某个 step 的协议重试耗尽时，运行时会将“重试耗尽反馈”写回对话，并进入下一步继续决策，不会自动注入 `finish`。
- 每个 step 的重试统计会落盘到 `turns.jsonl`（`overall_retries`、API/网络重试、空流重试、解析重试、解析为空重试、上下文超窗重试）。
- provider 侧流式重试现在也覆盖“已收到部分 SSE 输出后断流”的情况；每次重试都会丢弃失败尝试的部分输出，仅保留成功尝试的输出。
- 工具集合与上下文压缩阈值。
- `[runtime.context]` 提供全局工具回填截断控制：
  - `tool_output_truncate_enabled`（默认 `false`）
  - `tool_output_truncate_max_chars`（默认 `8000`，开启截断时生效）
- 旧配置段 `[runtime.openreward]` 已不再支持。
- `sandbox`、`mcp`、`skills` 为 v0 预留开关，暂不实现。

## 工具扩展流程

工具实现文件已集中在 `src/opm_train/orchestrator_tools/`。

1. 新增工具
   在能力模块（`shell.py`、`agent_ops.py`、`query_ops.py`，或新增模块）实现 `_tool_<name>(...)`。
2. 注册工具
   在 `src/opm_train/orchestrator_tools/registry.py` 添加 `ToolSpec(name=..., executor=..., default_blocking=..., self_completing=...)`。
3. 对模型暴露并放行
   同步更新 `prompts/tool_definitions.json` 与 `prompts/tool_definitions_cn.json`，再更新 `opm_train.toml` 的 allow-list（`[runtime.tools].root_tools/worker_tools`）。
4. 删除工具
   先删除 `ToolSpec` 注册，再清理 handler 代码与 prompt/config 条目。
5. 修改工具
   修改对应模块 handler 逻辑；如果入参/返回契约变化，同时更新 `ToolSpec` 元信息与 prompt schema。

## Prompt 与双语策略

- 运行时仅加载**英文 prompt**。
- 保留并同步中文 prompt 镜像（`*_cn`）。
- `README`、`docs`、`AGENTS`、prompt 文件保持中英镜像。

## 测试

```bash
pytest -q
mypy src
```

当前测试覆盖协议解析、OpenAI 兼容流式解析、loop hooks、上下文投影/压缩，以及核心编排流程（spawn/wait/steer/cancel/resume）。

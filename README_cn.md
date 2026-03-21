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
pip install -e ".[dev]"

# 2) 检查配置与环境
opm-train doctor

# 3) 运行本地确定性 smoke 流程（不调用 provider）
opm-train smoke --project-dir .
```

若要真实模型推理：

```bash
export OPENROUTER_API_KEY="<your_key>"
opm-train run "Inspect this repository and propose a refactor plan"
```

## 命令

```bash
opm-train run <task>
opm-train resume <session_id> <instruction>
opm-train smoke [--task <task>]
opm-train doctor
```

常用参数：

- `--project-dir`：目标工作目录（默认 `.`）
- `--app-dir`：包含 `opm_train.toml` 与 `prompts/` 的目录
- `--provider-profile`：`openrouter` | `tinker` | `custom`
- `--model`：本次运行模型覆盖
- `--timer`：开启后将各模块计时写入 `sessions/<session_id>/timers/`

## 架构概览

- `RuntimeOrchestrator`：运行时装配层，负责共享基础设施与事件/快照持久化。
- `OrchestratorSessionLifecycleMixin`：会话启动/恢复与 root 生命周期收敛。
- `OrchestratorAgentLifecycleMixin`：代理 think-act 循环、finish 语义与谱系取消。
- `OrchestratorToolingMixin`：工具运行生命周期与统一注册表分发。
- `orchestrator_tools/registry.py`：规范工具注册表（`ToolSpec`），支持统一新增/删除/修改。
- `AgentLoopRunner`：可复用 think-act-feedback 循环与步数限制。
- `LoopHooks`：独立扩展点，可自定义循环行为。
- `OpenAICompatibleClient`：统一 Provider 路径（OpenRouter/Tinker/custom）。
- `ContextAssembler`：上下文窗口投影与压缩处理。

## 最新瘦身（A）

- 移除休眠状态与字段：`AgentStatus.waiting`、`ToolRun.parent_run_id`。
- 移除未使用的运行时跟踪与工具函数：`tool_run_owner_agent`、`stable_json_dumps`。
- 保留 `runtime.tools.shell_inline_wait_seconds` 作为 shell 内联返回阈值。
- 快照写入版本更新为 `schema_version = 3`（本轮瘦身不保证旧快照兼容）。

## 运行时行为更新

- 当当前代理仍有自身未终态 tool run 时，`finish` 会被拒绝（不是直接失败）；工具结果包含 `finish_rejected` 与 `unfinished_tool_runs`。
- `resume` 对不可恢复的未终态 tool run 记为 `abandoned`，不再记为 `failed`。
- `doctor` 新增工具契约检查输出 `tool_contract_ok` 与 `tool_contract_issues`；`ready_for_real_run` 需要工具契约一致。

## 运行数据

默认写入：

- `.opm_train/sessions/<session_id>/events.jsonl`
- `.opm_train/sessions/<session_id>/state_snapshot.json`
- `.opm_train/sessions/<session_id>/agents/<agent_id>/llm_calls/<index>_request.json`
- `.opm_train/sessions/<session_id>/agents/<agent_id>/llm_calls/<index>_response.json`
- `.opm_train/sessions/<session_id>/agents/<agent_id>/context_compressions/<index>.json`
- `.opm_train/sessions/<session_id>/logs/runtime.log`
- `.opm_train/sessions/<session_id>/logs/errors.jsonl`
- `.opm_train/sessions/<session_id>/timers/module_timings.jsonl`（启用 `--timer` 时写入）

`events.jsonl` 是用于分析/训练的规范原始轨迹日志。
在 `resume` 前，运行时会严格校验 snapshot 与 event-tail（`seq` 连续且数量一致）。

## 配置

通过 `opm_train.toml` 配置：

- Provider 选择（`openrouter`、`tinker`、`custom`）统一走 OpenAI 兼容客户端。
- 运行限制（子代理数量、并发代理数、步数预算）。
- 模型协议解析失败重试控制（`max_protocol_retries`、`protocol_retry_backoff_seconds`）。
- 工具集合与上下文压缩阈值。
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

# 进度

## 2026-03-29

- 为 `opm-train batch-run` 扩展 `--dataset openreward` 模式，新增 OpenReward 参数：`--environment`、`--split`、`--task-index`、`--start`、`--stop`、`--variant`、`--base-url`、`--openreward-tool-format`、`--max-steps`。
- 增加 OpenReward 模式参数校验：`--environment` 必填、`--task-index` 与 `--start/--stop` 互斥、非 OpenReward 数据集仍要求 `--input`、`--max-steps` 必须为正数。
- 在 `batch_runner` 新增 OpenReward 执行后端：通过 `list_tasks/get_task/get_task_range` 选题，使用 `session/call_tool` 执行工具回合，并接入 OpenAI 兼容模型循环与确定性终止条件（`finished`、无工具调用、步数上限）。
- 增加按 provider profile 的工具格式默认策略（`openrouter` -> `openrouter`，其余 -> `openai`），并支持参数显式覆盖。
- 增加 OpenReward 专用产物与指标：
  - `.opm_train/batches/<batch_id>/openreward_results.jsonl`（`environment/split/variant/task_key/task_index/reward_total/finished/tool_calls/turns/session_status/error`）。
  - `.opm_train/batches/<batch_id>/openreward_summary.json`（`total/completed/finished/failed/total_reward/avg_reward`）。
- 增加 OpenReward 续跑能力：以稳定任务键去重（优先 `task_id`，回退 `task_index`/顺序），同时保持 `gsm8k/simple_math/mixed` 现有路径行为不变。
- 新增可选依赖 `.[openreward]`，同步更新 README/README_cn（含 OfficeQA 示例与其他环境切换示例），并补充 OpenReward backend/CLI 参数与输出的测试覆盖。
- 新增 OpenReward 混合选择能力：通过可重复 `--task-spec`（`<split>` 或 `<split>:<start>:<stop>`）在一次批跑中组合多个 split 与区间任务。
- 增加混合选择参数校验（`--task-spec` 与旧选择器 `--task-index` / `--start` / `--stop` 互斥）及多 split/区间展开测试。
- 增强 OpenReward SDK 兼容性：环境解析优先使用显式 `variant` 选择，并在 client 初始化时对 `api_key/base_url` 组合做逐级回退重试。

## 2026-03-25

- 调整 `spawn_agent` 容量上限行为（`max_active_agents` / `max_children_per_agent`）：运行时不再以工具执行异常直接抛错，而是向调用方 agent 返回结构化 `status: rejected` 结果；对应 spawn tool run 记为 `completed` 并带拒绝细节。
- 增加容量受限 spawn 的回归测试，确保不会创建子代理，且拒绝信息可通过 tool 结果稳定观测。
- 在 `opm_train.toml` 中恢复 root/worker 两侧 `shell` 工具放行，和默认运行时工具契约重新对齐。
- 加固 `SessionStorage` 的 JSONL 读取（`load_events`、`load_turns`）：改为按物理行流式读取，不再使用 `splitlines()`，避免 `U+2028` 内容破坏 JSON 解码。
- 加固 SSE 解析：仅按协议 LF 边界切分，不再使用 `splitlines()`，避免 `data:` JSON 内 Unicode 行分隔符导致事件被截断。
- `doctor` 契约检查新增“核心工具集缺项”守卫（`shell` + agent/tool 查询控制 + `compress_context` + `finish`）。
- 修复 spawn 拒绝事件日志的 mypy 类型问题，并补充 shell allow-list、Unicode JSONL/SSE、doctor 核心工具缺项检测的回归测试。

## 2026-03-24

- 修复 OpenAI 兼容 tool-call 重放结构：assistant `tool_calls` 现会在跨轮保留必需的 `type/function` 字段，避免 `messages[*].tool_calls[*].type` 缺失导致的 provider 400。
- 收紧 tool-call 协议为仅 OpenAI 兼容格式（`type=function`、`function.name`、`function.arguments`）；不再接受旧格式 `name/arguments_json`。
- 更新每个 agent 产物目录命名为在前缀和 id 后缀之间加入 name slug（例如 `agent-tester-08277a53f952`），不兼容旧目录名。
- 增加对应回归测试：覆盖 assistant tool-call 重放结构与协议解析兼容性。
- 新增按 step 落盘的轮次索引 `.opm_train/sessions/<session_id>/turns.jsonl`，固定字段覆盖 `turn_id`、step 范围、协议尝试引用、动作/结果、finish 负载与 step 错误。
- 新增 step 级运行时事件：`agent_step_started`、`agent_step_finished`、`llm_call_request_recorded`、`llm_call_response_recorded`。
- 快照写入版本升级为 `schema_version = 4`。
- 新增模块化轨迹导出子系统 `src/opm_train/trajectory`（`loader`、`filter`、`formatter`），支持 session/agent/agent-step 作用域导出。
- CLI 新增 `opm-train export --session-id ... --mode raw|sft`，支持可选 `--agent-id` 与 `--step`。
- 新增导出版本门禁：快照 schema `< 4` 的旧会话会被拒绝导出。
- 增加回归测试：turns 存储与过滤、运行时 turn 索引落盘、轨迹 raw/sft 导出、CLI 导出链路与旧 schema 拒绝。
- 同步更新 README/README_cn：补充 export 用法、turns 索引说明与 schema v4 说明。

## 2026-03-21

- 在 `pyproject.toml` 增加 `sft` 可选依赖组（`tinker`），并将环境安装更新为 `.[dev,sft]`。
- 扩展 README/README_cn：补充 SFT 数据组织方式、JSONL 行级约束与训练数据示例。
- 为每次推理的 LLM request/response 产物新增统一元数据字段：`inference_provider`、`inference_endpoint`、`inference_model`、`inference_parameters`。
- 增加对应集成回归断言，确保推理元数据字段长期稳定。
- 新增模块化 SFT 子系统 `src/opm_train/sft`，包含后端协议/注册表、JSONL 加载器与产物落盘 runner。
- 新增内置 `tinker` SFT 后端：按需加载 SDK、循环 batch、加权 loss 统计与可选训练后采样。
- CLI 新增 `opm-train sft` 子命令，支持 backend/base-model/input/优化器参数，并输出结构化 JSON。
- 新增 SFT 产物目录 `.opm_train/sft_runs/<run_id>/`（`config.json`、`metrics.jsonl`、`result.json`）。
- 更新默认 `provider.tinker` 运行配置为 Tinker OpenAI 兼容推理端点与 sampler 路径模型格式。
- 增加对应回归测试：SFT 解析/后端执行/CLI 输出，以及 Tinker 默认 profile 配置接线。
- 新增可扩展数据子系统 `src/opm_train/data`，提供通用契约（`DatasetSample`、`PreparedTask`、`ValidationResult`、`BatchItemResult`）与适配器注册表。
- 将数学数据适配重构为共享 `MathVerifyDatasetAdapter`，并新增内置 `simple_math` 适配器。
- `gsm8k` 加载阶段改为先提取规范化数值答案用于校验，同时保留原始答案文本。
- 修复 JSONL 解析：改为按物理行流式读取（不再使用 `splitlines()`），避免 Hugging Face 合法样本字段内含 Unicode 分隔符（如 `U+2028`）时误判为坏行。
- 新增批量执行器 `opm_train.batch_runner`，支持并发可配置、单样本容错、实时 JSONL 追加写入与汇总 JSON 输出。
- CLI 新增 `opm-train batch-run` 子命令，支持 `--dataset/--input/--concurrency/--limit/--batch-id/--resume`，输出结构化批量指标与产物路径。
- 新增 `--batch-id` + `--resume` 续跑能力，基于 `results.jsonl` 中已完成 `sample_id` 跳过并继续未完成样本。
- 新增混合数据路由能力（`--dataset mixed` + 每行适配器键），按行分派到对应 adapter 执行。
- 新增 `batch-run --smoke`，无需外部 API Key 可验证端到端批量链路。
- 删除静态 `contract_samples` 样例，避免写死 payload 与运行时契约漂移；契约校验仍由 `doctor` 与工具 schema 驱动。
- 增加对应测试：GSM8K/simple-math 解析与校验、数据注册扩展能力、混合路由、实时写入/续跑行为、batch-run CLI 集成路径（本地 stub 运行时）。
- 增加 active own tool run 的 finish 保护语义：`finish` 不再直接打失败，而是返回结构化拒绝结果（`finish_rejected`、`unfinished_tool_runs`）。
- 增加 `ToolRunStatus.abandoned`，并将 resume 对不可恢复未终态 tool run 的处理改为 `abandoned`（附明确 reason）。
- 增强 `doctor` 输出：新增工具契约校验字段（`tool_contract_ok`、`tool_contract_issues`），并将其纳入 `ready_for_real_run` 判定。
- 增加对应回归测试：finish 拒绝+wait 收敛、仅阻塞当前 agent 自身 tool run、resume 后 abandoned 状态、doctor 契约不一致上报。
- 新增 `mypy` 开发依赖与基础 `[tool.mypy]` 配置，并在 README 增补本地类型检查命令（`mypy src`）。
- 新增 GitHub Actions 工作流 `.github/workflows/ci.yml`，在 push/PR 上执行 `pytest -q` 与 `mypy src`。
- 精简工具注册实现：将逐工具转发函数收敛为声明式方法绑定，并补充执行器参数绑定回归测试。
- 修复 `python -m opm_train.cli` 模块直跑路径：为 `cli.py` 增加 `__main__` 入口并补充回归测试。
- 将工具分发与 LLM 回调中的 awaitable 判定从 `hasattr(__await__)` 统一为 `inspect.isawaitable`。
- 为每个 session 增加 agent 级产物目录：
  - `agents/<agent_id>/llm_calls/`：按顺序保存 request/response。
  - `agents/<agent_id>/context_compressions/`：按顺序保存上下文压缩记录。
- 增加结构化运行观测文件：
  - `logs/runtime.log` 与 `logs/errors.jsonl`。
- CLI 新增 `--timer`（run/resume/smoke），将模块计时写入 `timers/module_timings.jsonl`。
- 增加对应测试，覆盖产物落盘、UTF-8/中文记录、计时输出与错误日志可见性。
- 将 telemetry 相关方法从 `RuntimeOrchestrator` 抽离到 `OrchestratorTelemetryMixin`，模块边界更清晰。
- 在 `SessionStorage` 中为 agent 产物顺序号增加惰性缓存，减少重复目录扫描开销。
- 工具/动作执行错误（含未知 `agent_id`）改为返回结构化错误结果，并同时记录错误日志与 failed tool run，不再通过异常冒泡导致整个 agent loop 失败。
- shell 后台执行新增晚到异常兜底：内部子进程启动等异常会被消费并将对应 run 标记为 failed，避免 run 卡在 `running`。
- `spawn_agent` 在 `blocking=true` 时，顶层 `status` 改为反映子代理终态（`completed` / `failed` / `cancelled`）。
- `cancel_tool_run` 继续只作用于 tool run 生命周期；取消 `spawn_agent` 的 tool run 不会终止已经运行中的子代理。
- 将 spawn/list 相关运行时逻辑拆成更小的 helper，后续扩展 agent/tool 编排更容易。
- SSE 解析器已支持 CRLF/LF 混合分隔的流式响应。
- 新增 `src/opm_train/prompt_assets` 打包提示词资源，默认路径支持安装态回退，并补充 package-data 声明。
- 增加回归测试：未知代理取消、blocking spawn 状态映射、CRLF SSE 解析、打包提示词回退加载。

## 2026-03-20

- 完成全新 `opm-train` 包与 CLI（`run`、`resume`）初始化。
- 增加 `doctor` 与本地确定性 `smoke` 命令，支持“开箱可跑通一遍”。
- 完成 OpenAI 兼容 Provider 客户端与 profile 配置。
- 完成核心编排循环、工具运行时、上下文压缩、JSONL 事件日志与快照恢复。
- 去除冗余的自定义 suppress 实现，并补强 root 任务收尾清理。
- 瘦身 A：移除休眠状态 `AgentStatus.waiting`、无消费字段 `ToolRun.parent_run_id`、无读路径跟踪 `tool_run_owner_agent` 与无调用函数 `stable_json_dumps`。
- 恢复 `runtime.tools.shell_inline_wait_seconds` 并接入 shell 语义：阈值内完成则内联返回，超阈值转后台，`get_tool_run` 可查看累计输出。
- 快照写入 schema 版本提升为 `3` 以匹配瘦身 A 结构。
- 完成 prompt/docs/README/AGENTS 中英镜像。
- 增加 v0 核心行为的单元测试与集成测试。

# 进度

## 2026-03-21

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
- `cancel_agent` 对未知 `agent_id` 改为抛出显式 `ValueError`，不再静默 no-op。
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

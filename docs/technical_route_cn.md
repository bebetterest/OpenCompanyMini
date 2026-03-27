# 技术路线

## 当前基线

- 仓库根目录全新实现 `opm-train`。
- `OpenCompany/` 作为参考实现，继续忽略，不纳入跟踪代码。
- 终端优先、运行时核心模块化。

## v0 核心方向

- 多代理编排，基于显式工具调用与并行子代理执行。
- 工具层按生命周期与能力模块化拆分（session/agent/tools），并通过统一注册表分发工具。
- OpenAI 兼容 Provider 抽象，内置 `openrouter`、`tinker`、`custom`。
- 新增监督微调模块，提供可插拔后端端口（内置 `tinker`）与 JSONL 产物落盘能力。
- 上下文管理采用“自动阈值压缩 + 手动 `compress_context`”。
- 以 JSONL 为规范事件日志，并通过快照实现 resume。
- 增加 step 级 turn 索引（`turns.jsonl`）与按 `session / agent / agent+step` 的轨迹导出链路，便于训练数据筛选。
- resume 前执行严格的 snapshot/event-tail 校验（序号连续且尾部一致）。
- 瘦身 A 去除休眠状态与无消费字段，并保留 shell 内联等待预算 `shell_inline_wait_seconds`。

## 延后模块

- v0 不实现 `sandbox`、`mcp`、`skills`。
- 预留稳定的 no-op 扩展接口供后续版本接入。

## 工程原则

- 遵循 Bitter Lesson 与第一性原理。
- 最小策略分支与显式运行状态。
- 保持循环行为可重载（`LoopHooks`）。

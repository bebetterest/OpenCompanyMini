# OPM-Train 代理原则

- 将 `opm-train` 构建为终端优先、模块化的多代理运行时，服务研究与训练。
- 遵循 Bitter Lesson 与第一性原理：优先简单、可扩展的基础原语，避免脆弱的人为流程分支。
- 开发与验证统一使用 Conda 环境 `OpenCompany`（Python 3.12）。
- Root 以组织为主，Worker 以执行为主。
- 为每个代理保留显式的父子关系与生命周期状态。
- 将编排限制显式配置化：子代理数量、活跃代理数、步数预算。
- 工具集保持最小可组合：shell、spawn/steer/cancel、运行查询/等待/取消、上下文压缩、finish。
- 核心循环独立在 `AgentLoopRunner`，并通过 `LoopHooks` 扩展。
- Prompt 统一集中在 `prompts/`，运行时仅使用英文 prompt。
- Prompt/docs/README/AGENTS 维持中英镜像（`*_cn`）结构和事实同步。
- 以 JSONL 持久化规范原始轨迹，并用轻量快照支持 resume。
- 恢复会话前严格校验快照与事件尾一致性。
- v0 不实现 `sandbox`、`mcp`、`skills`，但预留扩展接口。
- 结合模块测试和系统测试，避免重复踩坑。
- 行为变化时同步更新文档及中英镜像。

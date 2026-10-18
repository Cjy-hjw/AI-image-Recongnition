# AGENTS.md — 大赛仓存档架构索引（竞赛已收官，2026-10-18 起只读）

> 本仓 = 2026 医保影像 AI 大赛赛道七交付仓（初赛 145.03/全国 10，已收官存档）。本文件只做**模块规则索引**，各模块职责/硬约束/实测口径在对应 AGENTS.md。

| 模块 | 规则文件 |
| :--- | :--- |
| 全局规则与验证门 | `/rules.md`、`/QA_VP.md` |
| 模型与损失 | `/src/models/AGENTS.md` |
| 训练管线 | `/src/training/AGENTS.md` |
| 数据加载 | `/src/dataset/AGENTS.md` |
| 推理管线 | `/src/inference/AGENTS.md` |
| gatekeeper | `/src/gatekeeper/AGENTS.md` |
| 服务层 | `/src/service/AGENTS.md` |
| 脚本 | `/scripts/AGENTS.md` |
| 测试计数单一权威 | `/tests/AGENTS.md` |
| CI 与 import 契约 | `/.github/workflows/ci.yml`、`/.importlinter` |
| 会话自动化护栏 | `/.claude/hooks/AGENTS.md` |

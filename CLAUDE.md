# CLAUDE.md — 项目级指令(每次会话自动加载)

> 甲状腺超声多任务诊断系统(2026 全国医保影像AI识图大赛 · 赛道七)。
> 下方 `@` 导入的运行规则与质量门**强制生效**,等同正文。

## 强制规则(导入)

@rules.md

## 按需读(不常驻)

- `QA_VP.md` — 代码审查 / 验证门细则,由 `rules.md §2` 在跑 Gate 2 时触发去读。
- 各目录 `AGENTS.md` — 模块级规则(架构 / 训练 / 推理 / 数据 / 测试 / gatekeeper),**改某模块前先读其 `AGENTS.md`**;总索引见根目录 `AGENTS.md`。

## 项目要点

- **比赛约束**:赛道七**仅「自有模型组」**,交付为离线推理容器(`predict.py` 单图 + `video_predict.py` 动态图,`entrypoint.sh` 按扩展名自动分发),打包 gatekeeper + TaskAlignmentAdaptor + CBM 报告生成器。
- **当前卡点**:官方数据 **2026-08-01 初赛启动**后才可用、**禁本地、JPG、含动静态图**;在此之前核心三项(真实性 / 规范性 / 良恶性)无法真训。进度见 `docs/之前表现.md`、`docs/比赛对齐与风险.md`。
- **环境**:训练在服务器 A800;本地 Python 见全局 `~/.claude/CLAUDE.md`。
- **代理配置(本地访问外网必需)**:访问 HuggingFace/GitHub/外网资源时,**PowerShell 必须先设置代理环境变量**:
  ```powershell
  $env:http_proxy="http://127.0.0.1:1203"
  $env:https_proxy="http://127.0.0.1:1203"
  $env:NODE_OPTIONS="--dns-result-order=ipv4first"
  ```
  适用场景:`pip install`、`huggingface_hub` 下载、`git push/fetch`(见 `rules.md` §5 已有临时环境变量规则)、`curl`/`wget` 等。

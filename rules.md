# Repository Operational Rules & Verification Protocol

## 1. Resource Pre-checks (Avoid Resource Contention)
Before launching any training script on the server, you **MUST** run:
```bash
nvidia-smi
ps aux | grep python
```
* **Rule**: If other users have active training jobs utilizing the same GPU resources (high VRAM/compute usage), notify the team or wait for resources to clear. Do not double-schedule.

## 2. Compile & Shape Verification & Code Review (Pre-flight Checks)
Before deploying or starting full training, you **MUST** execute the double-gate pre-flight checks:
1. **Gate 1: Quantifiable Code Verification (Test Scripts)**: Run `verify_model.py` to perform a single forward/backward pass with standard normalized input dummy tensors. It must successfully check tensor shape compatibility and verify valid, non-zero, non-NaN gradient generation.
2. **Gate 2: Non-quantifiable Logic Review (Subagent Review)**: For complex mathematical pipelines (such as dual minimax optimization, variational mutual information decoupling, parameter registration in optimizers, gradient leaking across subnets), the agent **MUST** spawn a specialist `reviewer` subagent to audit code changes and confirm dynamic logic correctness.
*   **Protocol Reference**: The detailed quality gates, assertion guidelines, and review templates for dynamic gradient routing, optimizer binding, and BatchNorm freeze are defined in **`QA_VP.md`**. You MUST read and enforce `QA_VP.md` as the safety sub-protocol of this rulebook.
*   **Zero-Hallucination Diagnostic Loop (自诊断闭环)**:
    * If `verify_model.py` fails any Gate (Exit Code != 0), the AI agent **MUST NOT** output standard raw stack tracebacks to the human developer. Instead, it must print a structured diagnostic report (containing the failed Gate, numerical discrepancy evidence, and recovery guidance), execute patches locally in a sandbox, and verify correctness before requesting a git diff review.
* **Rule**: Never run standard training without both Gate 1 (Test verification) and Gate 2 (Subagent review) passing first.
* **Rule (CI 只验证已 push 的 commit)**: 声称「CI 通过」前,必须跑 `gh run list --branch <branch> --limit 1`(走代理)核对其 `headSha` **正是本轮 commit** 且 `conclusion=success`——`pull_request` 事件的 run message 取自 PR 而非最新 commit,只能凭 SHA 判定。本地未 push 的改动 CI 从未覆盖,不得用本地 pytest 全绿或历史 commit 绿灯冒充;对不上则如实说明「本地绿、CI 未覆盖本轮」。
* **OOM Preventative Rule → `src/models/AGENTS.md` §4**: See model architecture constraints for spatial attention resolution limits. **官方平台算力硬上界(答疑第4期 Q33):单卡 32G 显存、16 核 CPU、96G 内存、419 TFLOPS(FP16 稀疏)**——batch/分辨率/AMP 规划不得超此显存;本地 A800(80G)上不 OOM 不代表平台 32G 不 OOM,交付前须按 32G 复核。

## 3. Early Failure & Short-circuiting (Early Stopping Validation)
To prevent wasting GPU hours on diverging models, we enforce a training gatekeeper rule:
* **Rule**: If `val_dice` is `< 0.10` or classification metrics are essentially random after Epoch 3, the training script must auto-abort. Do not let dead runs finish.

## 4. GradScaler & AMP Rules → `src/training/AGENTS.md` §2
*See `src/training/AGENTS.md` §2 "GradScaler Protocol" for mixed precision rules.*
## 5. Execution & Lifecycle Management
* **Rule (Long-term Runs)**: If a training session is confirmed to be stable (i.e. model compiles, gradient flows, and metrics like Dice/Loss improve steadily over the first 5+ epochs), you **MAY** yield and stop the task session to save token resource. If it is a quick test, you must wait and monitor it inline.
* **Rule (Deep Research Gate)**: Before any major architectural iteration or complex optimization strategy change, the agent **MUST** generate a `提示词.md` and instruct the user to run a Deep Research query (via o1/o3-mini or specialized agents) to fetch the latest medical imaging constraints.
* **Rule (Deep Research Grounding)**: To prevent mathematical formulation flaws, grid/shape alignment mismatches, and numpy/pytorch type coercion bugs (e.g. bitwise ufunc errors), the agent **MUST** feed the exact relevant codebase implementation snippets (especially morphological renderers, data shapes, and tensor slicing indices) as grounding context into the Deep Research query prompt. Abstract theoretical queries are strictly discouraged.
* **Rule (Dynamic Maintenance 代码-文档同步)**: `rules.md` 与各级 `AGENTS.md` 必须随依赖、路径、约束的演进动态更新。**任何改变模块行为/接口/测试数/检测方案的代码改动,在同一提交内必须同步更新所有引用该行为的文档**(模块 `AGENTS.md`、根 `AGENTS.md` 索引、`docs/` 技术报告、测试统计等);**提交前必须自检**:`grep` 受影响的关键词(如算法名、旧测试数、旧 xfail/能力描述),确认无文档仍停留在改动前的旧状态。漏同步等同未完成改动。
  - 反例(必须避免):重写 `detect_splicing` 后只改了实现,却遗漏 `tests/AGENTS.md`、根 `AGENTS.md`、`src/gatekeeper/AGENTS.md` 三处仍写「xfail / 68 passed / Sobel 方差」的旧记录。正确做法:改完代码立即全仓 `grep` 旧关键词,一并修正再提交。
  - **过时文档三态处置(改动后必判)**:代码改动后,对受影响文档(`docs/`、各级 `AGENTS.md`、`reference/` 等)逐份判定并执行三选一——① **该改则改**:仍有价值但叙述/数值过时(如测试数、能力描述、主结论被新结论推翻),原地更新使其与现状一致(遵循「细节优先」不丢硬约束);② **该删则删**:整份内容已被取代且无独立历史价值(如已落地功能的临时提示词、被新报告完全覆盖的旧调研),连同其全仓引用一并删除;③ **不该动则保持**:官方原文(赛题/答疑)、自带「旧口径警告」的历史记录、与本次改动无关的文档,一律不动。**判定有疑则归为「改」或「保持」,不轻易删**——删除是不可逆动作,宁可留过时标注也不误删有用历史。
* **Rule (Phased Refactoring)**: 涉及多项复杂 ML 改动(CEM/KAN/损失重构/路径门控等)时**禁止单体重写**。必须拆成顺序、低风险、可独立验证的阶段(如 Phase 1 移除一致性损失 → Phase 2 门控优化 → Phase 3 CEM/MLP 瓶颈),每阶段跑 `verify_model.py` + Holdout 评估通过后方可进入下一阶段。
* **Rule (Detail-Preserving Edits 细节优先)**: 精简或重构任何文档(`rules.md`/`AGENTS.md`/`QA_VP.md`/`docs/`)时,**最大限度保留技术细节**——命令、数值阈值(如 FP16 上限 65504、clamp ±15、泄漏阈值 1e-7)、assert 代码、字段清单、以及规则背后的「为什么」均不得为求简洁而删除。只允许删:① 确定的跨文件重复;② 已落地的过时叙述;③ 纯修辞性赘语。**有疑则保留**,宁可冗长不可丢失硬约束。
* **Rule (Git Push 走代理端口)**: 本机访问 GitHub **必须经本地代理 `127.0.0.1:1203`**,否则 `git push`/`git fetch`/`gh` 会卡住或超时。**不要写进 `git config`(保持仓库干净)**,而是**用时临时带环境变量**:
  ```bash
  HTTP_PROXY=http://127.0.0.1:1203 HTTPS_PROXY=http://127.0.0.1:1203 git push origin <branch>
  ```
  仅访问外网(push/fetch/clone/gh)时加;本地 commit/diff/log 无需代理。

## 6. Code Modularity & Maintainability
* **Rule (Functional Decomposition)**: Every Python module must maintain a clear single responsibility. When a file exceeds 300 lines or contains more than 3 distinct functional domains (e.g., model architecture + physical priors + classification heads), it **MUST** be decomposed into separate modules.
* **Implementation Standard**: 
  - Use `__init__.py` re-export shims to maintain backward compatibility
  - Group related classes/functions by domain (e.g., `losses/segmentation.py`, `losses/classification.py`)
  - Keep training logic separate from model definitions (e.g., `src/training/` vs `src/models/`)

## 7. Official Category Constraint (Project-Level Business Rule)
* **Rule (Official Category Constraint)**: As per 2026 official regulations, Track 7 does NOT offer a "Self-built Model Group" (自建模型组). Our team is strictly registered under the **"Own Model Group" (自有模型组)**. The deployment must package our custom gatekeeper, TaskAlignmentAdaptor cross-task alignment, and ConceptEmbedding CBM report generator into a standard offline inference container via `predict.py`.

## 8. 子Agent模型选择策略
* **Rule**: 不重要或低风险的子任务(如搜索文件、简单检查、格式化)创建子agent时使用便宜模型(`model="sonnet"`);重要任务(如代码审查、逻辑验证、复杂重构)使用主会话同型号(默认)。节省 token 预算。
# Repository Operational Rules & Verification Protocol

## 1. Resource Pre-checks (Avoid Resource Contention)
Before launching any training script on the server, you **MUST** run:
```bash
nvidia-smi
ps aux | grep python
```
* **Rule**: 确认无其他用户高占用（高 VRAM/高计算）后再提交训练任务；有活跃任务则通知团队或等待资源释放。

## 2. Compile & Shape Verification & Code Review (Pre-flight Checks)
Before deploying or starting full training, you **MUST** execute the double-gate pre-flight checks:
1. **Gate 1: Quantifiable Code Verification (Test Scripts)**: Run `verify_model.py` to perform a single forward/backward pass with standard normalized input dummy tensors. It must successfully check tensor shape compatibility and verify valid, non-zero, non-NaN gradient generation.
2. **Gate 2: Non-quantifiable Logic Review (Subagent Review)**: For complex mathematical pipelines (such as dual minimax optimization, variational mutual information decoupling, parameter registration in optimizers, gradient leaking across subnets), the agent **MUST** spawn a specialist `reviewer` subagent to audit code changes and confirm dynamic logic correctness.
*   **Protocol Reference**: The detailed quality gates, assertion guidelines, and review templates for dynamic gradient routing, optimizer binding, and BatchNorm freeze are defined in **`QA_VP.md`**. You MUST read and enforce `QA_VP.md` as the safety sub-protocol of this rulebook.
*   **Gate2 审查搭配规范（通用 8 角度 × `QA_VP.md` 清单，二者互补，缺一都漏）**:
    - **对抗性审计前提(2026-08-11 新增,统摄两步)**: 所有 review 子agent 一律以「**这段代码有 bug,我的任务是找出来**」为默认前提,而非「确认它没问题」——举证责任反转、主动构造反例对拍、不采信注释/文档声称、判 PASS 也要给「试过什么反例才排除」的依据。细则见 `QA_VP.md §2.0`,派 reviewer 时必须把该心态写进其 prompt。
    - **第一步 → 通用 8 角度扫描（独立 sonnet 子agent,`effort=xhigh`,禁 haiku）**: 8 角度(行级 / 移除行为 / 跨文件调用 / 复用 / 简化 / 效率 / altitude / CLAUDE.md 约定)——擅长抓通用缺陷(反转条件/off-by-one/调用契约破坏/死代码),但**不懂本项目领域红线**。**2026-08-11 起改用独立子agent,不再用 `/code-review` skill**(skill forked 后台执行太慢、易 API 中断,本 session 连续两次异常);独立 agent 可控/可并行/可 SendMessage 追问。见 `memory/code-review-skill-too-slow-use-independent-agent.md`。
    - **第二步 → `QA_VP.md §1/§2` 领域清单**: 第一步扫完后，对照 §1.1 梯度隔离 / §1.2 优化器绑定 / §1.3 FP16 数值 / §1.4 患者泄漏 / §1.5 损失维度 / §1.6 DICOM 脱敏 / **§1.7 推理侧红线**（误报/维度契约/降级路径） / §2.1 符号极性 / §2.2 广播，逐条核对**改动触及的**项（未触及的跳过）。
    - **谁做**: 高风险改动（梯度路由/优化器绑定/损失数学/gatekeeper 降级路径）→ 派 `model="sonnet"` 及以上子 agent 审；低风险改动 → 主会话自审。**Gate2 禁用 haiku**（能力不足）。
    - **反例（本次教训）**: 只跑 skill 跳过 QA 清单 = 漏领域红线；只走清单跳过 skill = 漏通用缺陷。
    - **反例二（2026-07-30 真实违规·谎报审查）**：#87 高风险改动（`authenticity.py` 的 `EXPECTED_CKPT_SHA256` 封印）只跑测试就 `mark_gate2_pass.py` + commit，**reviewer 子agent 整个跳过**；被问"review 了？"时用上一轮结果冒充本轮。漏洞本质：`gate2_enforce.py` 只验"`.gate2_pass.json` 存在且未过期"，但标记是模型自己写的＝**自己给自己签发通行证**——hook 防"忘了标记"，防不了"没真审就标记"。→ 已给 `mark_gate2_pass.py` 加 `--evidence` 必填（拒空）、`gate2_enforce.py` 校验 evidence 非空（旧标记视为未审强制重标）。**边界**：evidence 防不了"伪造证据"，这是 hook 层天花板——真正兜底是执行纪律 + 下条"逐轮对应"铁律。
    - **回答"review了？"的铁律（2026-07-30 新增）**：被问"代码 review 了？"时，**必须逐轮对应**——只能回答"本轮 reviewer agent 结论是 XX（task <id>）"或"本轮 `/code-review` 结论是 XX"，**禁止用上一轮的结果冒充本轮**。若本轮确实没跑，如实说"没跑，现在补"——谎报比漏审恶劣得多（漏审是疏忽，谎报是欺骗）。
*   **Zero-Hallucination Diagnostic Loop (自诊断闭环)**:
    * If `verify_model.py` fails any Gate (Exit Code != 0), the AI agent **MUST** print a structured diagnostic report (containing the failed Gate, numerical discrepancy evidence, and recovery guidance), execute patches locally in a sandbox, and verify correctness before requesting a git diff review.
* **Rule**: Gate 1（`python scripts/verify_model.py` 全通过）**且** Gate 2（reviewer 子agent 审核通过）后，才启动正式训练。
* **Rule (Gate2 触发时机 · 推理侧/交付侧代码)**: 上一条的触发点写的是「启动正式训练」，但**推理容器代码（`predict.py`/`video_predict.py`/`src/inference/**`/`src/gatekeeper/**`/`entrypoint.sh`）永远不会触发训练**——若照字面理解，这类改动等于没有 Gate2 触发点。**对推理侧/交付侧改动，Gate2 的触发点是「commit 之前」**，不是训练前、也不是 push 后补做。
  - **反例（2026-07-29 真实违规）**：改完概率输出 + gatekeeper 软化后，只跑了搭配规范**第二步**（QA_VP 清单 + reviewer），**跳过第一步 `/code-review` skill** 就 commit+push；事后补跑 skill 查出 6 问题，其中一条是修 Gate2 nits 时**自己引入的回归**（`_assert_finite` 在逐帧循环里无隔离，单帧 NaN 拖垮整段视频）。教训：① **两步缺一都漏**（执行纪律问题非规则缺陷）；② **修复本身也要过 Gate2**——按审查意见改完的代码与原始改动同等风险，本次正是"补丁引入新回归"。
  - **顺序不可颠倒**：先 `/code-review` skill（通用 8 角度）→ 再 `QA_VP.md` 领域清单。颠倒或只做一步都视为未完成 Gate2。
* **Rule (CI 只验证已 push 的 commit)**: 声称「CI 通过」前,必须跑 `gh run list --branch <branch> --limit 1`(走代理)核对其 `headSha` **正是本轮 commit** 且 `conclusion=success`——`pull_request` 事件的 run message 取自 PR 而非最新 commit,只能凭 SHA 判定。本地未 push 的改动 CI 从未覆盖,不得用本地 pytest 全绿或历史 commit 绿灯冒充;对不上则如实说明「本地绿、CI 未覆盖本轮」。
* **OOM Preventative Rule → `src/models/AGENTS.md` §4**: See model architecture constraints for spatial attention resolution limits. **官方平台算力硬上界(答疑第4期 Q33):单卡 32G 显存、16 核 CPU、96G 内存、419 TFLOPS(FP16 稀疏)**——batch/分辨率/AMP 规划不得超此显存;本地 A800(80G)上不 OOM 不代表平台 32G 不 OOM,交付前须按 32G 复核。

## 3. Early Failure & Short-circuiting (Early Stopping Validation)
To prevent wasting GPU hours on diverging models, we enforce a training gatekeeper rule:
* **Rule**: Epoch 3 后 `val_dice < 0.10` 或分类指标随机时，训练脚本自动中止并记录原因，释放 GPU 资源。

## 4. GradScaler & AMP Rules → `src/training/AGENTS.md` §2
*See `src/training/AGENTS.md` §2 "GradScaler Protocol" for mixed precision rules.*
## 5. Execution & Lifecycle Management
* **Rule (Long-term Runs)**: If a training session is confirmed to be stable (i.e. model compiles, gradient flows, and metrics like Dice/Loss improve steadily over the first 5+ epochs), you **MAY** yield and stop the task session to save token resource. If it is a quick test, you must wait and monitor it inline.
* **Rule (Deep Research Gate)**: Before any major architectural iteration or complex optimization strategy change, the agent **MUST** generate a `提示词.md` and instruct the user to run a Deep Research query (via o1/o3-mini or specialized agents) to fetch the latest medical imaging constraints.
* **Rule (Deep Research Grounding)**: Deep Research 查询必须附带相关代码片段（形态学 renderer 实现、数据形状、tensor 切片索引）作为 grounding 上下文，确保数学公式、grid/shape 对齐、numpy/pytorch 类型强制均基于实际实现。
* **Rule (长输出落文件)**: 凡**调研提示词、Deep Research 结论、大段方案/报告**，一律创建 `.md` 文件（提示词进 `docs/提示词_<主题>.md`，调研结论进 `docs/`），**不直接刷屏输出到对话**。理由：① 可复用、可 diff、可追溯；② 不污染对话上下文触发压缩。对话里只留一句"已写入 `<路径>`"+关键结论摘要。
* **Rule (先证后改·别猜)**: 遇到**反常数字或矛盾现象**（如 probe 0.71 vs 训练 0.50、指标与预期不符），**第一反应必须是对拍实验/质疑评估协议/Deep Research 查经典坑，用数字锁死根因后再动手**，严禁逐个改超参/改结构盲试。血泪教训：malignancy 预训练曾因轻信随机 KFold 的 probe 0.705（实为 patient 泄漏虚高，无泄漏真值 0.50），一路在分类头上瞎改（绕CBM→linear_head→BatchNorm→冻结backbone）全是错方向，折腾大半天；一次对拍实验（同特征只换划分方式，0.71→0.50）即锁死真相。详见 `memory/patient-leakage-and-probe-trap.md`。
  - **医学影像评估铁律**: 患者级标签的数据，**所有评估一律按患者分组**（`patient_grouped_split` / `GroupKFold(patient_id)`），**禁用随机 KFold / train_test_split**——同患者多帧分进 train+val 会造成泄漏、AUC 虚高，是最常见的自欺陷阱。
* **Rule (Dynamic Maintenance 代码-文档同步)**: `rules.md` 与各级 `AGENTS.md` 必须随依赖、路径、约束的演进动态更新。**任何改变模块行为/接口/测试数/检测方案的代码改动,在同一提交内必须同步更新所有引用该行为的文档**(模块 `AGENTS.md`、根 `AGENTS.md` 索引、`docs/` 技术报告、测试统计等);**提交前必须自检**:`grep` 受影响的关键词(如算法名、旧测试数、旧 xfail/能力描述),确认无文档仍停留在改动前的旧状态。漏同步等同未完成改动。
  - 反例(必须避免):重写 `detect_splicing` 后只改了实现,却遗漏 `tests/AGENTS.md`、根 `AGENTS.md`、`src/gatekeeper/AGENTS.md` 三处仍写「xfail / 68 passed / Sobel 方差」的旧记录。正确做法:改完代码立即全仓 `grep` 旧关键词,一并修正再提交。
  - **过时文档三态处置(改动后必判)**:代码改动后,对受影响文档(`docs/`、各级 `AGENTS.md`、`reference/` 等)逐份判定并执行三选一——① **该改则改**:仍有价值但叙述/数值过时(如测试数、能力描述、主结论被新结论推翻),原地更新使其与现状一致(遵循「细节优先」不丢硬约束);② **该删则删**:整份内容已被取代且无独立历史价值(如已落地功能的临时提示词、被新报告完全覆盖的旧调研),连同其全仓引用一并删除;③ **不该动则保持**:官方原文(赛题/答疑)、自带「旧口径警告」的历史记录、与本次改动无关的文档,一律不动。**判定有疑则归为「改」或「保持」,不轻易删**——删除是不可逆动作,宁可留过时标注也不误删有用历史。
* **Rule (Phased Refactoring)**: 涉及多项复杂 ML 改动(CEM/KAN/损失重构/路径门控等)时**禁止单体重写**。必须拆成顺序、低风险、可独立验证的阶段(如 Phase 1 移除一致性损失 → Phase 2 门控优化 → Phase 3 CEM/MLP 瓶颈),每阶段跑 `verify_model.py` + Holdout 评估通过后方可进入下一阶段。
* **Rule (Detail-Preserving Edits 细节优先)**: 精简或重构任何文档(`rules.md`/`AGENTS.md`/`QA_VP.md`/`docs/`)时,**最大限度保留技术细节**——命令、数值阈值(如 FP16 上限 65504、clamp ±15、泄漏阈值 1e-7)、assert 代码、字段清单、以及规则背后的「为什么」均不得为求简洁而删除。只允许删:① 确定的跨文件重复;② 已落地的过时叙述;③ 纯修辞性赘语。**有疑则保留**,宁可冗长不可丢失硬约束。
* **Rule (Git Push 走代理端口)**: 本机访问 GitHub **必须经本地代理 `127.0.0.1:1203`**,否则 `git push`/`git fetch`/`gh` 会卡住或超时。代理通过**临时环境变量**传入（git config 保持不变，不写入仓库）：
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
* **Rule (Official Category Constraint)**: As per 2026 official regulations, Track 7 does NOT offer a "Self-built Model Group" (自建模型组). Our team is strictly registered under the **"Own Model Group" (自有模型组)**. The deployment packages our custom gatekeeper, TaskAlignmentAdaptor cross-task alignment, and ConceptEmbedding CBM report generator into a **resident HTTP inference service** (container `ENTRYPOINT = python -m src.service.app`, uvicorn:8000, batch-directory `/call` + `/health` probe + callback; per-case `{PatientId}/prediction.json` + per-nodule `.nii.gz`). `predict.py`/`video_predict.py`/`entrypoint.sh` are demoted to local/CI single-file debug bypass, no longer the production entry.

## 8. 子Agent委派策略(强制优先,勿事必躬亲)
* **Rule (默认委派 Delegate-by-Default)**: 遇到**可独立界定的子任务**——搜索/定位文件、批量读取多文件后归纳、代码审查、文档冗余排查、格式化、跑测试并汇总——**必须优先派子agent处理**,不得默认亲历亲为。理由有三,缺一不可:
  1. **省额度**:简单/低风险**非审查**子任务用便宜模型(`model="haiku"`),把 Opus 额度留给主线决策(审查类禁 haiku,见下条模型选型);
  2. **不污染主上下文**:子agent 只回传结论,大量文件内容/中间过程不进主会话窗口,避免主线被稀释、触发压缩;
  3. **并行提速**:相互独立的子任务在同一条消息里一次性并发派发(多个 Agent 调用同发),墙钟时间取最慢的一个,而非串行累加。
* **Rule (模型选型)**: 低风险**非审查**子任务(搜索、定位文件、格式化、跑既有测试并汇总)→ `model="haiku"` 可用;中等子任务(文档归纳、批量读文件)→ `model="sonnet"`；**高风险子任务(数学管道/梯度路由/优化器绑定审查 Gate2、复杂重构、逻辑正确性验证)→ `model="sonnet"` 或主会话同型号**。**一切 review/审查类任务(Gate2 代码审查、plan 审查、文档一致性审查、冗余排查)一律禁用 haiku**(2026-08-11 用户明确指令:haiku 能力不足以理解领域红线/梯度路由/契约,审查用它等于没审),必须 sonnet 及以上。见 §2 Gate2 必须派 `reviewer` 子agent 的硬性要求。
* **Rule (何时才自己做)**: 仅当任务**极短**(1-2 步可完成)、**强依赖主线实时上下文**(拆出去反而要重新灌大量背景)、或**需连续交互决策**时,才在主会话直接做。其余一律优先委派。
* **Rule (并行写代码 worktree 隔离)**: 多个**互不依赖的代码模块**需同时编写时(如同时补测试、同时改多个独立 AGENTS.md、同时写不相关的工具脚本),派子agent 并行写代码并传入 `isolation="worktree"` 参数——每个子agent 在独立 git worktree 分支上操作,互不冲突;主线收到结论后合并。**不需要 worktree 的情况**:有依赖顺序、或改动量极小(1-2 文件)的任务,直接串行更简洁。
* **Rule (允许子Agent 再派子Agent · 套娃 · 2026-08-11 放开)**: 子agent **可以**再开自己的子agent(嵌套委派),不再限制单层。适用:子agent 接到一个大任务后,自行拆成并行子任务下派(如"审查整个模块"→子agent 再派多个 agent 分文件审)。约束仍在:嵌套的审查类子agent 同样禁 haiku;嵌套要有实际收益(拆分/并行),别为套娃而套娃徒增开销。
* **反例(必须避免)**:一次性顺序读十几个文件做审查、把整个代码库/多文档拉进主上下文人工比对冗余、串行跑多组独立测试、主线亲自写低风险辅助脚本——这些都应拆成并行子agent,否则既费额度又污染上下文。
* **Rule (审查子agent prompt 必带对抗心态 · 2026-08-11 新增)**: 派任何 review/审查类子agent(Gate2 代码审查、plan 审查、文档一致性审查)时,**其 prompt 里 MUST 显式写明对抗性审计前提**——「默认这段代码/plan 有 bug,你的任务是主动找出来,不是确认它没问题」,并要求:举证责任反转(默认错、由证据证对)、主动构造反例/对拍实测、不采信被审对象的注释/文档/commit 声称、判 PASS 也要给「试过什么反例才排除」的依据。**只说"审查一下 X"而不带对抗心态 = 默认放行式轻审,视为未按规范派审**。细则见 `QA_VP.md §2.0`。喂 code-review findings 给复核 agent 时,让其**独立裁决真假**而非复述——防串供、防确认偏误。
* **Rule (Plan 退出前必审 · 2026-08-10 新增)**: **每次进 plan mode 写完 plan、`ExitPlanMode` 之前,必须派子agent(`model="sonnet"` 及以上,禁 haiku)审查该 plan 的技术正确性**——核对 plan 的关键技术判断是否与代码事实一致(签名/契约/控制流/依赖假设),据其结论修正 plan 后才退出。理由:plan 是后续所有实现的蓝图,一处与代码矛盾的假设(如"某函数返回 X"实则不返回)会让整个阶段返工;子agent 用干净上下文对照真实代码审 plan,能抓出主线在规划热度中漏掉的硬伤。审查子agent 须实际读 plan 引用的每个事实源文件,不得凭 plan 自述下结论,且按上一条带对抗心态审。
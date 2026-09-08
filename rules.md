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
    - **第一步 → 通用 8 角度扫描（独立 sonnet 子agent,`effort=xhigh`,禁 haiku,**必须分多路并行、每路只管 1-2 个角度,禁单 agent 包揽 8 角度 —— 见 §8 同名规则与实证**）**: 8 角度(行级 / 移除行为 / 跨文件调用 / 复用 / 简化 / 效率 / altitude / CLAUDE.md 约定)——擅长抓通用缺陷(反转条件/off-by-one/调用契约破坏/死代码),但**不懂本项目领域红线**。**2026-08-11 起改用独立子agent,不再用 `/code-review` skill**(skill forked 后台执行太慢、易 API 中断,本 session 连续两次异常);独立 agent 可控/可并行/可 SendMessage 追问。见 `memory/code-review-skill-too-slow-use-independent-agent.md`。
    - **第二步 → `QA_VP.md §1/§2` 领域清单**: 第一步扫完后，对照 §1.1 梯度隔离 / §1.2 优化器绑定 / §1.3 FP16 数值 / §1.4 患者泄漏 / §1.5 损失维度 / §1.6 DICOM 脱敏 / **§1.7 推理侧红线**（误报/维度契约/降级路径） / §2.1 符号极性 / §2.2 广播，逐条核对**改动触及的**项（未触及的跳过）。
    - **谁做**: 高风险改动（梯度路由/优化器绑定/损失数学/gatekeeper 降级路径）→ 派 `model="sonnet"` 及以上子 agent 审；低风险改动 → 主会话自审。**Gate2 禁用 haiku**（能力不足）。
    - **反例（本次教训）**: 只跑第一步(独立 agent 8 角度)跳过 QA 清单 = 漏领域红线；只走清单跳过第一步 = 漏通用缺陷。
    - **🔴 审查 agent 的证据必须不可伪造（2026-08-12 新增，本轮 6 条误报实证）**：审查 agent 报的 finding **不得直接采信**。这轮独立 sonnet 报 5 条，**3 条误报**；memory 整理 agent 又误判 1 条。归类后病根有三：① **读片段就外推**（说「gamma 不在优化器」但没读排除条件是 `"vclub" in name`；说「opencv 要 numpy≥2 会冲突」但只看本地已装版本的元数据、没跑 pip 解析；说「runner.py 已加 encoding」但 grep 实为零命中）② **概念性误解**（说 `_seed_worker` 在 `persistent_workers` 下「压窄训练多样性」——设种子不减少单次运行内的随机性）③ 🔴 **声称验证但结论相反**（说「`WeightedRandomSampler` 不消费 generator」，**自称读过 torch2.8 源码**，而源码 `__iter__` 明写 `torch.multinomial(..., generator=self.generator)`）。第三类最危险：**要求 agent 写「你做了什么验证」，反而给了它编造证据的模板**。
      - **操作细则(交产物不交自述 / 标 evidence 类型 / 实测须有区分力 / 主会话独立重跑 / 不让 agent 判 commit)统一见 `QA_VP.md §2.0`**，此处不重复。以下两条是本规则特有的判断原则：
      - **分工摆正**：sonnet 的价值在**广度覆盖**（本轮确实抓到 2 条主会话不会注意的真 bug：测试里 `WEIGHT_DECAY` 写成 `1e-4` 而 config 是 `5e-4`、`src/models/AGENTS.md` 残留已被提取的内联公式），**不在判断力**。深度裁决归主会话。
      - **「过时」标注同样要实跑核实**：memory 整理时曾把一条**正确**的 GBK 记录标成「已修复、可本地跑」（依据是 grep 到某字符串），实跑 `verify_model.py` 当场 `UnicodeDecodeError`。**差点反向误导后人**——判定「已过时」与判定「是 bug」同等需要证据。（该 GBK bug 已于 2026-08-12 真修复并实跑 EXIT=0 核实，Gate1 现可本地跑；本条保留的是**方法论**，不是那个 bug 的状态。）

    - **反例二（2026-07-30 真实违规·谎报审查）**：#87 高风险改动（`authenticity.py` 的 `EXPECTED_CKPT_SHA256` 封印）只跑测试就 `mark_gate2_pass.py` + commit，**reviewer 子agent 整个跳过**；被问"review 了？"时用上一轮结果冒充本轮。漏洞本质：`gate2_enforce.py` 只验"`.gate2_pass.json` 存在且未过期"，但标记是模型自己写的＝**自己给自己签发通行证**——hook 防"忘了标记"，防不了"没真审就标记"。→ 已给 `mark_gate2_pass.py` 加 `--evidence` 必填（拒空）、`gate2_enforce.py` 校验 evidence 非空（旧标记视为未审强制重标）。**边界**：evidence 防不了"伪造证据"，这是 hook 层天花板——真正兜底是执行纪律 + 下条"逐轮对应"铁律。
    - **回答"review了？"的铁律（2026-07-30 新增）**：被问"代码 review 了？"时，**必须逐轮对应**——只能回答"本轮独立 reviewer 子agent 结论是 XX（task <id>）"，**禁止用上一轮的结果冒充本轮**。若本轮确实没跑，如实说"没跑，现在补"——谎报比漏审恶劣得多（漏审是疏忽，谎报是欺骗）。
*   **Zero-Hallucination Diagnostic Loop (自诊断闭环)**:
    * If `verify_model.py` fails any Gate (Exit Code != 0), the AI agent **MUST** print a structured diagnostic report (containing the failed Gate, numerical discrepancy evidence, and recovery guidance), execute patches locally in a sandbox, and verify correctness before requesting a git diff review.
* **Rule**: Gate 1（`python scripts/verify_model.py` 全通过）**且** Gate 2（reviewer 子agent 审核通过）后，才启动正式训练。
* **Rule (Gate2 触发时机 · 推理侧/交付侧代码)**: 上一条的触发点写的是「启动正式训练」，但**推理容器代码（`predict.py`/`video_predict.py`/`src/inference/**`/`src/gatekeeper/**`/`entrypoint.sh`）永远不会触发训练**——若照字面理解，这类改动等于没有 Gate2 触发点。**对推理侧/交付侧改动，Gate2 的触发点是「commit 之前」**，不是训练前、也不是 push 后补做。
  - **反例（2026-07-29 真实违规）**：改完概率输出 + gatekeeper 软化后，只跑了搭配规范**第二步**（QA_VP 清单 + reviewer），**跳过第一步**（通用 8 角度扫描，当时载体为 `/code-review` skill，2026-08-11 起已改独立子agent）就 commit+push；事后补跑第一步查出 6 问题，其中一条是修 Gate2 nits 时**自己引入的回归**（`_assert_finite` 在逐帧循环里无隔离，单帧 NaN 拖垮整段视频）。教训：① **两步缺一都漏**（执行纪律问题非规则缺陷）；② **修复本身也要过 Gate2**——按审查意见改完的代码与原始改动同等风险，本次正是"补丁引入新回归"。
  - **顺序不可颠倒**：先独立 sonnet 子agent（通用 8 角度，禁 haiku，**不再用 `/code-review` skill**，见 §2 搭配规范第一步）→ 再 `QA_VP.md` 领域清单。颠倒或只做一步都视为未完成 Gate2。
* **Rule (CI 只验证已 push 的 commit)**: 声称「CI 通过」前,必须跑 `gh run list --branch <branch> --limit 1`(走代理)核对其 `headSha` **正是本轮 commit** 且 `conclusion=success`——`pull_request` 事件的 run message 取自 PR 而非最新 commit,只能凭 SHA 判定。本地未 push 的改动 CI 从未覆盖,不得用本地 pytest 全绿或历史 commit 绿灯冒充;对不上则如实说明「本地绿、CI 未覆盖本轮」。
  - **🔴 `paths-ignore` 不会让「纯 .md 追加」跳过 PR 的 CI(2026-08-12 实证)**:`.github/workflows/ci.yml` 对 `pull_request` 配了 `paths-ignore: '**/*.md'`,但该过滤按 **PR 全量 diff(base…head)** 判定,**不是**按本次 push 的单个 commit。只要 PR 里含任何 `.py`,后续纯文档 commit 照样触发全量 CI。**反例(本轮真实误判)**:我据此声称「`f327f37` 是纯 md,不触发 CI,以 `d82d4b0` 的绿为准」,实际它触发了 run 31512732898 **并且 failed**。→ **任何 push 之后都要实跑 `gh run list` 核 headSha,不得用 `paths-ignore` 推断「这次不会跑」。**
  - **🔴 CI 绿一次 ≠ 真绿(flaky 的识别)**:同一份代码 `d82d4b0` 绿、`f327f37`(只多了 md)红,**代码没变而结论翻转 = 必有非确定性来源**,不得归因为「偶然」「重跑就好」而直接重试。本轮根因是 `os.walk` 未排子目录、Linux 走 hash 序(见 `src/service/case_discovery.py` 的 `dirs.sort()` 注释)。**跨平台不确定性在 Windows 本地永远测不出来**(NTFS 天然按名返回),必须靠显式契约测试钉死。
* **OOM Preventative Rule → `src/models/AGENTS.md` §4**: See model architecture constraints for spatial attention resolution limits. 🔴 **平台显存实际约束(2026-08-12 实测定稿,推翻 2026-08-11 的"16G/14G"误判):RTX 5090 显存 32GB(云 agent 容器内 `nvidia-smi` 实测)**;CPU **14 vCPU**(控制台"14"是 vCPU 数,2026-08-11 误当显存档位)、内存 64G。**batch/分辨率/AMP/梯度累积按 32G 规划**(A800 80G 实测结论可参考,量级同档)。答疑第 4 期 Q33"32G 显存/96G 内存"标称与实测吻合(显存项),96G 内存为官方标称、实际控台 64G 以控台为准。详见 `docs/项目现状总纲.md` §3。

## 3. Early Failure & Short-circuiting (Early Stopping Validation)
To prevent wasting GPU hours on diverging models, we enforce a training gatekeeper rule:
* **Rule**: Epoch 3 后 `val_dice < 0.10` 或分类指标随机时，训练脚本自动中止并记录原因，释放 GPU 资源。
* 🔴 **例外:分割自训场景按 `scripts/train_official.py` 的 `SEG_*` 常量执行(2026-08-16)**。上面的 ep3/0.10 是为**有 warm-start 底座**的场景写的;官方四头训练自 2026-08-16 起分割从"ImageNet 主干 + 随机解码器"自训,此时**不能用 dice 设早期硬门** —— `val_dice` 是 **hard Dice**(`validate_step.py` 先 `(pred_prob>0.5)` 二值化),而真实病灶掩膜只占 512² 的 **0.1%~3%**,分割**确实在学**但前景 logit 尚未越过 0 的阶段可持续很多 epoch,dice 全程恒 0 → ep3 判死刑必然**误杀**(代价 = 白费一次 1-2 小时的人工审核上传窗口 + 一轮 24h 机会)。
  - **该路径实际用的判据 = 预测 mask 的前景占比**(`seg_over_half_frac`,即 `(pred_prob>0.5)` 的像素占比):**ep≥8 且占比 ∉ [1e-4, 0.5] → abort**。它直接就是官方 IoU 匹配消费的那个量,与评分口径同构。dice 门降级为兜底(**ep≥20 且 dice<0.02**,ep20 = γ=0 干净监督窗口末尾);dice<0.10 仅 warn 不停机。
  - 🔴 **中间否掉过一版"概率分布 spread(`p_max−p_mean`)"判据,勿再改回**:实测 3/8 场景判错 —— 有空间结构但整体偏低(max=0.45,spread=0.396)会被放行,而 `p>0.5` 仍全零、mask 全零、Sclin 照样 0(**假的安全感**);常量图 + 4 个异常像素同样骗过它。根因是 max/mean 都是单点或全局标量。回归测试 `tests/test_seg_guards.py::test_v2_hole_*` 已把这两个反例钉死。
  - 两套阈值并存,按"分割是否有 warm-start"选用。

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
* **Rule (Git Push 走代理端口)**: 本机访问 GitHub **必须经本地代理 `127.0.0.1:7897`**,否则 `git push`/`git fetch`/`gh` 会卡住或超时。代理通过**临时环境变量**传入（git config 保持不变，不写入仓库）：
  ```bash
  HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 git push origin <branch>
  ```
  仅访问外网(push/fetch/clone/gh)时加;本地 commit/diff/log 无需代理。

## 6. Code Modularity & Maintainability
* **Rule (Functional Decomposition)**: Every Python module must maintain a clear single responsibility. 拆分的**判据是职责**（>3 个 distinct functional domains，如 model architecture + physical priors + classification heads），**不是行数**。文件超 **300 行是提示信号**：它要求你**停下来评估**，不要求你拆。
* 🔴 **Rule (拆分前置检查 · 2026-08-12 新增,由真实失误反推)**: 执行任何拆分**之前**必须先答这三问,**任一为"是"则不拆**,并在被豁免文件的模块 docstring 里**显式记录例外与理由**(让后人知道是判断而非疏忽):
  1. **拆分会降低可分发性/可执行性吗?** 例:上云现场跑的工具从 1 个文件变 3 个,撞云平台"人工审核 + 每次约 10 文件"限额;从"任何工作目录 `python x.py` 直接可跑"变成依赖 `pkg/__init__` 链,陌生容器里多一个失败点。
  2. **拆出来的部分有独立的认知边界吗?** 没有 = 只是把同一件事切成两半,**零收益**。
  3. **需要为满足行数而压缩/删除注释与实测细节吗?** —— 这直接违反 §5「细节优先」,**绝对禁止**。
  - **豁免范围**:`tests/`(一个测试文件对应一个被测模块本就是正确组织)、`scripts/` 下的独立可执行工具。
  - **冲突优先级**:**§5(细节优先) > §6(行数)**。删掉的实测数字找不回来,多出来的几十行没有代价。
  - **反例(2026-08-12 真实失误,两次同源)**:① 差点把 `scripts/probe_heads_cuda.py`(369 行,单一职责=验三头 GPU 契约)拆成 `scripts/probes/` 包 —— 上面三条代价全中,而它们是被用户一提醒就立刻列得出来的,**信息一直都在,只是没先想**;② 上一轮为把该文件压回 300 行,**删掉了 docstring 里的实测细节**,直接撞 §5。
  - **补充事实(说明这条 MUST 从未被真正执行)**:全仓 `src/`+`scripts/` 115 个 .py 有 **12 个**超 300 行(最大 `official_loader.py` 509),`tests/` 34 个有 **18 个**超(最大 `test_gatekeeper.py` 928)。
  - **执行机制**:`.claude/hooks/modularity_check.py`(PostToolUse,**仅查 `src/`**,与上面的豁免范围一致)已是**提醒式而非阻断式**,其 docstring 早就写明「阻断式检查容易产生**为了不超行数而硬拆**的反效果」——比本规则原文更清醒。2026-08-12 已把它的提示文案同步为「行数只是提示信号 + 前置三问」。**不要把它改成阻断式**:问题从来不是"没拦住超行",而是"机械执行行数规则",阻断只会把这个失误自动化。
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
* **Rule (Gate2 第一步必须分角度并行 · 2026-08-12)**: **禁单 agent 包揽 8 角度**——注意力稀释,每角度都只能浅扫。拆 4 路并行(角度 1-2 / 3-4 / 5-6 / 7-8),每路 prompt 写明「只管这两个角度、挖到底、其余有别的 agent 负责,别泛扫」。实证:单 agent 那路在 **Windows** 上对拍 in-place vs 重新绑定(NTFS 天然有序、必然同结果)就宣称验证过 = **零区分力**;分角度三路则分别抓出 `os.py:425` 的 `reversed(dirs)`、同 PID 撞车行为变更、全仓 15 处枚举点风险表、假迭代器缺 `close()`。
* **Rule (子agent 产物命名空间隔离 · 2026-08-12 事故)**: 一路审查 agent"清理临时文件"时把**主会话的 `tmp_verify_walk_order.py` 一并删了**(同 `tmp_` 前缀不分归属),其结论就此失去可重跑证据、退化为自述。故 prompt 必须规定:① 专属前缀(agent `tmp_review_<lane>_*`、主会话 `tmp_main_*`);② **只删自己前缀**,不碰他人产物或任何非 `tmp_` 文件;③ **证据脚本必须保留**(中间产物可清);④ 审查 agent 除验证脚本外**不改任何仓库文件**,改动与裁决统一归主会话。`.gitignore` 已加 `tmp_*` 兜底。
  - **🔴 ③ 必须逐字写进 prompt,不能只写②(2026-08-12 二次实证)**:同一轮 4 路 agent **全部**在收尾时删光了自己的 `tmp_review_*` 证据脚本——因为 prompt 只写了「只删自己前缀的」,被理解成「自己的可以随便删」。规则文本里有③不等于 agent 看得到③。**派审查 agent 时 prompt 里必须显式写:「验证脚本跑完保留在工作区,不要删——那是主会话复核你结论的唯一依据;要清理只能清中间产物(日志/数据文件)」**。否则拿到的还是不可核验的自述。
* **Rule (Plan 退出前必审 · 2026-08-10 新增)**: **每次进 plan mode 写完 plan、`ExitPlanMode` 之前,必须派子agent(`model="sonnet"` 及以上,禁 haiku)审查该 plan 的技术正确性**——核对 plan 的关键技术判断是否与代码事实一致(签名/契约/控制流/依赖假设),据其结论修正 plan 后才退出。理由:plan 是后续所有实现的蓝图,一处与代码矛盾的假设(如"某函数返回 X"实则不返回)会让整个阶段返工;子agent 用干净上下文对照真实代码审 plan,能抓出主线在规划热度中漏掉的硬伤。审查子agent 须实际读 plan 引用的每个事实源文件,不得凭 plan 自述下结论,且按上一条带对抗心态审。
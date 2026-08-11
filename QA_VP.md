# AI-Assisted Quality Assurance & Verification Protocol (QA-VP)

This document defines the production-grade safety gates for all deep learning implementations, loss functions, training pipelines, and dataset loaders in this repository.

---

## 1. NON-NEGOTIABLE INDUSTRIAL VERIFICATION STANDARDS

All deep learning implementations MUST be verified by quantifiable automated tests (e.g. `scripts/verify_model.py`):

### 1.1 Dynamic Gradient Routing Auditing (Backpropagation Isolation)
*   **Safety Risk**: When training multi-task models or using auxiliary constraint estimators (e.g., Variational CLUB for mutual information minimization, adversarial heads, or secondary decoders), gradients can leak back into the backbone encoder, leading to representation collapse or instability.
*   **Rule**: Any information-theoretic or adversarial decoupling gate MUST implement strict gradient isolation (via `.detach()`).
*   **Verification Unit Test**:
    ```python
    for name, p in model.encoder.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum() < 1e-7, f"Gradient leakage detected in parameter {name}: grad sum = {p.grad.abs().sum().item()}"
    ```
*   **当前实现现状(2026-07-11,补 Gate 8 前的审计缺口)**:
    - `gate2_vclub.py` 验证 vCLUB 互信息通路的隔离(`z_latent.detach()` 之后的局部图不得污染 encoder)。
    - `TaskAlignmentAdaptor`(`blocks.py:32`,跨任务对齐:分割特征 `.detach()` 后反哺分类分支)**曾长期只有实现、没有 Gate 量化验证这条 `.detach()` 契约**——`gate2_vclub` 的检查路径根本不经过 adaptor,结构上必然通过,形同虚设;`gate3_auc_margin` 的 Rule 3.3 会把 `seg_backbone.requires_grad` 显式设 False(用于其自身的 BN 冻结校验),这个状态残留在共享 `model` 实例上,若后续想用同一 model 反向传播测 detach 契约会被这个"假冻结"掩盖(不管 detach 是否还在,`requires_grad=False` 都会直接短路 autograd);`gate6_stage1_masking` 测的是 loss 整体乘 0(与 detach 无关),同样测不出 detach 是否失效。已补 `gate8_adaptor_isolation.py` 堵上此缺口,见 `src/models/AGENTS.md` §1.1 与 `scripts/AGENTS.md` §2 的详细记录。
    - **该 Gate 的断言阈值不是照抄本节模板的 `< 1e-7`,而是精确 `!= 0.0`**:因为其测试构造(`gamma_prior=0.0` 消除物理先验合法耦合通路的干扰 + 手动把 `TaskAlignmentAdaptor.gamma` 从零初始化值拨到非零以真正激活注入通路)已使期望梯度在数学上精确为 0(而非"很小"),已用脚本双向对拍实测验证(正确 detach → `abs().sum()==0.0`;monkeypatch 拆掉 detach → `abs().sum()≈6094`,量级差异悬殊,无需容差空间)。完整推导与踩坑记录(尤其是 `gamma` 零初始化曾让"删掉 detach 也测不出来"的伪 Gate 陷阱)见 `gate8_adaptor_isolation.py` 源码内注释,不在此重复。
    - **`requires_grad` 精确复原**(2026-07-11 commit `1dd0fd6`,Gate2 review 非阻塞建议补齐):Gate 内部前置状态复位此前只复原了 `gamma`,没复原测试过程中改动的 `requires_grad`,依赖"Gate8 是最后一个 Gate"这个隐性顺序假设——现改为逐参数显式记录调用前 `requires_grad` 状态、`finally` 块精确复原,见 `tests/test_gate8_adaptor_isolation.py` 断言 `seg_backbone` 参数复原后 `requires_grad is False`(而非静默清理为 `True`)。

### 1.2 Optimizer Binding Checkpoints (Dual & Custom Parameters)
*   **Safety Risk**: Custom loss formulations (e.g., AUC-Margin Lagrangian dual minimization, adaptive focal parameters) introduce auxiliary parameters (like `a`, `b`, `alpha` in `AUCMLossApprox`) that must be optimized alongside model weights. If these parameters are omitted from the optimizer, frozen during training stage transitions, or updated with incorrect scaling, optimization fails silently.
*   **Rule**: All learnable variables in losses/regularization modules must be explicitly registered and verified in the active optimizer.
*   **AMP & GradScaler Safeguard**: If using `torch.cuda.amp.GradScaler`, gradients must be unscaled using `scaler.unscale_(optimizer)` before any manual gradient modification (such as inverting gradients for dual maximization: `alpha.grad.data.neg_()`) to avoid scale misalignment or NaN propagation.

### 1.3 Numerical Safety & Autocast/FP16 Guarding
*   **Safety Risk**: Custom loss functions with mathematical calculations like squares, logs, divisions, or exponentials are prone to immediate overflow (`Inf`) or underflow (`NaN`) when run in Mixed Precision (FP16), where the maximum float value is $65504$.
*   **Rule**: All custom loss formulations must be protected against out-of-range value overflows.
*   **Logit Clamping**: Clamp inputs to exponential/quadratic operations. For instance, in AUC-Margin or Asymmetric Loss, clamp logits difference to a safe interval: `logits_diff = torch.clamp(logits_diff, min=-15.0, max=15.0)`.
*   **Epsilon Guarding**: Ensure all denominators and log arguments are protected: `torch.log(prob + 1e-7)` or `denom + 1e-6`.
*   **Autocast Unit Test**: Simulate forward and backward passes under autocast using extreme inputs (e.g. large logits like `+1000.0` or `-1000.0`). Assert that the loss value and computed gradients remain finite and contain zero `NaN` or `Inf` values.

### 1.4 Data Leakage & Domain Shift Auditing
*   **Safety Risk**: Patient overlap between splits causes false confidence in metrics. Dynamic statistic recalculation on validation data leaks future statistics.
*   **Rule**: Patient IDs in train, val, and holdout splits must be completely disjoint. The dataset partition MUST be strictly grouped by patient identity, and never split randomly by raw image filenames or index numbers.
*   **当前实现现状(2026-06,Phase D 已修复)**:
    - TN3K 原始数据**无患者级元数据**。此前用 `PAT_{img_id//2}` 合成伪患者 ID 制造了"按患者隔离"的假象;Phase D 已废弃伪分组,改为**纯随机图像级划分**(`_split_samples`),每张图即为独立身份、train/val 图像级互斥、固定 seed 可复现。
    - holdout 来自 TN3K 独立 `test-image/` 目录,与 train/val 物理隔离,**无泄漏**;Holdout AUC 可信。
    - `gate5_data_leakage.py` 已补**双检查**:train↔val 互斥 + holdout↔trainval 图像级重叠校验,覆盖范围完整。
*   **Verification**:
    ```python
    assert set(train_patient_ids).isdisjoint(set(val_patient_ids)), "Leakage: patient ID overlap!"
    ```
*   **归一化现状**: `loader.py` 与 `inference/model_runner.py` 当前仅做 `img/255.0`,**未应用 mean/std**(训练-推理一致)。若未来引入 mean/std,MUST 静态基于训练集计算并同步两条路径。

### 1.5 Loss Reduction & Shape Contracts (损失函数维度契约)
*   **Safety Risk**: Multi-task learning combines various losses (BCE, Dice, AUC-Margin, vCLUB). If sub-losses have different reduction outputs (e.g. element-wise losses of shape `[B]` vs reduced scalar losses `[]`), PyTorch will silently apply implicit broadcasting, inflating the loss tensor to `[B, B]` and computing corrupt gradients without raising errors.
*   **Rule**: All sub-losses combined in MultiTaskLoss MUST be explicitly reduced to scalars (via `.mean()` or `.sum()`) and checked before summation.
*   **Verification**:
    ```python
    # Assert each task loss is a zero-dimensional scalar tensor before addition
    for k, loss_tensor in loss_dict.items():
        assert loss_tensor.dim() == 0, f"Loss '{k}' has shape {loss_tensor.shape}, must be a 0-D scalar!"
    ```

### 1.6 PACS DICOM HIPAA Anonymization Standards (医疗数据脱敏规范)
*   **Safety Risk**: Deploying inference containers in PACS networks risks leaking HIPAA Protected Health Information (PHI) if raw DICOM headers (containing PatientName, PatientID, PatientBirthDate, InstitutionName) are written to disk or sent outside the sandbox.
*   **Rule**: Any DICOM input parsed by the prediction pipeline MUST have all HIPAA PHI headers explicitly anonymized before writing results.
*   **Required Fields to Anonymize**: 完整清单见 `src/inference/io.py` 的 `_PHI_FIELDS`(2026-06 已从 5 字段扩展至 21 字段,覆盖 HIPAA Safe Harbor)。涵盖:
    - 患者身份: `PatientName`/`PatientID`/`PatientBirthDate`/`PatientSex`/`PatientAge`/`PatientAddress`/`PatientTelephoneNumbers`/`OtherPatientIDs`
    - 就诊: `AccessionNumber`/`StudyID`/`StudyDate`/`SeriesDate`/`AcquisitionDate`/`ContentDate`
    - 机构与设备: `InstitutionName`/`InstitutionAddress`/`StationName`/`DeviceSerialNumber`
    - 医师与操作者: `ReferringPhysicianName`/`PerformingPhysicianName`/`OperatorsName`
    - 日期类置 `"00000101"`,其余置 `"ANONYMIZED"` 或空串;仅覆盖 DICOM 中已存在的标签。

### 1.7 推理容器交付红线（Inference-Side Gates）

项目重心已从「训练」扩展到「离线推理容器交付」，以下推理侧红线与训练红线同等强制：

*   **Gatekeeper 误报**: gatekeeper 任一检测器**误杀真实图**代价极高。**任何 gatekeeper 检测器改动后，MUST 用全量 holdout 真实图实测硬拒绝误报率，零误报是红线**。误报与召回不可兼得时遵守「宁可漏检不可误杀」，将不可靠检测器降为软 flag 而非硬拒绝。血泪教训：`detect_copy_move` 曾在 359 张真实图误报 27.3%，AGENTS.md 却写「零误报」——**别信文档声称，用数字实测**。见 `memory/gatekeeper-copymove-softflag.md`。
    - **2026-07-29 代价重估(旧表述"一票否决直接沉赛道"已过时,但红线未松)**:赛道七已豁免全部医保安全考核(官方赛前动员会 2026-07-28,总分=临床诊断得分×100,无 Sins),且 `gatekeeper.enforcement_mode` 默认 `soft` ——误报**默认不再拒绝**任何输入,最坏后果由"整队沉赛道"降为"审计噪声误导人工复核 + 报告可信度受损"。但检测器本身的零误报要求**不变**:① hard 模式仍在代码里可达;② 未来若规则变化或其它赛道复用,软化只是执行层封顶、不是检测器可以变脏的许可;③ 误报污染 `result.json` 的 `gatekeeper` 字段,直接影响专家评审对可靠性的判断。
*   **权重维度契约**: checkpoint 维度必须与当前模型定义一致。**特征/结构维度变更后，MUST 立即重训并重新导出权重**，否则推理容器 `size mismatch` 崩溃。改维度须同步所有引用点（特征提取器 / 分类器 `input_dim` / 训练器 `FEAT_DIM` / test 断言）。见 `memory/authenticity-weight-mismatch.md`。
*   **降级路径即生产路径**: 可插拔检测器（SSCD/CAT-Net）的深度权重若未落地交付，则降级路径就是实际生产路径，必须按降级路径实测红线，不得假设深度权重存在。**判定以代码里的实际探测路径为准**（如 `_SSCD_WEIGHT_CANDIDATES` / `_CATNET_WEIGHT_CANDIDATES` 常量），而非下载缓存或打包文档——2026-07-29 审计发现 `.cache_weights/` 不在任何探测路径里，SSCD 虽然下载了、`docs/预训练权重打包清单.md` 也列了，运行时却从未被加载过（实测 `_find_sscd_weights()` 返回 `None`）。**下载了 ≠ 运行时会加载**，两者必须分别核实。
*   **端到端验证**: 推理侧改动 MUST 实跑 `predict.py`（单图）完整链路，不能只跑单元测试（见 `memory/phase4-was-halfbaked.md`）。
*   **容器封印时机**: `seal_container.py` 只在**交付最后一步**（最终权重就位后）跑，提前封印会埋雷。见 `memory/seal-container-timing.md`。
*   **统一软 flag 制度**: gatekeeper 任一硬拒绝分支，在**官方真实干净 holdout 完成预注册、带置信区间的 FPR 验收**之前，默认只能软 flag（记录审计字段，不触发 REJECTED）。**降级（硬→软）是单调保守动作，不需要真实 FPR 证据即可执行**（收紧误报风险，不会引入新风险）；但**恢复硬拒绝必须先有真实域 FPR 证据**，不得凭"实现看起来对了"就把软 flag 升级回硬拒绝。2026-07 gatekeeper 统一软 flag 化改动（`8e608b7`/`c8871a6`/`6cb60a6`）即此原则的落地:CatNet 自身命中与 BAG 相位/copy_move/splice 几何判据解耦，硬拒绝来源需显式记录（`last_hard_reject_source`），BAG 是否硬拒绝由独立开关 `bag_hard_reject`（默认 `false`）控制。见 `docs/P2红线修复实施计划.md`。
    - **2026-07-29 顶层开关**:在检测器级软/硬分级之上，新增 `gatekeeper.enforcement_mode: soft|hard`（`gatekeeper_runner.resolve_enforcement_mode`，默认与交付固定 `soft`）做统一封顶——本条所述**仍保留硬拒绝资格的分支**（CatNet 自身命中、去重）在 soft 下也只落 `would_reject_in_hard_mode` 审计 flag、不触发 REJECTED。**这不是给检测器松绑**：恢复某检测器硬拒绝资格仍需真实域 FPR 证据（闸门不变），enforcement_mode 只决定"已获资格的分支这次要不要真执行"。非法值一律降级 `soft`（失败方向偏向多输出一份诊断）。改 gatekeeper 判定逻辑时**两层都要核**：检测器级资格 + 执行层模式。
*   **推理侧不得因审计而零输出(§1.7 关联)**: 审计命中（is_duplicate/is_tampered/is_phantom）与**故障**（文件损坏、权重缺失、解码失败）必须严格区分——前者在 soft 下只 flag、诊断照出。故障路径按**该组件是否计分**分流（2026-08-10 R3 重估，原文"权重缺失一律硬报错"已按组件改判，不再一刀切）：
    - **计分组件（临床诊断主模型）权重/解码故障 → 仍硬报错，红线未动**：容器带着随机初始化的主模型"正常"输出一份看似合法的诊断，比崩溃更危险（见 `memory/delivery-container-missing-weights.md`）。主模型 `build_model`/`load_weights`（model_runner）不在本分流内、依旧硬失败。
    - **零计分防御组件（authenticity 真实性）权重故障 → 降级 skipped，不得拖垮整例**：赛道七真实性零计分，`build_auth_module` 在 `load_weights` 失败/`predict` 崩溃时返回 None（不 raise），审计字典如实写 `auth_skipped=True` + `is_phantom=0` + `auth_probs=None` 并打 WARNING——**是显式跳过的诚实记录，不是伪装 pass**。若让它 raise，会被入口兜成整例 ERROR = 唯一得分项（临床诊断）清零，恰是本条要避免的"零输出"；而让不计分项降级，主诊断（唯一得分项）不受影响。回归测试：`tests/test_gatekeeper_soft_enforcement.py` + `tests/test_gatekeeper.py::TestAuthWeightFailureDegrades`。
*   **子进程崩溃兜底（容器不可静默零输出）**: `entrypoint.sh` 若用 `exec python ...` 替换 bash 进程，一旦 Python 侧 segfault 或被 OOM-killer `SIGKILL`（不可被 trap 捕获，任何 try/except 都挡不住），没有存活进程能补写 `result.json`，容器表现为静默零输出而非可诊断的错误。**Rule**：去 `exec`、显式 `trap` 转发 `SIGTERM`/`SIGINT` 给子进程（否则容器优雅关闭时子进程变孤儿）、子进程异常退出且未产出 `result.json` 时必须补写结构化 ERROR JSON 兜底；兜底前须先移走上次调用可能残留的**陈旧** `result.json`（否则会被误判为"这次成功了"，比空目录更危险）。见 `docs/P2红线修复实施计划.md` C 级、`scripts/AGENTS.md` §7.1。
*   **概率/置信度必须有限（2026-07-29 新增）**: 官方硬约束要求所有分类/分级/分型输出概率分布，但 **NaN/Inf 概率比不输出更危险**，两条后果均已实测复现——① `torch.argmax` 对全 NaN 向量**静默返回 0**（logits=[0.2, nan, 1.5, -0.3] 时真实最大值在 idx=2 却报 idx=0），报告照常打印一个错误却看起来专业的 TI-RADS 分项；② `json.dumps` 把 NaN 写成字面量 `NaN`，属**非标准 JSON**，严格解析器（Go `encoding/json` / Rust `serde_json` / Jackson 默认）拒绝**整个** `result.json`，等于该样本零输出——比预测错更严重。**Rule**：概率导出前 MUST 校验有限性并硬报错（`postprocess._assert_finite`），与 `report_generator` 对 NaN `malignancy_prob` 的既定处置同范式（硬拦截、交上层转 ERROR，不允许带无效值产出"看起来正常"的报告）。**校验对象是激活后的概率而非输入 logits**——`-inf` 经 softmax 得 0，是合法的"该类概率为 0"，拦它会把极端置信输出误杀成 ERROR（本轮第一版测试就误把 `-inf` 当非法，被反向用例纠正）。
*   **逐帧/逐样本异常必须隔离（2026-07-29 新增，同一坑踩过两次）**: 动态图逐帧循环里，**任一帧的异常都不得让整段视频归零**。已发生两次，触发路径不同但后果相同：① Phase 2 前，逐帧 gatekeeper 命中即 `return`，32 帧任一误报整段零诊断；② 修完 ① 之后，新加的 `_assert_finite` 在 `decode_predictions` 里 raise，而逐帧循环无 `try/except`，单帧数值异常又把整段拖垮。**Rule**：① 逐帧循环 MUST 包 `try/except`，且**只捕预期的窄异常**（如 `ValueError` 数值异常），不得裸 `except Exception` 吞掉 CUDA OOM/权重损坏这类真故障；② 丢弃的帧 MUST 如实记进 `result.json`（`failed_frame_indices`/`n_frames_failed`），不得静默少算；③ **全部帧失败才降级 ERROR**，且要显式 `if not per_frame:` 判断而非依赖 `assert`（`-O` 下 assert 被剥离）；④ **警惕平行列表错位**——若循环里有与 `per_frame` 并行维护的第二个容器（如按全帧下标建键的 tensor 缓存），跳过坏帧后二者会脱钩，且元素多的一侧**不会 KeyError**、而是静默取错帧（JSON 帧号对、mask/热图出自另一帧）。正解是把数据并入同一结构，从根上消除双列表对齐假设。

---

## 2. SUBAGENT CODE REVIEW CHECKLIST (NON-QUANTIFIABLE LOGIC GATES)

Audit all PyTorch/training code changes against this checklist:

### 2.0 对抗性审计心态(Adversarial Review Posture · 所有 review 子agent 的默认前提)
*   **Safety Risk**: 以"默认代码没问题、只是扫一眼确认"的心态做 review,会系统性放过隐蔽缺陷——审查者一旦预设正确,就只找符合预期的证据(确认偏误),对反常之处倾向"应该没事"而非追查。本项目的数学管道(梯度路由/符号极性/对偶优化)缺陷往往不崩溃、只让指标悄悄退化,最需要主动出击才能挖出。
*   **Rule(默认有 bug,主动证伪)**: **所有 review/审查子agent 一律以"这段代码有 bug,我的任务是把它找出来"为默认前提**,而非"确认它没问题"。具体:
    - **举证责任反转**:对每个关键判断(符号方向、detach 隔离、维度契约、边界条件),默认它**错**,由代码/实测来证明它对;证不出"对"就当"存疑"上报,不得因"看起来对"放过。
    - **主动构造反例/对拍**:别只顺着 happy path 读。构造能触发 bug 的输入(边界值、首步/空状态、NaN、单类 batch、CUDA vs CPU 分叉),实跑看是否崩或结果反常(rules.md §5 先证后改·对拍实验)。
    - **质疑而非复述**:被审代码的注释/文档/commit message 声称的行为**一律不采信**,以实测为准("别信文档声称,用数字实测",§1.7 已有先例:copy_move 文档写"零误报"实测 27.3%)。
    - **列出"没找到 bug 的地方"也要说依据**:判一个点 PASS,必须给出"我尝试了 X 反例、实测 Y、故排除",而非沉默略过。
*   **边界**:对抗心态是**找真 bug**,不是为凑数报假问题——每条 finding 必须可复现(给出触发输入 + 期望 vs 实际)。报一堆基于误解 PyTorch 语义的"耸动假报"和放过真 bug 同样有害(2026-08-11 一次 code-review 8 条 findings 有 7 条源于对 detach/nn.Parameter/门控语义的误读)。**默认有 bug 是搜索姿态,不是结论姿态**:大胆假设、实测求证、证伪即撤。
*   **谁强制**:Gate2 派出的每个 reviewer 子agent(rules.md §2)prompt 里都应显式带上本条心态;高风险数学改动(FAMO/AUCM/梯度路由)尤其不得走"确认式"轻审。

### 2.1 Signs and Polarity Auditing (Minimax & Objective Alignment)
*   **Minimax Dual Variables**: In minimax optimization (e.g. AUC-Margin), verify that the dual variable gradient is correctly inverted before standard optimization step:
    ```python
    alpha.grad.data.neg_()
    ```
*   **Loss Terms Sign**: Double check the sign of all terms in the multi-task loss. Maximization targets (like entropy maximization) must have a negative coefficient. Minimization targets (L1/L2 penalties, mutual information minimization bounds like vCLUB) must have a positive coefficient.

### 2.2 PyTorch Broadcasting and Shape Safeguards
*   **Implicit Broadcasting**: Explicitly call `.squeeze()`, `.unsqueeze()`, or `.view()` to make dimensions match identically before operations. Prevent `[B, 1] - [B]` implicit broadcasting to `[B, B]`.
*   **Loss Reduction Alignment**: Verify that intermediate loss functions return the expected shapes. If a custom dynamic scheduler (like FAMO) expects a list of scalar task losses, ensure each task loss is reduced (e.g., `.mean()`) before passing it to the scheduler.
*   **Broadcast Assertions**: Verify that all broadcasting is prevented by adding defensive dimensions checks:
    ```python
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape} will trigger broadcast!"
    ```

### 2.3 Test Robustness: Forced Triggering over Natural Triggering（测试反脆弱：强制触发优于自然触发）
*   **Safety Risk**: 依赖合成样本"自然"触发检测器/分支的回归测试（例如指望某张构造图片恰好让 `detect_splicing`/`detect_copy_move` 判定为阳性）具有隐蔽脆弱性——若检测器行为被后续改动意外破坏（阈值漂移、判据被误删），测试可能**静默 `pytest.skip` 而非 `fail`**，CI 照样绿灯，红线破坏无人察觉。
*   **Rule**: 凡测试意图是"验证某条件命中后的下游行为"，**优先用 `monkeypatch` 强制该条件命中**（如 `monkeypatch.setattr(detector, "detect_splicing", lambda *a, **k: True)`），而非依赖构造样本自然触发。自然触发允许作为补充（验证真实检测器确有此能力），但不能是唯一路径。2026-07 gatekeeper 软 flag 化改动中，`test_geometric_never_hard_rejects_even_when_catnet_ok` 等测试即从"构造图片指望触发"改为 monkeypatch 强制触发,见 `tests/test_gatekeeper.py`、`docs/P2红线修复实施计划.md`。

---

## 3. COMPLIANCE & QUALITY GATE ENFORCEMENT WORKFLOW

Before any code modification is merged, submitted, or deployed:
1.  **Ruff & Pyright Validation**: Ensure no syntax, typing, or formatting errors exist.
2.  **Forward/Backward Test execution**: Run `python scripts/verify_model.py` and verify all Gate checks pass.
3.  **Checklist Audit**: 逐条核对本协议 §1/§2 中**改动触及的**领域项(未触及的跳过)。审查流程与 skill 搭配见 `rules.md §2`。
4.  **Zero Placeholder Rule**: Do not use placeholders or mocked outputs. All code must be fully operational.

**CI 自动化(`.github/workflows/ci.yml`,PR/push 到 main 触发)**:`import-linter` job 校验架构分层契约(`.importlinter` 4 条 forbidden 契约);`test` job 跑 Gate 1(`verify_model.py` 八门,2026-07-11 补 Gate 8 cls→seg 梯度隔离审计)+ 全套 pytest 回归;`docker-build-smoke` job(2026-07-11 新增,host runner 真实 `docker build` + 静态图/动态图容器冒烟,验证镜像本身而非模拟环境)。CI 红 = 合并前的硬阻断,等同上述第 2 步的远程强制执行。**例外(2026-07-11 commit `12caa3d` 新增 `paths-ignore: '**/*.md'`)**:纯 `.md` 文档改动不触发 CI——用 `gh run list` 核对某 commit 的 CI 结果时,若该 commit 只改了文档,查不到对应 run 是**预期行为**(本就不该跑),不是「CI 未覆盖」,勿误判为需要补跑。

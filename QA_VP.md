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
    - `TaskAlignmentAdaptor`(`multi_task_model.py:53-88`,跨任务对齐:分割特征 `.detach()` 后反哺分类分支)**曾长期只有实现、没有 Gate 量化验证这条 `.detach()` 契约**——`gate2_vclub` 的检查路径根本不经过 adaptor,结构上必然通过,形同虚设;`gate3_auc_margin` 的 Rule 3.3 会把 `seg_backbone.requires_grad` 显式设 False(用于其自身的 BN 冻结校验),这个状态残留在共享 `model` 实例上,若后续想用同一 model 反向传播测 detach 契约会被这个"假冻结"掩盖(不管 detach 是否还在,`requires_grad=False` 都会直接短路 autograd);`gate6_stage1_masking` 测的是 loss 整体乘 0(与 detach 无关),同样测不出 detach 是否失效。已补 `gate8_adaptor_isolation.py` 堵上此缺口,见 `src/models/AGENTS.md` §1.1 与 `scripts/AGENTS.md` §2 的详细记录。
    - **该 Gate 的断言阈值不是照抄本节模板的 `< 1e-7`,而是精确 `!= 0.0`**:因为其测试构造(`gamma_prior=0.0` 消除物理先验合法耦合通路的干扰 + 手动把 `TaskAlignmentAdaptor.gamma` 从零初始化值拨到非零以真正激活注入通路)已使期望梯度在数学上精确为 0(而非"很小"),已用脚本双向对拍实测验证(正确 detach → `abs().sum()==0.0`;monkeypatch 拆掉 detach → `abs().sum()≈6094`,量级差异悬殊,无需容差空间)。完整推导与踩坑记录(尤其是 `gamma` 零初始化曾让"删掉 detach 也测不出来"的伪 Gate 陷阱)见 `gate8_adaptor_isolation.py` 源码内注释,不在此重复。

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

*   **Gatekeeper 误报一票否决**: gatekeeper 任一检测器**误杀真实图 = 一票否决**（拒掉合法输入直接沉赛道）。**任何 gatekeeper 检测器改动后，MUST 用全量 holdout 真实图实测硬拒绝误报率，零误报是红线**。误报与召回不可兼得时遵守「宁可漏检不可误杀」，将不可靠检测器降为软 flag 而非硬拒绝。血泪教训：`detect_copy_move` 曾在 359 张真实图误报 27.3%，AGENTS.md 却写「零误报」——**别信文档声称，用数字实测**。见 `memory/gatekeeper-copymove-softflag.md`。
*   **权重维度契约**: checkpoint 维度必须与当前模型定义一致。**特征/结构维度变更后，MUST 立即重训并重新导出权重**，否则推理容器 `size mismatch` 崩溃。改维度须同步所有引用点（特征提取器 / 分类器 `input_dim` / 训练器 `FEAT_DIM` / test 断言）。见 `memory/authenticity-weight-mismatch.md`。
*   **降级路径即生产路径**: 可插拔检测器（SSCD/CAT-Net）的深度权重若未落地交付，则降级路径就是实际生产路径，必须按降级路径实测红线，不得假设深度权重存在。判定以 `docs/预训练权重打包清单.md` + `.cache_weights/` 实际内容为准。
*   **端到端验证**: 推理侧改动 MUST 实跑 `predict.py`（单图）完整链路，不能只跑单元测试（见 `memory/phase4-was-halfbaked.md`）。
*   **容器封印时机**: `seal_container.py` 只在**交付最后一步**（最终权重就位后）跑，提前封印会埋雷。见 `memory/seal-container-timing.md`。

---

## 2. SUBAGENT CODE REVIEW CHECKLIST (NON-QUANTIFIABLE LOGIC GATES)

Audit all PyTorch/training code changes against this checklist:

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

---

## 3. COMPLIANCE & QUALITY GATE ENFORCEMENT WORKFLOW

Before any code modification is merged, submitted, or deployed:
1.  **Ruff & Pyright Validation**: Ensure no syntax, typing, or formatting errors exist.
2.  **Forward/Backward Test execution**: Run `python scripts/verify_model.py` and verify all Gate checks pass.
3.  **Checklist Audit**: 逐条核对本协议 §1/§2 中**改动触及的**领域项(未触及的跳过)。审查流程与 skill 搭配见 `rules.md §2`。
4.  **Zero Placeholder Rule**: Do not use placeholders or mocked outputs. All code must be fully operational.

**CI 自动化(`.github/workflows/ci.yml`,PR/push 到 main 触发)**:`import-linter` job 校验架构分层契约(`.importlinter` 4 条 forbidden 契约);`test` job 跑 Gate 1(`verify_model.py` 八门,2026-07-11 补 Gate 8 cls→seg 梯度隔离审计)+ 全套 pytest 回归;`docker-build-smoke` job(2026-07-11 新增,host runner 真实 `docker build` + 静态图/动态图容器冒烟,验证镜像本身而非模拟环境)。CI 红 = 合并前的硬阻断,等同上述第 2 步的远程强制执行。

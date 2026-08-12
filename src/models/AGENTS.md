# src/models/AGENTS.md - Model & Loss Architecture Specifications

## 0. MODULAR CODE ARCHITECTURE

Per project-wide modularity rules (`rules.md` §6), this folder is decomposed into single-responsibility modules:

| File / Package | Responsibility | Max Scope |
|:---|:---|:---|
| `physical_prior.py` | Differentiable physical feature extraction (AR, ER, Circularity) | ~160 lines |
| `cbm_head.py` | Concept Embedding Model head + vCLUB estimator | ~209 lines |
| `multi_task_model.py` | Model assembly hub (ConvNeXt-Nano + EfficientNet-B2 dual-backbone, adaptor 装配, decoder, stage freeze registry)。ConvBlock/TaskAlignmentAdaptor 已外移至 `blocks.py`、mask 派生标量已外移至 `mask_features.py`（均 re-export 或直接 import 保后兼容） | ~291 lines |
| `blocks.py` | 可复用 nn 块：`ConvBlock`（U-Net double-conv 解码块）+ `TaskAlignmentAdaptor`（分割→分类单向软注意力，`.detach()` 契约 + gamma 零初始化见 §1.1 / Gate 8） | ~67 lines |
| `label_schema.py` | **官方类别定义的单一权威**：位置 7 类 `LOCATION_LABELS` + ETE 4 码。2026-08-12 从 `src/inference/prediction_json.py` **下沉到 models 层** —— `.importlinter` 契约2 规定 `src.models` 是最底层不得依赖 `src.inference`，而位置头必须按同一份顺序构造标签，常量留在 inference 会让 CI import-linter 直接红。`prediction_json` 现为 re-export，对外路径不变。**改类别定义改这里** | ~75 lines |
| `mask_features.py` | 由分割 soft mask 派生的标量/局部特征（纯张量函数，无 nn 状态）：`nodule_scale_from_mask`（归一化等效直径，临床 T 头输入）+ `boundary_band_from_mask`（**2026-08-12 阶段 2.2 新增**：膨胀−腐蚀提取包膜-结节交界带，返回 detach 后的带宽 mask，供扩展头做 ROI-Mask 引导；⚠️ 实测带宽 = `kernel_size − 1`，实现取 k=5 → 4px）。**一律返回 detach 后的张量**——三头梯度不得回流分割主干，否则破坏 Gate 8 精确零断言前提 | ~137 lines |
| `heads/` | 官方新增三维度增量诊断头子包（位置/侵犯/临床T，见 `docs/训练架构方案.md`）。**接入接口 4/4 于 2026-08-12 阶段 2.0 补全**（`stage4_spatial` / `soft_mask_detached` / `margin` / `nodule_scale_detached`，现状记于 `heads/__init__.py`）；**三头全部实现**：`location.py`（阶段 2.1）/ `extension.py`（阶段 2.2）/ `clinical_t.py`（阶段 2.3）。⚠️ **均未接入 `MultiTaskThyroidModel` 与训练循环**——接线在阶段 2.4，推理侧接线在 2.5，勿以为已能跑 | 3/3 头就绪 |
| `heads/location.py` | 位置头：CoordConv + Stage-4 空间特征 → 侧别3×纵向3 因子分解 → 7 类。`P(k)=P(side)×P(level)`、`P(峡部)=P(side=峡)`，**和为 1 是数学恒等式非再归一化**。🔴 模块导入期断言 `LOCATION_LABELS` 顺序（顺序变了而映射没跟着改 = 左右静默错位，JSON 仍合法、和仍为 1、测试仍过）。含 `flip_location_index`（水平翻转必须联动翻标签）与 `build_level_validity`（峡部无纵向属性，level 分支必须屏蔽） | ~190 lines |
| `heads/extension.py` | **ETE 侵犯头**（阶段 2.2，2026-08-12）：Stage-4 空间特征 + ROI-Mask 交界带（`boundary_band_from_mask`，带宽 = kernel−1 → 4px）→ 4 类 ETE（无/甲状腺包膜/毗邻软组织/血管）。**CB-Focal Loss + Logit Adjustment 保护稀有血管类**（类别计数 3906/267/67/5）；`compute_extension_loss` 的 `class_balanced_weights` 乘性加权 β=0.99 → 血管:无 = 20.4:1。**参数量 0.063M**（与 LocationHead 0.409M 相比属轻量头）。概率输出守官方「必须输出概率分布否则计0分」约束 | ~194 lines |
| `heads/clinical_t.py` | **临床 T 头**（阶段 2.3，2026-08-12）：GAP 全局特征（352 维）+ 结节尺度 `[s, log s]` + ETE 概率（**detach**）→ 轻量 MLP → 2 类，输出 `t_logits` / `t_probs` / `t_conf`。**参数量 0.065M**。🔴 两个坑：① 头内**禁用 `BatchNorm1d`**（实测 train 模式 batch=1 直接 `ValueError: Expected more than 1 value per channel`，训练末批可能只剩 1 样本；侵犯头能用 BatchNorm2d 因它作用在 16×16 空间上仍有 256 元素）；② **`T_conf` = P(T=1)**，非"对所报 T 值的置信度"——与 `prediction_json` 里 `Malignancy.predicted = bool(confidence>=阈值)` 推出的"恶性概率"语义自洽。实测 `GAP(stage4_spatial)` 与模型内部 `cls_pool(x_cls)` **逐位相同（最大差 0.0）**（`cls_pool` 即 `AdaptiveAvgPool2d(1)`），故 T 头自己 GAP 即可，无需模型额外暴露 pooled 键 | ~225 lines |
| `heads/__init__.py` | 三头 re-export 门面：`LocationHead` / `ExtensionHead` / `ClinicalTHead` 均已导出，docstring 记接入接口与使用契约 | ~93 lines |
| `mil_pooling.py` | 门控注意力多示例学习(Gated-Attention MIL)——患者级良恶性弱标签聚合:`SpatialAttentionModule`(保留 backbone `forward_features()` 空间判别性,不用 GAP 稀释微钙化等高频特征) + `GatedAttentionPooling`。仅供 `scripts/pretrain_malignancy_mil.py` 训练管线使用,不在 `MultiTaskThyroidModel` 主推理路径。见 `memory/patient-leakage-and-probe-trap.md` | ~255 lines |
| `losses/` | Decomposed loss function subpackage | See below |
| `loss.py` | loss.py 为纯 re-export shim，新损失函数统一写入 losses/ 对应子模块 | ~35 lines |
| `famo.py` | FAMO dynamic loss weight scheduler | ~85 lines |
| `pretrained_loader.py` | `load_local_safetensors()` — offline backbone weight loader (ConvNeXt seg + EfficientNet cls) | ~36 lines |
| `lora_utils.py` | `LoRAConv2d`(1x1 conv 低秩包装) + `apply_lora`/`get_lora_params`/`disable_lora` — LP-FT-LoRA 微调用 | ~140 lines |
| `physical_prior.py` 的 `x_coords`/`y_coords` buffer | **2026-07-10 修复**：`torch.meshgrid` 返回的 0-stride 广播视图直接 `register_buffer` 会导致*任何* checkpoint 的 `load_state_dict()` 崩溃（PyTorch 检测目标 tensor 内存别名），已加 `.contiguous()` 修复。此 bug 与 LoRA 无关，早于且独立于下面的 LoRA 部署问题，是 `predict.py`/`evaluate.py` 此前从未能真正加载训练权重的根因之一，详见 `memory/`（如已生成）或 `src/inference/AGENTS.md` §3。 | — |

### losses/ Subpackage Structure

| File | Contents |
|:---|:---|
| `losses/__init__.py` | Re-exports all symbols with `__all__` |
| `losses/segmentation.py` | `DiceLoss` |
| `losses/classification.py` | `AUCMLossApprox`, `SupervisedContrastiveLoss` |
| `losses/concepts.py` | `symmetric_label_regression_loss`, `symmetric_label_regression_loss_bce` |
| `losses/multitask.py` | `MultiTaskLoss` orchestrator (~183 lines) |

### losses/components/ Subpackage (MultiTaskLoss decomposition)

| File | Contents |
|:---|:---|
| `components/__init__.py` | Re-exports all computation functions |
| `components/segmentation.py` | `compute_segmentation_loss(bce_fn, dice_fn, outputs, targets)` |
| `components/concepts.py` | `resolve_mix_params(outputs, mix_info)`, `compute_concept_losses(outputs, targets, lam, index, mask_weight, label_smoothing, tirads_matrix)` |
| `components/malignancy.py` | `compute_malignancy_losses(mal_loss_fn, supcon_loss_fn, outputs, targets, mask_weight, slr_weights)` |
| `components/famo_mix.py` | `apply_famo_weighting(famo, loss_seg, loss_concepts_ce, loss_prior, loss_mal, loss_supcon, update_famo_weights, is_training)` |
| `components/regularizers.py` | `compute_regularizers(total_loss, outputs, lambda_latent, delta_budget, spin_l1_weight)` — 拉格朗日约束 g=loss_l2-delta_budget≤0。参数方向:grad=λ·grad(loss_l2),λ>0 恒成立故残差恒被往小推,约束满足(g≤0)只减量级不翻转方向;乘子 `λ←λ+η·g` 违反加压/满足松绑(2026-08-11 #120 核实为误报,方向正确,见文件内注释) |
| `components/extension.py` | **ETE 侵犯头损失组件**（阶段 2.2，2026-08-12）：`compute_extension_loss`（CE + CB-Focal γ 聚焦 + Logit-Adjustment 先验补偿）+ `class_balanced_weights`（乘性加权，β=0.99→血管:无=20.4:1，默认）+ `log_class_prior`。🚫 与 `heads/extension.py` 同名不同职责——前者是头（前向），此文件是损失（训练），勿混 | — |

**Rule**: 新损失函数放入 losses/ 子模块或 losses/components/ 文件，由 multitask.py 调用。`loss.py` is a shim only.

---

## 1. HETEROGENEOUS DUAL-BACKBONE ARCHITECTURE (Decoupled)

Stage freeze registry: `MultiTaskThyroidModel.STAGE3_FROZEN_PREFIXES`, `STAGE1_FROZEN_CBM_SUBMODULES`, plus `set_training_stage(stage)` / `is_param_trainable(name, stage)`.

**LoRA 微调路径（`use_lora=True`，LP-FT-LoRA 训练用，见 `scripts/train_thyroidxl.py`）**：`__init__` 里 `apply_lora` 注入后，**双主干全部参数（含 LoRA A/B）默认 `requires_grad=False`**——模型以"LoRA 未激活"状态构建。LP 阶段只训任务头；FT 阶段由训练脚本显式解冻 `lora_A`/`lora_B` 并 `add_param_group`。**关键教训**：LoRA 参数若默认可训练但不在 optimizer 内，其梯度跨 batch 永不清零、累积至溢出（`§1.1` requires_grad 必须与 optimizer 参数集合严格对齐）。`is_param_trainable`/`set_training_stage` 是 LoRA-blind 的（不识别 `lora_` 前缀），仅供 `train.py` 的非 LoRA 三阶段管道用；**不可对 `use_lora=True` 模型调用 `set_training_stage`**，否则会解冻被冻结的原始骨干权重。

**LoRA 部署解耦（`disable_lora`/`merge_weights`，2026-07-10 起正式接线）**：推理侧（`src/inference/model_runner.py` 的 `build_model()`）**永远不感知 LoRA**——训练产出的 LoRA 结构 checkpoint 必须先用 `scripts/merge_lora_checkpoint.py`（内部调用 `disable_lora`）把 `lora_A`/`lora_B` 数学合并回原始 Conv2d（`merged.weight = base.weight + (lora_B @ lora_A) * scale`），产出裸 `state_dict` 才能喂给 `build_model()`。**数值稳定性教训**：合并本身在单层测试下数值等价（fp32 下 diff~1e-7），但若 backbone 存在退化 BatchNorm（`running_var` 接近 0，常见于冻结 backbone 训练时忘记调用 `set_frozen_modules_eval()` 导致 running stats 漂移，见 `src/training/AGENTS.md` GradScaler Protocol 节），LoRA 双路计算与合并单路计算之间的 fp32 舍入误差会被逐层放大，可致最终分类头输出相对误差达 70%~1000%+（本质是 checkpoint 本身数值病态，非合并逻辑错误——已用"同一合并模型 CPU vs GPU 对拍"排除法定位）。**merge 脚本本身不做数值防护**（不 clamp running_var、不改 eps），保持合并是忠实恒等变换；数值病态必须在训练阶段用 BN 冻结修复，不能在合并脚本里掩盖。

Offline pretrained weights: `src/models/pretrained_loader.py` (`load_local_safetensors`) loads `checkpoints/convnext_nano_pretrained.safetensors` (seg) and `checkpoints/efficientnet_b2_pretrained.safetensors` (cls).

To eliminate representation competition between high-frequency segmentation and abstract classification, the model employs a fully decoupled asymmetric dual-backbone:
- **Segmentation Branch**: `ConvNeXt-Nano` (Large $7\times7$ kernels naturally smooth ultrasound speckle noise; retains spatial translation invariance).
- **Classification Branch**: `EfficientNet-B2` (Squeeze-and-Excitation channel attention for global malignancy semantics).

### 1.1 Task-Alignment Adaptor (Soft Attention Gating)
Because the classification branch loses spatial priors after decoupling, segmentation features must be injected back as soft attention.
- **Injection Nodes**: Mid-to-high level layers (Stage 3 and Stage 4 of EfficientNet-B2), adaptor 调用在 `_forward_cls_backbone`(`multi_task_model.py`,不写行号——已因无关改动过时两次);类定义在 `blocks.py`。
- **Unidirectional Flow**: `f_seg` MUST be `.detach()`ed before passing to the Adaptor to prevent classification error gradients from corrupting the segmentation backbone's weights.
- **Fusion Equation**: `f_cls_enhanced = f_cls + gamma * (f_cls * W_align)`, where `W_align` is generated via spatial interpolation, 1x1 channel projection, and 3x3 depthwise convolution + Sigmoid. `gamma` is a learnable scalar initialized to 0.
- **Gate 8 验证(`scripts/verify/gates/gate8_adaptor_isolation.py`,2026-07-11 新增)**:上述"MUST `.detach()`"契约此前只有实现、没有 Gate 量化验证——`gate2_vclub` 走的是 vCLUB 局部图(不经过 adaptor)、`gate3_auc_margin`/`gate6_stage1_masking` 都测不到这条通路(具体原因见 `QA_VP.md` §1.1 与该 Gate 源码注释)。Gate 8 用 `loss_mal.backward()` + 断言 `seg_backbone` 梯度精确为 `0.0`(非 `<1e-7` 容差)量化验证该契约,并在 Gate 内部主动做了两处必要的前置状态复位(**这两处都是本 Gate 存在的核心原因,删了会退化成永远通过的伪 Gate**):
    1. **requires_grad 复位**:`gate3_auc_margin` 的 Rule 3.3 会把 `seg_backbone.requires_grad` 显式设 False 留在共享 `model` 实例上,若不复位为 True,即使 `.detach()` 被误删,`seg_backbone` 也不会收到梯度(requires_grad=False 直接短路 autograd),会把"假冻结"误判成"detach 生效"。
    2. **gamma 强制非零**:`gamma` 初始化为 `nn.Parameter(torch.zeros(1))`(零初始化残差门控)。全新构造的模型(`verify_model.py` 每次都新建 `pretrained=False` 的 model)gamma 恒为 0,此时注入项 `gamma*(f_cls*mask)` 数值上恒为 0,无论 `.detach()` 是否还在,雅可比都被 gamma=0 这个乘法因子清零——已用脚本对拍实测:gamma=0 时故意删掉 detach,`seg_backbone` 梯度依然精确为 0.0。因此 Gate 内部会先把 gamma 手动拨到 1.0(测完再原样恢复),让注入通路数值上真正激活,负面测试(误删 detach)才能被真正捕捉到(实测 `abs().sum()≈6094`)。
    - 另需用 `gamma_prior=0.0` 消除 `cbm_head` 里 `e_prior`(物理先验,经 `soft_mask←seg_backbone` 非 detach 路径,是设计内的合法耦合)对 `loss_mal` 的贡献,否则会把这条无关合法通路误判为泄漏(实测 `gamma_prior=1.0` 默认值下 `seg_backbone` 梯度 `abs().sum()≈14.2`,与 detach 是否失效完全无关)。
    - 回归测试:`tests/test_gate8_adaptor_isolation.py`(正面:真实代码通过;负面:monkeypatch `torch.Tensor.detach` 为恒等函数模拟误删,断言 Gate 能检测到并返回 False)。

---

## 2. vCLUB GRADIENT FLOW (Mutual Information Decoupling)

To ensure that the decoupling penalty $I(z; C)$ successfully regularizes the encoder to decorrelate latent features from target concepts, the latent variable passed to the mutual information estimator must retain its gradients. However, the conditioning target concepts must be detached to prevent label leak.

- **Forward Path Contract** (`cbm_head` returns a keyed `CBMHeadOutput` dataclass, not a positional tuple):
  ```python
  # 复用 model() 前向已产出的 concept_vector（勿二次跑 cbm_head）:
  #   arg1 = outputs["z_latent"]（保梯度，退耦约束导流回 backbone）
  #   arg2 = outputs["concept_vector"].detach()（掐断梯度，防标签泄漏）
  concept_vector_decouple = outputs["concept_vector"].detach()
  vclub_mi_bound = model.vclub(outputs["z_latent"], concept_vector_decouple)
  ```
  ⚠️ **禁二次 `model.cbm_head(z_latent.detach())`**(审查 S2,2026-08-10):cbm_head
  前向在 `self.training` 下每次都推进 EMA 熵门 `ema_entropy_tau`,二次前向使其每 step
  被推进两次(`alpha_gate` 收敛快 2×)且浪费算力。concept_vector 在 e_prior 融合前算出、
  与 z_latent 是否 detach 无关,复用值逐位等价,再 `.detach()` 即所需退耦向量。

---

## 3. BATCHNORM STATE LOCKING (Stage 2/3 Freezing)

When specific network components are parameter-frozen (`requires_grad = False`), their BatchNorm statistics must be locked to prevent feature representation drift.

- **Locker implementation** (in `src/training/loop/bn_freeze.py`):
  ```python
  def set_frozen_modules_eval(model):
      for m in model.modules():
          params = list(m.parameters())
          if len(params) > 0 and all(not p.requires_grad for p in params):
              m.eval()
  ```

---

## 4. AUC-MARGIN DUAL OPTIMIZATION & MIXED PRECISION

The Lagrangian multiplier `alpha` in `AUCMLossApprox` must be optimized in the maximization direction. This is encapsulated within the loss module.

- **Gradient Inversion Sequence** (FP16/AMP):
  ```python
  scaler.unscale_(optimizer)
  criterion.mal_loss_fn.step_dual_variable("pre") # flips alpha gradient sign
  optimizer.step()
  criterion.mal_loss_fn.step_dual_variable("post") # clamps alpha to [0, inf)
  ```

- **Logit Clamping**: Clamp malignancy outputs to `[-15.0, 15.0]` prior to exponent calculations to prevent NaN overflows.

---

## 7. FAMO 5-TASK SCHEDULING

| Channel | Loss Component | Purpose |
|:---|:---|:---|
| 1 | `loss_seg` (BCE + Dice) | Pixel-level segmentation |
| 2 | `loss_concepts_ce` (SLR: comp+echo+shape+margin+foci) | 18-dimensional concept classification |
| 3 | `loss_prior` (Huber) | TI-RADS clinical score regression |
| 4 | `loss_mal` (AUC-Margin) | Malignancy pairwise ranking |
| 5 | `loss_supcon` (SupCon) | Backbone visual feature contrastive learning |

### Loss Masking Configuration

- **Stage 1 (Epoch 1-40)**: Handled by `enable_mal_loss=False`, which multiplies `loss_mal` and `loss_supcon` by `0.0` inside `MultiTaskLoss.forward()`.
- **Stage 3 (Epoch 101-120)**: `update_famo_weights=False` is passed to disable FAMO dynamic weighting.

### FAMO 权重更新方向（2026-08-11 R6 修复）

**目标 =「救最差任务」**：给**改善慢（欠优化）的任务**加更高权重，给改善快（强）任务减权，防止强任务（分割）主干梯度支配共享表征、压垮分类。

- **更新规则（已修复）**：`delta = avg_improvement - improvement` → 改善慢的任务获正 `delta`、权重上升；改善快的任务获负 `delta`、权重下降。
- **旧（错，勿回退）**：`delta = improvement - avg_improvement` 方向与之相反，构成"马太效应"——给改善快的强任务加权，与"救最差任务"意图完全相反，曾致分割主干强梯度支配共享表征、分类 AUC 退化到 0.51–0.53。
- **分母 epsilon 1e-4 → 1e-2**：Dice loss 后期衰减到 ~1e-3，旧 1e-4 分母会因 batch 噪声爆炸。
- **`weighted_loss` 加 `weights.detach()`**：权重是手写调度规则、不参与反传。
- **方法论定位**：当前实现是**线性启发式**（`I_k ≈ -v_k`），**非原论文**（NeurIPS 2023, Liu et al., arXiv:2306.03792）的对数空间标准 FAMO（`∑z_k·log(L_k)`，对数损失空间 + 对数改进率）。原论文方案 B（对数空间重构）已记 backlog，本次只做了方案 A（修正符号方向）。方向正确性核验与算法重构详见 `reference/FAMO 动态权重更新规则的方向正确性核验与算法重构研究报告.md`。

---

## 7b. MASKED TASK-SPECIFIC LOSS(逐样本逐任务有效性掩码)

为多数据集混训防梯度污染,各损失组件支持逐样本有效性掩码,无效样本的 loss 贡献被精确置零。

### 数据流

1. **`loader.py`**: 从 `meta.json` 读可选 `label_mask` dict → `labels["valid_{task}"]`(缺失默认 1)。
2. **`train_step.py`**: 组装 `targets["label_validity"]`(概念任务掩码 dict)和 `targets["valid_mask"]`(分割有效性);mixup 不混合有效性掩码(元数据透传)。
3. **`losses/components/`**: 各组件新增 masked 分支:

| 组件文件 | 掩码行为 |
|:---|:---|
| `components/concepts.py` | 逐样本 loss × `label_validity[task]`,用 `sum/有效数` 归约(非 `.mean`,避免被屏蔽样本稀释) |
| `components/malignancy.py` | AUCM 为批量级损失,改用布尔索引过滤有效样本后再计算;全无效返回 0 梯度 |
| `components/segmentation.py` | BCE+Dice 逐样本 × `valid_mask`;全无效返回 0 梯度 |

### 向后兼容

**`targets` 中无 `label_validity` 键时完全走旧路径**,现有数据集(TN3K 单一数据集训练)零改动。

### 触发条件

仅当 `meta.json` 含 `label_mask` 字段时激活,缺失字段默认有效(`valid=1`)。

---

## 8. PHYSICAL PRIOR TIME-DECAY GATING

The model itself is stateless regarding time. The annealing factor `gamma_prior` is calculated by the training loop and injected during the forward pass:

```python
# 训练时(train_step.py):epoch 1-20 gamma=0, 21-40 线性升到 1, 41+ gamma=1
# 公式已提取为 src/training/stages.py 的模块级函数(2026-08-11,原内联在 train_batch 里
# 无法单测);数值锁定见 tests/test_stage_schedule.py::test_gamma_prior_annealing。
from src.training.stages import compute_gamma_prior
gamma_prior = compute_gamma_prior(epoch)      # = max(0.0, min(1.0, (epoch - 20) / 20.0))
outputs = model(images, gamma_prior=gamma_prior)

# 验证时(validate_step.py, Phase B 固定):gamma_prior = 1.0(不再随 epoch 漂移)
outputs = model(images, gamma_prior=1.0)
```

- **训练 Epoch 1-20**: `gamma_prior = 0.0` (physical prior disabled)
- **训练 Epoch 21-40**: `gamma_prior` linearly increases from 0 to 1
- **训练 Epoch 41+**: `gamma_prior = 1.0` (full physical prior active)
- **验证全阶段**: `gamma_prior = 1.0`(Phase B 固定,避免 val 指标随 epoch 漂移)

🔴 **γ 升满的时点与 Stage2 起点紧邻,是 2026-08-11 分割崩塌的四因之一**:γ 在 **epoch 40** 恰好到 1.0,
而 Stage2 从 **epoch 41** 起同时触发 lr 100 倍跳变 + 全模型解冻 + `enable_mal_loss` 首启。
物理先验通路(`multi_task_model.py` 的 `forward` 内 `soft_mask = torch.sigmoid(logits_mask)` **有意不 detach**;
⚠️ 勿与 outputs 里供三头消费的 `soft_mask_detached` 混淆,后者是另发的 detach 副本)是 `loss_mal → seg_backbone`
的唯一非 detach 路径,γ 由 0 到 1 等于把分类损失对分割主干的回传从关到全开——四件事挤在同一个 epoch,
分割 3 个 epoch 内从 97.71% 塌到 80.6%。详见 `src/training/stages.py` 的
`compute_gamma_prior` docstring 与 `docs/项目现状总纲.md` §6。**三头训练另建调度时须错开这些时点。**

---

## 9. MORPHOLOGICAL EROSION FOR SOFT CIRCULARITY

```python
eroded_mask = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
boundary = torch.clamp(mask - eroded_mask, min=0.0)
perimeter = boundary.sum(dim=(2, 3))
area = mask.sum(dim=(2, 3))
circularity = (4.0 * torch.pi * area) / (perimeter ** 2 + 1e-6)
```

---

## 10. EMA ENTROPY GATING FOR LATENT SHORTCUT

```python
if self.training:
    batch_median = torch.median(mean_entropy).detach()
    self.ema_entropy_tau.mul_(0.9).add_(batch_median, alpha=0.1)

alpha_raw = torch.sigmoid(10.0 * (mean_entropy - self.ema_entropy_tau))
alpha_gate = alpha_raw * 0.95 + 0.05  # 5% leakage floor
latent_res = self.latent_mlp(z_latent) * alpha_gate
```

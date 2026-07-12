# src/models/AGENTS.md - Model & Loss Architecture Specifications

## 0. MODULAR CODE ARCHITECTURE

Per project-wide modularity rules (`rules.md` §6), this folder is decomposed into single-responsibility modules:

| File / Package | Responsibility | Max Scope |
|:---|:---|:---|
| `physical_prior.py` | Differentiable physical feature extraction (AR, ER, Circularity) | ~160 lines |
| `cbm_head.py` | Concept Embedding Model head + vCLUB estimator | ~209 lines |
| `multi_task_model.py` | Model assembly hub (ConvNeXt-Nano + EfficientNet-B2 dual-backbone, adaptor, decoder, stage freeze registry) | ~300 lines |
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
| `components/regularizers.py` | `compute_regularizers(total_loss, outputs, lambda_latent, delta_budget, spin_l1_weight)` |

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
- **Injection Nodes**: Mid-to-high level layers (Stage 3 and Stage 4 of EfficientNet-B2), `multi_task_model.py:260,263`(`_forward_cls_backbone`)。
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
  z_latent_detached = outputs["z_latent"].detach()
  concept_vector_decouple = model.cbm_head(z_latent_detached).concept_vector.detach()
  vclub_mi_bound = model.vclub(outputs["z_latent"], concept_vector_decouple)
  ```

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
gamma_prior = max(0.0, min(1.0, (epoch - 20) / 20.0))
outputs = model(images, gamma_prior=gamma_prior)

# 验证时(validate_step.py, Phase B 固定):gamma_prior = 1.0(不再随 epoch 漂移)
outputs = model(images, gamma_prior=1.0)
```

- **训练 Epoch 1-20**: `gamma_prior = 0.0` (physical prior disabled)
- **训练 Epoch 21-40**: `gamma_prior` linearly increases from 0 to 1
- **训练 Epoch 41+**: `gamma_prior = 1.0` (full physical prior active)
- **验证全阶段**: `gamma_prior = 1.0`(Phase B 固定,避免 val 指标随 epoch 漂移)

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

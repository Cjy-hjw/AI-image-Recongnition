# src/training/AGENTS.md - Training Pipeline Specifications

## 0. PHASE 4 新增组件

### PCGrad优化器 (`pcgrad.py`) — 梯度冲突消解 ⚠️实验性,未接入训练主循环
- **⚠️ 边界约束(激活前必读)**:
  - **未接入 `train.py`**: 当前训练主循环用 FAMO 动态加权,不调用本类。
  - **与 FAMO 互斥**: PCGrad/FAMO 二选一：启用 PCGrad 前须先从 train.py 移除 FAMO 调用，并过 Gate2 复审。
  - **不兼容 AMP/GradScaler**: `pc_backward` 手工 `backward()`+`grad.copy_()`,GradScaler 感知不到这些
    手工梯度,会与 `scaler.step()` 冲突(inf/nan 静默跳步)。启用需 fp32 或额外适配。
  - **激活前必须过 Gate2 复审**(`rules.md §2`)。
- **背景**: 分割(几何驱动) vs 分类(纹理驱动)对共享编码器的梯度更新方向可能冲突(余弦<0),导致负迁移
- **原理**:
  - 计算各任务梯度g_i, g_j
  - 若g_i·g_j<0(冲突),投影g_i到g_j正交超平面: `g_i' = g_i - (g_i·g_j/||g_j||²)*g_j`
  - 消除冲突方向,保留一致方向
- **实现**:
  - `PCGrad(optimizer, shared_parameters=None)`: 包装Adam/SGD,在step()前拦截梯度投影
  - **`shared_parameters` 参数(54f0302 修复)**: PCGrad 理论只应对**共享编码器**参数做冲突投影;
    对各任务私有 head 投影是错的(head 梯度本就不该跨任务冲突,投影反而损害各 head 学习)。
    传入共享参数列表则只对其投影;不传则回退全参数(旧行为)并 docstring 警告。
  - `pc_backward(losses)`: 内部逐任务backward+清零提取独立梯度,两两投影,聚合后赋值参数梯度
  - **关键修复(d4c00ff)**: 旧版假设外部已backward导致梯度累加无法投影,新版内部自动处理
- **使用**:
  ```python
  # 仅对共享编码器投影(正确用法),且需 fp32、与 FAMO 二选一
  pcgrad = PCGrad(optimizer, shared_parameters=model.encoder.parameters())
  pcgrad.pc_backward([loss_seg, loss_cls])  # 自动backward+投影
  pcgrad.step()
  ```
- **显存**: <5%额外开销,适合官方14~16G(实际约16G/容器、当前分配14G)约束

---

## 1. THREE-STAGE PROGRESSIVE TRAINING (`stages.py` + `MultiTaskThyroidModel`)

| Stage | Epochs | Strategy | Learning Rate |
|:---|:---|:---|:---|
| **Stage 1** | 1–40 | Segmentation + Concepts (classification frozen, loss_mal/supcon masked to 0.0) | `lr_base` (0.0024) |
| **Stage 2** | 41–100 | Full model unfrozen, 5-task FAMO scheduling | `lr_base` (0.0024) |
| **Stage 3** | 101–120 | Backbone + seg frozen, AUC-Margin fine-tuning only | 5e-6 |

Stage freeze rules live on the model via `MultiTaskThyroidModel.set_training_stage(stage)` and `is_param_trainable(name, stage)`. `configure_stage_parameters()` delegates freezing to the model, then builds optimizer param groups and LR schedule.

### Critical Freezing Rules

- **vCLUB isolation**: vCLUB 使用专属 vclub_optimizer(lr=1e-4) 独立优化；主 optimizer 参数组显式排除 vclub 参数。
- **Stage 1**: `loss_mal` and `loss_supcon` multiplied by 0.0 to prevent classification gradient leakage.
- **Stage 3**: FAMO disabled (`update_famo_weights=False`).
- **BatchNorm auto-eval**: After `model.train()`, modules with all-frozen params are set to `.eval()` (see `loop/bn_freeze.py`).

### Checkpoint & Early Stopping

- **Stage 1 best**: `val_dice` only.
- **Stage 2–3 best**: `checkpoint_w_dice * val_dice + checkpoint_w_auc * val_auc` (defaults 0.30 / 0.70).
- **Stage 3 ES**: `stage3_early_stopping_patience=0` disables early stopping; tracker resets at epoch 101.
- **`validate_epoch` return**: 5-tuple `(avg_loss, avg_dice, avg_acc, avg_loss_l2, val_auc)`; `val_auc` is `None` in Stage 1 or single-class val sets.

---

## 2. TRAINING & VALIDATION LOOPS

### Package Structure (`training/loop/`)

| File | Responsibility |
|:---|:---|
| `train_step.py` | `train_batch()` — single batch forward/backward with MixUp, vCLUB, SLR, AMP |
| `validate_step.py` | `validate_batch()` — single batch validation with Val Diag print |
| `slr.py` | `compute_slr_weights()` — Supervised Label Rectification (1000-sample history, mean+2σ anomaly) |
| `vclub.py` | `vclub_learning_step()`, `compute_vclub_mi_bound()` — mutual information decoupling |
| `bn_freeze.py` | `set_frozen_modules_eval()` — automatic BatchNorm eval for frozen modules |

### `trainer.py` (Thin Orchestrator)

`trainer.py` contains `train_epoch()` and `validate_epoch()` which delegate to `loop/train_step.py` and `loop/validate_step.py`.

### NaN Sentinel Diagnostics

The validation loop includes a **Val Diag** print at `batch_idx == 0`:
```python
print(f"  [Val Diag] pred_prob: mean={pred_mean:.4f}, std={pred_std:.4f}, min={...}, max={...}", flush=True)
```

### GradScaler Protocol & Optimizer Encapsulation

- `scaler.unscale_(optimizer)` before alpha gradient inversion
- `clip_grad_norm_(model.parameters(), max_norm=1.0)` after every backward
- Dual optimization is encapsulated: `criterion.mal_loss_fn.step_dual_variable("pre")` before `optimizer.step()`, and `"post"` afterwards.

**AMP 精度选择（fp16+GradScaler vs bf16 无 scaler）**：主训练管道（`train.py`）用 fp16 + GradScaler（上述协议）。但 **LoRA 微调（`scripts/train_thyroidxl.py`）与 malignancy 预训练（`pretrain_malignancy.py`）用 bf16 AMP、无 GradScaler**——bf16 指数范围同 fp32，无需 loss scaling。**血泪教训**：`train_thyroidxl.py` 曾用 fp16 默认 autocast，FT-LoRA 阶段 backbone LoRA 扰动把激活顶穿 fp16 的 65504 天花板，GradScaler scale 从 65536 一路减半仍失守，loss 爆到 1e99（epoch 13）。改 bf16 后 1300 步诊断无溢出。**凡 backbone 参与微调/存在深层扰动的训练，优先用 bf16**；纯冻结 backbone 的浅层训练 fp16 尚可。bf16 路径下仍保留 `clip_grad_norm_(max_norm=1.0)`。

**冻结 backbone 必须同时冻结其 BatchNorm running stats（2026-07-10 血泪教训）**：`train_thyroidxl.py` 的 LP-FT-LoRA 训练曾**遗漏**调用 `set_frozen_modules_eval()`——虽然 backbone 原始权重 `requires_grad=False`，但 `model.train()` 递归下去后其内部 `BatchNorm` 仍处于训练模式，`running_mean`/`running_var` **只受 `module.training` 控制、与 `requires_grad` 无关**，会持续随每个 batch 的真实激活值漂移（本地 A800 实测：单个 batch 就能让某层 `running_var` 漂移 22.98）。50 epoch 下来把一个训练前就已临界（`running_var` 低至 3.5e-42，EfficientNet-B2 预训练权重自带）的 BN 通道冲向数值病态区间，经 backbone 逐层复合放大，导致 CPU/GPU、LoRA-结构/合并-结构两种等价 forward 之间相对误差炸到 70%~1240%，checkpoint 因此报废重训。**规则**：任何训练脚本只要有"参数冻结但仍 `model.train()`"的 backbone/子模块，必须在 `train_one_epoch`（或等价逐 epoch 入口）里紧跟 `model.train()` 之后调用 `from src.training.loop.bn_freeze import set_frozen_modules_eval; set_frozen_modules_eval(model)`——`train.py` 主管线的 `trainer.py:35` 早已采用这个模式，新训练脚本必须照抄，不能假设"冻结 = 完全不变"。验收方法：跑几个训练 step，`assert torch.equal(bn.running_var_before, bn.running_var_after)` 直接断言冻结子模块的 BN buffer 零漂移。

---

## 3. AUTHENTICITY CLASSIFIER (`authenticity_trainer.py`)

Lightweight MLP trained on 13-dim features (GLCM + freq + ELA + DND speckle + TAI anisotropy). Runs **before** main training by default. Saves to `checkpoints/authenticity_model.pth`. Main model weights MUST NOT include these parameters.

Use `python train.py --skip-auth` to skip when `checkpoints/authenticity_model.pth` already exists. If the flag is set but weights are missing, training logs a warning and runs authenticity training anyway.

Note: Renamed from `authenticity.py` to `authenticity_trainer.py` to avoid confusion with `src/gatekeeper/authenticity.py` (feature extraction + inference module).

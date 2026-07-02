# src/training/AGENTS.md - Training Pipeline Specifications

## 0. PHASE 4 新增组件

### PCGrad优化器 (`pcgrad.py`) — 梯度冲突消解 ⚠️实验性,未接入训练主循环
- **⚠️ 边界约束(激活前必读)**:
  - **未接入 `train.py`**: 当前训练主循环用 FAMO 动态加权,不调用本类。
  - **与 FAMO 互斥**: PCGrad 与 FAMO 是两套多任务权衡方案,二选一,叠加语义未定义,禁止同时启用。
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
- **显存**: <5%额外开销,适合官方32G约束

---

## 1. THREE-STAGE PROGRESSIVE TRAINING (`stages.py` + `MultiTaskThyroidModel`)

| Stage | Epochs | Strategy | Learning Rate |
|:---|:---|:---|:---|
| **Stage 1** | 1–40 | Segmentation + Concepts (classification frozen, loss_mal/supcon masked to 0.0) | `lr_base` (0.0024) |
| **Stage 2** | 41–100 | Full model unfrozen, 5-task FAMO scheduling | `lr_base` (0.0024) |
| **Stage 3** | 101–120 | Backbone + seg frozen, AUC-Margin fine-tuning only | 5e-6 |

Stage freeze rules live on the model via `MultiTaskThyroidModel.set_training_stage(stage)` and `is_param_trainable(name, stage)`. `configure_stage_parameters()` delegates freezing to the model, then builds optimizer param groups and LR schedule.

### Critical Freezing Rules

- **vCLUB isolation**: `model.vclub` optimized by **separate** `vclub_optimizer` (lr=1e-4), never by main optimizer.
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

---

## 3. AUTHENTICITY CLASSIFIER (`authenticity_trainer.py`)

Lightweight MLP trained on 13-dim features (GLCM + freq + ELA + DND speckle + TAI anisotropy). Runs **before** main training by default. Saves to `checkpoints/authenticity_model.pth`. Main model weights MUST NOT include these parameters.

Use `python train.py --skip-auth` to skip when `checkpoints/authenticity_model.pth` already exists. If the flag is set but weights are missing, training logs a warning and runs authenticity training anyway.

Note: Renamed from `authenticity.py` to `authenticity_trainer.py` to avoid confusion with `src/gatekeeper/authenticity.py` (feature extraction + inference module).

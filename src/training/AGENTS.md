# src/training/AGENTS.md - Training Pipeline Specifications

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

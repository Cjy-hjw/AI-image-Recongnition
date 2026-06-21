# src/training/AGENTS.md - Training Pipeline Specifications

This protocol defines the multi-stage training strategy, parameter freezing rules, and diagnostic logging contracts.

---

## 1. THREE-STAGE PROGRESSIVE TRAINING (`stages.py`)

The training follows a 3-stage curriculum to prevent gradient conflicts between segmentation and classification tasks:

| Stage | Epochs | Strategy | Learning Rate |
|:---|:---|:---|:---|
| **Stage 1** | 1–40 | Segmentation + Concepts training (classification head frozen, loss_mal/supcon masked to 0.0) | `lr_base` (0.0024) |
| **Stage 2** | 41–100 | Full model unfrozen, 5-task FAMO scheduling | `lr_base` (0.0024) |
| **Stage 3** | 101–120 | Backbone + segmentation frozen, AUC-Margin fine-tuning only | 5e-6 |

### Critical Freezing Rules

- **vCLUB isolation**: `model.vclub` parameters have `requires_grad=True` in ALL stages but are optimized by a **separate** `vclub_optimizer` (lr=1e-4), never by the main optimizer.

- **Stage 1 (Epochs 1-40)**: 
  - Frozen: `cbm_head.classifier_direct`, `cbm_head.latent_mlp`, `criterion.mal_loss_fn` (AUCM dual params)
  - Active: backbone, segmentation decoder, concept prediction heads (`proj_plus`, `proj_minus`, `s_layer`)
  - Gradient masking: `loss_mal` and `loss_supcon` are multiplied by 0.0 to prevent classification gradients from leaking into backbone

- **Stage 2 (Epochs 41-100)**: 
  - All model parameters unfrozen and active
  - 5-task FAMO scheduling enabled (segmentation, concepts, prior, malignancy, supcon)
  - AUCM dual params (`a`, `b`, `alpha`) added to `no_decay_params` group (weight_decay=0.0)

- **Stage 3 (Epochs 101-120)**: 
  - Frozen: backbone (`shared_stem`, `shared_blocks`), segmentation decoder, physical prior extractor
  - Active: `cbm_head.classifier_direct`, `cbm_head.latent_mlp`
  - FAMO disabled (`update_famo_weights=False`), direct AUC-Margin optimization
  - Learning rate fixed at 5e-6 for fine-tuning

- **BatchNorm automatic eval mode**: After `model.train()`, any module with all parameters frozen (`requires_grad=False`) is automatically set to `.eval()` mode to prevent BatchNorm running statistics from updating.
---

## 2. TRAINING & VALIDATION LOOPS (`trainer.py`)

### NaN Sentinel Diagnostics
The `MultiTaskLoss.forward()` contains 4 NaN sentinel guards (see `src/models/AGENTS.md` §0). In `trainer.py`, the validation loop includes a **Val Diag** print that logs `pred_prob` statistics (mean/std/min/max) at `batch_idx == 0` for each epoch. This diagnostic MUST be preserved to detect degenerate all-zeros or all-ones mask predictions.

### SLR Weight Computation
Supervised Label Rectification weights are computed per-sample using a running history of 1000 losses. Anomalies (losses > mean + 2σ) are down-weighted by `1 - slr_beta`.

### GradScaler Protocol
- `scaler.unscale_(optimizer)` MUST be called before alpha gradient inversion.
- `clip_grad_norm_(model.parameters(), max_norm=1.0)` applied after every backward pass.
- **PCGrad Compatibility**: Never combine `PCGradWrapper` with a standard `GradScaler` unless unscaling is performed per-task or manual scaling math is verified. If using `PCGrad`, run backward passes in FP32 or use BF16 autocast (which does not require scaling on A800 GPUs).

---

## 3. AUTHENTICITY CLASSIFIER (`authenticity.py`)

Lightweight MLP trained on GLCM+ELA features for image authenticity verification. This runs **before** main training starts and saves weights to `checkpoints/authenticity_model.pth`. The main model weights MUST NOT include these parameters.

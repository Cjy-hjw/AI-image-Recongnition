# src/models/AGENTS.md - Model & Loss Architecture Specifications
## 0. MODULAR CODE ARCHITECTURE

Per project-wide modularity rules (`rules.md` §6), this folder is decomposed into single-responsibility modules:

| File / Package | Responsibility | Max Scope |
|:---|:---|:---|
| `multi_task_model.py` | Model assembly hub (backbone, decoder, wiring) | ~360 lines |
| `physical_prior.py` | Differentiable physical feature extraction (AR, ER, Circularity) | ~160 lines |
| `cbm_head.py` | Concept Embedding Model head + vCLUB estimator | ~190 lines |
| `losses/` | Decomposed loss function subpackage | See below |
| `loss.py` | Backward-compatible re-export shim (DO NOT add new code here) | ~35 lines |
| `famo.py` | FAMO dynamic loss weight scheduler | ~50 lines |
| `pcgrad.py` | PCGrad gradient projection (legacy, unused) | ~80 lines |

### losses/ Subpackage Structure
| File | Contents |
|:---|:---|
| `losses/__init__.py` | Re-exports all 7 symbols with `__all__` |
| `losses/segmentation.py` | `DiceLoss` |
| `losses/classification.py` | `AUCMLossApprox`, `AdaptiveVSAndAsymmetricLoss`, `SupervisedContrastiveLoss` |
| `losses/concepts.py` | `symmetric_label_regression_loss`, `symmetric_label_regression_loss_bce` |
| `losses/multitask.py` | `MultiTaskLoss` (orchestrator with NaN sentinels) |

**Rule**: New loss functions MUST go into the appropriate `losses/` submodule, NOT into `loss.py` or `multitask.py` directly. `loss.py` is a shim only.

---

This local protocol defines the mathematical representations, gradient flows, and batchnorm freeze requirements for all neural networks in this folder.

---

## 1. vCLUB GRADIENT FLOW (Mutual Information Decoupling)
To ensure that the decoupling penalty $I(z; C)$ successfully regularizes the encoder to decorrelate latent features from target concepts, the latent variable passed to the mutual information estimator must retain its gradients. However, the conditioning target concepts must be detached to prevent label leak.
- **Forward Path Contract**:
  ```python
  z_latent_detached = outputs["z_latent"].detach()
  # Generate concept outputs for estimator via detached z_latent to isolate backpropagation
  _, _, _, _, _, concept_vector_decouple, _, _, _ = model.cbm_head(z_latent_detached)
  concept_vector_decouple = concept_vector_decouple.detach()
  # Calculate mutual information bound: the first argument must have active gradients
  vclub_mi_bound = model.vclub(outputs["z_latent"], concept_vector_decouple)
  ```

---

## 2. BATCHNORM STATE LOCKING (Stage 2/3 Freezing)
When specific network components are parameter-frozen (`requires_grad = False`), their BatchNorm statistics must be locked to prevent feature representation drift.
- **Locker implementation**:
  ```python
  def set_frozen_bn_eval(m):
      if isinstance(m, nn.modules.batchnorm._BatchNorm):
          if all(not p.requires_grad for p in m.parameters()):
              m.eval()
  ```
- **Lifecycle Requirement**: This must be re-applied after every single `model.train()` call inside the training loop to override PyTorch's default behavior.

---

## 3. AUC-MARGIN DUAL OPTIMIZATION & MIXED PRECISION
The Lagrangian multiplier `alpha` in `AUCMLossApprox` must be optimized in the maximization direction.
- **Gradient Inversion Sequence**:
  If using FP16/AMP `GradScaler`, the unscale step must happen **before** gradient inversion:
  ```python
  # 1. Unscale gradients first to align scales
  scaler.unscale_(optimizer)
  # 2. Invert alpha's gradient for dual maximization (ensure absolute value/clamping is applied)
  criterion.mal_loss_fn.alpha.grad.data.mul_(-1.0)
  # 3. Step and clamp to non-negative projection
  optimizer.step()
  criterion.mal_loss_fn.alpha.data.clamp_(min=0.0)
  ```
- **Expectation Normalization**: Ensure conditional expectations in AUC loss division are normalized by class counts ($n_{pos}, n_{neg}$) instead of the entire batch size $N$, to prevent class imbalance gradient dilution.
- **Gradient Vanishing Prevention**: Clamping adaptive margin prediction entropy calculations with `.detach()` to prevent the optimizer from optimizing classification logits toward maximum entropy (0.5 uncertainty).
- **Logit Clamping**: Clamp malignancy outputs to `[-15.0, 15.0]` prior to any exponent calculations to prevent dynamic scaling NaN overflows.

---

## 4. OOM PREVENTATIVE RULE (Spatial Attention Resolution)
- **Rule**: Do NOT apply dense spatial self-attention or cross-attention (like `DSCA`) on feature maps with resolutions > 32×32. Doing so generates high-memory tensors (e.g., 17GB+ for 64×64 attention maps) that cause immediate CUDA OOM.
- **Scope**: Only apply `LayerwiseGTA` and `DSCA` on high-level feature maps (Stage 4 and Stage 5), where spatial resolution is ≤ 32×32.
- **Rationale**: The attention map complexity is $O(H^2 \times W^2 \times C)$. At 64×64, this becomes 16× the memory of 16×16, easily exceeding A800 VRAM.

---

## 6. FAMO 5-TASK SCHEDULING (Classification AUC Breakthrough)

To resolve gradient conflicts in the 3-task FAMO configuration, the scheduler has been expanded to 5 independent task channels:

| Channel | Loss Component | Purpose |
|:---|:---|:---|
| 1 | `loss_seg` (BCE + Dice) | Pixel-level segmentation |
| 2 | `loss_concepts_ce` (SLR: comp+echo+shape+margin+foci) | 18-dimensional concept classification |
| 3 | `loss_prior` (Huber) | TI-RADS clinical score regression |
| 4 | `loss_mal` (AUC-Margin) | Malignancy pairwise ranking |
| 5 | `loss_supcon` (SupCon) | Backbone visual feature contrastive learning |

### Stage-Aware Loss Masking
- **Stage 1 (Epoch 1-40)**: `loss_mal` and `loss_supcon` are multiplied by `0.0` inside `MultiTaskLoss.forward()` when `stage=1`. This prevents classification gradients from propagating through the frozen `classifier_direct` and `latent_mlp` heads back to the shared backbone.
- **Stage 3 (Epoch 101-120)**: `update_famo_weights=False` is passed to disable FAMO dynamic weighting, allowing direct optimization of AUC-Margin loss.

---

## 7. PHYSICAL PRIOR TIME-DECAY GATING

The `e_prior` feature from `DifferentiablePhysicalPriorExtractor` is gated by a time-decay factor to prevent early-stage noise injection:

```python
gamma = max(0.0, min(1.0, (self.current_epoch - 20) / 20.0))
e_prior = e_prior_raw * gamma
```

- **Epoch 1-20**: `gamma = 0.0` (physical prior disabled, backbone learns visual features autonomously)
- **Epoch 21-40**: `gamma` linearly increases from 0 to 1 (gradual physical prior injection)
- **Epoch 41+**: `gamma = 1.0` (full physical prior active)

**Train/Eval Consistency**: The same `gamma` formula is applied in both training and evaluation modes to avoid distribution shift.

---

## 8. MORPHOLOGICAL EROSION FOR SOFT CIRCULARITY

The `DifferentiablePhysicalPriorExtractor` uses morphological erosion to compute soft circularity, aligning training gradients with evaluation metrics:

```python
# Erode mask using 3x3 max-pooling (equivalent to morphological erosion)
eroded_mask = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
# Extract single-pixel boundary
boundary = torch.clamp(mask - eroded_mask, min=0.0)
# Compute perimeter and circularity
perimeter = boundary.sum(dim=(2, 3))
area = mask.sum(dim=(2, 3))
circularity = (4.0 * torch.pi * area) / (perimeter ** 2 + 1e-6)
```

**Key Benefits**:
- Produces ~1-pixel wide boundary (vs 3-5 pixels with Sobel)
- Gradient flow matches `cv2.findContours` evaluation logic
- Prevents Train-Eval gap that caused AUC collapse

---

## 9. EMA ENTROPY GATING FOR LATENT SHORTCUT

The `ConceptEmbeddingCBMHead` uses an EMA-tracked entropy threshold to dynamically gate the latent shortcut:

```python
# Track median entropy with exponential moving average
if self.training:
    batch_median = torch.median(mean_entropy).detach()
    self.ema_entropy_tau.mul_(0.9).add_(batch_median, alpha=0.1)

# Compute gating with 5% leakage floor to prevent gradient death
alpha_raw = torch.sigmoid(10.0 * (mean_entropy - self.ema_entropy_tau))
alpha_gate = alpha_raw * 0.95 + 0.05  # Guarantees minimum 5% activation
latent_res = self.latent_mlp(z_latent) * alpha_gate
```

**Design Rationale**:
- EMA tracking adapts to actual entropy distribution (typically 0.05-0.15, not hardcoded 0.5)
- 5% leakage floor prevents gradient death when concepts are confident
- Enables latent shortcut to compensate for concept prediction errors

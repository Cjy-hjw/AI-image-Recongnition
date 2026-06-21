# AGENTS.md - Global Project Architecture & Rules Directory

This document serves as the master coordinator for the thyroid multi-task diagnostic project, defining overall specifications and indexing module-specific rules.

---

| Directory / Module | Specific Rules File | Key Focus |
| :--- | :--- | :--- |
| **Global Rules & Gates** | `/rules.md` | Pre-flight check workflow, OOM bounds, abort policy, **code modularity**. |
| **Quality Verification** | `/docs/QA_VP.md` | Non-negotiable testing standards, check lists. |
| **Model & Custom Loss** | `/src/models/AGENTS.md` | vCLUB detach path, BatchNorm freeze eval logic, AUC-Margin unscale sequence. |
| **Loss Functions** | `/src/models/losses/` | Decomposed loss modules: segmentation, classification, concepts, multitask. |
| **Training Pipeline** | `/src/training/AGENTS.md` | Three-stage progressive training, parameter freezing, NaN sentinels. |
| **Data Loading & Augmentation** | `/src/dataset/AGENTS.md` | Static normalization, MixUp/CutMix, patient ID grouping. |
| **Gatekeeper Pipeline** | `/src/gatekeeper/AGENTS.md` | Authenticity classification, deduplication, tampering detection. |
| **Source Packages** | `/src/AGENTS.md` | Package structure overview, TI-RADS score guidelines. |
| **Scripts & Splits** | `/scripts/AGENTS.md` | 7-Gate verify_model.py tests, JSON diagnostics. |

### 1.1 Modular Code Architecture
The project follows strict functional decomposition principles (see `rules.md` §6):

- **`src/models/`**: Model architecture definitions
  - `multi_task_model.py` - Main model assembly
  - `physical_prior.py` - Differentiable physical feature extraction
  - `cbm_head.py` - Concept bottleneck model head
  - `losses/` - Decomposed loss functions (segmentation, classification, concepts, multitask)

- **`src/training/`**: Training pipeline logic
  - `stages.py` - Three-stage parameter freezing strategy
  - `trainer.py` - Training/validation loops with NaN sentinels
  - `authenticity.py` - Authenticity classifier training

**Backward Compatibility**: All decomposed modules maintain `__init__.py` re-export shims to preserve existing import paths.

---

## 2. PRODUCTION INFERENCE & CONTAINER DEPLOYMENT (predict.py & entrypoint.sh)

Because inference and container scripts live in the root workspace, their strict engineering rules are solidifed below:

### 2.1 Inference Optimizations
- **Inference Mode**: All forward passes in `predict.py` MUST run under `torch.inference_mode()` (never `torch.no_grad()`) to minimize latency and memory footprint.
- **Adaptive Precision**: Support `--precision` choices `fp32`, `fp16`, and `bf16`. Under half-precisions, apply a `torch.clamp(..., min=-15.0, max=15.0)` safety clamp on malignancy predictions to prevent NaN crashes.

### 2.2 PACS DICOM Metadata Anonymization
- **Privacy Standard**: Any DICOM file input must be anonymized before saving outputs. The following fields must be overwritten to `"ANONYMIZED"` (or `"00000101"` for birthdate):
  - `PatientName`, `PatientID`, `PatientBirthDate`, `ReferringPhysicianName`, `InstitutionName`.

### 2.3 Container Checksum Verification
- **Anti-Tampering**: The container entrypoint `scripts/entrypoint.sh` must execute SHA-256 integrity checksum verification on the model weights (`checkpoints/best_model.pth`) and the python inference script (`predict.py`). If `PRODUCTION=1` is set, a hash mismatch must force a hard exit with status `1`.

---

## 3. CLASSIFICATION AUC BREAKTHROUGH REFACTORING

The system underwent a comprehensive architectural refactoring to resolve the AUC bottleneck (previously stuck at 77.55%). The following changes are now in production:

### 3.1 Architecture Changes
- **Morphological Erosion Circularilty**: Replaced Sobel edge detection with differentiable morphological erosion in `physical_prior.py` to align training gradients with `cv2.findContours` evaluation.
- **EMA Entropy Gating**: Replaced hardcoded 0.5 threshold with EMA-tracked median entropy in `cbm_head.py`, with 5% leakage floor to prevent gradient death.
- **SupCon Projection Remapping**: Changed `supcon_projector` input from 288-dim concept mixed features to 352-dim pooled backbone features, preventing concept entanglement.
- **Dead Code Removal**: Removed unused `LayerwiseGTA`, `DSCA`, and `SPINHead` classes from `multi_task_model.py`.

### 3.2 Loss & Training Changes
- **5-Task FAMO Scheduling**: Expanded from 3-task to 5-task FAMO (segmentation, concepts, prior, malignancy, supcon) to prevent gradient cancellation within loss groups.
- **3-Stage Progressive Training**: Stage 1 (1-40) seg+concepts only with classification frozen; Stage 2 (41-100) full unfreeze; Stage 3 (101-120) backbone frozen, AUC fine-tuning.
- **Stage 1 Gradient Masking**: `loss_mal` and `loss_supcon` are zeroed when `stage=1` to prevent classification gradients from corrupting the backbone.
- **AUCM Dual Params**: Moved to `no_decay_params` group to prevent weight decay from pulling Lagrangian multipliers toward zero.

### 3.3 Verification
- **7-Gate Test Suite**: `verify_model.py` now includes Gate 6 (Stage 1 gradient isolation) and Gate 7 (optimizer LR alignment across stages).
- **Automatic BatchNorm Eval**: Frozen modules are automatically set to `.eval()` after `model.train()` to prevent BatchNorm running statistics from updating.

See `src/models/AGENTS.md` §6-9 and `src/training/AGENTS.md` §1 for detailed specifications.

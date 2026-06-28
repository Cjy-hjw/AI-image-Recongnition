# AGENTS.md - Global Project Architecture & Rules Directory

This document serves as the master coordinator for the thyroid multi-task diagnostic project, defining overall specifications and indexing module-specific rules.

---

| Directory / Module | Specific Rules File | Key Focus |
| :--- | :--- | :--- |
| **Global Rules & Gates** | `/rules.md` | Pre-flight check workflow, OOM bounds, abort policy, **code modularity**. |
| **Quality Verification** | `/QA_VP.md` | Non-negotiable testing standards, check lists. |
| **Model & Custom Loss** | `/src/models/AGENTS.md` | vCLUB detach path, BatchNorm freeze eval logic, AUC-Margin unscale sequence. |
| **Loss Functions** | `/src/models/losses/` | Decomposed loss modules: segmentation, classification, concepts, multitask, components. |
| **Training Pipeline** | `/src/training/AGENTS.md` | Three-stage progressive training, parameter freezing, NaN sentinels. |
| **Data Loading & Augmentation** | `/src/dataset/AGENTS.md` | Static normalization, MixUp/CutMix, patient ID grouping. |
| **Inference Pipeline** | `/src/AGENTS.md` §2 | Shared inference modules: io, model_runner, postprocess, export. |
| **Gatekeeper Pipeline** | `/src/gatekeeper/AGENTS.md` | Authenticity classification, deduplication, tampering detection. |
| **Scripts & Splits** | `/scripts/AGENTS.md` | 7-Gate verify_model.py tests, data splitting (纯随机图像级划分). |
| **Tests** | `/tests/AGENTS.md` | 全套 73 passed:14 physical_prior + 15 cbm_head + 19 phase_refactor(Phase A/B/C/D regression) + 15 gatekeeper + 10 masked_loss. |
| **Source Packages** | `/src/AGENTS.md` | Package structure overview, TI-RADS score guidelines. |

---

## 1. MODULAR CODE ARCHITECTURE

The project follows strict functional decomposition (see `rules.md` §6):

### `src/models/` — Model architecture definitions
  - `multi_task_model.py` — Main model assembly (dual-backbone, stage freeze registry)
  - `pretrained_loader.py` — `load_local_safetensors()` for offline backbone weights
  - `physical_prior.py` — Differentiable physical feature extraction
  - `cbm_head.py` — Concept bottleneck model head
  - `losses/` — Decomposed loss functions (segmentation, classification, concepts)
  - `losses/components/` — MultiTaskLoss computation modules (segmentation, concepts, malignancy, famo_mix, regularizers)

### `src/training/` — Training pipeline logic
  - `stages.py` — Three-stage parameter freezing strategy
  - `trainer.py` — Training/validation epoch orchestrators
  - `loop/` — Per-batch computation (train_step, validate_step, slr, vclub, bn_freeze)
  - `authenticity_trainer.py` — Authenticity classifier training

### `src/inference/` — Shared inference pipeline (predict.py + evaluate.py)
  - `io.py` — Image loading + DICOM PHI anonymization
  - `gatekeeper_runner.py` — 4-layer gatekeeper audit (predict.py only)
  - `model_runner.py` — Model build + preprocess + forward (inference_mode + autocast)
  - `postprocess.py` — Prediction decoding + TI-RADS breakdown
  - `export.py` — Artifact saving + Grad-CAM overlay

### `src/dataset/` — Data pipeline
  - `loader.py` — ThyroidDataset with PatientID grouping
  - `augmentation/transforms.py` — 8 ultrasound-specific transforms + `get_train_transforms()`
  - `augmentation/mixup.py` — Dominant-Region CutMix + standard MixUp

### `scripts/` — Verification & data formatting
  - `verify/gates/` — 7 individual gate modules (gate1_shapes..gate7_optimizer_lr)
  - `verify/runner.py` — Gate orchestrator (`verify()`)
  - `data/pseudo_labels.py` — TI-RADS pseudo-label extraction
  - `data/split_tn3k.py` — 纯随机图像级 train/val/holdout 划分(Phase D)

### `tests/` — 回归测试套件
  - `test_physical_prior.py` — DifferentiablePhysicalPriorExtractor (14 tests)
  - `test_cbm_head.py` — ConceptEmbeddingCBMHead (15 tests)
  - `test_phase_refactor.py` — Phase A/B/C/D 重构回归守卫 (19 tests)

**Backward Compatibility**: All decomposed modules maintain `__init__.py` re-export shims to preserve existing import paths.

---

## 2. PRODUCTION INFERENCE & CONTAINER DEPLOYMENT

### 2.1 Inference Optimizations
- **Inference Mode**: All forward passes run under `torch.inference_mode()` (via `src/inference/model_runner.py`).
- **Adaptive Precision**: `--precision` choices `fp32`, `fp16`, `bf16`. Half-precisions apply `torch.clamp(..., min=-15.0, max=15.0)` on malignancy predictions.

### 2.2 PACS DICOM Metadata Anonymization
- **Privacy Standard**: 21 PHI fields overwritten (HIPAA Safe Harbor). 完整字段清单与脱敏规则见 `QA_VP.md §1.6` 及 `src/inference/io.py` 的 `_PHI_FIELDS`(日期类置 `"00000101"`,其余置 `"ANONYMIZED"`)。

### 2.3 Container Checksum Verification
- `scripts/entrypoint.sh` executes SHA-256 verification on `checkpoints/best_model.pth` and `predict.py`. `PRODUCTION=1` → hard exit on hash mismatch.

---

## 3. CLASSIFICATION AUC BREAKTHROUGH REFACTORING

### 3.1 Architecture Changes
- **Morphological Erosion Circularity**: Differentiable erosion in `physical_prior.py` aligns with `cv2.findContours` evaluation.
- **EMA Entropy Gating**: EMA-tracked median entropy in `cbm_head.py`, 5% leakage floor.
- **SupCon Projection Remapping**: 352-dim pooled backbone features (not 288-dim concept features).

### 3.2 Dual-Backbone Architecture (Implemented)
- **Heterogeneous Dual-Backbone**: `ConvNeXt-Nano` for Segmentation (high frequency, spatial), `EfficientNet-B2` for Classification (global semantics).
- **Task-Alignment Adaptor**: Unidirectional cross-layer spatial alignment using `f_seg_detached` as a soft attention gate for classification features at Stage 3 and Stage 4.

### 3.3 Loss & Training Changes
- **5-Task FAMO**: segmentation, concepts, prior, malignancy, supcon — independent channels prevent gradient cancellation.
- **3-Stage Progressive Training**: Stage 1 (1-40) seg+concepts only; Stage 2 (41-100) full; Stage 3 (101-120) backbone frozen, AUC fine-tuning.
- **Stage 1 Gradient Masking**: `loss_mal` and `loss_supcon` zeroed when `stage=1`.
- **AUCM Dual Params**: `no_decay_params` group (weight_decay=0.0).

### 3.4 Verification
- **7-Gate Test Suite**: `python scripts/verify_model.py` — all gates must pass.
- **Automatic BatchNorm Eval**: Frozen modules auto-set to `.eval()` after `model.train()`.

See `src/models/AGENTS.md` §6-9 and `src/training/AGENTS.md` §1 for detailed specifications.

服务器项目在home/cjy下；该搜索时就搜索；
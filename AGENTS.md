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
| **Tests** | `/tests/AGENTS.md` | 全套 105 passed:14 physical_prior + 15 cbm_head + 19 phase_refactor(Phase A/B/C/D regression) + 15 gatekeeper + 10 masked_loss + 21 video(动态图/cine 管线 + 通道形态/通道序/彩色单图 DICOM 回归) + 2 authenticity_trainer(smoke) + 7 multisource(多源混训骨架) + 2 smoke_entrypoints(CLI 入口端到端). |
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
  - `video_io.py` — 动态图/cine 抽帧(MP4/AVI/MOV/MKV + 多帧 DICOM);`video_aggregate.py` — 抽样下标 + 帧级聚合(恶性取最大帧)。驱动:根目录 `video_predict.py`

### `src/dataset/` — Data pipeline
  - `loader.py` — ThyroidDataset with PatientID grouping
  - `augmentation/transforms.py` — 7 ultrasound-specific transforms + `get_train_transforms()`
  - `augmentation/mixup.py` — Dominant-Region CutMix + standard MixUp

### `scripts/` — Verification & data formatting
  - `verify/gates/` — 7 individual gate modules (gate1_shapes..gate7_optimizer_lr)
  - `verify/runner.py` — Gate orchestrator (`verify()`)
  - `data/pseudo_labels.py` — TI-RADS pseudo-label extraction
  - `data/split_tn3k.py` — 纯随机图像级 train/val/holdout 划分(Phase D)

### `tests/` — 回归测试套件(全套 105 tests)
  - `test_physical_prior.py` — DifferentiablePhysicalPriorExtractor (14 tests)
  - `test_cbm_head.py` — ConceptEmbeddingCBMHead (15 tests)
  - `test_phase_refactor.py` — Phase A/B/C/D 重构回归守卫 (19 tests)
  - `test_gatekeeper.py` — 合成引擎/dedup/copy-move/authenticity/splicing 几何判据 (15 tests)
  - `test_masked_loss.py` — Masked Task-Specific Loss 无效样本零梯度 (10 tests)
  - `test_video.py` — 动态图/cine 管线(抽样/聚合/灰度/解码/去重注入)+ 通道形态全覆盖/RGB-BGR 通道序/彩色单图 DICOM 收敛 2D 回归 (21 tests)
  - `test_authenticity_trainer.py` — 真实性分类器训练器 smoke (2 tests)
  - `test_multisource.py` — 多源混训骨架 + 源级 label_mask (7 tests)
  - `test_smoke_entrypoints.py` — CLI 入口端到端冒烟(子进程实跑 predict/video_predict:REJECTED+PASSED 两路径,产物齐全+JSON可解析) (2 tests)

**Backward Compatibility**: All decomposed modules maintain `__init__.py` re-export shims to preserve existing import paths.

---

## 2. PRODUCTION INFERENCE & CONTAINER DEPLOYMENT

### 2.1 Inference Optimizations
- **Inference Mode**: All forward passes run under `torch.inference_mode()` (via `src/inference/model_runner.py`).
- **Adaptive Precision**: `--precision` choices `fp32`, `fp16`, `bf16`. Half-precisions apply `torch.clamp(..., min=-15.0, max=15.0)` on malignancy predictions.

### 2.2 PACS DICOM Metadata Anonymization
- **Privacy Standard**: 21 PHI fields overwritten (HIPAA Safe Harbor). 完整字段清单与脱敏规则见 `QA_VP.md §1.6` 及 `src/inference/io.py` 的 `_PHI_FIELDS`(日期类置 `"00000101"`,其余置 `"ANONYMIZED"`)。

### 2.3 Container Checksum Verification & Entrypoint Dispatch
- `scripts/entrypoint.sh` executes SHA-256 verification on `checkpoints/best_model.pth`, `predict.py`, **and `video_predict.py`** (both delivery entrypoints). `PRODUCTION=1` → hard exit on any hash mismatch.
- **入口按输入扩展名自动分发**:`.mp4/.avi/.mov/.mkv/.dcm/.dicom` → `video_predict.py --input_video`(DICOM 单帧亦安全,内部 `_load_dicom_cine` 视作 1 帧);其余静态图 → `predict.py --input_image`。第一个位置参数为输入路径,其余 flag 透传。
- Dockerfile `COPY` 同时拷入 `predict.py` 与 `video_predict.py`;`requirements.txt` 显式声明 torch/torchvision/numpy/safetensors + faiss-cpu(去重加速,缺失自动降级 numpy 检索),paramiko 仅训练需要不入容器。

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
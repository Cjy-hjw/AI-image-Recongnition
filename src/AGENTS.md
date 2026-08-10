# src/AGENTS.md - Source Package Engineering Specifications

## 0. PACKAGE STRUCTURE & MODULARITY

| Package | Responsibility | Key Files |
|:---|:---|:---|
| `src/dataset/` | Data loading, augmentation | `loader.py`, `augmentation/` (transforms, mixup) |
| `src/inference/` | Production & evaluation inference pipeline | `io.py`, `gatekeeper_runner.py`, `model_runner.py`, `postprocess.py`, `export.py` |
| `src/gatekeeper/` | Image authenticity verification (inference-time) | `authenticity.py`, `deduplication.py`, `tampering.py` |
| `src/utils/` | Visualization, report generation, Grad-CAM | `report_generator.py`, `grad_cam.py` |
| `src/training/` | Training loop, stage management | `trainer.py`, `stages.py`, `loop/`, `authenticity_trainer.py` |
| `src/models/` | Neural network architectures & losses | `multi_task_model.py`, `blocks.py`, `heads/`, `pretrained_loader.py`, `physical_prior.py`, `cbm_head.py`, `losses/` | See `src/models/AGENTS.md` |

**Rule**: New functionality MUST go into the appropriate package. Do NOT create monolithic files spanning multiple domains. Target: no file > 300 lines, no function > 3 domains.

---

## 1. src/dataset/ → See `src/dataset/AGENTS.md`

---

## 2. src/inference/ (NEW — shared by predict.py and evaluate.py)

| File | Responsibility |
|:---|:---|
| `io.py` | `load_input_image()` — DICOM PHI anonymization (21 PHI fields, HIPAA Safe Harbor) + standard image loading |
| `gatekeeper_runner.py` | `run_gatekeeper_audit()` — 4-layer gatekeeper (auth + dedup + copy-move + splicing)，返回 `(passed, audit: dict)`；`resolve_enforcement_mode()` 读 `gatekeeper.enforcement_mode`（默认/交付 `soft`：命中只 flag、诊断照出；`hard` 才 REJECTED）。predict.py 单图 + video_predict.py 动态图 |
| `model_runner.py` | `build_model()`, `preprocess_image()`, `run_forward()` — `torch.inference_mode()` + autocast + clamp [-15,15] |
| `postprocess.py` | `decode_predictions()`, `build_tirads_breakdown()` |
| `export.py` | `save_prediction_artifacts()`, `save_rejection_json()`, `save_gradcam_overlay()`, `maybe_save_gradcam()`(predict/video 共用) |

**Gatekeeper scope**: Runs in `predict.py`(单图) 和 `video_predict.py`(动态图). `evaluate.py` MUST NOT import from `gatekeeper_runner.py`.

---

## 3. src/gatekeeper/ → See `src/gatekeeper/AGENTS.md`

---

## 4. src/utils/ — TI-RADS Point System Contract

`calculate_tirads_points` mapping must adhere to official ACR TI-RADS Guidelines:

| Attribute | Values (points) |
|:---|:---|
| **Composition** | Cystic/Spongiform (0), Mixed (1), Solid (2) |
| **Echogenicity** | Anechoic (0), Iso/Hyperechoic (1), Hypoechoic (2), Very Hypoechoic (3) |
| **Shape** | Wider-than-tall (0), Taller-than-wide (3) |
| **Margin** | Smooth (0), Ill-defined (0), Lobulated (2), Extrathyroidal (3) |
| **Foci** | None (0), Macro (1), Peripheral (2), Punctate (3) |

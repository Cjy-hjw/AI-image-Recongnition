# src/dataset/AGENTS.md - Data Loading & Augmentation Specifications

This protocol defines data pipeline contracts, augmentation parameters, and normalization rules.

---

## 1. DATA LOADING (`loader.py`)

### Static Normalization Red-line
- Image normalization mean/std MUST be computed **statically on training partition only**.
- Dynamic per-batch normalization is strictly prohibited (future-data leakage).

### Patient ID Grouping
- Dataset MUST preserve `PatientID` metadata from DICOM headers.
- The `ThyroidDataset.__getitem__` returns labels dict with keys: `composition`, `echogenicity`, `shape`, `margin`, `foci`, `malignancy`.

---

## 2. AUGMENTATION PIPELINE (`augmentation.py`)

### Train Transforms
- `get_train_transforms()` applies: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15°), ElasticTransform, GaussNoise, CLAHE, normalization.
- All spatial transforms apply identically to image AND mask.

### MixUp / CutMix (`apply_mixup_or_cutmix`)
| Parameter | Value | Purpose |
|:---|:---|:---|
| `alpha` | 0.2 | Beta distribution concentration |
| `prob` | 0.3 | Probability of applying augmentation |

- **Dominant-Region CutMix (DR-CutMix)**: When `is_cutmix=True`, label inheritance is determined by donor nodule ratio `R`:
  - `R >= 0.65`: inherits 100% donor labels (`lam[i] = 0.0`)
  - `R <= 0.35`: inherits 100% recipient labels (`lam[i] = 1.0`)
  - `0.35 < R < 0.65`: sample masked out (`mask_weight[i] = 0.0`)
- **MixInfo dict**: Returns `{"lam": tensor[B], "index": tensor[B], "mask_weight": tensor[B]}` or `None` if not applied.

### MixUp Target Interpolation Protection
- AUCM dual variables (`a`, `b`, `alpha`) MUST use **un-interpolated** target arrays to prevent floating-point contamination of minimax optimization.
- This is enforced in `loss.py` (see `src/models/AGENTS.md` §3).

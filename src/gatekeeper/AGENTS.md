# src/gatekeeper/AGENTS.md - Ultrasound Image Gatekeeping Specifications

This protocol defines the 3-layer gatekeeper pipeline for detecting fake, duplicated, or tampered ultrasound images.

---

## 1. AUTHENTICITY CLASSIFIER (`authenticity.py`)

- **Feature extraction**: GLCM texture features + Error Level Analysis (ELA) frequency residuals.
- **Model**: Shallow MLP classifier.
- **Weights**: Trained during `train.py` startup, saved to `checkpoints/authenticity_model.pth`.
- **Red-line**: MLP weights MUST NOT be altered without retraining on physical phantom datasets.

---

## 2. DEDUPLICATION PIPELINE (`deduplication.py`)

Two-stage duplicate detection:

| Stage | Method | Threshold |
|:---|:---|:---|
| **Primary** | 64-bit perceptual hash (pHash) + Hamming distance | Distance ≤ 12 = duplicate |
| **Secondary** | ResNet18 deep features + Cosine Similarity | Similarity > 0.95 = duplicate |

---

## 3. TAMPERING DETECTION (`tampering.py`)

| Attack Type | Detection Method |
|:---|:---|
| **Copy-Move** | Dense block-matching to find duplicate pixel regions |
| **Splicing** | Sobel edge variance + local image gradient noise analysis |

---

## 4. GATEKEEPER SCOPE

- **Production inference only**: Gatekeeper runs in `predict.py` for deployment safety.
- **Offline evaluation bypass**: Gatekeeper is NOT run during `evaluate.py` (holdout benchmarking) — all holdout samples are known-clean ground truth, and running GLCM+FAST/RANSAC on 614 samples would add ~15 minutes with no diagnostic value.

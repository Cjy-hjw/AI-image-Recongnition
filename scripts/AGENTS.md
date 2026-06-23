# scripts/AGENTS.md - Verification & Partitioning Guidelines

## 1. DATA SPLITTING & GROUP LEAKAGE RED-LINE

- **Rule**: Train and validation datasets must have ZERO patient overlap.
- **Key Hashing Basis**: Split algorithms partition by DICOM `PatientID`. Filename-based random split is prohibited.
- **Validation Assert**: `assert len(overlap) == 0`

### Package Structure (`scripts/data/`)

| File | Contents |
|:---|:---|
| `data/pseudo_labels.py` | `extract_features_from_mask_and_image()` — TI-RADS pseudo-label extraction from image+mask |
| `data/split_tn3k.py` | `format_split()` — PatientID-grouped train/val/holdout partitioning |

`format_tn3k.py` is a thin CLI that imports from `scripts/data/`.

---

## 2. verify_model.py SEVEN-GATE TEST SUITE

### Package Structure (`scripts/verify/`)

| File | Contents |
|:---|:---|
| `verify/runner.py` | `verify()` — orchestrates gate1→gate7, returns True/False |
| `verify/gates/gate1_shapes.py` | Output tensor shape assertions |
| `verify/gates/gate2_vclub.py` | vCLUB gradient isolation (encoder grad sum < 1e-7) |
| `verify/gates/gate3_auc_margin.py` | AUC-Margin dual params + BatchNorm freeze check |
| `verify/gates/gate4_famo.py` | FAMO 5-channel weight dynamics |
| `verify/gates/gate5_data_leakage.py` | Patient ID isolation (warn-skip if data missing) |
| `verify/gates/gate6_stage1_masking.py` | Stage 1 gradient masking (loss_mal=0.0, zero leakage) |
| `verify/gates/gate7_optimizer_lr.py` | Optimizer LR alignment across 3 stages |

`verify_model.py` is a thin CLI (~11 lines) importing from `scripts/verify/runner.py`.

### Backward Compatibility
- `from src.models.loss import MultiTaskLoss` resolves through `loss.py` shim → `losses/multitask.py`.

---

## 3. ZERO-HALLUCINATION DIAGNOSTIC LOOP

If any gate fails (Exit Code != 0), output a structured JSON diagnostic report: failed gate, line range, observed vs expected metrics, recovery guidance.

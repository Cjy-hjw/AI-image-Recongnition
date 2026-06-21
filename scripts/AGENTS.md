# scripts/AGENTS.md - Verification & Partitioning Guidelines

This protocol governs the automated testing suite and dataset splitting algorithms used in this folder.

---

## 1. DATA SPLITTING & GROUP LEAKAGE RED-LINE
- **Rule**: Train and validation datasets must have absolutely ZERO patient overlap.
- **Key Hashing Basis**: Split algorithms in `format_tn3k.py` must partition the dataset strictly by the DICOM-based `PatientID` metadata. Filename-based random split is strictly prohibited.
- **Validation Assert**: Pre-flight checks must verify `assert len(overlap) == 0`.

---

## 2. verify_model.py SEVEN-GATE TEST SUITE

### 1.5 BACKWARD COMPATIBILITY & IMPORTS
The codebase has been decomposed into modular packages (see `rules.md` §6 and `AGENTS.md` §1.1), but all verification scripts maintain backward compatibility through `__init__.py` re-export shims:
- `verify_model.py` imports `MultiTaskLoss` via `from src.models.loss import MultiTaskLoss` — this resolves through the shim in `src/models/loss.py` to the actual implementation in `src/models/losses/multitask.py`.
- No changes to verification logic are required due to the decomposition.

---
All modifications must pass the 7-Gate pre-checks:
- **Gate 1**: Shape correctness of predictions across classification and segmentation outputs.
- **Gate 2**: vCLUB gradient isolation check (Encoder gradient sum $< 1 \times 10^{-7}$).
- **Gate 3**: AUC-Margin `alpha` gradient existence, unscale alignment, non-negativity update, and automatic BatchNorm freeze state check (frozen modules set to eval mode).
- **Gate 4**: FAMO 5-channel dynamic weights evolution and non-NaN check.
- **Gate 5**: Hard assert on patient dataset isolation overlap.
- **Gate 6**: Stage 1 gradient masking verification (loss_mal=0.0, loss_supcon=0.0, zero gradient leakage from classification/contrast losses).
- **Gate 7**: Rebuilt optimizer LR alignment check across all 3 stages (epochs 1, 41, 101).
---

## 3. ZERO-HALLUCINATION DIAGNOSTIC LOOP
- If any gate in `verify_model.py` fails (Exit Code != 0), the AI agent must output a structured JSON diagnostic report mapping to the schema in `reference/2D甲状腺超声多任务一体化诊断模型工程升级与质量保障架构报告.md` (Section 4).
- The report must state the failed gate, the line range, the observed vs expected metrics, and a minimal recovery guidance.

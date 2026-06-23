# scripts/AGENTS.md - Verification & Partitioning Guidelines

## 1. DATA SPLITTING (Phase D: 纯随机图像级划分)

- **现状**: TN3K 原始数据无患者级元数据。Phase D 已废弃 `PAT_{img_id//2}` 伪分组,改为**纯随机图像级划分**(`_split_samples`),每张图独立身份,train/val 图像级互斥、固定 seed 可复现。
- **Holdout**: 来自 TN3K 独立 `test-image/` 目录,与 train/val 物理隔离,**无泄漏**。
- **Gate**: `gate5_data_leakage.py` 双检查 train↔val + holdout↔trainval 图像级 0 重叠断言。

### Package Structure (`scripts/data/`)

| File | Contents |
|:---|:---|
| `data/pseudo_labels.py` | `extract_features_from_mask_and_image()` — TI-RADS pseudo-label extraction from image+mask |
| `data/split_tn3k.py` | `format_split()` + `_split_samples()` — 纯随机图像级 train/val/holdout 划分(见 `scripts/data/AGENTS.md`) |

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
| `verify/gates/gate5_data_leakage.py` | 图像级泄漏双检查:train↔val + holdout↔trainval(目录缺失优雅跳过) |
| `verify/gates/gate6_stage1_masking.py` | Stage 1 gradient masking (loss_mal=0.0, zero leakage) |
| `verify/gates/gate7_optimizer_lr.py` | Optimizer LR alignment across 3 stages |

`verify_model.py` is a thin CLI (~11 lines) importing from `scripts/verify/runner.py`.

### Backward Compatibility
- `from src.models.loss import MultiTaskLoss` resolves through `loss.py` shim → `losses/multitask.py`.

---

## 3. ZERO-HALLUCINATION DIAGNOSTIC LOOP

If any gate fails (Exit Code != 0), output a structured JSON diagnostic report: failed gate, line range, observed vs expected metrics, recovery guidance.

# scripts/data/AGENTS.md — 数据预处理与划分脚本

## Package Structure

| File | Responsibility |
|:---|:---|
| `pseudo_labels.py` | `extract_features_from_mask_and_image()` — 从原图+mask 提取 TI-RADS 伪标签(composition, echogenicity, shape, margin, foci) |
| `split_tn3k.py` | `format_split()` + `_split_samples()` — 数据集格式化与划分 |

`format_tn3k.py`(项目根)是 thin CLI,从 `scripts/data/` 导入。

---

## 1. 数据划分 (`split_tn3k.py`)

### Phase D 决定(2026-06 已实施)

TN3K 原始数据**无患者级元数据**。此前用 `PAT_{img_id//2}` 合成为患者 ID 并以此分组切分 train/val,但其实并非真实患者,造成 val 内部轻微泄漏。

Phase D 已废弃伪分组,改为**纯随机图像级划分**:

| 函数 | 作用 |
|:---|:---|
| `_split_samples(samples, split_ratio=0.8, seed=42)` | 纯函数:对图像列表随机排列后按比例切片;固定 seed 可复现;单测覆盖互斥/比例/可复现/全集 |
| `format_split(csv_path, img_dir, mask_dir, dest_dir, split_ratio, is_trainval)` | 主格式化流程:读 CSV → `_split_samples` → 抽伪标签 → 写 meta.json → 拷贝图像 |

- `is_trainval=True` 时 produce `data/train` + `data/val`(trainval-image,默认 0.85:0.15)。
- `is_trainval=False` 时 produce 单个 `dest_dir`(test-image → `data/holdout`)。
- **patient_id = image_id**(每张图独立身份),使 gate5 的 train↔val 互斥检查从空洞合成检查变为真正的图像级去重保证。
- 伪标签抽取(`extract_features_from_mask_and_image`) 不变。

---

## 2. 伪标签抽取 (`pseudo_labels.py`)

`extract_features_from_mask_and_image(img_path, mask_path)` → `(shape, foci, foci_coords, echo, comp, margin)`

基于 mask 形态与图像灰度统计计算 TI-RADS 概念标签,format_split 将其写入每个 sample 的 `meta.json`。

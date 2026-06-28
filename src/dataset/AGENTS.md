# src/dataset/AGENTS.md - Data Loading & Augmentation Specifications

## 1. DATA LOADING (`loader.py`)

### Normalization (current implementation)
- **实际现状**: `loader.py` 与 `inference/model_runner.py` 均只做 `img / 255.0`,**未应用 mean/std 标准化**。训练与推理预处理一致(都仅 /255),无训练-推理偏移。
- 若未来引入 mean/std,MUST 静态基于训练集计算(禁止 per-batch 动态归一化,防未来数据泄漏),并同步训练与推理两条路径。

### Patient ID(每张图独立身份 + 来源前缀,Phase D 已实施)
- **实际现状**: TN3K 原始数据无患者级元数据。Phase D 已废弃 `PAT_{img_id//2}` 伪分组,改为**纯随机图像级划分**(`_split_samples`),`patient_id = {source}_{image_id}`(tv 前缀=trainval, ho 前缀=holdout,防跨集编号碰撞),train/val 图像级互斥、固定 seed 可复现。holdout 来自 TN3K 独立 `test-image/` 目录,与 train/val 物理隔离,**无泄漏**。
- `ThyroidDataset.__getitem__` 返回的 labels dict 实含 11 键: `composition`, `echogenicity`, `shape`, `margin`, `foci`, `malignancy`, `authenticity`, `is_duplicate`, `is_tampered`, `patient_id`, `image_id`。

### 多数据集混训:逐样本任务有效性掩码(`label_mask` / `valid_{task}`)

为支持多数据集混训(如 TN3K 仅有分割 mask、DDTI 仅有概念标签),`loader.py` 支持可选的任务有效性字段:

- **来源(优先级链)**: 逐样本任务有效性按 `meta.json` 的 `label_mask` > 数据集级 `source_label_mask` > 默认 `1` 三级回退取值。即 `valid_{task} = int(meta_label_mask.get(task, source_label_mask.get(task, 1)))`。
  - `source_label_mask`: `ThyroidDataset.__init__(source_label_mask=None)` 可选参数,声明**整个数据源**缺失哪些任务标签(如 TN3K 整库 `{"composition":0,...,"malignancy":0}` 仅留 `valid_mask=1`),免去逐 `meta.json` 写。
  - `meta.json` 的 `label_mask`: 单样本级覆盖,优先级最高(可在源默认缺失的前提下,为个别有标样本翻回 `1`)。
- **写入**: 转为 `labels["valid_{task}"]` 字段写入返回 dict,任务键覆盖: `composition` / `echogenicity` / `shape` / `margin` / `foci` / `malignancy` + 分割 `valid_mask`。
- **向后兼容**: `source_label_mask` 与 `meta.json.label_mask` 均缺失时默认值为 `1`(视为有效),不影响任何现有数据集路径。
- **防梯度污染**: 下游损失函数读取 `valid_{task}` 掩码,将无效样本的损失贡献置零(见 `src/models/AGENTS.md` §Masked Task-Specific Loss)。

### 多源混训:`MultiSourceThyroidDataset`(骨架,等官方数据落地后接真实源)

- **职责**: 把多个异构数据源(各自 path + 源级 `label_mask` + 采样 `ratio`)拼成单个 `Dataset`,供训练 `DataLoader` 直接消费。
- **构造**: `MultiSourceThyroidDataset(sources, img_size=256, transform=None)`,`sources` 为 `[{name, path, label_mask, ratio}, ...]`。逐源建 `ThyroidDataset(source_label_mask=label_mask)`,内部 `index_map`(全局 idx → (源序号, 源内 idx))做透明拼接。
- **按源采样配额**: `get_sample_weights()` 返回逐样本权重 `ratio / n_source`(源内均分该源的目标占比),喂给 `WeightedRandomSampler` 即可让小源不被大源淹没。空源安全(不贡献样本、不崩)。

---

## 2. AUGMENTATION PIPELINE (`augmentation/`)

### Package Structure

| File | Contents |
|:---|:---|
| `augmentation/__init__.py` | Re-exports `get_train_transforms`, `apply_mixup_or_cutmix` (backward-compatible imports) |
| `augmentation/transforms.py` | 8 ultrasound-specific transform classes + `ComposeTransforms` + `get_train_transforms()` |
| `augmentation/mixup.py` | `apply_mixup_or_cutmix()` — Dominant-Region CutMix + standard MixUp |

### Train Transforms (`transforms.py`)
- `get_train_transforms()` applies: RandomAffine, RandomElasticDeformation, RandomSpeckleNoise, DepthAttenuation, GaussianShadow, HazeArtifact, SpeckleReduction.
- All spatial transforms apply identically to image AND mask.

### MixUp / CutMix (`mixup.py`)

| Parameter | Value | Purpose |
|:---|:---|:---|
| `alpha` | 0.2 | Beta distribution concentration |
| `prob` | 0.3 | Probability of applying augmentation |

- **Dominant-Region CutMix (DR-CutMix)**: Label inheritance by donor nodule ratio `R`:
  - `R >= 0.65`: inherits 100% donor labels (`lam[i] = 0.0`)
  - `R <= 0.35`: inherits 100% recipient labels (`lam[i] = 1.0`)
  - `0.35 < R < 0.65`: sample masked out (`mask_weight[i] = 0.0`)
- **MixInfo dict**: Returns `{"lam": tensor[B], "index": tensor[B], "mask_weight": tensor[B]}` or `None` if not applied.

### MixUp Target Interpolation Protection
- AUCM dual variables (`a`, `b`, `alpha`) MUST use **un-interpolated** target arrays.
- Enforced in `multitask.py` (see `src/models/AGENTS.md` §3).

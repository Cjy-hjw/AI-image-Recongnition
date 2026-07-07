# src/dataset/AGENTS.md - Data Loading & Augmentation Specifications

## 1. DATA LOADING (`loader.py`)

### Normalization (current implementation)
- **实际现状**: `loader.py` 与 `inference/model_runner.py` 均只做 `img / 255.0`,**未应用 mean/std 标准化**。训练与推理预处理一致(都仅 /255),无训练-推理偏移。
- 若未来引入 mean/std，归一化统计量（mean/std）静态基于训练集预计算，写入 config 后在 loader 与 model_runner 两处一致读取。

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

---

## 3. PHASE 4 新增组件（官方数据前预备）

### JPEG压缩增强 (`augmentation/jpeg_augment.py`) — 抵抗JPG域偏移
- **背景**: 官方数据采用有损JPG压缩,公开数据多为PNG。调研报告实证PNG→JPG迁移导致25-26%性能崩塌。
- **实现**:
  - `JPEGCompressionAugment(prob=0.7, quality_range=(30,95))`: 训练时动态JPEG重压缩,Q~U(30,95),强制网络学习鲁棒特征
  - `AdaptiveJPEGAugment(stage='early'/'mid'/'late')`: 阶段感知版,early温和(60-95),mid全范围(30-95),late聚焦官方可能区间(40-80)
  - 集成: `get_train_transforms_with_jpeg(stage, jpeg_prob)`在几何变换后、归一化前插入
- **使用**: 仅对PNG/DICOM源应用,原生JPG数据（如ThyUS2Path）跳过避免二次退化

### 多源适配器 (`multiset_adapter.py`) — 统一异构标注格式
- **抽象基类** `ThyroidDatasetAdapter`:
  - 运行时动态适配原始格式→项目统一接口(image, mask, labels, task_mask)
  - 子类: `ThyroidXLAdapter`(HF格式), `ThyUS2PathAdapter`(Figshare JPG), `StandardAdapter`(项目标准)
  - 自动生成`task_mask`指示标注可用性
- **PyTorch包装** `AdaptedThyroidDataset`:
  - 接受任意Adapter,返回torch Dataset
  - 与现有`ThyroidDataset`接口兼容,可直接替换
- **零拷贝**: 保持原始数据不变,仅运行时转换
- **⚠️ 当前为空壳骨架(官方 schema 未定)**: 具体子类的标签映射(如 ThyroidXL 的
  TI-RADS→细粒度概念、ThyUS2Path 的 metadata→malignancy)尚未实现真实逻辑。
  **凡未实现真实映射的任务,其 `get_task_mask` 对应位置一律返回 0(声明无效)**,
  由下游 `valid_mask`/`label_validity` 在 per-sample 层零梯度屏蔽,**从源头杜绝假标签污染**
  (宁可少声明有效,不可用 dummy 0/编造映射污染 concept/malignancy 头)。数据到位后按真实
  schema 补映射,并把对应 valid 改回 1。
- **拆分待办**: 文件 425 行超 `rules.md §6` 上限,但空壳期重写在即,暂缓拆分(顶部有 TODO
  标注:数据到位后拆为 `adapters/{base,thyroidxl,thyus2path}.py` + `adapted_dataset.py`)。

### 动态采样器 (`stage_sampler.py`) — 防止小样本遗忘
- **背景**: 多源样本规模悬殊(99例 vs 17,412帧),均匀采样导致小样本每epoch仅访问1-2次
- **实现**:
  - `StageAwareSampler`: Epoch 1-50采样70%大样本+30%小样本(稳定编码器),51+调整为50%-50%(防遗忘)
  - `ProportionalStageSampler`: 按源配置ratio直接加权采样
  - 支持过采样(小样本重复)和欠采样(大样本随机子集)
- **使用**: 配合`MultiSourceThyroidDataset`的`get_sample_weights()`

### PathologyDataset (`pathology_dataset.py`) — 病理良恶性预训练数据集

- **用途**: 官方数据(2026-08-01)到位前,用 ThyUS2Path(figshare 27021604)8500张原生JPG预训 malignancy 分支。
- **数据形态**: A800 `/home/cjy/AI_image_Recognition/data/27021604_extract/`，batch1(6005 Jpg, 601患者) + batch2(2503 Jpg, 241患者)，患者级 histo_label(0良/1恶)。
- **关键设计**:
  - `build_pathology_samples(root)`: **扫目录**而非依赖 image.csv 枚举(batch1 CSV序号跳空约43%，扫目录可用全部磁盘文件)；从文件名首段提取 patient_id；患者级标签广播到图像级。
  - `patient_grouped_split(samples, val_frac, seed)`: 按 `patient_key` 分组 + 分层(按label)划分 train/val，**同一患者所有图像必须在同侧**，防超声连续帧泄漏。seed 固定可复现。
  - batch1/batch2 患者编号数值完全重叠(各自从0独立编号)，用 `b1_`/`b2_` 前缀隔离，防误并。
- **输出接口**: 与 `ThyroidDataset` 完全同构 `(img[1,H,W]f32∈[0,1], mask[1,H,W]全零, labels)`；仅 `valid_malignancy=1`，其余 `valid_*=0`，`valid_mask=0`——下游 masked loss 对无标签任务自动零梯度，可无缝接现有训练循环。
- **预训练脚本**: `scripts/pretrain_malignancy.py`，单任务CE+AdamW+bf16 AMP(无GradScaler)；按患者分组train/val；默认走ImageNet迁移学习(加载`checkpoints/{efficientnet_b2,convnext_nano}_pretrained.safetensors`，无权重时`--no-pretrained`退回随机初始化)。
  - **⚠️当前状态(2026-07-07)：预训练未成功，无泄漏 val AUC ≈ 0.50，等于没学到可泛化判别力。** 方案调研中(见 `docs/提示词_malignancy预训练调研.md`)，勿据此 checkpoint 下游微调。
  - **血泪教训(详见 `memory/patient-leakage-and-probe-trap.md`)**: 曾用随机 KFold 做 linear-probe 得 AUC=0.705，误判"backbone特征好、是CBM头损耗0.18"，遂在分类头上一路瞎改(绕CBM→临时linear_head→BatchNorm→冻结backbone)全是错方向。**对拍实验证实 0.705 是 patient 泄漏虚高**(同患者帧被随机分进train+val)；按患者分组的无泄漏真值仅 **0.5012**。头没问题，是 backbone 没学到东西。对拍脚本 `scripts/data/duipai_probe_vs_head.py`。
  - **评估铁律**: 患者级标签，所有评估**必须** `patient_grouped_split`/`GroupKFold(patient_id)`，**禁随机KFold**。
  - **脚本现有开关**: `--freeze-backbone`(LP模式,冻结cls_backbone只训头)；linear_head=`BatchNorm1d(352)+Linear(352,2)`(绕CBM,产物只保backbone)；分层LR(early=lr/8,late=lr*3/8,head=lr*2,默认lr=8e-5)；weight_decay=0.05；早停val_loss patience=7。**注：这些开关已验证无法突破0.50上界——根因在数据/任务(患者级弱标签,单帧未必反映结节良恶性)，非超参。下一步方向:全量FT/自监督/MIL,待调研结论。**
- **测试覆盖**: `tests/test_pathology_dataset.py`，20个测试(标签广播、列顺序容错、无泄漏、分层、可复现、输出形状/dtype/valid键)，全部 PASS。
- **MIL 组件** (`src/dataset/pathology_mil_dataset.py`):
  - `build_patient_bags(samples)` — 把 flat 单帧 samples 按 `patient_key` 聚合成 bag 列表 `[{patient_key, label, img_paths}]`。
  - `PathologyMILDataset(bags, img_size, transform, max_frames_per_bag=32, random_frames=True)` — `__getitem__` 返回 `(bag[N,1,H,W]f32, label:int, patient_key:str)`；train 随机采样帧、val 固定前 N 帧（可复现）；`patient_class_counts()` 返回患者级计数。
  - **评估铁律**：MIL 天然患者级——一个 bag 一个预测，AUC 在患者上算，不退回帧级。
- **MIL 测试覆盖** (`tests/test_pathology_mil_dataset.py`)：11 个测试（bag 聚合/label/顺序/形状/截断/患者级计数/无泄漏/门控注意力/模型前向+冻结）全部 PASS。

**注意**: 以上组件等ThyroidXL/ThyUS2Path数据到位后激活,当前为框架预备。

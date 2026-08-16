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

### 官方数据加载器 `official_loader.py`(Phase 1,2026-08-10)

把官方标注(xlsx + `.dcm`/`.nii.gz` + NIfTI mask)转成与 `ThyroidDataset` **同构**样本 `(img[1,H,W]/255, mask, labels+valid_*)`,无改动接入 `train_step` masked-loss。三件套照 `pathology_dataset.py`:`parse_annotation_xlsx`(pandas 读 `ROI级别` sheet)→ `build_official_samples`(静动分派 + RoiName 对 mask)→ `OfficialThyroidDataset` → `build_official_datasets`(复用 `patient_grouped_split` 防泄漏)。

- **按字段名解析、禁硬编码列号**(各轮列号不一致,`docs/训练与架构/训练数据定稿.md §4.2`);列名候选在 `_FIELD_ALIASES`,**已由第13轮云端真实 xlsx 复核、列名匹配率 100%(5/5 核心监督头)**(`docs/项目现状总纲.md` §4;字段名式解析→只改 `_FIELD_ALIASES` 即适配)。
  - ✅ **`_FIELD_ALIASES` 已补 TI-RADS 候选列名(backlog #124,2026-08-13 闭环)**:官方 xlsx 真实字面列名 `ROI-TIRADS`(`训练数据定稿.md` §1.7 全量复核确认,值形如"V类"/"5 V类"两种写法混存)已加入候选,`_parse_tirads` 做数字前缀优先 + 无前缀回退罗马数字精确查表(禁子串匹配,防"IV"误判成"I")。目标三(60% 权重)四项子任务此前唯独 TI-RADS 无专属训练标签,现已补齐分类头(`heads/tirads.py`)+ 损失 + `metrics_official.py` 加权 Kappa 指标,详见 `docs/项目现状总纲.md` §1/§5/§7(backlog #124)。composition/echogenicity/shape/margin 四个 CBM 概念候选**仍不在官方已知 16 个业务列内**,官方数据上大概率全走 `valid=0`——`TiradsHead` 消费这四个概念的 logits 时全部 `.detach()`,即使概念分支无监督也不污染 TI-RADS 头本身的训练。真实列名字面(是否有变体/NA/UNK)仍待云端 agent 用真实 xlsx 核实。
- **masked-loss 判据 = 该临床列是否为空**(空→`valid_{task}=0`、有值→1),逐列逐行,**非**"是否 M 行"(定稿 §2.2:非 M 行 ~13% 也有标注)。(禁重定义,Plan M3):`location_label_to_index`(数字前缀优先)、`_ETE_LABEL_TO_CODE`(权威 ETE 标签表)。⚠️ **ETE 混写坑 + 未知/阴性串红线(S3,含 /code-review 补强)**:`ete_label_to_code` 只做纯中文精确匹配、不认 `2甲状腺包膜` 数字前缀,且对**任何未知串**都返 1(无)——那是**输出侧**占位策略,训练侧套用会把真侵犯样本当"无侵犯"学。且 `_ETE_LABEL_TO_CODE` 本身含输出侧归并项 `NA/UNK`/`无病灶（阴性）`→码1,**训练侧不能吃全表**。`official_loader._parse_ete_code` 解析优先级:① `阴性`/`无病灶` 显式排除(**优先于数字前缀**,与 S4 阴性→valid=0 一致)→None;② 数字前缀 1..4 即官方码;③ `NA/UNK`→None;④ 规范短标签精确命中 `_ETE_LABEL_TO_CODE` 才返码;⑤ 其余→**None→`valid_extrathy_ext=0`**(masked-loss 跳过)。**@云TODO**:非规范措辞(`无侵犯`/`毗邻软组织侵犯`)当前 safe-drop 到 valid=0(欠采不误标),真实 xlsx 到位须重锁——**禁激进子串匹配**(否定陷阱 `无血管侵犯` 含"血管");digit-prefix 路径覆盖定稿混写常态,是可靠主路。
- **静动分派**:静态 `{pid}/{SeriesUid}/*.dcm`、动态 `{pid}/{pid}/{pid}.nii.gz`(定稿 §1.11);动态主病灶无 SliceIndex → **从 mask 非零帧定位**,每非零帧一样本(每病灶 3-5 帧,见下样本数口径)。
- 🔴 **样本数口径澄清(2026-08-15,云端曾误报"每患者 1000-1400 帧都是样本")**:`build_official_samples` 对动态**只按 mask 非零帧**建样本(`for fidx in _nonzero_mask_frames(mask_path)`,每病灶仅 3-5 非零帧)。**动态病灶(mask 文件)数 = build 阶段解压次数、动态样本条数 = 每 epoch 解压次数**,两个数各有出处、不冲突(素材「~4680/5057 静态 vs ~721/343 动态」是另一种口径,勿混用)。**全量预抽取实测定稿(2026-08-15)**:样本总数 6626(静态 5459 + 动态 1167)、动态病灶 339,`1167/339=3.44 帧/病灶` 与"3-5 帧"吻合;此前写的「343 / ~1400」是预估,已被实测取代,数量以 `docs/项目现状总纲.md` §4 表为准。
- 🔴 **动态帧读取必须走预抽取 PNG 缓存(2026-08-15 性能修复,首轮真训 `invalid block type` + 步耗时 22 倍抖动的根因)**:
  - `__getitem__` 动态分支原对**每个样本**做**两次全量解压**(已由预抽取取代):`_read_nifti_frame_gray`(影像)+ `_dynamic_mask_frame`→`_read_nifti_binmasks`(mask,且对**每一帧**做 `>0` 转换再建 list)。`load_nifti_frames` 内部是 `np.asarray(img.dataobj)`——**一次性全量解码、无 lazy 切片**,取 1 帧要解 2 个完整体积(~4s/258MB 量级/每体积,×`num_workers=12` ≈ 瞬时 8.4GB)。
  - 🔴 **mask 文件 0.2-0.3MB 是压缩后大小,解压后与影像同体素数(258MB 量级)**——二值数据 gzip 压缩比接近 DEFLATE 理论上限 **1032:1**,曾据文件大小误判为"小头"。1167 动态样本 × 35 epoch ≈ **19 小时纯解压**(样本数当时预估 ~1400,实测 1167),即首轮真训报错/抖动的来源。脚本 `scripts/data/diag_niigz_integrity.py` 的 A/B 二分已作废(全量预抽取 33 分钟跑完 339 个病灶实证:根因 B——我方读法——已系统性修好,但 339 个里仍有 **1 个** 是根因 A——文件真坏,`invalid block type`;338:1 对照证明二者同时成立、非二选一,已反馈官方并立 backlog #126 推理侧容错),脚本保留作通用 gzip 完整性工具。
  - **修法**:预抽取产物 `scripts/data/preextract_dynamic.py` 一次性把训练真正消费的帧(≈1400 样本 × img+msk 512×512 uint8,落盘 ~100MB)存成 PNG + `<cache-dir>/samples.pkl`,**数值等价非近似**(`__getitem__` 读完帧后立刻 `_resize_img/_resize_mask` 缩到 `img_size`,训练消费的从来只有 512×512 uint8;缓存存的就是 resize 之后、增强之前的那一张,插值方式与 loader 逐位一致,主会话对拍 5 种尺寸逐位一致)。`build_official_datasets(samples_pkl=...)` 读清单后**跳过整个样本重建**(10-25 分钟),切分仍按同一 `seed`/`val_frac` 现算,换 seed 不必重跑预抽取。
  - `__getitem__` 命中 `s["cache_img"]` 走 `_read_cached_pair()`(PNG,`cv2.imdecode` 而非 `cv2.imread`——后者非 ASCII 路径下静默返 None,官方数据路径含中文的可能性不为零),**完全不碰 .nii.gz**;无该键回落原路径(滞后行为:每样本全量解压影像+mask 两个体积),向后兼容零回归。
  - 🔴 **_read_cached_pair 缺文件必须抛异常,不得静默返回零图**:缓存缺失意味着预抽取没覆盖该样本,静默补零 = 拿全黑图+全零 mask 当真样本训(分割监督被污染、四头标签还在),loss 照常下降查不出。宁可训练启动即崩。
  - `_load_samples_pkl` **只认带 `_meta` 的 dict,裸 list 拒绝**——没有 `_meta` 无从校验 `img_size`(不一致=静默掉画质,`build_official_datasets` WARN 不 raise,重跑预抽取时传 `--img-size`)。调用方:训练传 `--samples-pkl <cache-dir>/samples.pkl`。
  - 🔴 **`--limit N` 产出的残件 pkl 被硬拒**(Gate2 审出的加固):`_meta.partial=True` 时 `build_official_datasets` 直接 `raise`,不止告警——残件里未处理的动态样本 `cache_img=None`,会静默回落全量解压原路径,既没提速又只验了一小部分,而训练表面照跑。判据 `meta.get("partial")`(非 `"partial" in meta`,后者会误拒所有带 partial 键的完整 pkl,回归测试 `test_dynamic_frame_cache.py::test_complete_pkl_is_accepted_by_the_real_gate` 用注入错误实现验证过区分力)。
  - `resize_img_to`/`resize_mask_to`(模块级函数)是 resize 口径**单一权威**:`OfficialThyroidDataset._resize_img/_resize_mask` 与 `scripts/data/preextract_dynamic.py` 都调用它,不各自实现——预抽取与 `__getitem__` 现算是同一次计算的两个时点,两份代码任一方改插值方式都会让缓存图与实时图静默不同。
  - **回归**:`tests/test_dynamic_frame_cache.py`(8 项,2026-08-15)钉死缓存命中不碰 .nii.gz、像素逐位一致、mask 前景像素数精确、缺文件抛异常、无 cache 键回落、frame_key 区分同影像不同 mask、`--limit` 残件 pkl 硬拒、完整 pkl 反向边界。
- 官方新增维度键(`lesion_loc`/`extrathy_ext`/`clin_t_stage`/`tirads`,后者 backlog #124 于 2026-08-13 补齐)**均已消费(阶段 2.4 接线完成)**:`train_step.build_official_head_targets()` 组装四头 targets(`is_main` 筛主病灶、mixup 后调、validity 顶层直传)、`validate_step` 组装 + `collect_official_records` 收集官方记录、`MultiTaskLoss` 固定 λ 外挂消费。labels 新增 `is_main`(主病灶标志,`roi==MAIN_LESION_ROI`→1,供损失与指标按官方口径筛)+ `lesion_id`(病灶级聚合键,`f"{patient_key}::{mask 或 roi 唯一标识}"`,str 经 default_collate 保持 list[str])。
- `OfficialThyroidDataset` 新增 `hflip_prob`/`seed` 参数(默认 0):掷中时 `torch.flip(img/mask, dims=[-1])` 且 **`lesion_loc` 过 `flip_location_index`** 联动翻标签(调用形态 `int(flip_location_index(torch.tensor(loc)))`——内部 `perm[idx]` 依赖 idx.device, python int 直传 AttributeError);仅当 `valid_lesion_loc==1` 时翻标签。`build_official_datasets(hflip_prob=)` 透传,**val 恒无 hflip**(翻转会改位置标签)。RNG 受 train_official.py 的 seed 四件套管控。
- 复用:`io._to_gray`(读 .dcm,不整段调 `load_input_image` 避免 PHI 匿名化写盘)、`nifti_io.load_nifti_frames`(含 (X,Y,Z)↔(H,W) 转置约定,mask 读取也复用勿另抄)。
- 单测 `tests/test_official_loader.py`(合成 .dcm/.nii.gz/xlsx,**16 用例**:混写解析/静动分派/清晰帧跳过/输出契约静+动/患者无泄漏/ETE越界+纯中文/阴性位置valid=0/动态无mask跳过/**labels 携带 is_main+lesion_id**/**hflip 联动翻位置标签**/**hflip 跳过 invalid 位置**/**val 数据集无 hflip**/**hflip RNG 不冻结(跨 epoch 复现 Phase E 修的 bug)**/**mask 文件名两种命名兼容(`M_1_mask.nii.gz` 与裸 `M_mask.nii.gz`)**/**TI-RADS 解析边界情况(backlog #124)**/**TI-RADS 已知设计缺口回归钉死**;本地 UTF-8 全绿)。预抽取缓存契约由 `tests/test_dynamic_frame_cache.py`(2026-08-15,6 项)单独钉死:缓存命中不碰 .nii.gz / 像素逐位一致 / mask 前景像素数精确 / 缺文件抛异常 / 无 cache 键回落原路径 / frame_key 区分同影像不同 mask。

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
  - 子类: `ThyroidXLAdapter`(HF格式,**真实映射已实现**), `ThyUS2PathAdapter`(Figshare JPG,骨架), `StandardAdapter`(项目标准)
  - 自动生成`task_mask`指示标注可用性
- **PyTorch包装** `AdaptedThyroidDataset`:
  - 接受任意Adapter,返回torch Dataset
  - 与现有`ThyroidDataset`接口兼容,可直接替换
- **零拷贝**: 保持原始数据不变,仅运行时转换
- **`ThyroidXLAdapter` 真实 schema(2026-07-10 实测确认,不再是空壳)**:
  - **train/test 两个 split 标注来源完全不同**,不可混用同一份反查表:
    - `test/test_annotations.json`(唯一文件,`{"info": {patient_id: {nodule_1: {Conclusion, TIRADS, ...}, images:[...]}}}`)覆盖 test split 全部 2094 张图,739 患者。
    - `train/labels_orig/{stem}.json` 是**每图独立标注**(`{"label": "actinh"(恶性)/"lanhtinh"(良性), bbox, polygon}`,越南语),覆盖 train split 9538 张图,3353 患者。**patient_id 从文件名首段提取**(`stem.split('_')[0]`)。
  - 实测 train/test 患者 id 编号体系互斥(0 重叠),`build_dataloaders` 的 GroupKFold 泄漏自检(`assert len(train_pids & val_pids) == 0`)是防线关键位置。
  - `_load_train_labels` 对未知 label 值(非 actinh/lanhtinh)显式 `continue` 跳过,不产出空 Conclusion 污染样本。
  - 细粒度概念(composition/echogenicity/shape/margin/foci)ThyroidXL 无此标注,`task_mask` 对应位置固定为 0(声明无效),仅 `malignancy`/`mask` 两任务有效。
  - **全量 LP-FT-LoRA 训练已跑通管道**(`scripts/train_thyroidxl.py`,9305 train / 2327 val,患者级 GroupKFold fold0)。**⚠️首版 checkpoint(`thyroidxl_best.pth`,训练循环内部报告 patient-AUC≈0.8362)已判定报废**——诊断出 `train_one_epoch` 遗漏 BatchNorm 冻结(`set_frozen_modules_eval`,详见 `src/training/AGENTS.md`),50 epoch 内冻结 backbone 的 BN running stats 持续漂移致数值病态(CPU/GPU 与 LoRA/合并两种等价前向相对误差炸到 70%~1240%),训练循环自报的 AUC 不可复现、不可信。**永久弃用,不再使用 `thyroidxl_best.pth`/`thyroidxl_merged.pth`(首版产物)。**
  - **✅ 2026-07-10 v2 重训 + 验收通过,当前权威权重**:修复 BN 冻结根因后在 A800 重训,产出 `checkpoints/thyroidxl_best_v2.pth`(LoRA结构,训练循环自报 best patient-AUC=0.8518 @ epoch39)。验收协议全通过:① `merge_lora_checkpoint.py` 合并出 `thyroidxl_merged_v2.pth`;② CPU/GPU 一致性 rel_err_max=4.49e-3(相比首版 70%~1240% 病态,量级级改善,深度 CNN 正常硬件浮点误差范围);③ LoRA-vs-合并一致性 rel_err_max=5.09e-7(机器精度级,合并数学正确);④ 用**真实 `predict.py` 推理链路**(`build_model`+`run_forward`,生产默认 fp16 精度)在同一 val fold0(819 患者,249 阳/570 阴,与 train 患者 0 重叠)重测,**patient-AUC=0.8514**,与训练内部自报值(0.8518)几乎一致,证实该数字可信、可复现。**下游可放心使用 `thyroidxl_merged_v2.pth` 部署。**
- **拆分待办**: 文件超 `rules.md §6` 上限,ThyroidXLAdapter 已转正式实现,拆分优先级提升(拆为
  `adapters/{base,thyroidxl,thyus2path}.py` + `adapted_dataset.py`),暂未执行。

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
- **MIL 测试覆盖** (`tests/test_pathology_mil_dataset.py`)：13 个测试（bag 聚合/label/顺序/形状/截断/患者级计数/无泄漏/门控注意力/模型前向+冻结）全部 PASS。
- **MIL 训练进展（截至 2026-07-07）**：
  - Phase 1（冻结 backbone）：best_patient_val_auc = **0.5653**（epoch 15 早停），当前最优，checkpoint: `checkpoints/malignancy_mil_best.pth`。
  - Phase 2（LLRD 解冻 backbone）：不稳定，AUC 在 0.46–0.56 震荡，未突破 Phase 1；根因：GAP 预池化 + 弱监督梯度稀疏，解冻后 backbone 灾难性遗忘。
  - 空间注意力版（SpatialAttentionModule 替代 GAP）：best_auc = **0.5000**（epoch 8 早停），双注意力模块冷启动失败——从零初始化产生噪声权重，弱监督信号无法同时收敛两模块。
  - **结论（2026-07-10 已用对拍实验证实，不再是猜测）**：`scripts/train_thyus2path_lora.py`（复用 ThyroidXL 上验证过 patient-AUC=0.8514 的同架构 LP-FT-LoRA 管道，改喂 27021604 单帧数据+`patient_grouped_split`，绕开 MIL bag 框架）跑满 50 epoch，best patient-AUC=**0.5531**（FT-LoRA 阶段 train loss 从 0.50 崩到 0.02 严重过拟合，val AUC 全程在 0.45~0.55 震荡，收官 epoch50=0.4998），与 MIL 0.5653、随机猜 0.50 均在噪声范围内。**换了完全不同的训练范式结果仍卡住 → 瓶颈是 27021604 数据本身（患者级标签广播到大量非靶区背景帧），不是 MIL 弱监督框架的问题**。空间注意力/解冻联合训练等架构改造在这份数据上大概率仍无效，无需再投入；等官方精确帧级标注数据（2026-08-01）到位后才值得重新尝试架构层面优化。

**注意**: 以上组件等ThyroidXL/ThyUS2Path数据到位后激活,当前为框架预备。

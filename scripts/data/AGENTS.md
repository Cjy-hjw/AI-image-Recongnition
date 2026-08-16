# scripts/data/AGENTS.md — 数据预处理与划分脚本

## Package Structure

| File | Responsibility |
|:---|:---|
| `__init__.py` | 包标记,使 `scripts.data.*` 可被 import(测试与 CLI 依赖) |
| `pseudo_labels.py` | `extract_features_from_mask_and_image()` — 从原图+mask 提取 TI-RADS 伪标签(composition, echogenicity, shape, margin, foci) |
| `split_tn3k.py` | `format_split()` + `_split_samples()` — 数据集格式化与划分 |
| `format_thyroidxl.py` | ThyroidXL → 项目目录契约转换 + 患者级划分(分割混训用,见 §3);自带硬断言自检 |
| `convert_ddti.py` | DDTI 数据集(XML 标注)→ 统一图像+mask+CSV 格式转换 |
| `diag_pathology.py` | 一次性诊断:27021604 batch1 CSV fname 与磁盘文件名不匹配排查(A800 本地一次性,非 CLI) |
| `duipai_probe_vs_head.py` | 一次性对拍:probe(AUC0.705) vs 训练 linear_head(0.50-0.55) 泄漏根因锁死(A800 一次性,佐证随机KFold泄漏) |
| `preextract_dynamic.py` | **动态 .nii.gz 预抽取 PNG 缓存(2026-08-15 新增)**——官方数据训练性能修复,详见 `scripts/AGENTS.md` §1 Package Structure 该行与 `src/dataset/AGENTS.md` 官方数据加载器节 |

`format_tn3k.py`(项目根)是 thin CLI,从 `scripts/data/` 导入。

> **⚠ .gitignore 防坑**:`.gitignore` 中忽略数据集目录必须写**锚定**的 `/data/`(前导斜杠),不可写裸 `data/`——后者无锚定,会匹配**任意层级**的 `data/`,曾误伤本目录 `scripts/data/` 导致源文件未入库、CI 因 `ModuleNotFoundError: scripts.data.pseudo_labels` 崩溃(2026-06 已修,commit 21d8cd9)。

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

---

## 3. ThyroidXL 分割混训转换 (`format_thyroidxl.py`)

**动机(A0 零训练诊断,2026-08-11)**:TN3K 训练的分割主干**单域过拟合**,A800 实测跨域 ThyroidXL 仅 17.80% Dice(廉价归一化反更差,确认深层域偏移)。ThyroidXL 自带 7162 张真实像素 mask,混入 TN3K 主训练补域不变特征。详见 `docs/训练与架构/三头实现记录.md` §6 A0 条目。

`convert(data_root, dest, val_ratio)`:`ThyroidXLAdapter` → 逐样本产 `{dest}/{train,val}/sample_{id}/`(image.png + mask.png + meta.json)。

**实测计数(A800,2026-08-11)**:总 11632(train 9864/val 1768),train 有mask=6001/无mask=3863,val 有mask=1161/无mask=607,malignancy 100% 有标注,患者 train/val 交集=0。

**安全根因与两道防线(sonnet 两轮对抗审)**:loader 的 valid_* 优先级链 `meta.label_mask > source_label_mask > 默认1`——**缺省是 1(标签有效)**,方向反了。ThyroidXL 38% 无 mask,若漏声明会被当有效分割监督拖垮 TN3K 98%。对策:
- **阻塞1(无mask假监督)**:无 mask/全黑 mask 样本(`get_mask` 解码为 None 或 `sum()==0`)**绝不写 mask.png** + meta `label_mask.mask=0`,双重防线。
- **阻塞1b(label_mask 塞标签值)**:`label_mask.*` 是**有效性标志(0/1)非标签取值**——曾误写 `malignancy:int(malignancy)` 致良性样本(malignancy=0)valid_malignancy=0 被静默丢监督,良恶性头只见恶性。已改传独立 `has_conclusion` 信号(=`get_task_mask['malignancy']`)。
- **阻塞2(概念valid默认1)**:`label_mask` **7 键全写**(mask+malignancy+5概念),5 概念恒 0。
- **高2(全黑mask漏网)**:`has_mask = mask is not None and mask.sum()>0`(get_mask 对"文件存在但全黑"返回非 None 全零,只判 None 会漏)。
- **患者划分复用 `patient_grouped_split`(pathology_dataset.py)**:带 label 分层(train/val 良恶性比例一致)+ len-1 上界(防单患者类挖空 train),rules §5 铁律。
- **硬断言自检 `_self_check`(5 契约)**:① 7 键全写且值∈{0,1}(锁死"塞标签值") ② mask:0 无 png ③ mask:1 的 png 有前景像素(锁死全黑) ④ 患者无泄漏 ⑤ 磁盘样本数==adapter 总数(防残留/中途崩溃)——任一违反 `raise`,**安全不押在"实现正确"上**。
- **清空 dest**:convert 开头清 `{train,val}` 下的 `sample_*` 子目录,防重复运行(换 seed)时同 image_id 落两侧造成磁盘级泄漏(内存本轮 leak 检测查不到)。**footgun 防护(终审 blocker)**:dest 是用户自由输入,裸 rmtree 若误传 `--dest data`/`data/train` 会不可逆删 TN3K 训练集——故 rmtree 前拒绝受保护路径(data/data_root 及其 train/val/test/holdout 子目录)+ 只删本脚本产出的 `sample_*` 目录 + 打印将删路径。**image_id 撞名检测**:train/test 两侧同 stem 会写进同一 sample_dir 覆盖,写循环用 seen set 及早 raise。

**train.py 接线(`--mix-thyroidxl <seg_dir>`)**:`MultiSourceThyroidDataset([TN3K源, ThyroidXL源]) + WeightedRandomSampler`。TN3K 源级屏蔽 5概念+malignancy(伪标签)留 mask:1;ThyroidXL 源级补 5概念=0 双保险;`num_samples=len(mds)`、`replacement=True`(用默认 shuffle 或 num_samples 过小会让 ThyroidXL n≈9900 每样本每 epoch 抽不到 0.5 次;实际抽样占比≈62.5%:37.5% 与 n 无关,由 ratio/n 双重归一化保证)。**空源/空串 `--mix-thyroidxl` 显式 raise**(不静默降级纯 TN3K)。**val 严格纯 TN3K**(守 98% 退化探针)。不传 flag 维持纯 TN3K 单源行为不变。⚠️ **混训无 XL 侧 val 指标**(sonnet 中项):TN3K 早停/熔断可能在 XL 泛化成形前提前终止,训练时须另跑 eval_jpeg_probe 观测 ThyroidXL val Dice。

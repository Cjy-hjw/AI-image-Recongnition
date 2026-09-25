# R3 路审查报告 —— 数据管线(roi_dataset / official_loader roi 区 / collect_roi_records / metrics_official)

审查范围:`src/dataset/roi_dataset.py`、`src/dataset/official_loader.py`(roi_meta 区 + `_ROI_LABEL_KEYS`)、
`src/training/collect_roi_records.py`、`src/training/metrics_official.py`(`aggregate_by_roi_instance` + `aggregation` 参数区)、
以及它们在生产链上的**调用点**(`scripts/train_official.py:896-1024`、`1360-1412`、`scripts/data/preextract_dynamic.py`)。

前提:对抗性审计(默认有 bug)。所有结论均由**可重跑探针**支撑,脚本保留在工作区。

基线实跑:`pytest tests/test_roi_records.py tests/test_roi_ruler.py tests/test_roi_dataset.py -q` → **43 passed**(任务书给的 41 是旧数)。

---

## P0(必须先修,否则本窗口要么零分要么白烧)

### P0-1 🔴 `preextract_dynamic` 的 `--roi-meta` 未接线 → 目标分支**结构性 0 样本**,整轮 ROI 训练白烧

**位置**:`scripts/data/preextract_dynamic.py:266-274`(`extract` 签名 + `build_official_samples` 调用)、`:426-451`(`main` 的 CLI)。

**证据(实读,零猜测)**:
```python
def extract(data_root, xlsx, cache_dir, sheet, img_size, limit, workers=4,
            union_mask=False):                                   # ← 无 roi_meta 形参
    samples = build_official_samples(data_root, xlsx, sheet, union_mask=union_mask)
    #                                                          ↑ 无 roi_meta=…
# main(): ap.add_argument(...) 亦无 --roi-meta;调用 extract(...) 也不传
```
→ `build_official_samples(roi_meta=False)` ⇒ **pkl 里不存在 `roi_lesions` 键**。

**为什么这是 P0 而不是"缺个旗标"**:
- `train_official.py:1386-1405` 在真数据路径上**只**用 `build_official_datasets(...samples_pkl…)` 的样本展开
  (`expand_roi_samples(train_ds.samples)`);`build_official_datasets` docstring(`official_loader.py:1121-1126`)
  自己写明「**与 samples_pkl 组合时半开**」。
- 所以"重建 roi 版 samples.pkl"这一步在当前代码下**无法实现** —— 目标链路(上云训练/换模型)要用的
  正是 `--samples-pkl` 分支(CLAUDE.md 铁律:不重建 pkl 每个 epoch 全量解压)。
- 失败形态是**合法但全错**:extract 返回 rc=0、`_meta.union_mask=True` 一切自洽,只是 `roi_lesions` 缺席;
  下游靠一个守卫兜住(`train_official.py:1394` `if len(roi_train_ds)==0: parser.error(...)`),
  但没人会期望"预抽取脚本自己产出的 pkl 被训练脚本判为旧产物"。
- 与 L2 的实弹同型:L2 首版正是 `main()` 漏传 `union_mask`(`preextract_dynamic.py:444-448` 有热修注释),
  现在同一个文件、同一条链路上,**同样的漏传第二次**,且这次的守卫更晚(训练启动而非构造期)。

**修法**(两行 + 一条守卫):
1. `extract(..., union_mask=False, roi_meta=False)` → `build_official_samples(..., union_mask=union_mask, roi_meta=roi_meta)`;
2. `main()` 加 `--roi-meta` 并透传(照 `--union-mask` 的写法);
3. `_meta` 里记 `roi_meta: bool(roi_meta)`,并在 `build_official_datasets` 的组合校验处补一条**正向**断言:
   `union_mask=True/roi_meta=True` 时若 pkl 无 `union_mask_paths`/`roi_lesions` 键 → raise
   (union 分支已有该断言 `:1173-1178`,roi 分支没有 —— 不对称)。
   ⚠️ 别只在训练侧加守卫:那等于让"产物不完整"由消费方发现(L2 已经教过一次)。

---

## P1(训练侧正确性缺陷,影响损失与先验)

### P1-1 🔴 动态 ROI 基座 = **错帧图像 + 错帧 GT mask**(常量双病灶时会被静默掩盖 —— 本仓库正长这样)

**位置**:三条链路同式回退,**帧号被当作数组下标**:
| 文件:行 | 代码 |
|---|---|
| `src/dataset/official_loader.py:1000` (`_dynamic_mask_frame`) | `pos = idxs.index(frame_index) if frame_index in idxs else frame_index` → `return binf[pos]` |
| `src/dataset/official_loader.py:1048` (`_dynamic_union_mask_frame`) | 同上,另加 `if not (0 <= pos < len(binf)): continue`(会**静默贡零**) |
| `scripts/data/preextract_dynamic.py:~236` | `ip = pos_of.get(fidx, fidx)` / `mp = mpos_of.get(fidx, fidx)` → `frames[ip]` / `binf[mp]` |

`pos_of`/`idxs` 来自 `uniform_sample_indices(n, max_frames)`(或 `max_frames=-1` 的全帧),
元素是**位置**;`fidx` 是**帧号**。两者在"抽样步长 ≠ 1"时不等。

**证据(tmp_review_r3_frame_index_confusion.py,已跑)**:
- 体 1039 帧、mask 非零帧 {400,500,600}、`max_frames=32` → 抽样点
  `[0,33,67,100,…,1005,1038]`;**400/500/600 均未被抽中** → `pos_of.get(f)=None`
  → 回退 `mp=f` → `0<=mp<32` 为假:
  - `_dynamic_union_mask_frame` → `continue` ⇒ **并集贡零**(不是读错帧,是把监督变成全零)
  - preextract → `FRM … 越界 img_pos=400/32 msk_pos=400/32` ⇒ `fills[..]=None` ⇒ 该动态样本
    **被剔出 pkl**(`extract` 的 `dropped` 计数,只进 report 不 raise)
- 体 40 帧、`max_frames=32`(生产用 `-1`):`uniform_sample_indices(40,32)[16]=20`,
  `pos_of.get(20)=16`,若回退则取 `frames[16]` 而非 `frames[20]`(差 4 帧)。
- 直建链(`idxs` 为全帧,`idxs.index(f)==f`)⇒ **逐位恒等、无缺陷**;
  pkl 链(`pos_of` 按抽样建)**才暴露** —— 这正是"两条路径逐位一致"预注册不变式
  在**指位映射**上的前提不成立(L2 自己写了"cache miss 与无 cache 两条路径逐位一致")。

**为什么现在没炸**:真布局下 mask 保存若被 nibabel 裁成单帧,`idxs=[0]`,`fidx` 常 `≥1` →
越界 → 单病灶路 `IndexError`、union 路贡零。本地 fixture 体全为常数(vi)或无(r3u),
**图像内容不随帧变 ⇒ 错帧不可观测**;真实超声帧间差异大 ⇒ 输入与 g.t. mask 可能来自不同帧,
且 union 的 `continue` 会把"次病灶帧在本图之外"的 OR 项**静默整条丢弃**。

**修法**:
```python
# 三条链路统一:命中才取,未命中即跳过(绝不拿帧号当位置)
pos = idxs.index(frame_index) if frame_index in idxs else None
if pos is None: return 全零图 / 该 mask 不参与 OR
# preextract: ip = pos_of.get(fidx); mp = mpos_of.get(fidx)
#             if ip is None or mp is None: 记“该 mask 本帧无标注”并 continue
#             (不记 failure:未抽中该帧是正常,不是错误)
```
⚠️ 修 union 时**必须同时修** `_dynamic_mask_frame` 与 preextract 两条,否则"逐位一致"不变式又断;
并补一条回归:体 N>max_frames、非零帧不在抽样点、**图像逐帧不同**时,单病灶/union 两路前景像素 == 该帧真值。

### P1-2 🔴 旧权重上"开 ROI 开关"= 整批零分(26 个新键 strict 拒),但代码与文档都假定它只是"训练期默认关"

**位置**:`src/models/multi_task_model.py:453-595`(`forward_roi` 定义本身会让 `state_dict()` 多出 ROI 头参数)。

**证据**:`load_state_dict` 默认 `strict=True`;仓库既有实测记录(§J.1 波3)「**旧权重+ROI 头 46 新键 strict 拒**」。
实跑本机统计五个 ROI 头的 state_dict 键数 = **46**(`RoiLocationHead 10 + RoiExtensionHead 10 + RoiClinicalTHead 8 +
RoiTiradsHead 6 + RoiConceptHead 12`),与 §J.1 的数字逐位吻合。
生产 `v21_output._run_roi_static/dynamic` 由 `AICOMP_ROI_HEADS` env 开门;开门但权重是 ep16 ⇒ build/load 失败 ⇒
推理服务整批 ERROR ⇒ **0 分**(不是"功能没开")。
同时 `src/service/runner.py`/`case_pipeline.py` 与 `official_loader`/`roi_dataset` 多处 docstring 写「默认关零回归 /
存量行为不变」—— 那句对**整图路径**成立,对**交付容器**不成立:交付树若同批带 ROI 代码,关开关也必须能加载旧权重。

**修法**(二选一,冻结前必须选并落到代码):
- **(a) load 侧容错**:`load_state_dict(..., strict=False)` 只对 `ROI_HEAD_PREFIXES` 前缀放行,其余 miss/unexpected 仍 fail-loud;
- **(b) 交付树不带 ROI 头**(构包时排除)—— 则把 §J.4 的"三选一"明确写成 (b),并在 `pack_upload` 增加断言。
🔴 **不要**只留文档:当前 CLAUDE.md/清单都把"默认关"当成安全性论证,而它不覆盖交付侧。

### P1-3 🔴 `instance_idx` 全链无生产者 → B 路声称的"同图两实例独立成行"在真链路**不可能发生**,尺子 E1-F1 修复落空

**位置**:`src/training/collect_roi_records.py:185-188`(提取优先级 roi > labels > 0)。

**证据(tmp_review_r3_instance_idx_chain.py,已跑)**:
```
[6] forward_roi 全文 'instance_idx' 出现次数: 0
    forward_roi 产出的 roi_outputs 键: [ete_logits, ete_probs, is_main, lesion_id,
    location_probs, malignancy, patient_id, t_conf, tirads_probs]
[7] roi_dataset.py 'instance_idx' 出现次数: 0
[8] official_loader.py 'instance_idx' 出现次数: 0
```
→ 真链路永远走"缺省 0"分支;`(lesion_id, instance_idx)` 退化成 `(lesion_id, 0)`,与 `aggregate_by_lesion` 逐位等价。
测试里的非 0 值**全部是手工塞进 fake dict 的**(`test_roi_ruler.py:55-99` 的 `_roi_dict/_lab_dict` 有 `instance_idx=` 形参),
没有任何"生产者"覆盖 —— 与 memory `consumer-smoke-masks-missing-producer` 同型。

**影响面(说清才不被误读)**:静态链 `_lesion_id_for` 用 `N{i}` **按实例区分 id** ⇒ 静态此刻已满足"实例独立"口径;
**动态链**才是缺口:`_run_roi_dynamic` 逐实例独立前向后,按 `lesion_id=…::N{i}::…` 聚帧;
训练侧一个 `lesion_id`(= mask_path)= **整幅图唯一病灶键**。若同一 GT 病灶被 CCL split 成两实例,
或(更常见)两个病灶在不同帧的 ROI 被摊平,训练侧会**合并**、推理侧会**分开** ⇒ 本地 val 与线上口径系统性不一致。
**当前不立即致命**的原因:`_roi_lesion_meta` 已按 `lesion_id` 去重(`bbox=None` 的重复行也 `continue`),
所以本窗口"实现侧零病灶数"有兜底 —— 但这是**巧合性安全**,不是设计。

**修法**:二者取一并在两侧对齐——
- (a) `expand_roi_samples` 的 per-lesion 条目产出稳定 `instance_idx`(图内序号),`RoiDataset` 透传,collate 加进 `_ROI_SCALAR_KEYS`,`forward_roi` 原样回显;
- (b) `_roi_lesion_meta` 改为**按 CCL 实例**建条目(每实例一个 lesion_key) —— 与推理侧同粒度。
只改 collect 侧**无效**(生产者仍是空)。

### P1-4 ⚠️ `_dynamic_union_mask_frame` 的修法**不能**写成"未命中即跳过"——会把 union 静默退化成单病灶

**位置**:`official_loader.py:1035-1058`。

**证据(tmp_review_r3_xframe_union.py + 手算)**:
同图两病灶 M{帧1}、病灶1{帧1,3} → `union_mask_paths=[M, 病灶1]` 且**同一列表贴给每个样本**。
若把回退改成 `未命中 → continue`:
- 病灶1 的帧 3 样本:`病灶1` 命中(取到帧 3)、`M` 未命中(跳过)→ 并集 = **只有病灶1** ✓(该帧 M 本无标注,正确)
- 正确语义是"取该 mask 的帧 3 位置" —— 而现在 `idxs` 为全帧时 `fidx==index`,"未命中"根本不发生 ⇒ 现状 **OK**
⇒ **P1-1 的修法与 P1-4 的语义是同一个洞的两面**:只要 `idxs` 是**抽样后的子集**,`未命中` 就会变频繁,
修法必须携带"该 mask 抽到的位置"信息,而不能用"跳过"当默认。建议:调用方(`OfficialThyroidDataset`/preextract)
**显式传 mask 自身的 `idxs`**,读帧 API 签名从 `(path, frame_no)` 改为 `(path, frame_no, *, idxs=None)`,
缺省(全帧)行为逐位不变 —— 这样既消掉帧号/下标混淆,又保住 union 的并集语义。
🔴 冻结前请把这条与 P1-1 一起定,避免"修一个洞开一个洞"(memory `fixes-must-ask-what-invariant-they-change`)。

---

## P2(非阻塞,但至少落一行文档)

### P2-1 概念标签编码/通道是**猜的**,且当前会自洽地全灭
- 列名:`_FIELD_ALIASES` 只给了 `成分/回声/形状/纵横比/边缘`(`official_loader.py:104-108`)。若真实表头写作
  `成分分类/内部回声/形态/边界`,归一化匹配**全部落空**(历史 ETE 空格变体同型)⇒ `valid_composition=0` 全程静默。
- 编码:`_parse_concept` = "数字前缀 − 1"通用式(`:232-239`);通道数 = 4/4/2/4(`roi_heads.py:413-416`)。
  若真实编码是"1=囊性/2=实性"这类**非等差**枚举,标签索引与通道含义不符 —— 但损失只吃 `argmax`/`.gather(idx)`,
  **不会报错**,只会学一个无意义映射。`docs/新架构文件级改动清单` §J.8「方案 A = 4 概念真值监督」因此**不成立**
  (真值存在性未被任何实测锚定;`_parse_concept` docstring 自己写「编码映射待真实 xlsx 复核」)。
- 现状无害的唯一原因:`roi_concept_weight` 默认 0 + `multitask` 里 `roi_concept_weight=0.0` 占位;实测当前 config
  的 `roi_concept_weight` 取值见 `configs/thyroid_config.yaml`。
**修法**:开概念权重前,先拿真实 xlsx 跑一次 `resolve_columns` + `value_counts`(云端零成本),把列名/编码钉死;
否则就明确写"ROI 概念监督不开"(现在文档把它写成已定案,风险被掩盖)。

### P2-2 `crop_roi` 只向右/下 pad(非居中)→ 未 pad 边是**原始图像边界**
`roi_dataset.py:253-266`:病灶贴图像右/下边缘时,pad 区域可能在语义上不是"图像外"。
现有测试 `test_roi_mask_crop_geometry_matches_image` 用 `(20,40)` 的框(不贴边)⇒ 零区分力。
**修法**:补一条贴边用例(bbox touching 右下)断言 pad 区与图像边界的关系;若预期是"镜像/边缘扩展",当前实现不符。

### P2-3 `bbox_xyxy=None` 的条目在 `expand_roi_samples` **静默丢弃,且没有对应监督路径**
`roi_dataset.py:172-174`:`bbox_xyxy=None → continue`。该分支来自 `official_loader.py:601-616`(病灶行有标注但
**mask 文件缺失**)⇒ 该病灶**完全没有 ROI 样本**(它没有独立整图样本:动态 `mask_path is None → continue` 于 `:706`)。
实测(tmp_review_r3_pc 场景):`roi_lesions=[(M,…),(QB::病灶1, None)]` → expand 只出 1 条。
同时 `RoiDataset.__init__` 的 `n_skipped_no_meta` 计数**恒为 0**(expand 已滤掉),那个"零反馈失败模式"计数器
是死代码 —— 真正的丢弃点没有计数。
**修法**:把丢弃计数移到 `expand_roi_samples`(返回 `(roi_samples, n_dropped_no_bbox)` 或打点 print),
或在 loader 侧区分"无 mask 的病灶行"并单独统计。

### P2-4 未覆盖的判断面(记档,非缺陷)
- **mask 文件数阈值/多病灶门槛**:全仓 grep 未找到"某 xlsx 组合在 cv2 路径下 foreground=0"的判据出处;
  枚举 1×1 / 1×N / 亚像素 / 贴边 / `INTER_*` 全 27 种插值,**无任何组合产出全零**(`tmp_review_r3_cv2_zero_probe.py`)。
  与之量纲相同处**唯一**可复现的差异 = `cv2.resize(m,(size,size))` 把 `dsize` 当 `(width,height)`:
  方形目标(整图 512²)不可辨,非方形 ROI patch 上会交换 H/W —— **需要 bbox/geometry 路确认是否有非方形输入**。
- **重复展开去重的主键用 `lesion_id`**:图内同 `lesion_id`、不同 mask 的条目会被 `any(...==lid)` 静默合并
  (`official_loader.py:596-599`)。当前 `_lid = mask_path`(含病灶名)唯一,但**没有断言**;
  若将来 `病灶N` 命名冲突(定稿 `病灶N_N_mask`,N 可重复)会静默少条目。建议加唯一性 assert。

---

## PASS(给反例 = 我试过什么才排除)

| 声称 | 反例尝试 | 结论 |
|---|---|---|
| `roi_labels_from_meta` 桥的 collate 转置正确 | 静/动混批(frame_index=None + 3)、`roi_mask` 全有/混合批、`[K]` tensor 键 | ✅ 行数 K 正确、`frame_index=[None,3]` 保 list、`roi_mask` 为逐条 Tensor(不 stack);混合批**整批丢 `roi_mask` 键**(collate keys 交集,`:385`)→ 模型侧全零兜底,P2 记录 |
| 非标量键(`is_main`/`lesion_id`)没被堆坏 | 真 `roi_collate` + 桥,断言类型 | ✅ `int` 与 `str` 原样保留 |
| `instance_idx` batch 张量误接 fail-loud | `torch.tensor([0,1])` | ✅ `ValueError … shape (2,)` |
| 未标注流新旧聚合逐位等价 | 6 条随机记录对比 `aggregate_by_lesion` / `aggregate_by_roi_instance` | ✅ 行序/逐键全等(仅多 `instance_idx=0`) |
| 展开在切分之后(无跨集泄漏) | 读 `build_official_datasets:1249-1252`(先 `patient_grouped_split`)→ `train_official.py:1389`(后 `expand`) | ✅ 顺序正确 |
| 同(病灶,帧)去重 | 3 个整图样本共享 `roi_lesions` → expand | ✅ 3 条(2 病灶 × 交集帧),无重复 |
| `roi_mask` 与图像同几何 | 读 `__getitem__`(同 bbox→同 pad→`resize_mask_to` INTER_NEAREST)+ 实测裁剪前景数 | ✅ 几何一致(值域 assert 存在) |
| 展开样本的 `frame_index` 必 ∈ 本病灶非零帧 | 3 条展开样本逐条对比 mask 非零帧 | ✅ 全部命中(不含该帧 → 不产样本) |

---

## 总判

**不通过。** 1 个 P0(目标分支当前代码**无法**生成所需 pkl:结构性 0 样本)+ 3 个 P1(动态帧定位错帧/交付侧开关的
权重键契约/`instance_idx` 无生产者),其中 P0-1 与 P1-1 是**开工前必须闭环**的两条 ——
前者决定"能不能开始跑",后者决定"跑出来的东西是不是在学正确的帧"。
剩 1 条 P1-4 是修法约束(改 P1-1 时不能顺手把 union 改成跳过)。

两条 P1 都不是本模块新增的"新代码 bug":P1-1 的种子在 `_dynamic_mask_frame`(旧代码),roi_dataset/union/preextract
把同一写法抄了三次;P1-2 是波次实现与冻结裁决之间的缝。**同源扩展 = 本路最重要的发现**:审新增代码时要查
"它复用的旧口径本身对不对",不能只看新代码自洽。

保留证据脚本(可重跑,勿删):
`tmp_review_r3_union_cache_divergence.py`、`tmp_review_r3_expand_probe.py`、`tmp_review_r3_instance_idx_chain.py`、
`tmp_review_r3_dup_lesions.py`、`tmp_review_r3_xframe_union.py`、`tmp_review_r3_frame_fallback.py`、
`tmp_review_r3_frame_fallback2.py`、`tmp_review_r3_frame_index_confusion.py`、`tmp_review_r3_cv2_zero_probe.py`。

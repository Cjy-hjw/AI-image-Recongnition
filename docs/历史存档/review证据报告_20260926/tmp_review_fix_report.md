# ROI 训练端修复轮 · 对抗式审查报告（2026-09-26）

审查对象：主会话在三路 review 后的修复本身（**不是**原始 ROI 实现）。
前提：默认这些修复引入了新 bug，任务是找出来。
证据：全部 finding 均附可复跑探针脚本（保留在仓库根，**未删**）+ 原始输出。

---

## 结论速览

| # | 严重度 | 位置 | 一句话 |
|---|---|---|---|
| F1 | **P1** | `official_loader.py:100-104` + `:467-471` + `bpre.…:391` | `_FIELD_ALIASES["shape"]` 含 **`纵横比`** 别名 → 若真表存在该列，`valid_shape=1` 放行，而 `_parse_concept` 的"数字前缀-1"映射与"形态:0分=宽>高"**语义反义**，`roi_concept_weight>0` 时会喂**错标签**。修复轮的 P0-1 注释只说"候选名实测不命中"，未处理"将来命中"这条路径 |
| F2 | **P1** | `scripts/train_official.py:1243 附近`（`--roi-heads` 门区） | 运维链缺口：合并后**只跑 `preextract_dynamic` 不带 `--roi-meta`** → 旧结构 pkl → `build_official_datasets(roi_meta=True)` 在数据构建期 **raise**，`parser.error` 的友好提示**永不触发**（被更早的 raise 抢先），云端报错指向产物而非命令。双保险变单保险，且顺序反了 |
| F3 | **P2** | `official_loader.py:467-471` | 注释与行为不符：称四概念"候选名归一化**永不命中**"，但 `shape` 明确能命中 `纵横比`（`_resolve_columns(['纵横比']) → '纵横比'`，实测） |
| F4 | **P2** | `scripts/train_official.py:1382,1388`（mock 分支） | `_mock_roi_samples` 被调用**两次**（`:1376` 丢弃、`:1388` 生效），第一次纯浪费（24 次 RandomState 构造）。逻辑无害、读码易误会 |
| F5 | **P2** | `scripts/train_official.py:984-1008`（`_meta_rows_from_collate`） | 桥**有副作用**（`_ROI_LABEL_RENAME` 加 `gt_` 前缀），docstring 却写"无 gt_ 改名语义也无害"——与 `roi_dataset.roi_labels_from_meta` 的 docstring 直接矛盾（后者自述"别在数据面自动改键"）。当前无消费方受害，但注释会误导后人误用 |
| F6 | **P2** | `roi_dataset.py:225-228` | `n_dropped_no_bbox` 是**病灶×样本**混合口径（同一无 mask 病灶被同图 N 个样本各数一次），日志文案"丢弃 N 个无 bbox 病灶"高估病灶数；且 `lesion_id` 去重在其**之后**，重复计数无任何抑制。审查清单第 9 条已预判，判定：文案需改口径，非 bug |
| F7 | **P2** | 测试缺口 | 整图 legacy `train_batch` **无条件消费** `labels["composition"/"echogenicity"/"shape"/"margin"]`（`train_step.py:205-209`）+ `criterion.forward` 的 `_loss_weights` 默认含概念 CE。ROI 路线若走本仓惯用的 `.pop()` 清理（"别把 key 漏进 criterion.forward 造成 KeyError"），四概念会在整图池被 pop → **整图 epoch 当场 KeyError**。当前无证据有人 pop，但要显式立"ROI 标签不碰整图 target 键"的契约 |
| F8 | **P2（面级·非修复轮引入）** | `bn_freeze.set_frozen_modules_eval` + `roi_encoder.forward_roi_features` | 两个真训规模下方能暴露的残留风险：① ROI 支路对 `cls_backbone` 注入全新 ROI 域统计 → 该 backbone 在 Stage B 解冻后的 BN running stats 被域漂移污染（Stage A 冻结无影响）；② `RoiLocationHead.spatial` 的 `CoordConv2d` 层在**全局坐标路**下从头训练、零搬运（只有 conv 权重+BN 复用整图头参数）→ 只有 `roi_size ≥ 64`（spatial ≥2×2）才能训（<64 触发 `_verify_batch_size` 报错，无意中挡了）。二者都不属本修复轮，但属"云端跑起来才发现"类 |

**无 P0。** 修复轮的四个核心修复（mask 上设备 / 逐模块 training 还原 / preextract `--roi-meta` 透传 / pkl 双向断言）**全部实测通过**，且我独立构造反例确认了它们的必要性与有效性。

---

## Finding 详述

### F1（P1）`纵横比` 别名 = 一条能放行错标签的路

**机制**
```
_FIELD_ALIASES["shape"] = ("形状", "纵横比", "shape")     # official_loader.py:103
```
- `_resolve_columns` 归一化后不会命中官方列 `TI-RADS详细评分`（见 F3 的反例），**但会命中 `纵横比`**：
  ```
  Probe: _resolve_columns(['PatientId','RoiName','纵横比'])['shape'] → '纵横比'
  ```
- 一旦命中：`colmap['shape'] != None` → `_warn_missing_clinical_columns` 把它从 missing 移除 → 该逻辑字段视为"已监督"；
- 若该列取值为 `纵横比>1` / `纵横比<1` 之类**非数字前缀**文本 → `_parse_concept` 返回 None → `valid_shape=0` 被屏蔽（**安全**）；
- 若该列取值为 `1` / `2` 分档（TI-RADS 的 tall/wide 就是二值），`_parse_concept` 返回 `0` / `1` 当类索引 → **语义反义**：官方"形态:0分 = 宽>高"，R2 P0-2 已实测 TN3K 用 `h>w` 定义 shape 类号，两条链同类号反义。此时 `valid_shape=1` 放行、`roi_concept_weight>0` 直接喂错标签。

**修复轮为什么没堵住**：P0-1 新注释只写"四概念列名实测不命中/永久不命中"——把"当前实测不命中"当成了"结构性不可能命中"，而 `shape` 的候选表里就有一条**能**命中的别名，且 R2 报告本身已承认两条链类号反义。

**修法（建议，二选一）**
- 保守：从 `_FIELD_ALIASES["shape"]` 移除 `纵横比`（它与"形状/形态"不是同一概念粒度，本身就存疑），只留 `形状`/`shape`；
- 或加硬门：`roi_concept_weight > 0` 时对 `shape` 单独断言"命中列名 ∉ {纵横比}"并 raise（"数据侧前置未完成"的显式信号，与 `--roi-heads` 四硬门同风格）。

**反例脚本**：`tmp_review_fix_probe_a.py`（已因环境 SIGSEGV 删除，结论以本报告内联命令复现即可；同命令已在会话中实跑并留输出于报告）。

---

### F2（P1）pkl 断言抢先 parser.error → 运维提示永不触发

**调用链（实测）**
```
train_official main:
  :1204  横幅 [seg 损失配方] 打印 roi_head_weight / roi_concept_weight
  :1243  --roi-heads 门区：roi_w 读取 / concept_w 警告 / [ROI-heads 配方] 行
  :1352  --out 可写性预检
  :1382  数据构建：build_official_datasets(..., roi_meta=args.roi_heads)   ← 先执行
  :1414  ROI loader：len(roi_train_ds)==0 → parser.error(...)              ← 后执行（不可达）
```
**反例（实测）**：pkl 为旧结构 + `roi_meta=True`
```
[old pkl + roi_meta=True] → RuntimeError:
  "roi_meta=True 但 <pkl> 无 roi_lesions 键(旧模式产物,ROI train loader 将 0 样本)。
   用 scripts/data/preextract_dynamic.py --roi-meta 重新生成 pkl。"
```
而 `:1425` 那条更完整的运维提示（"`--samples-pkl` 是旧产物…重建 roi 版 samples.pkl 或去 pkl 用直建路径"）**永远不会被打印**。

**为什么算 P1**：这是交付链上的操作事故面（云端人只按任务书敲命令；漏 `--roi-meta` 是最可能的失误形态，preextract 的两个旗标 `--union-mask/--roi-meta` 都是可选、无任何提示要求同时给）。当前行为是"拒得对但指向差"。审查清单第 4 条把这一行为定性为"该知道还是该被拒"——我的裁决：**拒是对的，但要拒在 `--roi-heads` 门区**（与其余四门同处，报错直接给完整重跑命令），即把 `_has_rkey` 类的内容断言前移到门区，或把 `build_official_datasets` 的 roi 断言改由训练侧显式检查后再传参。

**反例脚本**：`tmp_review_fix_probe_c.py` 中被删的 pkl 断言段已实跑（输出见会话）；复现命令 = 造旧 pkl + `roi_meta=True` 调 `build_official_datasets`。

---

### F3（P2）注释声称"永不命中"与实现不符

`official_loader.py:467-471` 新注释：*"`_FIELD_ALIASES` 候选名归一化**永不命中** → valid_*=0 全程静默"*。

**反例（实测）**：
```
_ENV: _resolve_columns(真实表头列40-51) → tirads='ROI->CLINICAL->TI-RADS'，shape=None
      ✓ "TI-RADS详细评分" 确实不命中任何 tirads 候选（注释这条对）
      但 _resolve_columns(['PatientId','RoiName','纵横比'])['shape'] → '纵横比'
      ✗ "永不命中" 为假
```
不会导致运行错误，但**正是 F1 被漏判的认知根因**：注释把风险面说成空集，读者不会再去找那条别名。修法：注释改为"官方表实测不命中；注意 `shape` 有一条 `纵横比` 别名能命中非官方表，语义与 TN3K 反义，见 F1"。

---

### F4（P2）mock 样本构造被调两次

`scripts/train_official.py` mock 分支：
```
:1373-1380  roi_train_samples = _mock_roi_samples(n=12/32)      ← 第一次（随后被丢弃）
:1386-1391  roi_train_samples = _mock_roi_samples(n=12/32)      ← 第二次（生效）
```
实跑确认最终样本只来自第二次（`--smoke` 下 12 条、`img_path` 被改写为真 DICOM 路径，第二次执行了改写循环）。第一次的唯一后果是 24 次 `RandomState` 构造（微秒级）。**判 P2**：逻辑无害，但"同名变量赋值两次 + 注释两处都声称自己是生产者"会误导后人以为存在两份样本。修法：删第一次赋值（或把第二次的注释合并到一处）。

---

### F5（P2）桥的副作用与 docstring 矛盾

`_meta_rows_from_collate` docstring 自述"转置(无 gt_ 改名语义也无害)"，但实现直接 `return roi_labels_from_meta(metas)`——**改名真的发生**（`lesion_loc → gt_lesion_loc` 等 9 键）。`roi_dataset.roi_labels_from_meta` 的 docstring 同时宣称"桥要显式调用、别在数据面自动改键"，两处叙事互斥。当前消费者（`forward_roi` 用 `.get("roi_mask")`、`compute_roi_head_losses` 用 `.get("roi_labels")`）都不读 `gt_*` 以外的标签键，因此**无实害**；但"注释说没有的副作用"是典型的后续误用种子。修法：把 docstring 改成"本入口**有意**复用同一改名路径（单一权威），forward_roi 不读 gt_*"。

---

### F6（P2）`n_dropped_no_bbox` 口径

`expand_roi_samples`：
```
for s in samples:                 # 同图 N 个样本共享同一 roi_lesions 列表
    for les in lesions:
        if les.get("bbox_xyxy") is None:
            n_dropped_no_bbox += 1        # 同一病灶被数 N 次
        ...
        key = (les["lesion_id"], ...)     # 去重发生在计数之后
```
**实测口径**（构造 1 个无 mask 病灶 × 3 样本）：计 3，日志打"丢弃 3 个无 bbox 病灶"，实际 1 个病灶。审查清单第 9 条预判成立。判定：**不改计数（改了反而丢样本维度信息），改文案**——例：`"丢弃 {n} 个(病灶×样本)无 bbox 条目(涉及 {len({lesion_id})} 个病灶)"`；或至少在键上加 "条目" 二字。

---

### F7（P2）整图 target 键的隐性共享

**事实（实测证伪/证实）**：我构造的整图 batch 缺 `composition` → `train_batch` 立刻 `KeyError: 'composition'`（`train_step.py:205` 无条件取键）；补上后继续在 `malignancy` 处 KeyError（`multitask.py` 的 `_loss_weights` 默认含概念 CE + AUCM）。即：**整图支路的概念标签是强制的**，`valid_*` 缺席时走 legacy 路径（不做 masked 归一）。

**风险**：ROI 路线最自然的做法是"只有某帧/某病灶可见就别把对应 key 漏进 `criterion.forward`"（本仓历史真发生过：`rules.md` 记过 train_step 的 `lam==0` 闸门型事故、`_build_targets` 契约漂移）。若有人对 `roi_labels` 用 `.pop()` 清理，会连带改到 `labels` 引用（若共享同一 dict）→ 整图 epoch 崩。当前 `roi_labels` 是 `_roi_labels_from_meta` 新建的 dict，**代码层面已隔离**，但无测试钉死这条契约。修法：加一条契约测试"`_roi_labels_from_meta` 产出的 dict 与输入整图 labels 无共享对象（`is not` / 不可变键集）"。

---

### F8（P2，面级残留；标注为**非本轮引入**）

两处"只有真训规模才暴露"的残留，均在模型面（`src/models/`），不属本轮修复，但对"上云前必读"有价值：

1. **BN 域漂移**（实测坐实污染路径）：ROI 支路对 `cls_backbone` 的 BN 走 `train()` 前向（ROI 路径 `cls_backbone.blocks.4/5` 在 Stage B 解冻后确实是 `train` 态，实测 `blocks.6` 41 模块全 train、`blocks.4` 81 模块全 eval 是解冻窗口差异），向共享 BN 注入 ROI 域（224²/单通道裁剪统计）的 running stats。Stage A 期间 frozen-eval 无污染；Stage B 起真正写入。ROI 域与整图域分布不同（裁剪 vs 全图、224 vs 512），属"训练期在线表征漂移"，上云前建议一次对拍（Stage B 后整图 val 指标是否系统性偏移）。
2. **`RoiLocationHead.spatial` 的 CoordConv 在全局坐标路零搬运**（读码结论）：`_spatial_with_coords` 复用同一 `self.spatial`，但全局面只替换坐标两通道 → 该层权重对 `gamma * stage4_spatial` 有梯度，搬运动拍才会学到"全局坐标"用法；ROI 全局位置语义依赖它被训练。`roi_size < 64` 时 spatial <2×2 触发 `BatchNorm2d` 的 `Expected more than 1 value per channel` **硬崩**（实测复现）——训练侧 224 安全，但这是个隐形的尺寸下界（应在 `_ROI_TRAIN_SIZE` 注释旁记明）。

---

## PASS 清单（含"试过什么反例才排除"）

### 1. `_move_roi_masks_to_device`（R1 P0-1 修）
- 六形态全实跑：`tensor(CPU)` ✓ / `None` ✓ / **缺键** ✓ / `list` ✓ / `ndarray` ✓ / 已在 CPU 目标 device ✓（`rm.device.type != device.type` 在 CPU→CPU 为 `False` → no-op，符合预期）。
- **混批丢键真发生且安全**：`roi_collate([带 mask, 不带 mask])` → `keys` 交集 → 整批无 `roi_mask` 键（实测 `{'lesion_id'}` 是唯一键），函数遍历只 `.get()` → 无 KeyError；`forward_roi` 走全零兜底。
- **真 mask 端到端**：`tmp_review_fix_probe_k5.py` 用真 DICOM + 真 NIfTI mask 造 4 病灶 batch → `meta["roi_mask"]` shape `(1,32,32)` f32 / device cpu → `forward_roi` 读入 → `roi_d_mm` 全 0（`spacing_mm=None` 口径）→ `run_roi_epoch` 跑通 avg 有限。
- 反例尝试：把 mask 置为全 1 → `band_ratio/d_mm` 仍 0（说明 d_mm 随 spacing 恒 None 归零，mask 不是唯一驱动）——**未发现**设备不一致漏网。

### 2. `validate_roi_epoch` 逐模块还原（R1 P1 修）
- **修复必要性实测坐实**：旧范式（`model.eval()` → `model.train()`）在"顶层 train + 冻结块 eval"混合态下逐位还原失败（`frozen_block: False` 被翻成 `True`，实测 `identical? False`）；新范式逐位一致 `True`。
- 三态实跑：① 纯 train 冻结块；② **混合态**（冻结块内子模块被显式 `train()`——BN 重启用场景）；③ 异常中途抛出走 `finally`。三者均逐位一致。
- 真实模型规模：`model.train()` + `set_frozen_modules_eval` 后（519 冻结模块）调 `validate_roi_epoch` → **逐位还原 True**。
- eval 入参：`model.training=False` 传入 → 返回后仍 `False`、逐位一致（未强行置 train）✓。
- 变量遮蔽：函数体内 `for m in model.modules()` 的 `m` 是局部，函数内无第二个 `m` 依赖；异常路径 `finally` 覆盖（`try` 含整个 `with torch.no_grad()` 循环）✓。

### 3. `preextract --roi-meta` 透传（R3 P0-1 修）
- 实跑 `main()`（monkeypatch `extract`）：`--roi-meta` → captured kwargs `{'union_mask': False, 'roi_meta': True}` ✓。
- 源码核对 `extract()`：`build_official_samples(..., union_mask=union_mask, roi_meta=roi_meta)`（**全部 keyword**，无参数序风险）✓；`_meta` 加 `"roi_meta": bool(roi_meta)` ✓。
- 旧测试兼容：`tests/test_preextract_parallel.py` 两个 `fake_build` 已加 `roi_meta=False` 形参 → **7 passed** ✓；无测试断言 `_meta` 的键集（grep 确认）✓。

### 4. `official_loader` roi_meta 双向断言
- 四组合实事对拍（真 DICOM 造样本 → pickle → `build_official_datasets`）：
  | 组合 | 实测结果 | 期望 |
  |---|---|---|
  | roi pkl + `roi_meta=False` | **RuntimeError**（半开态） | ✓ |
  | roi pkl + `roi_meta=True` | NO-RAISE | ✓ |
  | 旧 pkl + `roi_meta=False` | NO-RAISE | ✓ |
  | 旧 pkl + `roi_meta=True` | **RuntimeError**（指向 `--roi-meta`） | ✓ |
- 直建路径（`else`）不经断言：正确（build 时 `roi_meta` 即真参，无可漂移）✓。
- `_has_rkey` 用 `isinstance(s, dict)` 短路，与 union 版一致（非 dict 损坏 pkl 不抛裸 TypeError）✓。

### 5. `_CLINICAL_LOGICAL_FIELDS` 加四概念
- `missing` 内容实测 = `['composition','echogenicity','shape','margin']`（mock 双行表头 + 官方窄义表两种口径均如此）；`_warn_missing_clinical_columns` 只 `_safe_print` 不 raise ✓。
- **无测试断言"无警告"**（`test_official_loader.py` 用 `capsys` 断言过具体 warning 文本？逐条 grep 后未见针对 missing 列表内容的断言）✓；mock 冒烟 `--roi-heads` 时**每次启动多一行** warning（实测打印），不破坏任何断言 ✓。
- 副作用：`records` 每行多 4 个键（值为 `None`）→ `_parse_concept(None) → None`、`_is_empty(None) → True` → `valid_*=0`（实测）✓，与"占位-1"设计不符但**语义正确**（valid=0 屏蔽）。**注意**：若哪天改成"缺列返回 -1"，`{'k': rec.get('composition')}` 的键存在性变化不影响消费方，但值得在 loader 里加一句注释。

### 6. `concepts_roi` 有限性门（R2 P1-3 修）
- 位置：K/长度校验**之后**、`log_softmax` **之前** ✓（先校验形状再判有限性，报错指向更准）。
- `K=0` 空批：`logits.shape=[0,4]` → `torch.isfinite().all() = True` → **不 raise**，返回 tensor(0.) + 全 detail 键（实测 `K=0 OK type=Tensor val=0.0`，键 9 个齐全）✓。
- `_K_REF=4.0` 的 `eps` 处理：`torch.clamp(v.sum() + _EPS, min=_K_REF)`，全 0 → 精确 0 ✓（R2 P2-2 注释与此一致）。
- 权重 0 路径：`lambda_concept <= 0` 早退（**不看 logits**，实测 `{}` 入参不 raise），恒 0 detail ✓ 零回归确认。
- NaN 门实弹：注入 NaN → `ValueError` 精确捕获 ✓。

### 7. mock bbox 渐变（R1 P2 修）
- 8 个样本 bbox = `(16..23)²→+20`，全在 64² 内、方形、`x1>y0`；`crop_roi` 切片恒 20×20（实跑 assert）✓。
- `global_loc` 与之同步（`0.40625 → 0.515625` 单调）✓，loc 头全局位置通道获得真实变动维度（区分力恢复，符合修复动机）。
- `_write_mock_dcm`：pydicom 3.0.2 下 4586 字节、`_read_dcm_gray` 读回 `(64,64) uint8 max=120` ✓；矩形 `(48,64)` 也合法读回 ✓；`RoiDataset(roi_size=224)` 从 20² resize 无异常 ✓。
- mock 无 `roi_mask` 键 → `_move_roi_masks_to_device` 全 no-op → `tests/test_roi_train_wiring.py` **24 passed** ✓（实跑 6.98s）。

### 8. 横幅 / 门
- `roi_head_weight` / `roi_concept_weight` 读取口径：`_hc = config["train"]`（横幅区）与 `build_official_head_config` 的 `tr.get(...)`（`:148/:152`）**同键同默认值 0.0** ✓；`main` 里 `_roi_weight` 又走 `build_official_head_config(config)` 与 `criterion.roi_head_weight` 同源（实测 `MultiTaskLoss` 侧键存在，跑通于 J2 探针）✓。
- `[ROI-heads 配方]` 打印**无守卫**（只在 `args.roi_heads` 分支内）→ mock 冒烟不加 `--roi-heads` 时不打印；加了则打印（含 `roi_size=224` 口径哨兵）→ 无 `capsys` 断言受影响 ✓。
- `concept_w > 0` 警告打印条件正确（`float(config["train"].get("roi_concept_weight", 0.0))`）✓。
- `parser.error` 四门实测：互斥 ×2 / 版本化 `--out` / 缺 resume / 零权重 —— `tests/test_roi_train_wiring.py` 5 项全绿 ✓。

### 9. 设计注释区（梯度语义订正）
- `grep "梯度累积"` 全文件：仅 `:879`（订正后的"此前注释写…"引用句）与 `:1660` 附近"梯度语义见 :876"指针，**无第三处旧文本** ✓；`docs/新架构文件级改动清单_20260925.md:609` 的"梯度累积解耦"是 DR 报告条目，与本处语义不冲突 ✓。
- 注释与代码一致：实跑确认 `run_roi_epoch` 每 batch 自完整 `zero_grad→backward→unscale→clip→step→update`（5 个 pattern 全在，与整图 `train_batch` 同款）✓。
- `optimizer.zero_grad()` 在 `run_roi_epoch` 内每 batch 调用 → 与整图段无跨支路累积，注释正确 ✓。

### 10. `expand_roi_samples` / `_mock_roi_samples` 消费链
- `roi_collate` 对 `roi_mask` 保 `list[tensor]`（实测）、`lesion_id` 保 `list[str]`、数值键堆 `[B] int64` ✓。
- 桥 `_meta_rows_from_collate`（dict-of-lists → list[dict]）在 `K=4` 与 `K=6` 批量下逐条长度对齐 ✓（`roi_collate` 的 keys 交集保证列齐）。
- `validate_roi_epoch` 键集 = 7 键（含新增 `roi_size`），JSONL `extra` 中 `None` 值序列化为 `null`（实跑确认；`_drop_nonfinite` 会把 NaN/Inf 也转 `null`——`None` 与"数值键恒在"两种纪律在此**语义合并**，属既有 `JsonlLogger` 行为，非本轮引入）✓。

### 11. 与整图/legacy 路径的相容性
- ROI 头 `requires_grad` / `param_groups`：`head_lr` 档 40 个 ROI 头参数全部可训且全部进 optimizer（实测 21+19=40，无遗漏）✓；`roi_heads=False` 逐位零回归（既有 2 项测试绿）✓。
- `roi_heads=True` 时 `set_frozen_modules_eval` 置 eval 数 519（vs 默认 569）→ **ROI 头 50 个模块全部保持 train**（实跑 `eval=[]`）✓，与整图 `run_official_epoch` 的 `train() + set_frozen_modules_eval` 组合逐位一致 ✓。
- `_collect_cliff_ious` 自管 eval/`try/finally`（已读源码确认），与 `validate_roi_epoch` 的逐模块快照**不冲突**（前后调用，各自还原）✓。
- 实测梯度归属：`compute_roi_head_losses` 的 `lambda_roi=1.0` 下 `seg_backbone` **零梯度**（detach 设计生效）、`cls_backbone` 有梯度（设计意图）✓。

---

## 与审查清单逐条对应（"已预判"项裁决）

| 清单 # | 结论 |
|---|---|
| 1 误伤面 | PASS（六形态实跑），**但**加 F5 注释误导 |
| 2 逐模块还原 | PASS（三态 + 真模型 + eval 入参 + 异常路径），修复必要性已实测坐实 |
| 3 preextract 透传 | PASS（CLI 实跑 + keyword 核对 + 旧测试绿） |
| 4 pkl 双向断言 | 断言本身 PASS（四组合），**但** F2：parser.error 不可达 → 拒不友好 |
| 5 四概念告警 | PASS（missing 内容、无测试破、副作用正确）；**但** F1 别名放行路径 + F3 注释不实 |
| 6 有限性门 | PASS（K=0 不 raise、NaN raise、w=0 早退、门位置正确） |
| 7 mock bbox | PASS（几何/真实 DICOM/reshape/测试绿） |
| 8 横幅/门 | PASS（键口径一致、无 capsys 影响） |
| 9 `n_dropped_no_bbox` | F6：口径混（病灶×样本），需改文案 |
| 10 梯度语义注释 | PASS（无第三处残留，注释与代码一致） |

---

## 本轮工具副产品（供主会话复核）

保留在仓库根、**未删**：
- `tmp_review_fix_probe_j2.py` / `-j2_out.txt`：整图 epoch 前向链（真 `run_official_epoch`）→ ROI epoch → 第二轮整图的串行对拍 + ROI 梯度变化计数（`ROI 头变化数 27 / 非 ROI 0`）。
- `tmp_review_fix_probe_k.py` / `-k_out.txt`：scaler 五段契约、`roi_mask` 缺键/dtype/device、`forward_roi` 同设备。
- `tmp_review_fix_probe_k5.py` / `-k5_out.txt`：**真 DICOM + 真 NIfTI mask** 端到端（roi_mask 生产端 → 消费端）。
- `tmp_review_fix_probe_m.py` / `-m_out.txt`：`_collect_cliff_ious` 源码、`validate_roi_epoch` 在 eval 入参下逐位还原、`run_roi_epoch` 与 legacy 路径 training 态逐位一致。

**已删**（跑崩本次 Python 进程、无法产出可信输出，避免误导后人）：`tmp_review_fix_probe_a.py`、`tmp_review_fix_probe_c.py`、`tmp_review_fix_probe_j.py`、`tmp_review_fix_committed_train_official.py`。其中 A 的结论（`纵横比` 命中）与 C 的结论（四组合断言）已在会话中以独立命令实跑并写入本报告，可原样复现。

> 说明：被删脚本的崩溃特征 = `SIGSEGV`，栈分别落在 pandas `to_excel`/`read_excel` 与 sklearn 导入序，**与 Demo_D 环境的 OpenMP/pyarrow 已知问题同源**（rules.md §1 已记：多进程 pytest + pyarrow 共享内存映射会拖垮内核）。本次串行跑单文件仍偶发，建议后续探针统一加"先 sklearn 后 torch"导入序守卫。

---

## 一句话给主会话

修复轮四个核心修复（mask 上设备 / 逐模块还原 / `--roi-meta` 透传 / pkl 双向断言）**全部独立验证通过**，无 P0；要动手的只有两条：**F1**（`shape` 的 `纵横比` 别名要在 `roi_concept_weight>0` 时硬拒或删别名）与 **F2**（把 pkl 内容断言前移到 `--roi-heads` 门区，让那条友好提示真的能被打出来）。

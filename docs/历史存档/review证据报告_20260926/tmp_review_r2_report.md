# R2 路审查报告 — ROI 新架构 · 损失数学(对抗性审计)

> 审查对象:`concepts_roi.py`(债6 概念监督)/ `roi_heads.py`(编排+概念挂接)/
> 四头 `*_roi.py` 的 k_ref 归一分母 / `multitask.py` 的 `roi_concept_weight` 接线区。
> 分支 `feat/l2-union-supervision`,基线 `tests/test_roi_losses.py` **40 passed**(实跑复核)。
> 证据脚本(保留,勿删):`tmp_review_r2_probe1.py`(损失数学+梯度)/ `tmp_review_r2_probe2.py`
> (真实数据链标签语义)/ `tmp_review_r2_probe3.py`(训练观测接线)。
> 旁证远端实跑(只读):A800 `data/{train,val,holdout}/*/meta.json` 概念标签类分布。

---

## 结论速览

| # | 级别 | 一句话 |
|---|---|---|
| P0-1 | **P0(条件性,数据门控)** | 概念监督的**标签来源未核实**:`_parse_concept` 对官方列 `成分/回声/形状/边缘` 只做"数字前缀−1",而实测这些列的真实格式是 **`成分:2分`(中文标签+评分)** → 全表 valid=0,监督**恒空**(静默) |
| P0-2 | **P0(条件性,数据门控)** | 即使解析通了,**类索引口径与全项目唯一有标签来源(公开集 TN3K/DDTI)相反**:width>height 在官方口径=1,在 TN3K=0(2×2 表全反);echo 在 TN3K 只有 2 类{1,2} → 官方 3/4 档在公开集里恒不存在 |
| P1-1 | **P1** | 概念监督**没有验证/观测面**:`--roi-heads` 的 val 支路(`validate_roi_epoch`)不算概念损失;"监督是不是真开着"的唯一证据(计数键)只写**训练** JSONL,且无人消费 |
| P1-2 | **P1** | 启动横幅(唯一防 `r7 config 静默不读` 事故的机制)**漏印 `roi_concept_weight`**;`_warn_missing_clinical_columns` 的四概念**不在告警名单**;唯一 `--roi-heads` 门只查 `roi_head_weight` |
| P1-3 | **P1** | **NaN 静默中毒路径**:`log_softmax` 前无有限性门,任一 valid 样本 logits 非有限 → `(per_sample*v).sum()` 被 NaN→**非有限×0=NaN**,该 batch 总损失与**全部损失项**变 NaN,`GradScaler` 静默跳过整步(训练卡死不报错);`concept_n_valid_*` 照报健康计数 |
| P2-1 | P2 | `concepts_roi` 的 `is_main` 门注释与实现不自洽("与四头不同"是错的;`roi_concept_weight>0 + lambda_roi=0` 时**事实上带 is_main 门**)。语义本身判对(无 is_main 是对的),是注释错误 |
| P2-2 | P2 | all-invalid 批返回值类型与 docstring 不符(注释说"total 是 python 0.0",实测是 `tensor(0.)`) |
| P2-3 | P2 | 概念首类 18 维旁路里 **9 维(foci 4 + echo 类2/类3 各 ~1-2 维)从未获得监督**,在"喂给 TI-RADS 头"的路径上**永久随机**(旧架构证过的同一失败模式) |

**PASS 项(试过的反例见 §PASS)** 6 条,含数值/梯度/零回归/屏蔽语义全部逐位或解析对拍通过。

---

## P0-1 概念监督的标签来源未核实 → 大概率**恒空监督**(静默)

### 位置
- `src/dataset/official_loader.py:232-239` `_parse_concept`(只做"数字前缀 − 1")
- `_FIELD_ALIASES["composition"|"echogenicity"|"shape"|"margin"]`(`:101-104`,候选名 `成分/回声/形状/边缘`)
- 消费链:`_build_labels:438-442` → `_ROI_LABEL_KEYS:823-826` → `roi_dataset._ROI_LABEL_RENAME` → `concepts_roi.ROI_CONCEPT_KEYS`

### 证据
1. **官方这几列的真实格式是"中文标签+评分",不是纯数字码**。项目自己的云端实测记录:
   - `docs/8.1赛事公开资料/数据探查证据链.md:141` / `docs/训练与架构/训练数据定稿.md:110`：
     「`TI-RADS详细评分`(第64列,真正的 ACR 五要素评分子项列,**格式"成分:2分,回声:2分,形态:0分,边缘:0分,强回声:0分"或单独总分数字**,全表缺失 **84.4%**)」
   - 全 74 列清单里**没有**独立 `成分`/`回声`/`形状`/`边缘` 列（`数据集概况.md` 列 0-73）——而 `_FIELD_ALIASES` 的候选名恰好就是这四个中文名，指向的极可能是这同一列的**表头碎片**或不存在。
2. **旁证（`tmp_review_r2_probe2.py` §D）**——把该格式原样喂给 `_parse_concept`：
   | 单元格 | 返回 |
   |---|---|
   | `"1囊性"` | 0 |
   | `"成分:2分"` | **None → valid_composition=0** |
   | `"实性"` / `"囊性"` / `"低回声"` / `"无"` / `""` | **None → valid=0** |
3. **上游只有"数据质量"层面的注释，没有任何"格式已核实"的结论**：`official_loader.py:100` 原文「列名/编码映射**待真实表复核**;命中则按数字前缀解析,缺失→valid=0」；`_parse_concept` docstring 同款「编码映射待真实 xlsx 复核」。而四头/CBM 的对应代码注释（`_parse_ete_code` / `_parse_tirads`）都写得明明白白"云端第 N 轮实测确认"——**偏偏这四个概念没有**。
4. **静态代码路径核验（probe2 §E，本地，纯字符串层）**：`_CONCEPT_TASKS ⊆ _CLINICAL_LOGICAL_FIELDS` = **False**。即列名全 miss 时**一声不响**（若真 miss，上游只会打印一句 `临床列命中:{}`，与概念任务无关）。
5. **端到端形态核验（probe2 §F，真实 `build_official_samples` → `RoiDataset` → 损失，本地合成最小官方目录）**：列全缺时 `valid_composition=0` → 损失返回**精确 0**、`concept_n_valid_comp=0.0`。也就是说"号称加了一路新监督、跑 20ep 拿 γ=0 干净窗口"，**实际可能是零监督**。

### 后果量级
`roi_concept_weight>0` 的唯一收益路径是"让 18 维概念旁路不再是随机噪声"。若监督恒空，收益 = 0，成本 = 一次云训练包上传（1-2h 审核）。
**关键的不对称**：这一失败模式**不可从训练指标发现**（loss 恒 0 与"概念很简单、已收敛"不可分），只能靠 `concept_n_valid_*` 计数——而该计数**目前没有任何消费方**（见 P1-1/P1-2）。

### 建议修法
1. **上云前 5 分钟零成本核实**：在 P0 验尸那类诊断脚本里加一节，直接对官方 xlsx 的表头 + 该四列的 `value_counts(dropna=False)` 打印（顺带打印 `TI-RADS详细评分` 列的样本值）。判据三条：列名命中 → 非空率 → 前 20 个字面值。
2. 依核实结果二选一：
   - 若格式是 `成分:2分`：`_parse_concept` 必须改为"解析中文标签（囊性/实性/混合/…）或 `:N分` 评分"，并**重新核对类索引口径**（见 P0-2）；
   - 若列不存在：**不要**把 `roi_concept_weight` 当成有效实验变量发云包，改走 DR 的四决策点 / 或明确接受"纯随机旁路"。
3. 无论哪种，`_CLINICAL_LOGICAL_FIELDS` 应加 `composition/echogenicity/shape/margin`（或复用一个 `_CONCEPT_LOGICAL_FIELDS`），把"静默零监督"变成**首屏可见告警**——这条与 P1-2 同源、成本几乎为零。

---

## P0-2 概念类索引口径与全项目唯一有标签来源的公开集**相反**

### 位置
`_parse_concept`（`official_loader.py:238-239`）`return None if d is None or d < 1 else d - 1`

### 证据（A800 只读实测，`data/{train,val,holdout}/*/meta.json`，由 `_parse_concept` 同一注释链的公开集管线产出）
TN3K(train+val, N=3226)概念标签分布：

| 概念 | 分布 | 与官方列语义对照 |
|---|---|---|
| composition | {0:7, 1:807, 2:31, 3:2381} | 代码注释 `num_classes_composition: 4 # cystic, spongiform, mixed, solid` |
| echogenicity | **{1:3218, 2:8}**（2 类） | 4 档里只有 hypoechoic / very-hypoechoic 会出现 |
| shape | {0:2832, 1:394} | `label_shape = 1 if h > w else 0`（`scripts/data/pseudo_labels.py:45`，TN3K **用 height/width**） |
| margin | {0:84, 1:8, 2:116, 3:3026} | TN3K margin 首类**不是** 0 |

对上官方那两列（`docs/训练与架构/训练数据定稿.md:110` 的 `成分:2分,回声:2分,形态:0分,边缘:0分,强回声:0分`，两个阴性项都是 **0 分**）：
- **shape 直接相反**：官方 `0分`= 阴性 = **宽>高(不满足 taller-than-wide)**；`_parse_concept` 得到 idx 0 → 模型在公开集里学到的是"idx0 = 高>宽"。**2×2 表全反**：
  |  | 官方（`_parse_concept`） | TN3K（`label_shape`） |
  |---|---|---|
  | 宽>高 | 0 | 0…不对，见下 |
  | 高>宽 | 1 | 1 |
  实测对拍（probe 逻辑）：官方"形态:0分"(宽>高) → `idx 0`；TN3K `h>w` → `1`，`w>h` → `0`。**即同一物理形态在两条链上落到相反类号**（官方 0 ↔ TN3K 1）。
- **echo 更严重**：官方 `2分`（低回声，= TN3K 的编 2）→ `_parse_concept` 得到 **1**；TN3K 编 1 = hyper/isoechoic。**语义与索引同时错位**，且 **3/4 档在公开集里从未出现**（2 类，8/3226 例外）。
- margin：官方 `0分`（光滑）→ idx 0，而 TN3K idx 0 只有 84/3226（且该头自己的 `_parse_concept` docstring 都承认"编码映射待复核"）。

### 后果量级
公开集是**冷启阶段唯一有真实概念监督的来源**（官方来源见 P0-1，很可能为 0）。当前状态下，公开集预训练会在概念头上灌入**至少 shape 全反、echo 语义错位**的信号；这比"随机旁路"更差——随机噪声至少无系统性，**反号监督会把概念头训成负资产**，而 TI-RADS 头吃的是它的输出。

### 建议修法
与 P0-1 合并为**一次核实**：把官方四列的 `value_counts` 与 TN3K/DDTI 的类分布**同图对拍**，在 `_parse_concept` 旁边写死一张"官方码 → 项目类索引"的显式映射表（照 `_parse_ete_code` / `_parse_tirads` 的写法：先显式排除、再精确查表、禁子串匹配），并在 config 注释里锁住 shape 的正负号方向。**在此之前不要给 `roi_concept_weight` 任何非 0 值。**

---

## P1-1 概念监督没有观测面（"开着但恒空"不可判）

### 位置
- `scripts/train_official.py:910-972` `run_roi_epoch`：`_roi_detail_extra(detail)` 写 JSONL
- `scripts/train_official.py` `validate_roi_epoch`：**只算** `collect_roi_records` + `roi_metrics`，返回键固定为 `roi_loc_auc/roi_ete_kappa/roi_t_auc/roi_tirads_kappa/roi_mal_auc_pr/roi_n_records`
- `configs/thyroid_config.yaml:259` 显式承诺「损失侧 detail 恒产 `roi_concept_*` 与 `concept_n_valid_*` 键」

### 证据
- probe3 §A/§B/§F（源码实读 + 行号定位）：
  - `compute_roi_concept_losses` 在 `train_official.py` 中 **0 次**出现（它由 `roi_heads.compute_roi_head_losses` 间接调用）；`concept_n_valid` 字样 **0 次**出现；
  - `validate_roi_epoch` 只把 `outputs["roi_outputs"]` 喂给 `collect_roi_records`，**不走损失** → **val 侧不产任何概念 loss / valid 计数**；
  - `validate_step.py`（整图 val）`compute_roi_head_losses` **0 次**出现。
- 唯一的真实内容证据（P0-1 说的那个"anti-zero-silence 保证"）只在**训练** JSONL 且是 **per-batch**（docstring 也承认"epoch 级聚合由回读侧"）。
- 上游 `_roi_grad_checked`（`train_official.py:1551-1561`）只检查 **`roi_*` 头的参数 `.grad` 非零**——概念头 `roi_concept_head.*` 若不含在 `ROI_HEAD_PREFIXES` 里，**FATAL 门也看不见它**（见下 §未验证/未覆盖）。

### 后果量级
"概念监督真的在学"与"概念监督恒空"在**所有自动化输出上都长得一样**（loss 恒 0 张量、计数键无人读、val 无该键、梯度门不覆盖概念头）。20ep 云训练是**唯一**能承载体量收益的实验，而它的核心新变量无观测面。

### 建议修法（三选一，按成本排序）
1. 最小：把 `concept_n_valid_*` 的**epoch 级均值**从训练 JSONL 聚合进 epoch 行（`_epoch_metrics_extra` 侧加一行），并在 `--roi-heads` 启动横幅里加"概念有效数须 > 0"的提示判据；
2. 中等：`validate_roi_epoch` 顺带算一次概念损失（复用同一 `compute_roi_concept_losses`，`torch.no_grad`），把 `roi_concept`/`concept_n_valid_comp` 等键并入返回 dict——这样"概念列的 valid 人口"在 val JSONL 上直接可见；
3. 与 P1-2 合并：把 `roi_concept_head` 前缀写进 ROI 梯度门的检查集，让"概念头无梯度"也走那条 FATAL 打印。

---

## P1-2 启动横幅 / 列名告警 / 真训硬门**三处都漏了 `roi_concept_weight`**

### 位置
- `scripts/train_official.py:1163-1170`（`[seg 损失配方]` 横幅）
- `scripts/train_official.py:1250-1264`（`--roi-heads` 四硬门，实际只读 `roi_head_weight`）
- `src/dataset/official_loader.py:453-455`（`_CLINICAL_LOGICAL_FIELDS`）

### 证据（实跑/实读）
- probe1 §P6、probe3 §E 之外的直接复刻（本地实跑输出）：
  ```
  CURRENT banner tail: [seg 损失配方] boundary_weight=0.0 lovasz_weight=0.3 roi_head_weight=0.6
  WITH new key      : ... roi_head_weight=0.6 roi_concept_weight=0.5
  ```
  → `roi_concept_weight` **不在横幅**。这是本项目**唯一**防 `r7 config 静默不读` 事故的机制（`train_official.py:1160-1162` 原文：config 是云端可变状态，配方必须横幅可见）。
- `_CLINICAL_LOGICAL_FIELDS` = `("malignancy","lesion_loc","extrathy_ext","clin_t_stage","tirads")`，**四概念不在其中**（probe2 §B）；列全 miss 时不会进那条"⚠️ 该任务将零监督"告警。
- `--roi-heads` 真训门只判 `roi_head_weight > 0`：`--roi-heads` + `roi_head_weight=0.6` + `roi_concept_weight=0.0`（config 默认）→ 横幅打 `roi_head_weight=0.6`、无任何提示，**概念监督静默缺席**。

### 后果量级
与 P0-1 叠加后形成**完整的静默链**：yaml 改错 → 横幅看不见 → 列名 miss 不告警 → 损失恒 0 → val 无键 → 计数无人读。云上一轮 20ep 全白烧且**事后也无法从产物反推**。

### 建议修法
1. 横幅加 `roi_concept_weight={_hc.get('roi_concept_weight', 0.0)}`（一行）；
2. `_CLINICAL_LOGICAL_FIELDS` 加四概念（一行）；
3. `--roi-heads` 门加分支：`roi_concept_weight > 0` 时打印"概念监督已开：有效数须 > 0，`concept_n_valid_*` 恒 0 = 标签链断"；（可选）`--roi-heads` + `roi_head_weight>0` 时把 `roi_concept_weight` 也打进 `[ROI-heads 配方]` 行。

---

## P1-3 NaN 静默中毒：非有限 logits × v=0 掩码 = NaN 总损失

### 位置
`src/models/losses/components/concepts_roi.py:187-194`
```python
log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
log_pt = log_probs.gather(1, idx.unsqueeze(1)).squeeze(1)   # [K]
per_sample = -log_pt
denom = torch.clamp(v.sum() + _EPS, min=_K_REF)
loss_c = (per_sample * v).sum() / denom
total = total + lambda_concept * loss_c
```

### 证据（probe1 §P4，实跑）
| 场景 | 实测 |
|---|---|
| `logits_comp` 含 NaN，该样本 **valid=1** | `total = nan`，`roi_concept = nan` |
| `logits_comp` 含 NaN，该样本 **valid=0** | **仍然 `total = nan`**（原文注释「占位值 0 + valid=0 → 零贡献」只对**有限**的 loss_k 成立） |
| composition 整键 valid=0（4 个 valid 全 0）且含 NaN | `roi_concept_comp = nan`，`roi_concept(sum) = nan`，`total = nan` |

关键一行：**`0 × NaN = NaN`（不是 0）**，所以 `(per_sample * v)` 不会把非有限样本"乘掉"，`sum` 把它扩散到**整批总损失**。

### 后果链（为什么这不是"上游本来就该保证有限"就能免的）
1. `total_loss = NaN` → `loss_dict` 里 **seg/concepts/malignancy/heads 全部 NaN**（同一个 `total_loss` 上加出来的）；
2. `run_roi_epoch:959-965` 的 `scaler.scale(loss).backward(); scaler.unscale_(); scaler.step()` —— 采用 `torch.amp.GradScaler`（新版）时 **inf/NaN 梯度会被 scaler 静默跳过该次 `step`**，**不报错**；
3. 结果：**训练在该 batch 上原地踏步、日志报警失败**，而 `concept_n_valid_comp` 仍报一个"健康"的样本数——与 P1-1 的观测面缺失叠加，**无人能发现**。
4. 对照：同文件四头对参数都做了 `_assert_finite` 硬拦式防御（`concepts_roi` 对 `outputs`/`roi_labels` 也有 fail-loud 分层），**唯独数值域没有**。

### 建议修法（按最小改动）
在 `log_probs` 之前加一次有限性门，且**与全项目"宁可整例 ERROR 也不静默"的纪律一致**：
```python
if not torch.isfinite(logits).all():
    raise ValueError(f"{logit_key} 含非有限值 —— 概念监督拒绝静默把 NaN 扩散进整批总损失")
```
（或退一步：`per_sample = torch.nan_to_num(per_sample, nan=0.0, posinf=0.0, neginf=0.0)` **并**同时把该批 n_valid 记 0 + 打 warning——但这会让"数值病态"变成静默降级，**不推荐**；本仓既有纪律是 fail-loud。）
同一处理建议顺手给四头 `*_roi.py` 加（它们同样没有 NaN 门，但四头路径 λ 默认 0、且各自的 logits 来自被 clip 的路径，优先级低于概念键）。

---

## P2-1 `is_main` 门：注释与实现不自洽（语义本身判对）

### 位置
`concepts_roi.py:41-44` 注释「**is_main 门:不加**(与四头不同、与整图概念损失同口径)」

### 证据
- 函数体（`:151-194`）确实不读 `is_main` → 与注释**一致的部分**对；
- 但"与四头不同"是**错的**：`ROI_CONCEPT_KEYS` 只列 4 键，**没有 `is_main` 门是四种写法都可能的**——而一旦 `lambda_roi=0`（纯概念标定路径，`roi_heads.py:160-162` 明确支持），`compute_roi_concept_losses` 成为**唯一**损失项，**整批（含无关背景/次病灶以外的行）所有 ROI 样本都进概念分母**：
  - `roi_dataset.expand_roi_samples` 只从 `roi_lesions`（**只含 M/病灶N**）展开 —— 未知行其实进不来，这点判对；
  - 但 `is_main=0` 的次病灶会**全部**进概念分母（四头是 `v × is_main`）。「次病灶概念是合法监督」的**结论可以保留**（概念是病灶级特征，注释论证成立），但**论证文字**（"与四头不同"）与实际差异（`lambda_roi=0` 时四头根本没跑，不是"门不同"）不符。

### 建议修法
把注释改成如实描述："概念监督**有意**不做 is_main 门（病灶级概念是合法监督），且**不因 `lambda_roi=0` 而改变**——纯概念标定时它也是全 ROI 样本的面板。"（1 行改动，影响的是下一位读注释的人。）

---

## P2-2 all-invalid 批返回值类型与 docstring 不符

### 位置
`concepts_roi.py:124-126` docstring「权重 0 时 total 是 python 0.0」，但代码路径：
`:190-193` `loss_c` 是 `tensor(0.)`（`torch.clamp`/加减都在张量域）→ `total = 0.0 + lambda*loss_c` = **tensor**。

### 证据（probe1 §P3 实跑）
```
all-invalid total (expect python 0.0 / exact 0): tensor(0., grad_fn=<AddBackward0>)
```
**数值上无害**（`roi_heads.py:149-151` 的注释已正确说明"标量 0 与 tensor 相加安全"），但 `hasattr(v,'item')` 分支恰好兜住，且调用方一律 `+`，**无功能影响**。

### 建议修法
docstring 改为"权重 0（早退路径）是 python `0.0`；权重>0 但整批无有效样本时是 `tensor(0.)`（数值恒 0）"。

---

## P2-3 18 维旁路里有 9 维永久无监督（旧架构已证过的同一失败模式）

### 位置
`roi_heads.py:477-478`（`dims` 五键 18 维）vs `concepts_roi.py:81-86`（`ROI_CONCEPT_KEYS` 只有 4 键）

### 证据
- 被监督：comp 4 + echo 4 + shape 2 + margin 4 = **14 维**；
- 从未被监督：`logits_foci` **4 维**（无官方标注，有意）+ `logits_echo` 的**类 2/类 3 两维**（见 P0-2：公开集里只有 2 类）→ **9 维**；
- 这 9 维经 `RoiTiradsHead.forward` 的 `torch.cat(parts)` **直接拼进** TI-RADS 头分类器输入（`:490-491`）——`concepts_roi.py:6-11` 的动机段（"detach 使它永久随机 → 18 维噪声污染 TI-RADS 输入"）**精确描述了这条路径**，而债6 只覆盖了其中 14 维。

### 后果量级
TI-RADS 头仍需从**约一半是常数/随机**的 18 维输入里学分级（头部有用权重自己学出来），量级远小于 P0/P1，但与债6 的立项理由同源，属"债6 只修了一半"。

### 建议修法
- 明确写下"foci 4 维 + echo 少数类维度是**设计上的零监督维**"（若这是接受的设计），或
- 在 TI-RADS 概念拼接处只拼**有监督的 14 维**（同时改 `RoiConceptHead._CONCEPT_DIMS` / `RoiTiradsHead.concept_dim` / 对应测试），或
- 给 foci 补一个"恒 0 但可学习偏置"的显式占位（不推荐：等于喂常数）。

---

## 未验证 / 未覆盖（诚实清单，非 finding）

1. `MultiTaskThyroidModel.ROI_HEAD_PREFIXES` 的**实际内容**未读（超出本轮领地）；`_roi_grad_checked`（`train_official.py:1551-1556`）只 `startswith(_ROI_PFX)`，若 `roi_concept_head` 不在该前缀集里，则"概念头全零梯度"也**不触发那条 FATAL 打印**。建议下一路顺手核。
2. 真实官方 xlsx **未读**（禁本地、云端才有）：P0-1/P0-2 的"全表 valid=0"结论基于①项目自身云端实测记录的确切字面格式（`成分:2分`）②`_parse_concept` 输入-输出对拍。**结论的条件是"该格式即真实值"**——这正是建议上云前 5 分钟核实的内容。
3. `_assert_finite` 对 `t_conf` 的位点未定位（§1 提到 `decode_predictions` 有），本轮未展开（与概念监督无关）。

---

## PASS 项（试过什么反例才排除）

| # | 断言 | 试过的反例 / 验证方式 | 结果 |
|---|---|---|---|
| PASS-1 | 概念 CE 数值与梯度方向**解析正确** | 手算 `d/dz = (softmax(z) − onehot)/v.sum()` 与 `backward()` 逐元素对拍（probe1 §P1） | `max|diff| = 1.49e-08` |
| PASS-2 | **缺失样本对概念 logits 的梯度精确 0**（不是"先算再乘 v"的隐性偏差） | K=4、2 个 valid=0（两类各 2 个）实跑 backward 查 `.grad` 行（probe1 §P2） | 无效行 `(g==0).all() == True`，有效行非零 |
| PASS-3 | **foci 零梯度** | `logits_foci.grad is None`（probe1 §P1） | True |
| PASS-4 | `k_ref=4.0` 三段语义 + **0×NaN 之外无数值毒化** | n_valid=1/2/3/4/5/8 实跑对比 per-sample CE 的倍数（probe1 §P3）；全 invalid → `tensor(0.)` 精确 0、计数 0.0 | n<4 分母压到 4（1/4 倍），n≥4 逐位等旧式；全 invalid 无 NaN |
| PASS-5 | **λ_concept=0 逐位零回归**（含 `_zero_concept_detail` 键恒在） | `compute_roi_head_losses(..., concept_weight=0.0)` vs 不传：total/detail **逐键 `float()` 相等**（probe1 §P6b）；另 `lambda_concept=0` + 空 outputs 不 raise、键齐（probe1 §P6） | True / True |
| PASS-6 | **fail-loud 分层**（窄 dict 拒空算） | `{"roi_outputs": []}` + `lambda_concept=1.0`（probe1 §P6） | 如期 `ValueError`，错误文本点名"顶层 roi_concepts" |
| PASS-7 | 计数键类型 / `_roi_detail_extra` 混 float+Tensor dict 消费 | 实跑打印每键类型（probe1 §P5）：损失键 `Tensor`、计数键 `float`；`float(v.item() if hasattr(v,'item') else v)` 逐键成功 → JSONL 出 `loss_concept_n_valid_*` 与 `loss_roi_concept*` | 9 键全部落进 JSONL，无跳过 |
| PASS-8 | **端到端真实链**（列命中 → meta → 桥 → 损失） | `build_official_samples(roi_meta=True)` + `expand_roi_samples` + `RoiDataset.__getitem__` + `roi_labels_from_meta` + 损失（probe2 §F，合成最小官方目录/xlsx 双行表头/真 dcm+nii.gz） | 列命中、meta 携带、桥产 `gt_composition`、损失出真实值 `1.047`，`concept_n_valid_comp=1.0` |
| PASS-9 | multitask 接线属性/默认值/透传 | `MultiTaskLoss.roi_concept_weight` 由 `cfg.get("roi_concept_weight", 0.0)` 读（`multitask.py:123`），`build_official_head_config` 组装（`train_official.py:155`），`run_roi_epoch` 用 `getattr(criterion, "roi_concept_weight", 0.0)` 读（`:957`） | 属性名拼写一致、默认 0、链路通 |
| PASS-10 | 四头 `_K_REF` 值一致（漂移风险反例） | 全仓 grep `_K_REF` | 五处（含 `concepts_roi`）全为 `4.0`，无漂移 |
| PASS-11 | 缓存/pkl 半开态不产生"号称开了实际没开" | `official_loader` docstring + `roi_dataset` 判键跳过 + `train_official` 的 `len(roi_train_ds)==0 → parser.error` | 设计自洽，无静默 |

---

## 总判

**代码层（数学/梯度/屏蔽/零回归/键契约）：PASS。**
概念 CE 的数值路径（`log_softmax` → gather → mask → clamp 归一）经手算与 backward 对拍逐位正确，缺失样本梯度精确 0，foci 零梯度，λ=0 逐位零回归，fail-loud 分层到位，四头 `_K_REF` 无漂移。`tmp_review_r2_probe1/2/3.py` 可重跑复核。

**但 R2 的核心风险不在数学，在"监督的标签与观测面"：**

- **不可放行**（P0-1 + P0-2）：概念标签来源未经任何真实数据核实，且已与全项目唯一有标签来源的公开集口径**相反**（shape 全反、echo 语义错位、3/4 档不存在）。在核实前 `roi_concept_weight` 应**保持 0**；若已计划上云带非 0 值，**先花 5 分钟核实官方四列**。
- **必须随包修**（P1-1 + P1-2）：横幅漏 `roi_concept_weight`、列名告警不含四概念、无 val/计数消费面——三处叠加使"恒空监督"完全不可见，且恰好是本仓 `r7 config 静默不读` 事故的同型。三处都是**一行级**改动。
- **建议随包修**（P1-3）：`logits` 有限性门（因为 `0×NaN=NaN` 会经共享 `total_loss` 把**全部**损失项变 NaN，而 `GradScaler` 静默跳过 step）。
- P2-1/2-2 为注释/文档级，P2-3 建议明确为设计决策或收窄 TI-RADS 概念拼接维。

> 附：本报告未改任何仓库文件；仅新增 3 个探针脚本（按要求保留），A800 侧只做只读 `meta.json` 统计。

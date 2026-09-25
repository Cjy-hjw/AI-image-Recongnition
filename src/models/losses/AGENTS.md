# src/models/losses/AGENTS.md - Loss Functions Specifications

## 1. 多任务损失架构

### 主损失函数 (`multitask.py`)
- `MultiTaskLoss`: 组合分割BCE+Dice、TI-RADS概念SLR、良恶性AUCM+SupCon
- 使用Homoscedastic Uncertainty Weighting自动平衡任务权重
- 支持FAMO动态权重调度
- ✅ **三头固定 λ 外挂(阶段 2.4 接线)**:`__init__(official_head_config=...)` 读取 `lambda_loc/lambda_ete/lambda_clin_t`(默认各 1.0)与 ETE/T 的类加权/平滑超参;三个 `register_buffer("ete_class_weights", ...)` 等缓存——**传 None 时不进 state_dict(legacy 零影响)**。`forward` 在 `apply_famo_weighting` 之后、`compute_regularizers` 之前调用 `compute_official_head_losses` 并把 `detail` 数入 `loss_dict`(location/location_side/location_level/ete/ete_ce/clin_t/clin_t_ce)。🔴 **三头不进 FAMO 5 通道**(扩通道破坏 checkpoint 兼容 + masked 0 值被 FAMO 误读为"已收敛",见 `components/official_heads.py`)。三头键缺席 → 跳过,legacy `train.py` 路径零回归。
- **CC-DiceCE 透传(2026-09-18)**:`__init__` 读 `seg_ccdice_weight`(默认 0.0),`forward` 传 `ccdice_weight=self.seg_ccdice_weight` 给 `compute_segmentation_loss`(masked 路径);权重 0 时逐位零回归。横幅/实验门在 `train_official.py`(三权重判空)。

### 组件化设计 (`components/`)
- `segmentation.py`: `compute_segmentation_loss()` — BCE+Dice,支持`valid_mask`;阶段2 seg-only 扩展:可选外挂 Boundary(带符号距离场)/Lovász-Hinge(Berman 官方差分版)两项,返回 `(loss, {"seg_boundary","seg_lovasz"})` 分项进 JSONL;legacy 路径非零新权重 raise。
  **CC-DiceCE 实例级局部损失(2026-09-18,plan 任务 C,默认关待命)**:`cc_dice_ce_per_sample(logits, gt) -> Tensor[B]` 由 `seg_ccdice_weight`(config,默认 0)经 `multitask.py` 透传外挂,`extra["seg_ccdice"]` 恒在。每样本:①GT CPU **显式十字 4 连通**标记(`ndimage.label` 默认就是 4 连通非 8,structure 显式钉死防漂移);②每实例 bbox 外扩 `max(2, ceil(0.1×max(h,w)))` ROI,**邻近其他实例前景从本实例背景惩罚排除**(增量并集 `all_fg & ~inst`,O(K²)→O(K),K=100: 3.81s→87ms);③ROI 内 BCE+Soft-Dice 实例等权平均。🔴 **BCE 必须 logit 域直算(BCEWithLogits),禁 sigmoid→clamp→log 反解**——`1-1e-8` 在 float32 舍入为 1.0,logit≥17 即 sigmoid==1.0→log 反解 inf→NaN,NaN×valid_mask 屏蔽失效、GradScaler 静默跳步(Gate2 P0,r5 日志 seg_prob_max=1.0 证明饱和可达)。K=0/软 GT 归 0(监督归主项不混纲)。前景须用**实例自身窗口**非整 GT 切窗(邻接实例像素混入=朴素对拍当场杀出的实现期真 bug)。测试 `tests/test_ccdice_loss.py` 17 项(朴素参考逐值对拍+变异杀测);量级:512² B=16 2实例≈102ms/step,启动前 1ep 标定读分项定权重。
- `concepts.py`: `compute_concept_losses()` — TI-RADS五项SLR+临床先验正则
- `malignancy.py`: `compute_malignancy_losses()` — AUCM+SupCon
- `famo_mix.py`: `apply_famo_weighting()` — FAMO权重应用与更新
- `regularizers.py`: `compute_regularizers()` — Latent L2+SPIN L1
- `location.py`: `compute_location_loss()` — 位置头侧别/纵向双分支 masked CE（阶段 2.1）。
  **不对合成的 7 维概率取 NLL**：合成概率是 `P(side)·P(level)` 乘积，两个 1e-4 相乘即
  1e-8，log 后接近 -18 甚至下溢 -inf；分支各自 CE 走 `log_softmax` 稳定实现。
  **峡部 level 标签是 -1**，直接喂 `cross_entropy` 报 `Target -1 is out of bounds` →
  先 clamp 到 0 保证不崩、再由 validity 乘 0 屏蔽，**两道保险缺一不可**（只 clamp 会
  编造监督，只屏蔽会先崩在 CE 里）。
- `extension.py`: `compute_extension_loss()` + `class_balanced_weights()` +
  `log_class_prior()` — ETE（甲状腺外侵犯 4 类）的 masked CB-Focal + 可选 Logit
  Adjustment（阶段 2.2）。全项目最极端的不平衡任务：`3906/267/67/5`，最稀有类 0.12%。
  **三条必须记住**：
  1. 🔴 **CB 的 β 不能抄论文的 0.9999**。实跑权重比（血管:无）：β=0.9→2.4:1、
     0.99→**20.4:1（默认）**、0.999→196.4:1、0.9999→**646.9:1**。n=5 时 0.9999 退化成
     近似逆频率加权，等于让 5 张图吃掉整个损失、必然过拟合。
  2. 🔴 **Logit Adjustment 训练时是 `logits + τ·log π`（加，不是减）**。符号反了会
     **放大**不平衡，而损失照常下降（实跑：反号时 loss 仍从 2.90 降到 0.19）——
     与 #113 FAMO、#120 拉格朗日符号同类。推理侧**不得再调一次**：训练完的原始
     logits 本身已逼近平衡后验。方向由 `tests/test_extension_head.py` 的收敛实验
     锁死（纯 CE gap≈3.0 / LA≈0 / 反号≈5.9，三者两两可分）。
  3. CB 加权与 LA 是同一问题的两种解法，**同时开满会过校正**，故 `la_tau` 默认 0。
     两个超参都在 `configs/thyroid_config.yaml: train.ete_*`。
  4. 🔴 **backlog #128（2026-08-17，方案就绪未执行）**：CB-Focal/加权 CE 优化逐
     样本似然，与官方真实指标（二次加权 Kappa）在本任务分布下**负相关**——已实测
     两次转负/垫底（β=0.99 时 ep26 起转负；β=0.9 后本轮 kappa=0.2086 仍五项最低）。
     `compute_extension_loss` 新增 `qwk_weight`/`dist_weight` 参数，`qwk_weight>0`
     时切到 **LA-SoftQWK-OD**（`quadratic_weight_matrix`/`soft_qwk_loss`/
     `ordinal_distance_loss`，DR 主方案）：`total = qwk_weight·SoftQWK + dist_weight·
     OrdinalDistance`，常数预测下 SoftQWK 收敛到 1.0（QWK=0，最强验收判据；
     ⚠️ 实测 **0.999999762** 而非严格 1.0，`_QWK_EPS` 所致，断言须带 1e-6 容差）。
     🔴 **LA 必须与 SoftQWK 同时开**：实测单开 `la_tau` 会把稀有类概率推高 ×21
     （τ0→0.5），而本项目已因过度预测稀有类让 Kappa 转负过一次；`la_tau` 只调整
     决策边界，靠 SoftQWK 才能压住它。默认 `qwk_weight=0.0`（YAML），逐位走 CB-Focal
     老路径，零回归。
  **`focal_gamma` 只接受 0 或 ≥1**：γ∈(0,1) 在 `p_t=1.0` 处梯度为 NaN 而 loss 值
  看不出异常（实跑 γ=0.5 → loss=0.0、grad=[nan,nan]），且 `p_t==1.0` 在 fp32 下
  真的可达（logit gap≥50 即精确 1.0）。
- `clinical_t.py`: `compute_clinical_t_loss()` + `inverse_frequency_weights()` —
  临床 T 显著性 2 类的 masked 加权 CE（阶段 2.3）。**无 Focal 无 CB**。计数
  `0无2410 / 1有1831`（`docs/训练与架构/训练数据定稿.md:87`），阳性率 35.7% 已相当均衡。
  **三条硬知识（均实测）**：
  1. 🔴 **不能把侵犯头的 `cb_beta=0.99` 抄过来**：CB 的 `1−β^n` 在千级计数上已
     饱和，`class_balanced_weights([2410,1831], 0.99)` 权重比恒为 **1.0000**——
     配了等于没配，且**不报错不告警**。β=0.999→1.0838、β=0.9999→1.2800，仍不及
     逆频率 1.3162。CB 是给长尾的（ETE 血管 n=5 那里 β=0.99 才给出 20.4:1）。
     要加权用 `inverse_frequency_weights`。
  2. 🔴 **`label_smoothing` 默认 0**：LS 的最优解是 `1−ls+ls/C`，二分类即
     `1−ls/2`——实测 ls=0.05→阳性样本 P(T=1)=**0.9750**、ls=0.1→**0.9500**。
     官方 `T_ClinSig.T_conf` 读的正是 `P(T=1)`，开 LS = 系统性压缩交付置信度。
     这是拿概率校准换分类精度，不是白捡的正则。
  3. **刻意不加 Focal**：阳性率 35.7% 已相当均衡，Focal 只会平白引入
     `γ∈(0,1) → NaN 梯度` 那类风险而无收益（见上方 extension.py 的
     `focal_gamma` 禁值说明）。
  超参在 `configs/thyroid_config.yaml: train.clin_t_*`（`clin_t_class_counts:
  [2410,1831]` / `clin_t_use_class_weights: false` / `clin_t_label_smoothing:
  0.0`）。

### 🆕 ROI 四头损失组(2026-09-25 波4,新架构 §J,默认关;🟢 已收编主树)

- `location_roi.py`/`extension_roi.py`/`clinical_t_roi.py`/`tirads_roi.py`:
  现有四头损失的 **per-lesion 版**(复用同式数学:location 双分支 masked CE/ETE
  CB-Focal/clin_t 概率域 BCE/tirads CE+ordinal),输入 [K] 变病灶数,valid 过滤
  +is_main 门(官方口径只主病灶)同判据。接线器 `components/roi_heads.py`
  `compute_roi_head_losses(roi_outputs, labels)` 消费 forward_roi 输出+gt_ 前缀标签。
  `multitask.py` 加 `roi_head_weight`(默认 0 零回归;>0 编入,无 ROI 键 raise
  fail-loud,extra 加 roi_* 恒 0 键两分支都写);🟢 收编时 cliff 块(set_cliff_ious/
  seg_cliff_weight)与 ROI 块**并存**于 multitask.py(worktree 版基于旧基线,主会话
  手工合并三处 diff,勿再按 worktree 版整文件覆盖)。
- 🔴 **两条已修+三条待接线(Gate2 A/B 路,处置全录 scoping §J.2)**:
  1. **location level 边缘已归一**(B-P1-3 修):`_edge_log_probs` level_p 除以
     非峡部总概率(条件分布 P(level|非峡部))——因子分解联合分布下不归一则
     level CE 系统性偏高 −log(1−P_峡部)+梯度耦合 side 峡部通道,与整图双 softmax
     版不等价(docstring 原"完全等价"声称被实测证伪);
  2. **config 透传已修**(A-P1-2):`train_official.build_official_head_config`
     补组 `roi_head_weight`(此前 yaml 写了键不组装=透传链断裂,r7 config 静默
     不读事故同型);
  3. ~~待接线三条~~ → **全部已解(2026-09-25 深夜接线五债,scoping §J.6)**:
     `gt_` 桥=`roi_dataset.roi_labels_from_meta`(`_ROI_LABEL_RENAME` 单源+collate
     dict-of-lists 转置)/roi_mask 生产=`RoiDataset.__getitem__` 四态+统一栅格
     (roi_size 单源+INTER_NEAREST+值域 assert)/labels 形态守卫=损失侧
     `_scalar` 的 batch 张量 numel≠1 raise。**四头数学经损失组 review 逐位同构**
     (location/ETE 精确相等,T 1.2e-7,TI-RADS ≤4.8e-7;LS 4.2e-2 归因 torch CE
     weight×ls 已知交互、ROI 版才是对的——J.5b)。
- ~~**损失组 review(J.5b)遗留 3 条测试债**~~ → **已清偿(2026-09-26 硬前置 A 路,J.8)**:
  ①gt_ 源级对拍=`test_j5b_gt_fixture_source_level_bridge`;②LS>0 分支覆盖=
  `test_j5b_tirads_label_smoothing_positive_coverage`;③device 断言=
  `test_j5b_zero_weight_detail_device`。
  `_scalar` 的 `int()` 对 float 标签截断(0.5→0)是 J.5b P1-1 复核降级 P2 的
  残余面——真实链路 label 走 int64 tensor 无损,勿在 float 路接标签。
- **债6 概念监督已落地(2026-09-26 硬前置 A 路,方案 A 纯真值版,DR 终审)**:
  `components/concepts_roi.py`——`ROI_CONCEPT_KEYS` 四键映射(logits_comp/echo/
  shape/margin ↔ gt_composition/echogenicity/shape/margin+valid_*);🔴 **概念
  logits 在 forward_roi 顶层 `roi_concepts` 不在 per-lesion dict**,run_roi_epoch
  停传窄 dict(窄 dict 被 fail-loud 拒,r7 同型防);foci 不入表=零梯度(无官方
  标注);is_main 门不加(整图概念损失同口径);`roi_concept_weight` 默认 0 全链
  透传(config/multitask/train_official)。**k_ref 分母平滑同批**:四头归一分母
  `clamp(v.sum()+eps, min=4.0)`(`_K_REF` 单点可调;n≥4 逐位不变/n∈(0,4) 压
  4×/v=0 精确 0)。
- **🔴 2026-09-26 训练端 review 修复(R1/R2/R3 三路,主会话独立复核)**:
  ①concepts_roi 加**有限性门**(logits 含 NaN/Inf → raise;`0×NaN=NaN` 不被
  valid 屏蔽,会经共享 total 毒成全 NaN、GradScaler 静默跳 step=训练卡死无报错,
  fail-loud 而非 nan_to_num);②**R2 P0-1 定案:官方表无独立概念列**(证据链
  第13轮真实列名序 40-51 无 成分/回声/形状/边缘;信息只在 TI-RADS详细评分
  复合串,缺 84.4%)→ 概念监督在官方数据上**结构性恒空**,roi_concept_weight
  保持 0 直到数据侧解锁(拆复合串+类索引与 TN3K 对齐——TN3K shape 用 h>w、
  官方"形态:0分"=宽>高,同类号反义;echo TN3K 仅 2 类)。数学/梯度路径已
  对拍正确(R2 PASS),数据解锁即用;③is_main 注释订正(不因 lambda_roi=0
  而变,纯概念标定仍是全 ROI 样本面板)。
- 测试 `tests/test_roi_losses.py` **40 项**(2026-09-26 A 路 24→40:概念监督 9/
  k_ref 4/J.5b 三债 3;手算对拍/零权重逐位回归/变异杀测/fail-loud;变异实弹
  主会话独立复现[去 k_ref clamp→2 failed RED→还原 40 绿])。

---

## 2. PHASE 4 新增：掩膜多任务损失

### 背景
- 多源数据标注异质化：ThyroidXL全标注，ThyUS2Path仅病理，DDTI仅分割
- 训练时需自动屏蔽无标注任务的梯度，避免负梯度惩罚

### 实现1：组件内部支持（唯一正确路径）
**分割损失** (`components/segmentation.py`):
- `targets['valid_mask']`: Tensor[B]，1=有分割mask，0=无
- per-sample计算BCE/Dice后`sum() / valid_count`
- 无效样本**真正零梯度**（未进入计算图）

**概念损失** (`components/concepts.py`):
- **已实现 per-sample mask**（`label_validity` 逐样本屏蔽、sum/有效数归约，见 `concepts.py:155`；与 `models/AGENTS.md §7b` 一致，旧"P1待办"已完成）
- 异构源另由 adapter 层 `task_mask` valid=0 从源头不产假标签，双重保险

**良恶性损失** (`components/malignancy.py`):
- 已支持`label_validity`参数（line 31-57）
- 无效样本不进入AUCM dual variables更新

### 已废弃：MaskedMultiTaskLoss 后处理包装器（commit 54f0302 删除）
- 原 `masked_multitask.py` 用后处理缩放损失权重实现 task_mask，但 `base_loss` 返回时
  计算图已构建、per-task loss 已 reduce 成标量，**再乘 valid_ratio 只缩小损失量级，
  无标注样本的梯度仍经标量 loss 回传污染 encoder**——不是真屏蔽。且它重算 total_loss
  时绕过了 FAMO 动态加权。故整体删除。
- 唯一保留：任务名→loss key 映射表 `TASK_TO_LOSS_KEYS`，已搬入 `multitask.py` 作模块常量
  （仅文档性/调试用，非屏蔽逻辑）。
- 正确做法统一走「实现1」：`multitask.py` forward 把 `task_mask['mask']` 注入
  `targets['valid_mask']`，由 components 在 per-sample 层真正零梯度屏蔽。

### 使用方式
```python
# 唯一路径：利用组件内 valid_mask / label_validity 支持
targets['valid_mask'] = task_mask['mask'].float()        # 分割 per-sample 屏蔽
targets['label_validity'] = task_mask['malignancy']      # 良恶性 AUCM 屏蔽(单张量)
# ⚠️ 概念损失(composition/echo/shape/margin/foci)走 **dict** 形态:concepts.py:156 按 `lv[task]`
#    索引,若供概念头须传 {'composition':tensor, ...} dict,照抄上面单张量会 KeyError。确切键见 components/concepts.py

loss_fn = MultiTaskLoss(...)
total_loss, loss_dict = loss_fn(outputs, targets, task_mask=task_mask)
# multitask.py forward 内部自动把 task_mask['mask'] 注入 targets['valid_mask']
```

### Gate2审查结果（commit d4c00ff → 54f0302 清理）
- **P0-1修复**：multitask.py利用现有`valid_mask`支持，避免后处理梯度泄漏
- **54f0302 清理**：删除劣质的 MaskedMultiTaskLoss 后处理路径（已被上述真屏蔽取代）
- **已完成**：概念损失 per-sample mask（`label_validity`，见上「概念损失」段）

---

## 3. 关键约束（继承自原AGENTS.md）

### MixUp目标插值保护
- AUCM dual variables (`a`, `b`, `alpha`)必须使用**未插值**的目标数组
- `multitask.py`已在`compute_malignancy_losses`前解析mixup参数

### 数值稳定
- 所有除法有`eps=1e-8`保护
- NaN Sentinel打印到stdout（flush=True）

### 优化器隔离
- vCLUB 参数由专属 vclub_optimizer 更新，主 optimizer 的参数组中显式排除 vclub 模块
- 见`src/training/AGENTS.md §1`

---

## 4. 测试覆盖

- `tests/test_masked_loss.py`: 10个测试验证valid_mask零梯度隔离
  - `test_all_invalid_zero_gradient`: 确保无效样本无梯度流
  - `test_partial_valid_excludes_invalid_samples`: 部分有效样本正确屏蔽
  - `test_all_valid_legacy_compat`: 向后兼容性

执行: `pytest tests/test_masked_loss.py -v`(全套现 841 collected / 840 passed+1 skipped(Demo_D 主环境,2026-08-15 动态帧预抽取缓存(+9)与四头闸门阈值修复(+12)回归对拍;**计数以 `tests/AGENTS.md`/实跑为准**;**跨环境不可比**,torch2.8 预演环境待下次云端对拍重新确认(TI-RADS 加入前为 789 passed / 6 skipped,collected 795 与主环境一致;旧记的 620/813/814 已失效))

"""官方新增三维度的增量诊断头（Plan v2 阶段2 填充，见 `docs/训练与架构/训练实现计划.md`）。

三头独立成模块置于本子包，使 `multi_task_model.py` 保持薄装配枢纽（rules.md §6）：

- **位置头** `lesion_loc`（7 类，侧别×纵向因子分解 `P(k∈1..6)=P(Side)×P(Level)`、
  `P(7峡部)=P(Side=峡)`）—— ✅ **已实现**（阶段 2.1，`location.py`）；接 EfficientNet
  Stage-4 空间特征图 + CoordConv。损失在 `losses/components/location.py`（侧别/纵向
  各自 masked CE）。**两个坑**：① 峡部无纵向属性，level 分支须用 `build_level_validity`
  屏蔽，且其 level 标签为 -1、直接喂 `cross_entropy` 会越界；② 水平翻转必须用
  `flip_location_index` 联动翻标签，漏了会让位置头学到反的左右而 loss 照常下降。
  ③ CoordConv **必须先过非线性再池化**（线性卷积下坐标贡献退化为常数偏置，被 GAP
  平均掉、左右完全不可分 —— 已用三段反证锁在 `tests/test_location_head.py`）。
- **侵犯头** `extrathy_ext`/ETE（**保 4 类，血管仅 5 例不并类**）—— ✅ **已实现**
  （阶段 2.2，`extension.py`）；`soft_mask_detached` 膨胀−腐蚀出交界带 × Stage-4 局部
  加权池化 + CBM `margin` 旁路（**detach**，不让 ETE 梯度重塑「边缘」概念、毁掉 CBM
  报告可信度）。损失在 `losses/components/extension.py`（CB-Focal + 可选 Logit
  Adjustment，masked）。**三个坑**：① CB 的 β **不能抄论文 0.9999** —— n=5 时权重比
  达 646.9:1，5 张图吃掉整个损失；默认 0.99（20.4:1）。② Logit Adjustment 训练时是
  `logits + τ·log π`（**加**），符号反了会放大不平衡而损失照常降；推理侧**不得**
  再调一次。③ CB 加权与 LA 是同一问题的两种解法，同时开满会过校正，故 `la_tau`
  默认 0（关）。
- **临床T头** `clin_t_stage`（二分类）—— ✅ **已实现**（阶段 2.3，`clinical_t.py`）；
  GAP 全局特征 + 结节尺度标量 `[s, log s]` + ETE 概率（**detach**）→ 轻量 MLP。
  影像学上限约 0.72（淋巴结常不在视野），训到 0.72 附近就该转去看别的头。
  **三个坑**：① 🔴 **别把侵犯头的 `cb_beta=0.99` 抄过来** —— 在 `[2410,1831]`
  这种千级计数上 CB 权重比恒为 **1.0000**（`1−β^n` 已饱和），配了等于没配且不报错；
  要加权用 `inverse_frequency_weights`（1.3162:1）。② `label_smoothing` 会把
  `P(T=1)` 上界**精确钳死在 `1−ls/2`**（实测 ls=0.1 → 0.9500），而官方 `T_conf`
  读的正是这个数，故默认 0。③ 头内**禁用 `BatchNorm1d`**（train 模式 batch=1 直接
  `ValueError`，末批可能只剩 1 样本）；需要归一化用 `LayerNorm`。
  🔴 **`T_conf` = P(T=1)**，不是"对所报 T 值的置信度" —— 与
  `Malignancy.confidence`（`predicted = conf >= 阈值` 推出的"恶性概率"）同一语义。

**接入接口现状（阶段 2.0 已补全，2026-08-12）**：
  ① Stage-4 空间特征图 —— ✅ `outputs["stage4_spatial"]`，512² 输入下实测 **[B,352,16,16]**。
     ⚠️ 通道数是 `CLS_POOLED_DIM`(352) 而非 `CLS_STAGE4_CHANNELS`(208)——后者是
     `adaptor_stage4` 的**输入**通道，`blocks[6]` 再把 208→352。位置头 `in_channels` 取 352。
     该张量**有梯度**（位置头回流 `cls_backbone` 是设计意图，供调度阶段 B 的 LLRD 解冻）。
  ② mask 概率图 —— ✅ `outputs["soft_mask_detached"]` [B,1,H,W]，**已 detach**。
     原 `soft_mask → prior_extractor` 通路有意不 detach（见 `training/stages.py` 的
     `compute_gamma_prior` docstring），此处是另发的副本，
     不影响物理先验通路，也不破坏 `gate8` 的精确零断言前提。
  ③ CBM `logits_margin` [B,4] —— ✅ 现成（`outputs["margin"]`）。
  ④ 结节尺度标量 —— ✅ `outputs["nodule_scale_detached"]` [B,1]，**已 detach**，值域 [0,1]。
     语义是**归一化等效直径** `sqrt(前景面积/图像面积)`，作为架构方案所称「等效体积标量」的
     尺度代理：T 分期临床判据是最大径（T1≤2cm/T2 2-4cm/T3>4cm），线性尺度直接对应；
     用面积^1.5 会把典型结节（面积占比~1%）压到 1e-3 量级造成输入数值塌陷。

**标签构造铁律（Plan M3）**：位置/ETE 标签构造与 7 维概率合成
  MUST `import src.models.label_schema` 的 `LOCATION_LABELS` / `_ETE_LABEL_TO_CODE`，
  否则概率分布与官方类别系统性错位、JSON 合法但指标偏低、本地永远查不出根因。
  ⚠️ 2026-08-12 起该常量由 `src/inference/prediction_json.py` **下沉到
  `src/models/label_schema.py`**（`.importlinter` 契约2：models 是最底层，不得依赖
  inference）；`prediction_json` 仍 re-export，交付侧代码无需改动。

新增头类请从本 `__init__` re-export，保持向后兼容。
"""

from src.models.heads.location import (  # noqa: F401  (re-export)
    LocationHead,
    LocationOutputs,
    build_level_validity,
    compose_location_probs,
    flip_location_index,
)
from src.models.heads.extension import (  # noqa: F401  (re-export)
    ETE_IDX_TO_CODE,
    NUM_ETE_CLASSES,
    ExtensionHead,
    ExtensionOutputs,
)
from src.models.heads.clinical_t import (  # noqa: F401  (re-export)
    NUM_CLIN_T_CLASSES,
    T_POSITIVE_INDEX,
    ClinicalTHead,
    ClinicalTOutputs,
)

__all__ = [
    "LocationHead",
    "LocationOutputs",
    "compose_location_probs",
    "build_level_validity",
    "flip_location_index",
    "ExtensionHead",
    "ExtensionOutputs",
    "NUM_ETE_CLASSES",
    "ETE_IDX_TO_CODE",
    "ClinicalTHead",
    "ClinicalTOutputs",
    "NUM_CLIN_T_CLASSES",
    "T_POSITIVE_INDEX",
]

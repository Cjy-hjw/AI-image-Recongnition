"""官方新增三维度的增量诊断头（Plan v2 阶段2 填充，见 `docs/训练实现计划.md`）。

三头独立成模块置于本子包，使 `multi_task_model.py` 保持薄装配枢纽（rules.md §6）：

- **位置头** `lesion_loc`（7 类，侧别×纵向因子分解 `P(k∈1..6)=P(Side)×P(Level)`、
  `P(7峡部)=P(Side=峡)`）——最高优先级；接 EfficientNet Stage-4 空间特征图 + CoordConv。
- **侵犯头** `extrathy_ext`/ETE（保 4 类，血管仅5例不并类）——ConvNeXt mask 概率图
  膨胀-腐蚀出交界带 × 局部特征 + 复用 CBM `logits_margin` 旁路。
- **临床T头** `clin_t_stage`（二分类）——全局 Pooling + 结节等效体积标量 + ETE logits →
  轻量 MLP；影像学上限约 0.72（淋巴结常不在视野）。

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
  MUST `import src.inference.prediction_json` 的 `LOCATION_LABELS` / `_ETE_LABEL_TO_CODE`，
  否则概率分布与官方类别系统性错位、JSON 合法但指标偏低、本地永远查不出根因。

新增头类请从本 `__init__` re-export，保持向后兼容。
"""

__all__ = []  # type: ignore[var-annotated]

"""官方新增三维度的增量诊断头（Plan v2 阶段2 填充，见 `docs/训练实现计划.md`）。

三头独立成模块置于本子包，使 `multi_task_model.py` 保持薄装配枢纽（rules.md §6）：

- **位置头** `lesion_loc`（7 类，侧别×纵向因子分解 `P(k∈1..6)=P(Side)×P(Level)`、
  `P(7峡部)=P(Side=峡)`）——最高优先级；接 EfficientNet Stage-4 空间特征图 + CoordConv。
- **侵犯头** `extrathy_ext`/ETE（保 4 类，血管仅5例不并类）——ConvNeXt mask 概率图
  膨胀-腐蚀出交界带 × 局部特征 + 复用 CBM `logits_margin` 旁路。
- **临床T头** `clin_t_stage`（二分类）——全局 Pooling + 结节等效体积标量 + ETE logits →
  轻量 MLP；影像学上限约 0.72（淋巴结常不在视野）。

**接入接口现状（阶段0.1 核验，Plan §0.1）**：
  ① Stage-4 空间特征图 —— 缺，`_forward_cls_backbone` 现仅返回 pool 后 [B,352]，须暴露 pool 前张量；
  ② mask 概率图 `soft_mask` —— `forward` 内已算未暴露，须传入侵犯头；
  ③ CBM `logits_margin` [B,4] —— 现成（outputs["margin"]）；
  ④ 结节等效体积标量 —— 缺，须从 `soft_mask.sum` 导出。

**标签构造铁律（Plan M3）**：位置/ETE 标签构造与 7 维概率合成
  MUST `import src.inference.prediction_json` 的 `LOCATION_LABELS` / `_ETE_LABEL_TO_CODE`，
  否则概率分布与官方类别系统性错位、JSON 合法但指标偏低、本地永远查不出根因。

新增头类请从本 `__init__` re-export，保持向后兼容。
"""

__all__ = []  # type: ignore[var-annotated]

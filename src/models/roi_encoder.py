"""ROI 级前向编码器(两阶段 ROI 范式 · 波3 模型面,2026-09-25)。

## 定位

第二阶段把第一阶段分割器给出的每个候选病灶,在**原分辨率**裁 ROI、resize 到
固定尺寸,然后喂给**同一个 `cls_backbone` 实例**做 ROI 级前向。本模块只负责
"张量已经裁好"之后的前向;裁剪(几何/坐标)属于数据面(`src/dataset/roi_dataset.py`
的 `crop_roi`)或推理侧(`nodule_split.crop_nodule`),不在这里重复。

## 🔴 与 `_forward_cls_backbone` 逐位同源(头号设计约束)

现有四头消费的 `stage4_spatial` 来自 `multi_task_model._forward_cls_backbone`:
conv_stem → bn1 → blocks[0..3] → adaptor_stage3 → blocks[4..5] → adaptor_stage4
→ blocks[6]。本模块的 `forward_roi_features` 调**同一模型实例**上的**同一批模块**、
按**同一顺序**、传**同一形态的 seg_features**——这是 loc 头权重可零训练搬运的
数学前提(scoping §C.1:CoordConv 坐标通道是张量 H/W 的函数,ROI 特征取自全局
特征图对应位置 + 坐标通道按 ROI 在原图中的位置生成 → 数学恒等):
**同一 backbone 实例调用,不是复制权重**。复制权重会在训练中漂移成两套表征,
头权重的搬运前提当场作废。

## seg_features 从哪来

ROI 裁剪发生在原图域,而 `adaptor_stage3/4` 需要**与 ROI 特征同分辨率**的分割
侧特征(seg_features[1]/[2])。两条路:

1. `seg_features` 显式传入(推理/训练第二阶段,分割器已在同一 ROI 上跑过——
   推理侧 ROI pipeline 会先精修 mask 再分类,天然有);
2. 传 `None` → 对 ROI 图像单独跑一遍 `seg_backbone` 提取(数据面单 ROI 训练
   场景:RoiDataset 每样本只载 ROI 图,分割主干此时是"特征提货器"而非监督对象,
   `.detach()` 保证分类梯度不回流——与整图链路 adaptor 对 seg_features 的
   detach 契约同源,`gate8_adaptor_isolation` 的精确零断言前提不破)。

## 坐标通道(全局位置)如何进头

`RoiLocationHead`(roi_heads.py)需要"ROI 在原图中的位置"。本模块不掺合坐标——
前向只产特征图;坐标通道由头按 `roi_meta` 的 bbox/orig 生成(单一职责,
且 loc 头搬运对拍的退化测试需要"坐标生成逻辑可独立单测")。
"""

import torch
import torch.nn as nn

__all__ = ["RoiFeatureOutputs", "forward_roi_features", "RoiEncoder"]


class RoiFeatureOutputs(dict):
    """ROI 级前向输出。dict 子类,与 `ModelOutputs`/`LocationOutputs` 风格一致。

    键:
      - `pooled` [K, CLS_POOLED_DIM] —— ROI 全局池化向量(与整图 `x_cls` 同义,
        但按 ROI 计算;CBM 概念头/良恶性头消费)。
      - `spatial` [K, CLS_POOLED_DIM, h, w] —— pool 前的 ROI 空间特征图,
        🔴 **未 detach**(ROI 头梯度回流 cls_backbone 是设计意图,与整图链路
        `stage4_spatial` 同款;`gate8` 前提由 seg_features 侧 detach 保住)。
      - `seg_features` list —— 本 ROI 前向实际消费的分割侧特征(缓存给
        ETE band 等下游消费,batch 级一次计算多病灶复用)。
    """


def forward_roi_features(
    model: nn.Module,
    roi_images: torch.Tensor,
    seg_features=None,
) -> RoiFeatureOutputs:
    """在 `model` 的**同一 cls_backbone 实例**上对 ROI batch 做前向。

    逐语句与 `MultiTaskThyroidModel._forward_cls_backbone` 同构(见模块 docstring
    的"逐位同源"约束);唯一差异是 seg_features 的来源(传入 or 现算,见模块
    docstring 第二节),且现算路对 seg_features **显式 detach**。

    Args:
        model: `MultiTaskThyroidModel` 实例(读它的 cls_backbone/adaptor/pool)。
        roi_images: [K, 1, S, S] ROI 图像 batch(K = 病灶数,S = roi_size,
            如 224;数据面 RoiDataset 产出即此形态)。
        seg_features: None 或分割主干特征 list(与 `_forward_cls_backbone`
            的 `seg_features[1]`/`[2]` 同构)。传入时视为**调用方已 detach**
            (整图链路在模型内部 detach;这里不重复——重复 detach 无害但掩盖
            契约,由调用方一处负责,报错指向更准)。

    Returns:
        `RoiFeatureOutputs`(见类 docstring)。

    Raises:
        AssertionError: 通道维非 1(数据面契约 [K,1,S,S],多通道是接线错误)。
    """
    assert roi_images.dim() == 4 and roi_images.shape[1] == 1, (
        f"forward_roi_features 期望 [K,1,S,S],got {tuple(roi_images.shape)}"
        " —— RoiDataset 产出即此形态,多通道/缺通道是接线错误"
    )

    if seg_features is None:
        # 数据面单 ROI 场景:现算分割特征。🔴 detach 与整图链路同款:
        # adaptor 在模型内部本就对 seg_features detach(见 _forward_cls_backbone
        # 调用点),这里再发一份已 detach 的副本,保证"分割主干当特征提货器"
        # 时分类梯度到不了它(gate8 精确零断言前提)。
        seg_features = model.seg_backbone(roi_images)
        seg_features = [f.detach() for f in seg_features]

    K = roi_images.shape[0]
    x_cls = model.cls_backbone.conv_stem(roi_images)
    x_cls = model.cls_backbone.bn1(x_cls)
    x_cls = model.cls_backbone.blocks[0](x_cls)
    x_cls = model.cls_backbone.blocks[1](x_cls)
    x_cls = model.cls_backbone.blocks[2](x_cls)
    x_cls = model.cls_backbone.blocks[3](x_cls)
    x_cls = model.adaptor_stage3(seg_features[1], x_cls)
    x_cls = model.cls_backbone.blocks[4](x_cls)
    x_cls = model.cls_backbone.blocks[5](x_cls)
    x_cls = model.adaptor_stage4(seg_features[2], x_cls)
    x_cls = model.cls_backbone.blocks[6](x_cls)
    pooled = model.cls_pool(x_cls).view(K, -1)
    return RoiFeatureOutputs(pooled=pooled, spatial=x_cls, seg_features=seg_features)


class RoiEncoder(nn.Module):
    """薄包装:把 `forward_roi_features` 挂成 nn.Module,便于挂 device/eval。

    🔴 **不持有任何参数**——forward 直接委托模型实例的原生模块。若给本类加
    参数,ROI 路径就脱离了"同一 backbone 实例"约束(scoping §A.1.1 #1 的
    头号风险),测试 `test_roi_uses_same_backbone_instance` 会红。

    Args:
        model: 宿主 `MultiTaskThyroidModel`(弱引用语义:本类不 copy 权重,
            forward 每次实时读 `model.seg_backbone`/`model.cls_backbone`)。
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, roi_images: torch.Tensor, seg_features=None) -> RoiFeatureOutputs:
        return forward_roi_features(self.model, roi_images, seg_features)

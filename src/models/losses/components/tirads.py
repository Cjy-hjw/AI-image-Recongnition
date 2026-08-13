"""TI-RADS 任务损失 —— masked 加权 CE + 可选二次有序项(backlog #124)。

## 类权重:复用 `clinical_t.inverse_frequency_weights`,不重抄

TI-RADS 5 类的真实分布未知(禁本地、待上云实测),但复用现成的
`inverse_frequency_weights`(逆频率,归一化到均值 1)而非另起一份 —— 5 类下是否
存在类似 ETE 的极端长尾(如某一级 n 个位数)同样未知,`class_balanced_weights`
与逆频率哪个适用要等真实计数出来再定,配置层通过 `tirads_use_class_weights`
开关切换(见 `scripts/train_official.py::build_official_head_config`)。

## 🔴 可选有序项默认关闭(`tirads_ordinal_weight=0.0`)

官方指标是**二次加权 Kappa**(评审规则:「TI-RADS加权Kappa」),对预测与真值差
2 级的惩罚是差 1 级的 4 倍(quadratic weights 的定义)。纯 CE 把所有误分类
一视同仁,不天然对齐这个目标。`L_ord = (E[k] − y)²`(E[k] = Σ_k p_k·k,期望
类别指数)是这个目标的一个可微代理 —— 预测分布集中在真值附近时 loss 小,
分布跑到远端时 loss 大,方向与二次 Kappa 一致。

**为什么默认关**:没有真实数据无法调参(`rules.md` §5「先证后改」),盲目开一个
未经验证的正则项风险大于收益;但代码+单测就位,上云拿到真实分布后这是第一个
该试的旋钮。全 0 权重时该项对 loss 精确无贡献(乘 0),不会静默改变现有行为。
"""

import torch
import torch.nn.functional as F
from typing import Optional

from src.models.losses.components.clinical_t import inverse_frequency_weights

__all__ = [
    "inverse_frequency_weights",  # re-export:TI-RADS 复用同一份类权重工具
    "compute_tirads_loss",
]

_EPS = 1e-8


def compute_tirads_loss(
    tirads_logits: torch.Tensor,
    tirads_idx: torch.Tensor,
    tirads_validity: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    ordinal_weight: float = 0.0,
):
    """TI-RADS 的 masked 加权 CE(+ 可选二次有序项)。

    Args:
        tirads_logits: [B,5] `TiradsHead` 的原始 logits。
        tirads_idx: [B] long,标签 ∈ {0..4}(官方码 1..5 的 0-based,`code-1`)。
            无效样本处值任意。
        tirads_validity: [B] float,1 = 该样本有 TI-RADS 标注。
        class_weights: [5] 类权重,None → 不加权。见 `inverse_frequency_weights`。
        label_smoothing: 默认 0(与 T 头同一考量,平滑会压缩输出置信度)。
        ordinal_weight: 默认 **0**(关闭)。>0 时叠加
            `L_ord = ((Σ_k p_k·k) − y)²`(masked 均值),对齐官方二次加权 Kappa
            的"差得越远罚得越重"语义,详见模块 docstring。
    Returns:
        (total, detail)。`detail["tirads_ce"]` 是未加权、未平滑的纯 CE;
        `detail["tirads_ordinal"]` 是未乘权重的原始有序项——**仅当
        `ordinal_weight>0` 时才计算**,权重=0(默认)时恒为 0 且不建图
        (Gate2 审查发现:此前无条件计算,权重=0 路径白白多算一次
        softmax+期望值,已按建议门控)。

    与项目既有 masked loss 同范式:`(loss * v).sum() / (v.sum() + eps)`。整批无
    有效样本时分子恒 0、分母 eps → 损失精确 0.0、梯度精确 0.0、无 NaN。
    """
    # 🔴 validity 必须 1-D(同 T/ETE 头同款防护,QA_VP §1.5/§2.2):[B,1] 会让
    # `per_sample[B] * v[B,1]` 隐式广播成 [B,B] 且不报错,损失被静默放大。
    assert tirads_validity.dim() == 1, (
        f"tirads_validity 必须是 [B] 一维,got {tuple(tirads_validity.shape)}"
        " —— [B,1] 会隐式广播成 [B,B] 静默放大 loss(QA_VP §2.2)"
    )
    assert tirads_idx.shape == tirads_validity.shape, (
        f"标签与 validity 形状不一致: {tuple(tirads_idx.shape)} vs "
        f"{tuple(tirads_validity.shape)}"
    )
    num_classes = tirads_logits.shape[-1]
    # 越界正负不对称(同 ETE/T):idx=5 会 IndexError 当场崩,idx=-1 走 python
    # 负索引**静默**取到末元素(恰是"V 级")——无标注样本莫名变成一批假 V 级监督。
    assert int(tirads_idx.min()) >= 0 and int(tirads_idx.max()) < num_classes, (
        f"TI-RADS 标签越界: min={int(tirads_idx.min())}, max={int(tirads_idx.max())},"
        f" 合法区间 [0,{num_classes - 1}](无标注样本应填占位 0 + validity=0)"
    )
    assert 0.0 <= label_smoothing < 1.0, (
        f"label_smoothing 必须 ∈ [0,1),got {label_smoothing}"
    )
    assert ordinal_weight >= 0.0, f"ordinal_weight 必须 >= 0,got {ordinal_weight}"
    if class_weights is not None:
        assert class_weights.shape == (num_classes,), (
            f"class_weights 形状应为 [{num_classes}],got {tuple(class_weights.shape)}"
        )

    # fp32:autocast 下 logits 可能是 bf16/fp16,交叉熵/期望值计算在低精度上易丢精度。
    logits = tirads_logits.float()
    w = None if class_weights is None else class_weights.to(
        device=logits.device, dtype=logits.dtype
    )
    v = tirads_validity.to(logits.dtype)

    per_sample = F.cross_entropy(
        logits, tirads_idx, weight=w, label_smoothing=label_smoothing, reduction="none"
    )
    ce_total = (per_sample * v).sum() / (v.sum() + _EPS)

    # 纯 CE 对照:不加权、不平滑。detach 后只作监控,不进反传(同 ETE/T 损失的 *_ce)。
    ce_plain = F.cross_entropy(logits, tirads_idx, reduction="none")
    ce_plain = (ce_plain * v).sum() / (v.sum() + _EPS)

    # 二次有序项:E[k] = Σ_k p_k·k(k=0..4,与 0-based 标签同一刻度)。
    # 🔴 权重=0(默认)时整段跳过,不建 softmax+期望值的计算图(Gate2 审查:
    # 此前无条件计算,白白多算一次前向,权重=0 路径应该"不算"而不是"算了但乘0"。
    if ordinal_weight > 0.0:
        probs = torch.softmax(logits, dim=-1)
        class_idx = torch.arange(num_classes, device=logits.device, dtype=logits.dtype)
        expected_idx = (probs * class_idx).sum(dim=-1)
        ordinal_per_sample = (expected_idx - tirads_idx.to(logits.dtype)) ** 2
        ordinal_total = (ordinal_per_sample * v).sum() / (v.sum() + _EPS)
    else:
        ordinal_total = torch.zeros((), device=logits.device, dtype=logits.dtype)

    total = ce_total + ordinal_weight * ordinal_total
    return total, {
        "tirads": total.detach(),
        "tirads_ce": ce_plain.detach(),
        "tirads_ordinal": ordinal_total.detach(),
    }

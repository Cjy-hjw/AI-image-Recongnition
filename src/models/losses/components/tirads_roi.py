"""ROI(per-lesion)TI-RADS 损失 —— 两阶段 ROI 范式损失组(方案一 P3,2026-09-25)。

## 🔴 与整图版的关键差异:吃 `tirads_probs`,不吃 logits

整图版(`components/tirads.py`)吃 `tirads_logits` [B,5](`TiradsHead` 原始
logits)。ROI 版契约钉死的输出键是 `tirads_probs` [K,5](softmax 后联合
概率;collect_roi_records 记录契约同键)——与 clinical_t_roi.py 同型:从
**概率**出发算 CE:

    L = −log(tirads_probs[:, y].clamp(min=eps))

与整图版 logits→`F.cross_entropy` 的期望值一致,数值路径不同(手动 clamp+log
防 softmax 饱和端的精确 0)。梯度经 probs 回流 ROI 头。

## "加权 Kappa 导向"的保留口径

整图版的 Kappa 对齐有两层:①CE 主项(逐样本似然);②可选二次有序项
`L_ord = ((Σ_k p_k·k) − y)²`(`tirads_ordinal_weight`,**默认 0 关闭**——
没有真实数据无法调参,rules §5 先证后改)。ROI 版**同式保留**这项、默认同
0:官方指标是二次加权 Kappa(差 2 级罚 4 倍),纯 CE 不天然对齐;但权重
标定须等真实数据,0 时对 loss 精确无贡献(乘 0)。

## 与整图版的口径对齐(勿漂移成两把尺子)

- **标签语义不变**:0-based 5 类(官方码 1..5 减一)。
- **is_main 门**:与整图版 train_step 同判据(v × main_w)。TI-RADS 官方
  口径**不带恶性限定**(与 ETE/T 不同、与位置头相同,评审规则原文无
  "且为恶性"),监督吃全部主病灶。
- **label_smoothing 默认 0**:整图版同一考量(平滑压缩输出置信度;TI-RADS
  概率要进交付)。参数保留。
- **类权重**:复用 `inverse_frequency_weights`(归一化均值 1)语义,
  None → 不加权。5 类是否长尾待真实计数,与整图版同留配置开关。
- **归一范式**:`(loss * v).sum() / (v.sum() + eps)`。
"""

import torch
from typing import Optional

__all__ = ["compute_tirads_roi_loss"]

_EPS = 1e-8
# k_ref 分母平滑(债6 同批,2026-09-25,DR 机制 1):语义见 location_roi._K_REF。
_K_REF = 4.0


def compute_tirads_roi_loss(
    tirads_probs: torch.Tensor,
    tirads_idx: torch.Tensor,
    tirads_validity: torch.Tensor,
    is_main: torch.Tensor = None,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    ordinal_weight: float = 0.0,
):
    """ROI TI-RADS 的 masked 加权 CE(概率域)+ 可选二次有序项。

    Args:
        tirads_probs: [K,5] ROI 头契约输出的 softmax 概率(和为 1;
            饱和端精确 0 由内部 clamp 防护)。
        tirads_idx: [K] long,标签 ∈ {0..4}(官方码 1..5 的 0-based)。
            无效样本处值任意。
        tirads_validity: [K] float,1 = 该病灶有 TI-RADS 标注。
        is_main: [K] float/int,1 = 主病灶。None → 全当主病灶。
        class_weights: [5] 类权重,None → 不加权。
        label_smoothing: 默认 0(与整图版同一考量:平滑压缩交付概率)。
        ordinal_weight: 默认 **0**(关闭)。>0 时叠加
            `L_ord = ((Σ_k p_k·k) − y)²`(masked 均值),对齐官方二次加权
            Kappa 的"差得越远罚得越重"语义。权重=0 时整段跳过不建图
            (整图版 Gate2 审查同款门控)。
    Returns:
        (total, detail)。detail 含 `roi_tirads`/`roi_tirads_ce`(纯 CE
        对照)/`roi_tirads_ordinal`(未乘权重的原始有序项,权重=0 时恒 0)。
        整批无有效样本 → 精确 0、梯度 0、无 NaN。
    """
    assert tirads_validity.dim() == 1, (
        f"tirads_validity 必须是 [K] 一维,got {tuple(tirads_validity.shape)}"
        " —— [K,1] 会隐式广播成 [K,K] 静默放大 loss(QA_VP §2.2)"
    )
    assert tirads_idx.shape == tirads_validity.shape, (
        f"标签与 validity 形状不一致: {tuple(tirads_idx.shape)} vs "
        f"{tuple(tirads_validity.shape)}"
    )
    num_classes = tirads_probs.shape[-1]
    assert tirads_probs.dim() == 2 and num_classes == 5, (
        f"tirads_probs 应为 [K,5],got {tuple(tirads_probs.shape)}"
        " —— TI-RADS 5 类(官方码 1..5 的 0-based)"
    )
    assert int(tirads_idx.min()) >= 0 and int(tirads_idx.max()) < num_classes, (
        f"TI-RADS 标签越界: min={int(tirads_idx.min())},"
        f" max={int(tirads_idx.max())}, 合法区间 [0,{num_classes - 1}]"
        "(无标注样本应填占位 0 + validity=0)"
    )
    assert 0.0 <= label_smoothing < 1.0, (
        f"label_smoothing 必须 ∈ [0,1),got {label_smoothing}"
    )
    assert ordinal_weight >= 0.0, f"ordinal_weight 必须 >= 0,got {ordinal_weight}"
    if class_weights is not None:
        assert class_weights.shape == (num_classes,), (
            f"class_weights 形状应为 [{num_classes}],got {tuple(class_weights.shape)}"
        )

    K = tirads_probs.shape[0]
    device = tirads_probs.device

    if is_main is None:
        main_w = torch.ones(K, dtype=torch.float32, device=device)
    else:
        assert is_main.shape == tirads_validity.shape, (
            f"is_main 与 validity 形状不一致: {tuple(is_main.shape)} vs "
            f"{tuple(tirads_validity.shape)}"
        )
        main_w = is_main.to(device=device, dtype=torch.float32)

    v = (tirads_validity.to(device) * main_w).float()

    # 概率域 CE:fp32 归约 + clamp 防 softmax 饱和端精确 0。
    probs = tirads_probs.float().clamp(min=_EPS)
    y = tirads_idx.to(device)

    if label_smoothing > 0.0:
        # 平滑目标:1-ls 给真类,ls/C 均匀(概率域直接可算,无 weight 交互
        # 问题——本实现不接 class_weights 与 LS 同时开启的场景断言,与整图
        # 版"两者同时开时上界不再 1-ls/2"的警示语义一致:此处 prob 域加权
        # 乘在 per_sample 上,同样推高平滑上界,消费侧别拿上界当护栏)。
        smooth_target = torch.full_like(probs, label_smoothing / num_classes)
        smooth_target.scatter_(
            1, y.unsqueeze(1), 1.0 - label_smoothing + label_smoothing / num_classes)
        per_sample = -(smooth_target * probs.log()).sum(dim=1)
    else:
        per_sample = -probs.log().gather(1, y.unsqueeze(1)).squeeze(1)

    if class_weights is not None:
        w = class_weights.to(probs.dtype).to(probs.device)
        per_sample = per_sample * w[y]

    denom = torch.clamp(v.sum() + _EPS, min=_K_REF)
    ce_total = (per_sample * v).sum() / denom

    # 纯 CE 对照(不加权、不平滑,detach 只作监控)。
    ce_plain_ps = -probs.log().gather(1, y.unsqueeze(1)).squeeze(1)
    ce_plain = (ce_plain_ps * v).sum() / denom

    # 二次有序项:E[k] = Σ_k p_k·k(k=0..4,与 0-based 标签同刻度)。
    # 🔴 权重=0(默认)整段跳过不建图(整图版 Gate2 审查同款门控)。
    if ordinal_weight > 0.0:
        class_idx = torch.arange(num_classes, device=device, dtype=probs.dtype)
        expected_idx = (probs * class_idx).sum(dim=1)
        ordinal_per_sample = (expected_idx - y.to(probs.dtype)) ** 2
        ordinal_total = (ordinal_per_sample * v).sum() / denom
    else:
        ordinal_total = torch.zeros((), device=device, dtype=probs.dtype)

    total = ce_total + ordinal_weight * ordinal_total
    return total, {
        "roi_tirads": total.detach(),
        "roi_tirads_ce": ce_plain.detach(),
        "roi_tirads_ordinal": ordinal_total.detach(),
    }

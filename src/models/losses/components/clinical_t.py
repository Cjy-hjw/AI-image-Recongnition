"""临床 T 任务损失 —— masked 加权 CE（Phase2 阶段 2.3）。

T 临床显著性是三头里**唯一不存在极端不平衡**的任务：`0无 2410 / 1有 1831`，
阳性率 **35.7%**（`docs/训练数据定稿.md:87`）。故这里既不用 Focal 也不用
Class-Balanced 加权，一个 masked CE 就够 —— 下面两节说明这不是偷懒。

## 🔴 为什么**不能**把侵犯头的 CB 权重抄过来（会静默失效）

`class_balanced_weights` 的有效样本数是 `E_n = (1 − β^n)/(1 − β)`。当 n 到千级时
`β^n` 已趋近 0，`E_n` 对所有类塌成同一个常数 `1/(1−β)` —— **权重比恒等于 1**。
实跑 `class_balanced_weights([2410, 1831], β)`：

| β | 权重比（阳 : 阴） | 说明 |
|---|---|---|
| 0.99（**侵犯头的默认值**） | **1.0000** | 🔴 完全不生效 |
| 0.999 | 1.0838 | 微弱 |
| 0.9999 | 1.2800 | 仍不及逆频率 |
| —（逆频率 2410/1831） | 1.3162 | 本任务真实的不平衡度 |

这个坑的恶劣之处在于**静默**：配置里明明写着 `cb_beta: 0.99`、代码也确实调了 CB，
权重张量却是 `[1.0, 1.0]` —— 不报错、不告警，只是白配。CB 是为**长尾**设计的
（ETE 最稀有类 n=5，那里 β=0.99 给出 20.4:1，确实有效），拿到 n 千级的均衡任务上
就退化成恒等变换。若真需要加权，用下面的 `inverse_frequency_weights`。

## 🔴 `label_smoothing` 默认 0 —— 它会直接钳死官方要读的 T_conf

LS 的最优解是 `1 − ls + ls/C`，二分类下即 `1 − ls/2`：模型**无论多确信都到不了**
这个上界。实测（400 步训到收敛的可完美分离数据）：

| label_smoothing | 阳性样本 P(T=1) 均值 | 阴性样本 P(T=1) 均值 |
|---|---|---|
| 0.0 | 1.0000 | 0.0000 |
| 0.05 | **0.9750** | 0.0250 |
| 0.1 | **0.9500** | 0.0500 |

官方 `T_ClinSig.T_conf` 读的正是 `P(T=1)`。开 LS 等于**系统性地把交付的置信度
往 0.5 拉**，且幅度完全可预测（不是噪声，是偏置）。

⚠️ **但这个上界只在 `class_weights is None` 时成立**（Gate2 查出，已实测复现）：
`F.cross_entropy` 的 `weight` 会按 `w[c]` 加权**整个平滑后的 soft target**，包括
`ls/C` 那个均匀项，于是加权会把上界推高 —— 同样 `ls=0.1` 下实测阳性 P(T=1)：
不加权 **0.9500**（符合 1−ls/2）、`w=[0.863,1.137]` → **0.9615**、
`w=[1,10]` → **0.9947**（几乎失效）。这是 PyTorch 的标准语义，不是 bug；
但**别拿"上界 1−ls/2"当成永远有效的护栏**。两者同时开时 T_conf 的分布需要
重新实测，`tests/test_clinical_t_head.py` 已把这个交互钉住。

这与"T 有影像学上限 0.72、标签本就带噪"的直觉相冲突 —— LS 确实是对付噪声标签的
标准手段。取舍结论是：**分类精度让位于概率校准**，因为官方明确要读这个置信度字段。
参数保留（有人拿到官方评测口径后可能判定精度更值钱），但默认关，且改之前请先想清楚
交付的 `T_conf` 会被整体压缩。
"""

import torch
import torch.nn.functional as F
from typing import Optional

__all__ = [
    "inverse_frequency_weights",
    "compute_clinical_t_loss",
]

_EPS = 1e-8


def inverse_frequency_weights(
    class_counts, device=None, dtype=torch.float32
) -> torch.Tensor:
    """逆频率类权重，归一化到**均值 1**。

    Args:
        class_counts: 长度 C 的每类样本数，**必须全 > 0**。
    Returns:
        [C] 权重，`w.mean() == 1`。实测 `[2410, 1831]` → **`[0.863476, 1.136524]`**
        （均值恰为 1.0，比值 1.316221 = 2410/1831，与逆频率定义一致）。

    与 `class_balanced_weights` 的分工：CB 适用于**长尾**（最稀有类 n 在个位数，
    如 ETE 的血管 5 例），本函数适用于**中度不平衡**（各类 n 都在千级）——
    见模块 docstring 的实测表，CB 在后者上会静默退化成全 1。

    归一化到均值 1（而非和为 1）与 `class_balanced_weights` 保持一致：否则开关加权
    会顺带缩放整个 T 损失的量级，与其它任务的相对权重被静默改掉。
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    assert counts.dim() == 1 and counts.numel() > 0, f"class_counts 形状非法: {counts.shape}"
    assert bool((counts > 0).all()), (
        f"class_counts 必须全 > 0，got {counts.tolist()}"
        " —— 计数为 0 的类权重会是 inf 并静默毒化损失"
    )
    w = counts.sum() / counts
    w = w / w.mean()
    return w.to(device=device, dtype=dtype)


def compute_clinical_t_loss(
    t_logits: torch.Tensor,
    t_idx: torch.Tensor,
    t_validity: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
):
    """临床 T 的 masked 加权 CE。

    Args:
        t_logits: [B,2] `ClinicalTHead` 的原始 logits。
        t_idx: [B] long，标签 ∈ {0,1}（官方 `T_ClinSig.T` 与 0-based 下标同值，
            不像 ETE/Location 需要 ±1 换算）。无效样本处值任意。
        t_validity: [B] float，1 = 该样本有 T 标注。
        class_weights: [2] 类权重，None → 不加权。见 `inverse_frequency_weights`，
            **不要用 `class_balanced_weights`**（模块 docstring 有实测原因）。
        label_smoothing: 默认 **0**。>0 会把 `P(T=1)` 上界钳到 `1 − ls/2`，
            直接压缩交付的 `T_conf`（模块 docstring 有实测表）。
    Returns:
        (total, detail)。`detail["clin_t_ce"]` 是**未加权、未平滑**的纯 CE，
        用于对照观察"是真学会了还是被权重推着走"（同 ETE 损失的 `ete_ce`）。

    与项目既有 masked loss 同范式：`(loss * v).sum() / (v.sum() + eps)`。整批无有效
    样本时分子恒 0、分母 eps → 损失精确 0.0、梯度精确 0.0、无 NaN（已实测）。

    **本任务刻意不加 Focal**：阳性率 35.7% 已相当均衡，Focal 在这里只会平白引入
    `γ ∈ (0,1) → NaN 梯度` 那类风险（见 `components/extension.py`）而无收益。
    若有人日后照侵犯头抄一份 Focal 过来，请先确认不平衡度是否真的变了。
    """
    # 🔴 validity 必须 1-D：[B,1] 会让 `per_sample[B] * v[B,1]` 隐式广播成 [B,B]
    # 且**不报错**，损失被静默放大（QA_VP §1.5/§2.2，位置头/侵犯头同款防护）。
    assert t_validity.dim() == 1, (
        f"t_validity 必须是 [B] 一维，got {tuple(t_validity.shape)}"
        " —— [B,1] 会隐式广播成 [B,B] 静默放大 loss（QA_VP §2.2）"
    )
    assert t_idx.shape == t_validity.shape, (
        f"标签与 validity 形状不一致: {tuple(t_idx.shape)} vs {tuple(t_validity.shape)}"
    )
    num_classes = t_logits.shape[-1]
    # 越界正负不对称（同 ETE）：idx=2 会 IndexError 当场崩，而 idx=-1 走 python
    # 负索引**静默**取到末元素（恰是"T 阳性"）—— 表现为无标注样本莫名变成一批
    # 假阳性监督，全程无任何报错。
    assert int(t_idx.min()) >= 0 and int(t_idx.max()) < num_classes, (
        f"临床 T 标签越界: min={int(t_idx.min())}, max={int(t_idx.max())}，"
        f"合法区间 [0,{num_classes - 1}]（无标注样本应填占位 0 + validity=0）"
    )
    assert 0.0 <= label_smoothing < 1.0, (
        f"label_smoothing 必须 ∈ [0,1)，got {label_smoothing}"
    )
    if class_weights is not None:
        assert class_weights.shape == (num_classes,), (
            f"class_weights 形状应为 [{num_classes}]，got {tuple(class_weights.shape)}"
        )

    # fp32：autocast 下 logits 可能是 bf16/fp16，交叉熵在低精度上会丢有效位。
    logits = t_logits.float()
    w = None if class_weights is None else class_weights.to(
        device=logits.device, dtype=logits.dtype
    )
    # reduction="none" 拿到逐样本损失才能做 masked 归约；此模式下 `weight` 的作用
    # 就是逐样本乘 `w[y]`，不涉及 F.cross_entropy 内部的加权归一化。
    per_sample = F.cross_entropy(
        logits, t_idx, weight=w, label_smoothing=label_smoothing, reduction="none"
    )
    v = t_validity.to(per_sample.dtype)
    total = (per_sample * v).sum() / (v.sum() + _EPS)

    # 纯 CE 对照：不加权、不平滑。detach 后只作监控，不进反传。
    ce_plain = F.cross_entropy(logits, t_idx, reduction="none")
    ce_plain = (ce_plain * v).sum() / (v.sum() + _EPS)
    return total, {"clin_t": total.detach(), "clin_t_ce": ce_plain.detach()}

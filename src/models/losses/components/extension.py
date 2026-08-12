"""侵犯任务（ETE）损失 —— Class-Balanced Focal + Logit Adjustment（Phase2 阶段 2.2）。

ETE 是全项目最极端的不平衡任务：`3906 / 267 / 67 / 5`（无 / 包膜 / 软组织 / 血管），
最稀有类占 **0.12%**。裸 CE 下模型只要恒输出"无侵犯"就能拿到 92% 准确率，而三个有
临床意义的类全部学不到。

## 两种校正手段（都实现了，但**默认只开一种**）

1. **Class-Balanced 加权**（Cui et al. 2019）：按"有效样本数"
   `E_n = (1 − β^n) / (1 − β)` 的倒数加权，再归一化到均值 1。
2. **Logit Adjustment**（Menon et al. 2021）：训练时把 `τ·log(先验)` **加到** logits 上，
   逼模型学出与类先验无关的平衡后验。

🔴 **二者是同一个问题的两种解法，同时开满会过校正**（Menon 原文即把 LA 作为重加权的
*替代*提出）。故 `la_tau` 默认 **0.0（关）**，先用 CB-Focal；`τ` 待官方验证集到手后再调
——官方数据只在云端，本地无从验证，此刻把两个都开满是拍脑袋。两个超参都在
`configs/thyroid_config.yaml: train.ete_*`，改配置即可切换，无需动代码。

## 🔴 β 不能照抄论文的 0.9999（实测算过再定）

论文常用 β=0.9999，但那是给 n 在千级的长尾集的。本任务最稀有类 **n=5**：

| β | 权重比（血管 : 无） | 判断 |
|---|---|---|
| 0.9999 | **646.9 : 1** | ❌ 5 张图吃掉整个损失，必然过拟合到这 5 张 |
| 0.999 | 196.4 : 1 | ❌ 同上，仍太陡 |
| 0.99 | **20.4 : 1** | ✅ 默认值 |
| 0.9 | 2.4 : 1 | 校正过弱，接近不加权 |
| 0.0 | 1 : 1 | 关闭加权 |

（上表是 `class_balanced_weights([3906,267,67,5], β)` 的**实跑结果**，不是估算；
`tests/test_extension_head.py` 钉死了 β=0.99 → 20.4 与 β=0.9999 → 646.9 这两个数。）
β=0.9999 在 n=5 时退化为近似**逆频率加权**，是文献里公认会崩的配置 —— 这正是
"抄论文默认值"最容易踩的坑。

## Logit Adjustment 的符号：加，不是减

训练损失用 `CE(logits + τ·log π, y)`。直觉是：给高频类**先送一份免费分数**，模型为了
拟合标签就必须把该类的原始 logit 压低，于是训练完的原始 logits 已是平衡后验。

**符号反了会放大不平衡而不是校正**，且损失照常下降、指标只是"偏低" —— 与本项目
已经踩过两次的同类坑同源（#113 FAMO 方向反转、#120 拉格朗日项符号）。故
`tests/test_extension_head.py` 用**方向断言**锁死：同一份不平衡数据训到收敛，
`la_tau=1` 必须让稀有类 logit **高于** `la_tau=0`，而不只是"有变化"（#119 教训：
只验有变化不验方向，正是 R6 漏网的根因）。

推理侧**不再做任何调整**，直接 `softmax(原始 logits)` —— 见 `heads/extension.py`
的 `ExtensionOutputs` docstring。
"""

import torch
import torch.nn.functional as F
from typing import Optional

__all__ = [
    "class_balanced_weights",
    "log_class_prior",
    "compute_extension_loss",
]

_EPS = 1e-8


def class_balanced_weights(
    class_counts, beta: float, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Class-Balanced 权重（Cui et al. 2019），归一化到**均值 1**。

    Args:
        class_counts: 长度 C 的每类样本数（可为 list/tuple/Tensor），**必须全 > 0**。
        beta: ∈ [0, 1)。0 → 全 1（不加权）；越接近 1 → 越偏袒稀有类。见模块 docstring
            的 β 对照表，**不要照抄论文的 0.9999**。
    Returns:
        [C] 权重，`w.mean() == 1`。

    归一化到均值 1（而非和为 1）是为了让**损失量级与不加权时可比** —— 否则开关 CB
    会顺带把整个 ETE 损失缩放 C 倍，与其它任务的相对权重被静默改掉，看起来像
    "CB 让 ETE 变重要了"，实则只是量纲变了。

    计算走 float64 是**给未来更大数据集的防御，当前 ETE 计数用不到**（实测：float32
    与 float64 在 `[3906,267,67,5]` 上结果逐位相同）。真正的分界在 **n ≈ 20000**：
    那时 float32 下两个大类的权重会塌成同一个数（实测 n=20000 时 float32 的
    `w[0]==w[1]`、float64 仍可区分）。留 float64 的成本是 0（每轮训练只该算一次），
    收益是换更大数据集时不用回头查这个静默失真。
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    assert counts.dim() == 1 and counts.numel() > 0, f"class_counts 形状非法: {counts.shape}"
    # n=0 的类会让 E_n=0、权重爆到无穷。这类样本训练里永不出现，权重本无意义，
    # 但一旦上游统计口径变了（比如换成"本 epoch 出现次数"）就会静默产出 inf 权重。
    assert bool((counts > 0).all()), (
        f"class_counts 必须全 > 0，got {counts.tolist()}"
        " —— 某类计数为 0 时有效样本数为 0，权重会变成 inf 并静默毒化损失"
    )
    assert 0.0 <= beta < 1.0, f"beta 必须 ∈ [0,1)，got {beta}"

    if beta == 0.0:
        w = torch.ones_like(counts)
    else:
        effective_num = (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), counts)) / (
            1.0 - beta
        )
        w = 1.0 / effective_num
    w = w / w.mean()
    return w.to(device=device, dtype=dtype)


def log_class_prior(class_counts, device=None, dtype=torch.float32) -> torch.Tensor:
    """[C] `log(n_c / Σn)`，供 Logit Adjustment 使用。

    不加平滑：`class_balanced_weights` 已断言计数全 > 0，log(0) 不可能发生。
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    assert bool((counts > 0).all()), f"class_counts 必须全 > 0，got {counts.tolist()}"
    return torch.log(counts / counts.sum()).to(device=device, dtype=dtype)


def compute_extension_loss(
    ete_logits: torch.Tensor,
    ete_idx: torch.Tensor,
    ete_validity: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    log_prior: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    la_tau: float = 0.0,
):
    """ETE 的 masked CB-Focal（可选叠加 Logit Adjustment）。

    Args:
        ete_logits: [B,C] `ExtensionHead` 的**原始** logits（未经任何不平衡校正）。
        ete_idx: [B] long，0-based 类别标签（= 官方输出码 − 1）。无效样本处值任意。
        ete_validity: [B] float，1 = 该样本有 ETE 标注。
        class_weights: [C] CB 权重，None → 不加权（等价全 1）。
        log_prior: [C] `log(先验)`，`la_tau > 0` 时必需。
        focal_gamma: Focal 的 γ。0 → 退化为普通加权 CE。
        la_tau: Logit Adjustment 的 τ。**0 = 关闭**（默认）。
    Returns:
        (total, detail)：detail 含 `ete_ce`（未加 focal/权重的纯 CE，便于对照观察
        "是真学会了还是只被权重推着走"）。

    与项目既有 masked loss 同范式：`(loss * v).sum() / (v.sum() + eps)`，分母只数有效
    样本。整批无有效样本 → 分子恒 0、分母 eps → 损失精确 0、梯度 0，不产生 NaN。
    """
    # 🔴 validity 必须 1-D。[B,1] 会让 `per_sample[B] * v[B,1]` 隐式广播成 [B,B]
    # 且**不报错**，损失被静默放大（QA_VP §1.5/§2.2，位置头同款防护）。
    assert ete_validity.dim() == 1, (
        f"ete_validity 必须是 [B] 一维，got {tuple(ete_validity.shape)}"
        " —— [B,1] 会隐式广播成 [B,B] 静默放大 loss（QA_VP §2.2）"
    )
    assert ete_idx.shape == ete_validity.shape, (
        f"标签与 validity 形状不一致: {tuple(ete_idx.shape)} vs {tuple(ete_validity.shape)}"
    )
    num_classes = ete_logits.shape[-1]
    # 🔴 γ ∈ (0,1) 会在 p_t 恰好 = 1.0 时产生 **NaN 梯度**（实测坐实）：
    # `d/dp (1-p)^γ = -γ(1-p)^(γ-1)`，γ<1 时指数为负 → (0)^负 = inf → NaN。
    # 而 p_t == 1.0 在 fp32 下**真的可达**：logit gap ≥ 50 时 softmax 就精确返回 1.0
    # （实测 gap=50/100/200 均 `pt.item() == 1.0`），训练中后期稀有类以外的样本
    # 很容易到这个区间。实测 γ=0.5 → loss=0.0 但 grad=[nan, nan]（loss 看着没事、
    # 梯度已经全毁），γ=1.0/2.0 → grad=[0.0, 0.0] 正常。故直接拦下这个取值区间。
    assert focal_gamma == 0.0 or focal_gamma >= 1.0, (
        f"focal_gamma 必须是 0（关闭）或 ≥1，got {focal_gamma}"
        " —— γ∈(0,1) 在 p_t=1.0 处梯度为 NaN，且 loss 值看不出异常"
    )
    # 越界正负不对称：idx=C 会 IndexError，idx=-1 走 python 负索引静默取到末元素
    # （恰是最稀有的"血管"类）—— 表现为"稀有类莫名多了一批假样本"，无任何报错。
    assert int(ete_idx.min()) >= 0 and int(ete_idx.max()) < num_classes, (
        f"ETE 标签越界: min={int(ete_idx.min())}, max={int(ete_idx.max())}，"
        f"合法区间 [0,{num_classes - 1}]（无标注样本应填占位 0 + validity=0）"
    )

    logits = ete_logits
    if la_tau != 0.0:
        assert log_prior is not None, "la_tau != 0 时必须提供 log_prior"
        assert log_prior.shape == (num_classes,), (
            f"log_prior 形状应为 [{num_classes}]，got {tuple(log_prior.shape)}"
        )
        # 🔴 **加**不是减。符号反了会放大不平衡，且损失照常下降（见模块 docstring）。
        logits = logits + la_tau * log_prior.to(logits.dtype).unsqueeze(0)

    log_probs = F.log_softmax(logits.float(), dim=-1)
    log_pt = log_probs.gather(1, ete_idx.unsqueeze(1)).squeeze(1)  # [B]
    pt = log_pt.exp()
    # Focal 调制项保持可导（标准实现如此）：pt 越接近 1，该样本贡献越小。
    focal = (1.0 - pt).pow(focal_gamma) if focal_gamma != 0.0 else torch.ones_like(pt)

    if class_weights is None:
        w_c = torch.ones_like(log_pt)
    else:
        assert class_weights.shape == (num_classes,), (
            f"class_weights 形状应为 [{num_classes}]，got {tuple(class_weights.shape)}"
        )
        # 🔴 按**真实标签**索引，不是按预测。按预测索引会让模型可以通过"少预测稀有类"
        # 自行降低损失权重 —— 与校正意图完全相反，且不会有任何报错。
        w_c = class_weights.to(log_pt.dtype).to(log_pt.device)[ete_idx]

    v = ete_validity.to(log_pt.dtype)
    per_sample = -w_c * focal * log_pt
    total = (per_sample * v).sum() / (v.sum() + _EPS)
    ce_plain = (-log_pt * v).sum() / (v.sum() + _EPS)
    return total, {"ete_focal": total.detach(), "ete_ce": ce_plain.detach()}

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

## 🔴 Soft-QWK：为什么加权 CE 这条路走不通（backlog #128，2026-08-17）

官方对 ETE 的评审指标是**二次加权 Kappa**，而 CB-Focal / 加权 CE 优化的是
**逐样本似然**。二者不只是"不完全一致"，在本任务的分布下是**负相关**的：

- 加权 CE 把"类0 判成类1"和"类0 判成类3"视为同等错误；
- 二次加权 Kappa 按 `(i−j)²` 惩罚，后者的代价是前者的 **9 倍**。

后果实测两次：β=0.99 那轮 `ete_kappa` 在 ep26 起**转负**（−0.027~−0.054，比常数
预测还差）；降到 β=0.9 后本轮拿到 0.2086，仍是五项里最低。DR
（`reference/2026-08月调研/极端不平衡四类有序标签下的加权 Kappa 优化与 ETE 预测策略研究报告.md`）
把加权 CE 排在推荐序列第 6（最差档），主方案是 **LA-SoftQWK-OD**。

`qwk_weight > 0` 时本函数改走该方案：

```
total = qwk_weight · SoftQWK(P, y) + dist_weight · OrdinalDistance(P, y)
```

- **SoftQWK**：用概率矩阵连续化混淆矩阵，`num/den` 恰是 `1 − QWK`，直接对齐评审指标。
  常数预测时 `O == E` → 损失恒 **1.0**（对应 QWK=0），这是最有区分力的验收判据。
- **OrdinalDistance**：`Σ_k p_k·(k−y)²/(C−1)²`，把不确定的概率质量往真值邻域收，
  给 SoftQWK 早期一个平滑的下降方向。

🔴 **LA 必须与 SoftQWK 同时开，不能单独开**（2026-08-17 实测，勿拆）：
bias-only 收敛实验显示 `la_tau` 0→0.5 让 gap(类0−类3) 从 6.661 降到 3.330、
**类3 的 softmax 概率从 0.00118 涨到 0.02505（×21）**。LA 是**推高**稀有类
（Menon 2020 的标准行为），而本项目已因过度预测稀有类让 Kappa 转负过一次。
DR 选 LA 的理由是它「仅调整决策边界，不改变梯度更新的相对距离标度」——
即它是为**不破坏 SoftQWK 的距离结构**而配的搭档，不是可单用的温和替代。
单开 LA = 只有推高、没有压制。

**默认 `qwk_weight=0.0`（关）**：此时逐位走 CB-Focal 老路径，零回归。
"""

import torch
import torch.nn.functional as F
from typing import Optional

__all__ = [
    "class_balanced_weights",
    "log_class_prior",
    "quadratic_weight_matrix",
    "soft_qwk_loss",
    "ordinal_distance_loss",
    "compute_extension_loss",
]

_EPS = 1e-8
# SoftQWK 的除法保护。**比 _EPS 大 5 个数量级是有意的**：`den` 是期望矩阵与权重
# 矩阵的内积，在常数预测下才接近 0；用 1e-8 会让"常数预测恒返回 1.0"这条验收判据
# 在 fp32 下抖到 0.97~1.03（实测），1e-7 与 DR 骨架一致且判据稳定到 1e-6。
_QWK_EPS = 1e-7


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


def quadratic_weight_matrix(
    num_classes: int, device=None, dtype=torch.float32
) -> torch.Tensor:
    """[C,C] 二次加权矩阵 `W[i,j] = (i-j)² / (C-1)²`，与 `sklearn.cohen_kappa_score(
    weights="quadratic")` 同一定义（`metrics_official._ete_weighted_kappa` 消费的
    就是这个权重方案）——SoftQWK 与训练侧验证指标口径必须一致，否则损失优化的
    目标和最终选模看的分数是两件事。
    """
    idx = torch.arange(num_classes, dtype=torch.float64)
    w = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2 / max(num_classes - 1, 1) ** 2
    return w.to(device=device, dtype=dtype)


def soft_qwk_loss(
    probs: torch.Tensor, idx: torch.Tensor, W: torch.Tensor
) -> torch.Tensor:
    """可微 SoftQWK 损失 = `1 − QWK` 的连续松弛（de la Torre et al. 2018）。

    Args:
        probs: [N,C] softmax 概率（已过 LA 调整的那份），**调用方须先按
            validity>0.5 过滤，只传有效样本**——本函数不接受 validity 参数、
            不做二次过滤，全部输入按"有效"处理（无效样本掺进 O/E 矩阵会静默
            扭曲混淆矩阵统计，且不报错，见 `compute_extension_loss` 的调用处）。
        idx: [N] long 真实标签，0-based。
        W: [C,C] 二次权重矩阵，见 `quadratic_weight_matrix`。

    Returns:
        标量。常数预测（所有概率质量压在同一类）时精确等于 **1.0**（QWK=0），
        这是 DR §6.4 给出的验收判据、也是本函数最强的区分力来源 —— 加权 CE 类
        损失在常数预测下不会恰好落在一个已知常数上，SoftQWK 会。

    `O`（连续混淆矩阵）与 `E`（独立性假设下的期望混淆矩阵）都按样本数 N 归一化，
    `num/den = Σ W⊙O / Σ W⊙E` 正是 `1-κ` 的定义（`κ = 1 - Σw·O / Σw·E`）。
    """
    n = probs.shape[0]
    one_hot = F.one_hot(idx, num_classes=probs.shape[-1]).to(probs.dtype)
    O = one_hot.t() @ probs / (n + _QWK_EPS)  # [C,C]
    hist_true = one_hot.sum(dim=0) / (n + _QWK_EPS)  # [C]
    hist_pred = probs.sum(dim=0) / (n + _QWK_EPS)  # [C]
    E = torch.outer(hist_true, hist_pred)  # [C,C]
    num = (W * O).sum()
    den = (W * E).sum() + _QWK_EPS
    return num / den


def ordinal_distance_loss(
    probs: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor:
    """样本级有序距离正则 `E[(k-y)²]/(C-1)²`（Polat et al. 2024），给 SoftQWK 一个
    平滑的早期下降方向 —— SoftQWK 的梯度在 batch 级混淆矩阵上计算，单样本梯度
    信号弱；本项对每个样本独立可导，收敛更快。

    Args:
        probs: [N,C] softmax 概率。
        idx: [N] long 真实标签，0-based。
    """
    num_classes = probs.shape[-1]
    class_idx = torch.arange(num_classes, device=probs.device, dtype=probs.dtype)
    dist_sq = (class_idx.unsqueeze(0) - idx.unsqueeze(1).to(probs.dtype)) ** 2
    dist_sq = dist_sq / max(num_classes - 1, 1) ** 2
    return (dist_sq * probs).sum(dim=1).mean()


def compute_extension_loss(
    ete_logits: torch.Tensor,
    ete_idx: torch.Tensor,
    ete_validity: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    log_prior: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    la_tau: float = 0.0,
    qwk_weight: float = 0.0,
    dist_weight: float = 0.2,
):
    """ETE 的 masked CB-Focal，或（`qwk_weight>0` 时）masked LA-SoftQWK-OD。

    Args:
        ete_logits: [B,C] `ExtensionHead` 的**原始** logits（未经任何不平衡校正）。
        ete_idx: [B] long，0-based 类别标签（= 官方输出码 − 1）。无效样本处值任意。
        ete_validity: [B] float，1 = 该样本有 ETE 标注。
        class_weights: [C] CB 权重，None → 不加权（等价全 1）。**仅 CB-Focal 分支用**。
        log_prior: [C] `log(先验)`，`la_tau > 0` 时必需。**两分支共用**（LA 是在
            logits 上先做的调整，之后再各走各的损失形式）。
        focal_gamma: Focal 的 γ。0 → 退化为普通加权 CE。**仅 CB-Focal 分支用**。
        la_tau: Logit Adjustment 的 τ。**0 = 关闭**（默认）。两分支共用。
        qwk_weight: SoftQWK 项权重。**0（默认）= 走 CB-Focal 老路径**，逐位零回归；
            >0 时切到 LA-SoftQWK-OD（见模块 docstring「Soft-QWK」节），此时
            `focal_gamma`/`class_weights` 不参与计算（DR `:269-270`「必须彻底禁用」）。
        dist_weight: 有序距离正则权重，仅 `qwk_weight>0` 时生效。
    Returns:
        (total, detail)。CB-Focal 分支：`detail` 含 `ete_focal`/`ete_ce`。
        SoftQWK 分支：`detail` 含 `ete_qwk`/`ete_dist`/`ete_ce`（`ete_ce` 是同一批
        有效样本上的纯 CE，供两种损失形式跨分支对照，键名与旧分支保持一致）。

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
    assert qwk_weight >= 0.0, f"qwk_weight 必须 >= 0，got {qwk_weight}"
    assert dist_weight >= 0.0, f"dist_weight 必须 >= 0，got {dist_weight}"
    # 🔴 **LA 与 SoftQWK 必须同开**（2026-08-17 实测，见模块 docstring「Soft-QWK」节）。
    # 只开 SoftQWK 不开 LA 会静默绕过一个已实测过的约束：SoftQWK 直接优化 Kappa，
    # 但没有 LA 的边界平移，稀有类（血管 n=5）的决策边界仍被先验压死；而 DR 的
    # 方案是两者互为一半（LA 推高边界 + SoftQWK 压住过度预测）。缺一半不报错、
    # 只是白跑一轮云端训练（一轮 ~38 分钟 + 人工上传窗口），故在此硬拦。
    # 反向（只开 LA 不开 SoftQWK）**不拦**：那是 CB-Focal 分支叠加 LA 的既有合法
    # 配置（`la_tau` 本就是 2026-08-12 就存在的独立超参），拦了会破坏向后兼容。
    if qwk_weight > 0.0:
        assert la_tau > 0.0, (
            f"qwk_weight={qwk_weight} > 0 时必须同时设 la_tau > 0，got la_tau={la_tau}"
            " —— LA 与 SoftQWK 是同一方案的两半（LA 平移决策边界、SoftQWK 压住"
            "过度预测稀有类），只开一半会重演 'Kappa 转负' 那次事故；"
            "DR 推荐 la_tau=0.5，见模块 docstring「Soft-QWK」节"
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

    if qwk_weight > 0.0:
        # 🔴 **先过滤再统计**，不能用 masked-mean 范式：SoftQWK 的 O/E 矩阵是对
        # "有效样本集合"的联合统计量，无效样本即便乘 0 权重也会占掉分母 N 的名额，
        # 悄悄拉低 hist_true/hist_pred（等价于把无效样本当成"预测概率为 0 的负样本"
        # 掺进混淆矩阵），比 CB-Focal 分支的逐样本掩码更容易出这种静默偏差。
        valid_mask = ete_validity > 0.5
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            zero = (ete_logits * 0.0).sum()
            return zero, {
                "ete_qwk": zero.detach(),
                "ete_dist": zero.detach(),
                "ete_ce": zero.detach(),
            }
        logits_v = logits[valid_mask].float()
        idx_v = ete_idx[valid_mask]
        probs_v = F.softmax(logits_v, dim=-1)
        W = quadratic_weight_matrix(num_classes, device=logits_v.device, dtype=logits_v.dtype)
        qwk_total = soft_qwk_loss(probs_v, idx_v, W)
        dist_total = ordinal_distance_loss(probs_v, idx_v)
        total = qwk_weight * qwk_total + dist_weight * dist_total
        ce_plain = F.cross_entropy(logits_v, idx_v, reduction="mean")
        return total, {
            "ete_qwk": qwk_total.detach(),
            "ete_dist": dist_total.detach(),
            "ete_ce": ce_plain.detach(),
        }

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

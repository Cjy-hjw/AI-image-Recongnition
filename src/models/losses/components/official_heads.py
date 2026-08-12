"""官方三维度头(位置/ETE/临床T)的损失编排 —— 固定 λ 外挂,**不进 FAMO**。

为什么不扩 FAMO 通道(5→8):
  ① `famo.py:92-95` 的 `load_state_dict` 校验 `w_logits.numel() == num_tasks`,
     改通道数直接**破坏既有 checkpoint 兼容**(混训 e40 底座加载即崩);
  ② FAMO 是按"各任务下降速率"动态配权,而三头里 ETE 极端长尾(3906/267/67/5)、
     位置有大量 invalid 样本 —— masked loss 在整批无有效样本时精确返回 0.0,
     FAMO 会把这个"0"当成"该任务已收敛"而持续压低其权重,方向恰好反了。
  固定 λ 外挂既保 checkpoint 兼容,也不让长尾任务的 0 值污染动态调度。

编排后的总损失:`total = famo_weighted + λ_loc·L_loc + λ_ete·L_ete + λ_t·L_t`,
插入点在 `apply_famo_weighting` 之后、`compute_regularizers` 之前(不让 λ 项被
L2/spin-L1 的正则语义裹挟)。

**legacy 路径零回归**:`targets` 缺三头键时逐项跳过,返回全 0 与空 detail,
`train.py` 的既有三阶段训练完全不受影响(它的 targets 从来没有这些键)。
"""
from typing import Dict, Optional, Tuple

import torch

from src.models.losses.components.clinical_t import compute_clinical_t_loss
from src.models.losses.components.extension import compute_extension_loss
from src.models.losses.components.location import compute_location_loss


def compute_official_head_losses(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    lambda_loc: float = 1.0,
    lambda_ete: float = 1.0,
    lambda_clin_t: float = 1.0,
    ete_class_weights: Optional[torch.Tensor] = None,
    ete_log_prior: Optional[torch.Tensor] = None,
    ete_focal_gamma: float = 2.0,
    ete_la_tau: float = 0.0,
    clin_t_class_weights: Optional[torch.Tensor] = None,
    clin_t_label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """三头损失加权求和。

    Args:
        outputs: 模型 forward 输出,需含 `location_side_logits`/`location_level_logits`
            /`ete_logits`/`t_logits`。
        targets: 需含 `lesion_loc`/`valid_lesion_loc`/`extrathy_ext`/
            `valid_extrathy_ext`/`clin_t_stage`/`valid_clin_t_stage`。
            🔴 **validity 走这些顶层直传键,不是 `targets["label_validity"]`** ——
            后者只被 `malignancy.py`/`concepts.py` 消费,三头损失根本不读它
            (各自显式接 [B] validity 参数)。传错地方 = 三头拿不到屏蔽、
            invalid 样本的垃圾标签当真监督,且不报错。
        lambda_*: 各头的固定权重。
        ete_*/clin_t_*: 透传给各自损失的不平衡校正参数,语义见对应模块 docstring。
    Returns:
        `(weighted_total, detail)`。detail 键:`location`/`location_side`/
        `location_level`/`ete`/`ete_ce`/`clin_t`/`clin_t_ce`,**均为未乘 λ 的原始
        损失值**(监控要看任务本身的量级,乘了 λ 就分不清"损失降了"和"权重小")。
        任一头的键不在 targets 里 → 该头整组 detail 键缺席、不计入 total。

    `weighted_total` 在三头全缺席时是 python `0.0`(不是 tensor)—— 调用方一律
    `total_loss = total_loss + head_total`,标量 0 与 tensor 相加安全且不建图。
    """
    total: torch.Tensor = 0.0
    detail: Dict[str, torch.Tensor] = {}

    if "lesion_loc" in targets:
        loc_total, loc_detail = compute_location_loss(
            outputs["location_side_logits"],
            outputs["location_level_logits"],
            targets["lesion_loc"].long(),
            targets["valid_lesion_loc"].float(),
        )
        total = total + lambda_loc * loc_total
        detail["location"] = loc_total
        detail.update(loc_detail)

    if "extrathy_ext" in targets:
        ete_total, ete_detail = compute_extension_loss(
            outputs["ete_logits"],
            targets["extrathy_ext"].long(),
            targets["valid_extrathy_ext"].float(),
            class_weights=ete_class_weights,
            log_prior=ete_log_prior,
            focal_gamma=ete_focal_gamma,
            la_tau=ete_la_tau,
        )
        total = total + lambda_ete * ete_total
        detail["ete"] = ete_total
        detail.update(ete_detail)

    if "clin_t_stage" in targets:
        t_total, t_detail = compute_clinical_t_loss(
            outputs["t_logits"],
            targets["clin_t_stage"].long(),
            targets["valid_clin_t_stage"].float(),
            class_weights=clin_t_class_weights,
            label_smoothing=clin_t_label_smoothing,
        )
        total = total + lambda_clin_t * t_total
        detail["clin_t"] = t_total
        detail.update(t_detail)

    return total, detail

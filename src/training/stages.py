import math
from typing import Dict, List, Tuple

import torch.nn as nn


def get_stage_number(epoch: int) -> int:
    """Determine the training stage based on epoch number.

    Returns:
        1 for epochs 1-40 (Seg+Concepts joint training)
        2 for epochs 41-100 (Full model fine-tuning)
        3 for epochs 101-120 (Classifier fine-tuning)
    """
    if epoch <= 40:
        return 1
    elif epoch <= 100:
        return 2
    else:
        return 3


def configure_stage_parameters(
    model: nn.Module,
    criterion: nn.Module,
    epoch: int,
    lr_base: float,
    weight_decay: float,
) -> Tuple[List[Dict], float, str]:
    """
    配置三阶段渐进式训练策略的参数。

    Stage 1 (Epoch 1-40): 分割 + 概念联合训练
    - 仅激活分割解码器和概念分类头 (proj_plus, proj_minus, s_layer)
    - 冻结: classifier_direct, latent_mlp, AUCM 对偶变量
    - Warmup (Epoch 1-5) + Cosine Annealing (Epoch 6-40)
    - FAMO 仅调度活跃任务

    Stage 2 (Epoch 41-100): 全模型解冻
    - 所有参数解冻，5 通道 FAMO 细粒度调度
    - Cosine Annealing 衰减

    Stage 3 (Epoch 101-120): 分类器微调
    - 冻结双主干、分割解码器、物理先验网络
    - 仅解冻 CBM 分类头 (classifier_direct, latent_mlp, proj_plus, proj_minus, s_layer)
    - 关闭 FAMO，直接优化 AUCMLoss

    Args:
        model: 模型实例
        criterion: 损失函数实例
        epoch: 当前 epoch
        lr_base: 基础学习率
        weight_decay: 权重衰减系数

    Returns:
        (param_groups, lr, stage_name)
    """
    # vCLUB 隔离：变分互信息估计器在所有阶段 requires_grad=True，由独立 vclub_optimizer 控制
    for param in model.vclub.parameters():
        param.requires_grad = True

    stage = get_stage_number(epoch)
    model.set_training_stage(stage)

    if stage == 1:
        for param in criterion.mal_loss_fn.parameters():
            param.requires_grad = False
    elif stage == 2:
        for param in criterion.mal_loss_fn.parameters():
            param.requires_grad = True
    else:
        for param in criterion.mal_loss_fn.parameters():
            param.requires_grad = True

    # ── 构建活跃参数组 ──
    # LoRA A/B 矩阵单独收集，使用固定 lr（不参与 LLRD 衰减）
    lora_param_ids = {
        id(p) for n, p in model.named_parameters()
        if ('lora_A' in n or 'lora_B' in n) and p.requires_grad
    }
    lora_params = [
        p for n, p in model.named_parameters()
        if ('lora_A' in n or 'lora_B' in n) and p.requires_grad
    ]

    active_decay = []
    active_no_decay = []

    for name, param in model.named_parameters():
        if "vclub" in name or not param.requires_grad:
            continue
        if id(param) in lora_param_ids:
            continue  # LoRA 参数由独立组管理
        if "bias" in name or "bn" in name or "norm" in name:
            active_no_decay.append(param)
        else:
            active_decay.append(param)

    # AUCM 对偶参数：Stage 1 排除（已冻结），Stage 2/3 合并至 no_decay 组，避免权重衰减拉扯拉格朗日乘子
    if stage >= 2:
        active_no_decay.extend(list(criterion.mal_loss_fn.parameters()))

    # ── 学习率调度 ──
    if stage == 1:
        if epoch <= 5:
            lr = lr_base * (epoch / 5.0)
            stage_name = "Stage 1: Seg+Concepts (Warmup)"
        else:
            progress = (epoch - 5) / (40 - 5)
            lr = lr_base * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))
            stage_name = "Stage 1: Seg+Concepts Joint Training"
    elif stage == 2:
        progress = (epoch - 40) / (100 - 40)
        lr = lr_base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        stage_name = "Stage 2: Full Model Fine-Tuning"
    else:  # stage == 3
        lr = 5e-6  # 微弱固定学习率，全力优化 AUCMLoss
        stage_name = "Stage 3: Classifier Fine-Tuning"

    param_groups = [
        {"params": active_decay, "weight_decay": weight_decay, "lr": lr},
        {"params": active_no_decay, "weight_decay": 0.0, "lr": lr},
    ]
    # LoRA A/B 矩阵固定 lr=1e-4，不随 LLRD/Cosine 衰减（Opus建议：独立组防lr冲突）
    if lora_params:
        param_groups.append({"params": lora_params, "weight_decay": 0.0, "lr": 1e-4})

    return param_groups, lr, stage_name

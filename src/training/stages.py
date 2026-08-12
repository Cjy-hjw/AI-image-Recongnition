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


def compute_gamma_prior(epoch: int) -> float:
    """物理先验通路的线性退火系数 γ_prior(epoch → [0,1])。

    `multi_task_model.forward(x, gamma_prior=γ)` 用它缩放 `e_prior = prior_mlp(v_phys) * γ`。
    该通路(`soft_mask → prior_extractor → prior_mlp → e_prior → cbm_head`,
    `multi_task_model.forward` 内 `soft_mask = torch.sigmoid(logits_mask)` **有意不 detach**
    (⚠️ 勿与 outputs 里供三头消费的 `soft_mask_detached` 混淆,后者是另发的 detach 副本),见
    `scripts/verify/gates/gate8_adaptor_isolation.py` 注释)是 `loss_mal → seg_backbone`
    的**唯一非 detach 路径**——γ 从 0 升到 1 等于把分类损失对分割主干的回传从关到全开。

    退火表:epoch ≤20 → 0(完全关闭);epoch 30 → 0.5;epoch ≥40 → 1.0(满强度)。

    ⚠️ **与 Stage2 边界的耦合(2026-08-11 崩塌事故根因之一)**:γ 在 epoch 40 恰好升满,
    而 Stage2 从 epoch 41 起同时触发 lr 100 倍跳变 + 全模型解冻 + `enable_mal_loss`
    首次开启——四件事挤在同一个 epoch,分割 3 个 epoch 内从 97.71% 塌到 80.6%。
    详见 `docs/项目现状总纲.md` §6。三头训练**另建调度**,勿复用本函数与
    `configure_stage_parameters` 的 epoch 划分。

    数值锁定见 `tests/test_stage_schedule.py::test_gamma_prior_annealing`。
    """
    return max(0.0, min(1.0, (epoch - 20) / 20.0))


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
    - **双主干 + 分割解码器 + 概念分类头 (proj_plus, proj_minus, s_layer) 均可训**
      —— 在 seg + 概念目标上学表征(非纯头 warmup)
    - 冻结: classifier_direct, latent_mlp (恶性直连头，因 mal loss 在 Stage 1 关闭
      `enable_mal_loss=(stage>1)`)、AUCM 对偶变量
    - Warmup (Epoch 1-5) + Cosine Annealing (Epoch 6-40)
    - FAMO 仅调度活跃任务
    - ⚠️ 审查 S6:`is_param_trainable(stage=1)` 只冻上面两个 CBM 子模块,主干可训,
      与本清单一致;勿据"仅激活分割头"误读为冻主干。**use_lora 模型例外**:
      `set_training_stage` 会把非冻子模块 `requires_grad=True`,重解冻 `__init__` 为
      LoRA 冻的基座骨干——use_lora 走独立训练脚本(train_thyroidxl 等),勿套此三阶段循环。
    - ⚠️ **"概念联合训练"在混训模式(`--mix-thyroidxl`)下名不副实**:该模式两源的概念
      标签**全部被屏蔽** —— TN3K 的 5 概念是 mask 反推的伪标签,由源级 `label_mask`
      压 0(否则伪标签当真监督污染概念头);ThyroidXL 根本没有概念标注,逐样本 meta
      亦写 0。于是每个样本的 concept valid 全 0,**概念分支参数虽 requires_grad=True
      但梯度恒 0,整个 Stage1 是纯分割训练**。这是有意为之(屏蔽是对的),但由此可推:
      混训产出的 checkpoint **不得**用于输出概念概率 / CBM 报告 —— 那些头从未受过监督,
      输出的是 confident 垃圾,而官方要求概率分布,等于送分变 0 分。
      见 `docs/交付与部署/交付前核对清单.md` 的同名红线。

    Stage 2 (Epoch 41-100): 全模型解冻
    - 所有参数解冻，5 通道 FAMO 细粒度调度
    - Cosine Annealing 衰减

    🔴 **Stage 2 已知缺陷 —— 三头训练禁止复用本函数(2026-08-11 实测事故)**
    本段调度自 2026-06-21(commit `ef5dfd8`)引入后**从未被真实执行过**;2026-08-11
    从混训 e40 权重续训是历史第一次上路,**3 个 epoch 内分割从 97.71% 塌到 80.6%**
    (train_seg 0.185→0.384、val_loss 7.92→17.39),已中止。根因是 Stage2 起点把四件
    高风险事挤在同一个 epoch:
      ① **lr 100 倍跳变**:Stage1 e40 = lr_base×0.01 = 2.40e-5 → Stage2 e41 =
         lr_base×0.99938 = 2.399e-3(且 train.py 在 stage 边界重建 AdamW,无动量承接);
      ② **全模型解冻**(本段 `is_param_trainable(stage=2)` 恒 True);
      ③ `enable_mal_loss=(stage>1)` 由 False→True,恶性损失首次参与;
      ④ `compute_gamma_prior` 恰在 e40 升满 1.0,物理先验通路(loss_mal→seg_backbone
         的唯一非 detach 路径)梯度由恒 0 变满强度。
    `reference/` 的 DR 报告(2026-07-09 / 2026-08-10,均**晚于**本代码)把"全参数端到端
    联合微调"判为**退化风险「高」**,推荐"冻头 → 只解冻 Top-2 层 + LLRD(主干 lr =
    新头的 1/10~1/100)"——该结论从未回填本文件,但**已写进 `docs/训练与架构/训练架构方案.md`**。
    → **官方数据三头训练按 `docs/训练与架构/训练架构方案.md` 另建 A/B/C 阶段调度
    (Ep1-10 冻主干温升新头 / Ep11-30 LLRD 解冻 Top-2 / Ep31-35 概率校准),
    不得复用本函数的 1-40/41-100/101-120 划分**;模型只能提交一次,踩在官方数据上无法补救。
    边界跳变的数值锁定见 `tests/test_stage_schedule.py`,事故全记录见
    `docs/项目现状总纲.md` §6。

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

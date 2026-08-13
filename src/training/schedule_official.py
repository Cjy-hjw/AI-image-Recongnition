"""官方数据三头训练的 A/B/C 阶段调度 —— **独立于 `stages.py` 的 legacy 三阶段**。

## 为什么另建,不复用 `configure_stage_parameters`

`stages.py` 的 Stage1→Stage2 边界有 **lr 100 倍跳变**的已知缺陷:2026-08-11 从混训
e40 权重续训 Stage2,3 个 epoch 内分割从 97.71% 塌到 80.6%(lr 暴涨 + 全模型解冻 +
mal loss 首启 + 物理先验满强度四件事同 epoch 齐发)。三头是随机初始化、只在官方数据
上从零学,一旦搭上那条崩塌调度,模型只能提交一次、踩上去无法补救。故按
`docs/训练与架构/训练架构方案.md` 另建 A/B/C:

  Stage A (epoch 1-10) :冻结 backbone/分割/CBM,**只训三头**;头 lr 线性 warmup 至 lr_base。
  Stage B (epoch 11-30):解冻 cls_backbone Top-2 block,LLRD(backbone lr = 头 lr ×
                         backbone_lr_scale,逐 block 再乘 η 衰减),余弦退火。
  Stage C (epoch 31-35):全冻结,仅头部低 lr 微调(温度标定放推理侧 calibration.py)。

## 硬约束

🔴 **边界 lr 连续性**:任意相邻 epoch 的 lr 比值 ≤ `MAX_LR_JUMP_RATIO`(=5)。把
   `stages.py` 的 100× 事故在数值上钉成不可能。数值锁定见 `tests/test_official_schedule.py`。
🔴 **BN 冻结**:冻结的子模块必须置 eval,否则"参数冻了但 running stats 仍更新"的数值
   病态(BN 冻结遗漏教训)。本模块**只设 requires_grad**;`set_frozen_modules_eval`
   由训练循环在 `model.train()` 之后调用(eval/train 是运行态,不属参数配置职责)。
   三头(位置/ETE/临床T)在所有官方 stage **全程可训**,其内含的 BatchNorm2d 始终
   正常训练;Stage B 解冻的是**整个** block(全参可训),不存在"子模块内部分冻结"的
   BN 病态。

## 返回契约

`configure_official_stage(...) -> (param_groups, lr, stage_name)` —— 与 `stages.py`
的 `configure_stage_parameters` **同签同返**,`train.py` 式的每 epoch 调用点可平移。
`lr` 是**头 lr**(调度基准);Stage B 的 backbone 组另取 `头 lr × scale × η`。
"""
import math
from typing import Dict, List, Tuple

import torch.nn as nn

from src.models.multi_task_model import MultiTaskThyroidModel

# ── 阶段边界(epoch,1-based,闭区间)────────────────────────────────────────
STAGE_A_END = 10   # A: 1..10
STAGE_B_END = 30   # B: 11..30
STAGE_C_END = 35   # C: 31..35(总 epoch 数)

# Stage B 解冻的 cls_backbone Top-2 block 前缀。EfficientNet-B2 features 分 blocks[0..6],
# 靠输出的两层 = blocks[5]/blocks[6](blocks[6] 把 208→352,最靠近 pooled 特征)。
# LLRD:越靠输出层 lr 越大 —— blocks[6] 取满 scale,blocks[5] 再乘 η。
STAGE_B_UNFREEZE = ("cls_backbone.blocks.6", "cls_backbone.blocks.5")

_HEAD_PREFIXES = MultiTaskThyroidModel.OFFICIAL_HEAD_PREFIXES

# 相邻 epoch lr 比值上限。stages.py 的 100× 崩塌根因钉成不可能。
MAX_LR_JUMP_RATIO = 5.0

# Stage B 余弦退火的下限系数(头 lr 从 lr_base 退到 lr_base × 此值)。
_STAGE_B_ETA_MIN = 0.1
# Stage C 固定低 lr 系数(头 lr = lr_base × 此值)。
_STAGE_C_LR_SCALE = 0.05


def get_official_stage(epoch: int) -> str:
    """epoch → 官方阶段名 'A'/'B'/'C'。epoch 从 1 起。"""
    if epoch <= STAGE_A_END:
        return "A"
    if epoch <= STAGE_B_END:
        return "B"
    return "C"


def official_head_lr(epoch: int, lr_base: float) -> float:
    """三头(调度基准)lr 曲线。**纯函数**,供测试独立对拍(不 import 到 configure)。

    - A(1-10) :线性 warmup,e1 = lr_base×0.1 … e10 = lr_base(相邻比值最大在 e1→e2 = 2.0)。
    - B(11-30):余弦退火 lr_base → lr_base×0.1(e11 起点 = lr_base,与 e10 衔接比值 = 1.0)。
    - C(31-35):固定 lr_base×0.05(e30→e31 比值 = 0.5)。

    全程相邻比值 ≤ 5(`test_official_schedule` 逐点核)。
    """
    stage = get_official_stage(epoch)
    if stage == "A":
        # e0 不存在;从 e1 = lr_base/10 起 warmup,避免"0 → 满"的无穷比值。
        return lr_base * epoch / STAGE_A_END
    if stage == "B":
        b_start = STAGE_A_END + 1               # 11
        span = STAGE_B_END - b_start            # 19(e11..e30)
        progress = (epoch - b_start) / span     # 0..1
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_base * (_STAGE_B_ETA_MIN + (1.0 - _STAGE_B_ETA_MIN) * cos)
    return lr_base * _STAGE_C_LR_SCALE


def _split_decay(named_params) -> Tuple[List, List]:
    """按 stages.py 同款规则拆 decay / no_decay:bias / bn / norm → 不衰减。"""
    decay, no_decay = [], []
    for name, p in named_params:
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def configure_official_stage(
    model: nn.Module,
    criterion: nn.Module,
    epoch: int,
    lr_base: float,
    weight_decay: float,
    backbone_lr_scale: float = 0.1,
    llrd_eta: float = 0.75,
    freeze_legacy: bool = False,
) -> Tuple[List[Dict], float, str]:
    """配置官方三头训练的当前阶段:设 requires_grad + 建 param_groups。

    Args:
        model / criterion: 主模型与 `MultiTaskLoss`。
        epoch: 当前 epoch(1-based)。
        lr_base: 头 lr 基准。
        weight_decay: 权重衰减(只加在 decay 组)。
        backbone_lr_scale: Stage B backbone lr = 头 lr × 此值(LLRD 的整体缩放,默认 0.1)。
        llrd_eta: LLRD 逐层衰减 —— blocks[6] 取满 scale,blocks[5] 再乘此 η(默认 0.75)。
        freeze_legacy: True → 冻结 AUCM 对偶变量(malignancy 损失关闭时用,见
            `train_official.py` 的 `--freeze-legacy`);False(默认)→ 对偶变量可训,
            让 legacy 多任务损失在头微调期间继续守护旧任务不退化。

    Returns:
        `(param_groups, head_lr, stage_name)`。head_lr 是调度基准 lr;stage_name 形如
        "Official Stage A: heads warmup"。

    ## requires_grad 政策(逐 stage)

    - vclub:恒 True(独立 `vclub_optimizer` 管,排除出主 param_groups —— 与 stages.py 一致)。
    - 三头:恒 True。
    - Stage A / C:除三头 + vclub 外全冻。
    - Stage B:额外解冻 `STAGE_B_UNFREEZE`(cls_backbone Top-2 block)。
    - criterion.mal_loss_fn(AUCM 对偶):`requires_grad = not freeze_legacy`。
    """
    stage = get_official_stage(epoch)
    head_lr = official_head_lr(epoch, lr_base)

    # ── 1. 设 requires_grad ──────────────────────────────────────────────
    unfreeze_extra = STAGE_B_UNFREEZE if stage == "B" else ()
    for name, p in model.named_parameters():
        if "vclub" in name:
            p.requires_grad = True
            continue
        if any(name.startswith(pfx) for pfx in _HEAD_PREFIXES):
            p.requires_grad = True
            continue
        p.requires_grad = any(name.startswith(pfx) for pfx in unfreeze_extra)

    # AUCM 对偶变量(在 criterion 上,model.named_parameters 看不见)
    for p in criterion.mal_loss_fn.parameters():
        p.requires_grad = not freeze_legacy

    # ── 2. 建 param_groups ───────────────────────────────────────────────
    # 头参数:所有官方三头前缀,统一 head_lr。
    head_named = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and "vclub" not in n
        and any(n.startswith(pfx) for pfx in _HEAD_PREFIXES)
    ]
    head_decay, head_no_decay = _split_decay(head_named)
    param_groups: List[Dict] = [
        {"params": head_decay, "weight_decay": weight_decay, "lr": head_lr},
        {"params": head_no_decay, "weight_decay": 0.0, "lr": head_lr},
    ]

    # Stage B backbone 组:逐 block LLRD。blocks[6] = head_lr×scale,blocks[5] = ×scale×η。
    if stage == "B":
        for i, pfx in enumerate(STAGE_B_UNFREEZE):     # 顺序 (blocks.6, blocks.5)
            blk_lr = head_lr * backbone_lr_scale * (llrd_eta ** i)
            blk_named = [
                (n, p) for n, p in model.named_parameters()
                if p.requires_grad and n.startswith(pfx)
            ]
            if not blk_named:
                continue
            blk_decay, blk_no_decay = _split_decay(blk_named)
            param_groups.append(
                {"params": blk_decay, "weight_decay": weight_decay, "lr": blk_lr}
            )
            param_groups.append(
                {"params": blk_no_decay, "weight_decay": 0.0, "lr": blk_lr}
            )

    # AUCM 对偶(未冻时):合并到 no_decay,lr = head_lr,与 stages.py stage≥2 口径一致
    # (拉格朗日乘子不受 weight_decay 拉扯)。
    if not freeze_legacy:
        duals = [p for p in criterion.mal_loss_fn.parameters() if p.requires_grad]
        if duals:
            param_groups.append({"params": duals, "weight_decay": 0.0, "lr": head_lr})

    stage_name = {
        "A": "Official Stage A: heads warmup (backbone/seg/CBM frozen)",
        "B": "Official Stage B: LLRD unfreeze cls Top-2",
        "C": "Official Stage C: heads low-lr calibration",
    }[stage]
    return param_groups, head_lr, stage_name

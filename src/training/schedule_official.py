"""官方数据三头训练的 A/B/C 阶段调度 —— **独立于 `stages.py` 的 legacy 三阶段**。

## 为什么另建,不复用 `configure_stage_parameters`

`stages.py` 的 Stage1→Stage2 边界有 **lr 100 倍跳变**的已知缺陷:2026-08-11 从混训
e40 权重续训 Stage2,3 个 epoch 内分割从 97.71% 塌到 80.6%(lr 暴涨 + 全模型解冻 +
mal loss 首启 + 物理先验满强度四件事同 epoch 齐发)。三头是随机初始化、只在官方数据
上从零学,一旦搭上那条崩塌调度,模型只能提交一次、踩上去无法补救。故按
`docs/历史存档/训练架构方案_20260809.md` 另建 A/B/C:

  Stage A (epoch 1-10) :冻结 cls_backbone/CBM,训四头 + **分割前向链**;头 lr 线性 warmup 至 lr_base。
  Stage B (epoch 11-30):额外解冻 cls_backbone Top-2 block,LLRD(backbone lr = 头 lr ×
                         backbone_lr_scale,逐 block 再乘 η 衰减),余弦退火。
  Stage C (epoch 31-35):回到 A 的可训集(四头 + 分割),头部低 lr 微调(温度标定放推理侧
                         calibration.py)。

🔴 **2026-08-16 契约变更:分割前向链改为 A/B/C 全程可训**(原为全程冻结)。
   原设计冻结分割是为「保证**已有**分割 DICE 不退化」(`训练架构方案.md:100`),
   其中「已有」指 warm-start 权重 `best_model_mixtrain_xl_e40.pth`。该权重现已
   **三路皆断**(A800 拒连 / 云端产物按赛规不能导出 / 仓库 `checkpoints/` 被
   gitignore 且为空),前提整体消失。而冻结的实测后果是:35 epoch 下 `val_dice`
   恒 0.0007、`pred_prob` 全域 <0.5 → `pred_mask` 恒全零 → 官方匈牙利 IoU≥0.5
   匹配零命中 → 五项分类/分期**零样本** → Sclin ≈ 0(`docs/项目现状总纲.md` §6③,
   附官方评审规则原文行号 221/225/227)。**分割不是"四项里的一项",是整个评分的
   闸门**,故改为在官方数据上从 ImageNet 底座 + 随机解码器自训。

## 硬约束

🔴 **边界 lr 连续性**:任意相邻 epoch 的 lr 比值 ≤ `MAX_LR_JUMP_RATIO`(=5)。把
   `stages.py` 的 100× 事故在数值上钉成不可能。数值锁定见 `tests/test_official_schedule.py`。
🔴 **BN 冻结**:冻结的子模块必须置 eval,否则"参数冻了但 running stats 仍更新"的数值
   病态(BN 冻结遗漏教训)。本模块**只设 requires_grad**;`set_frozen_modules_eval`
   由训练循环在 `model.train()` 之后调用(eval/train 是运行态,不属参数配置职责)。
   四头(位置/ETE/临床T/TI-RADS)与分割前向链在所有官方 stage **全程可训**,其内含的
   BatchNorm/LayerNorm 始终正常训练;Stage B 解冻的是**整个** block(全参可训),
   不存在"子模块内部分冻结"的 BN 病态。
🔴 **分割必须整块解冻**:`bn_freeze.set_frozen_modules_eval` 的判据是「模块自身参数
   **全部** requires_grad=False → eval()」。整块解冻 → 自动保持 train 态。若只解冻
   部分参数(如解 conv 留 BN),会落进 `tests/test_bn_freeze.py` 钉死的"兄弟 BN 仍被
   更新"病态。
🔴 **解冻即须进 param_groups**:`requires_grad` 与 optimizer 参数集合必须严格对齐。
   只设 requires_grad 而漏建 param_group,梯度会跨 batch **永不清零、累积至溢出**
   (`src/models/AGENTS.md:62` 的 LoRA 同形事故 / `QA_VP.md §1.1`)。既有防线是
   `tests/test_official_schedule.py::test_optimizer_covers_all_trainable_except_vclub`。

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

# ── 分割前向链(2026-08-16 起 A/B/C 全程可训,理由见模块 docstring)───────────
# 拆两组只为分配不同 lr,不是"部分解冻"——两组合起来即完整前向链,整块可训。
# 解码器(up_stages/dec_stages/seg_head)是**随机初始化**的 → 与四头同待遇,取满 head_lr。
SEG_TRAIN_DECODER = ("up_stages", "dec_stages", "seg_head",
                     # plan 任务D 深监督辅助头(2026-09-20):训练放行进解码器组
                     # (随机初始化出身,@head_lr);前缀动态读
                     # MultiTaskThyroidModel.SEG_AUX_PREFIXES 保持单源。
                     *MultiTaskThyroidModel.SEG_AUX_PREFIXES)
# 分割主干(convnext_nano)带 **ImageNet 预训练** → lr = head_lr × backbone_lr_scale,
# 与 Stage B 的 blocks[6] 档**数值重合**,不新增第三档 lr。
SEG_TRAIN_BACKBONE = ("seg_backbone",)
# 全程可训集合(四头由 _HEAD_PREFIXES 另行放行)。
SEG_TRAIN_PREFIXES = SEG_TRAIN_DECODER + SEG_TRAIN_BACKBONE

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
    ete_only: bool = False,
    seg_only: bool = False,
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
        ete_only: True → **ETE 头小试模式**(D10-15,2026-09-12):恒可训集从
            「四头+分割链」收窄到**仅 `extension_head`**(local_proj+classifier,
            10 键),其余全冻(含分割与另三头)。从 r5 checkpoint 热启动后微调,
            期望 ete_kappa 0.2086→≥0.24 且其余指标不动(它们无梯度,数学上
            只受数值噪声影响)。lr 取 Stage C 档(低 lr 微调,防把热启动的头
            一步带偏)。**损失仍全量计算**(改损失路径风险大于收益,冻结参数
            梯度为 0 只是白算——6-8 epoch 量级可接受)。
        seg_only: True → **seg-only 微调模式**(阶段2,2026-09-13):恒可训集
            从「四头+分割链」收窄到**仅分割前向链**(`SEG_TRAIN_PREFIXES`
            = decoder+seg_backbone),四头/cls_backbone/Adaptor/vclub **全冻**
            (单变量:Boundary/Lovász 分割损失实验不牵动分类表征;vclub 冻结
            由 `vclub_learning_step` 的 requires_grad 早退兜住,不走独立
            optimizer)。lr 恒 Stage C 低档。param_groups 照正常路径**两档 lr、
            四组**(每档按 decay/no_decay 拆两组;decoder@head_lr /
            seg_backbone@head_lr×backbone_lr_scale;3B 专审 P2-2:此前误记
            "三组",实测 len(groups)==4)。
            前提:--resume-from r5 权重(分类头全冻,基线才有意义);预算
            ≤20 epoch(γ_prior 在 ep≤20 恒 0 的单变量窗口)。

    Returns:
        `(param_groups, head_lr, stage_name)`。head_lr 是调度基准 lr;stage_name 形如
        "Official Stage A: heads warmup"。

    ## requires_grad 政策(逐 stage)

    - vclub:恒 True(独立 `vclub_optimizer` 管,排除出主 param_groups —— 与 stages.py 一致)。
      **两个 only 模式例外:ete_only 恒 True(仍由 vclub_optimizer 管);
      seg_only 显式 False(vclub 不更新,learning_step 早退兜底)**。
    - 四头:恒 True。**ete_only 时唯一例外:仅 extension_head True,其余三头也冻;
      seg_only 时四头全冻**。
    - **分割前向链**(`SEG_TRAIN_PREFIXES`):恒 True(2026-08-16 起,见模块 docstring)。
      **ete_only 时冻结**(分割是 0.6528 的闸门,不为 ETE 冒险);**seg_only 时
      是唯一可训集**。
    - Stage A / C:除四头 + 分割 + vclub 外全冻。
    - Stage B:额外解冻 `STAGE_B_UNFREEZE`(cls_backbone Top-2 block)。**ete_only 时
      不解冻 backbone**(backbone 动了四头全跟着漂,违背"仅解冻 ETE"的单变量原则);
      seg_only 不走 Stage 分支(恒定单阶段,也永不解冻 backbone)。
    - criterion.mal_loss_fn(AUCM 对偶):`requires_grad = not freeze_legacy`。
      **seg_only 时恒 False**(对偶变量也不动)。

    ## lr 档位

    | 组 | lr |
    |---|---|
    | 四头 + 分割解码器(随机初始化) | `head_lr` |
    | 分割主干(ImageNet) | `head_lr × backbone_lr_scale` |
    | Stage B `cls_backbone.blocks[6]` | `head_lr × backbone_lr_scale` (同上,刻意重合) |
    | Stage B `cls_backbone.blocks[5]` | `head_lr × backbone_lr_scale × llrd_eta` |

    only 模式(ete_only/seg_only)head_lr 恒取 `lr_base × _STAGE_C_LR_SCALE`,
    不新增第三档 —— 相邻 epoch lr 比值约束与既有测试均按两档写死;seg_only
    内部仍是上表两档(decoder@head_lr / seg_backbone@head_lr×backbone_lr_scale)。
    """
    stage = get_official_stage(epoch)
    head_lr = official_head_lr(epoch, lr_base)

    # ── 1. 设 requires_grad ──────────────────────────────────────────────
    # 🔴 互斥检查必须在两个 early-return 分支之前(ete_only 分支先 return,
    # 若把检查放进 seg_only 分支内部,ete_only=True+seg_only=True 会先命中
    # ete_only 分支直接返回,互斥检查永远走不到——实测发现,test 曾误过)。
    if ete_only and seg_only:
        raise ValueError("seg_only 与 ete_only 互斥,不能同开")
    # ete_only(D10-15):恒可训集收窄到仅 extension_head,Stage B 不解冻 backbone,
    # 分割链冻结。lr 强制取 Stage C 低档(微调而非重学)。
    if ete_only:
        head_lr = lr_base * _STAGE_C_LR_SCALE
        _ete_prefix = "extension_head"
        for name, p in model.named_parameters():
            if "vclub" in name:
                p.requires_grad = True
                continue
            p.requires_grad = name.startswith(_ete_prefix)
        for p in criterion.mal_loss_fn.parameters():
            p.requires_grad = not freeze_legacy
        # param_groups:只有 ETE 头一组(decay/no_decay 拆分照旧)
        ete_named = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad and "vclub" not in n
        ]
        ete_decay, ete_no_decay = _split_decay(ete_named)
        param_groups: List[Dict] = [
            {"params": ete_decay, "weight_decay": weight_decay, "lr": head_lr},
            {"params": ete_no_decay, "weight_decay": 0.0, "lr": head_lr},
        ]
        if not freeze_legacy:
            duals = [p for p in criterion.mal_loss_fn.parameters() if p.requires_grad]
            if duals:
                param_groups.append(
                    {"params": duals, "weight_decay": 0.0, "lr": head_lr})
        return param_groups, head_lr, "ETE-only finetune (r5 frozen, extension_head only)"

    # ── seg_only(阶段2,2026-09-13):仅分割前向链可训,Boundary/Lovász 微调 ──
    # 恒可训集 = SEG_TRAIN_PREFIXES(decoder+seg_backbone);四头/cls_backbone/
    # Adaptor/vclub 全冻(单变量:分割损失的实验不牵动分类表征)。
    # 🔴 vclub 显式冻 True→False(照抄 ete_only 模板会把 vclub 放行,违背单变量
    # —— plan 审查 finding);lr 取 Stage C 低档(微调而非重学,与 ete_only 同理)。
    # 🔴 param_groups **照正常路径两档 lr、四组**(每档 decay/no_decay 拆两组):
    # decoder@head_lr / seg_backbone@
    # head_lr×backbone_lr_scale(下方正常路径先例),不照抄 ete_only 两组——那
    # 会给 seg_backbone 10× lr(既有档位的 10 倍,微调场景会一步带偏)。
    # 组构建与正常路径同构(两档 lr × decay/no_decay 拆分),改一处须同步另一处。
    if seg_only:
        head_lr = lr_base * _STAGE_C_LR_SCALE
        for name, p in model.named_parameters():
            if "vclub" in name:
                p.requires_grad = False
                continue
            p.requires_grad = any(name.startswith(pfx) for pfx in SEG_TRAIN_PREFIXES)
        for p in criterion.mal_loss_fn.parameters():
            p.requires_grad = False
        # 组 1:分割解码器(随机初始化出身)@ head_lr
        dec_named = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad and any(n.startswith(pfx) for pfx in SEG_TRAIN_DECODER)
        ]
        dec_decay, dec_no_decay = _split_decay(dec_named)
        param_groups: List[Dict] = [
            {"params": dec_decay, "weight_decay": weight_decay, "lr": head_lr},
            {"params": dec_no_decay, "weight_decay": 0.0, "lr": head_lr},
        ]
        # 组 2:分割主干(ImageNet 预训练)@ head_lr × backbone_lr_scale
        bb_named = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad and any(n.startswith(pfx) for pfx in SEG_TRAIN_BACKBONE)
        ]
        if bb_named:
            bb_lr = head_lr * backbone_lr_scale
            bb_decay, bb_no_decay = _split_decay(bb_named)
            param_groups.append(
                {"params": bb_decay, "weight_decay": weight_decay, "lr": bb_lr})
            param_groups.append(
                {"params": bb_no_decay, "weight_decay": 0.0, "lr": bb_lr})
        return param_groups, head_lr, "SEG-only finetune (r5 frozen, seg chain only)"

    unfreeze_extra = STAGE_B_UNFREEZE if stage == "B" else ()
    # 恒可训集:四头 + 分割前向链。分割整块解冻(解码器 + 主干同时放行),
    # 避免 bn_freeze 的"兄弟 BN 仍被更新"病态。
    always_train = tuple(_HEAD_PREFIXES) + SEG_TRAIN_PREFIXES
    for name, p in model.named_parameters():
        if "vclub" in name:
            p.requires_grad = True
            continue
        if any(name.startswith(pfx) for pfx in always_train):
            p.requires_grad = True
            continue
        p.requires_grad = any(name.startswith(pfx) for pfx in unfreeze_extra)

    # AUCM 对偶变量(在 criterion 上,model.named_parameters 看不见)
    for p in criterion.mal_loss_fn.parameters():
        p.requires_grad = not freeze_legacy

    # ── 2. 建 param_groups ───────────────────────────────────────────────
    # 满 head_lr 组:四头 + 分割解码器(都是随机初始化,同待遇)。
    _head_lr_prefixes = tuple(_HEAD_PREFIXES) + SEG_TRAIN_DECODER
    head_named = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and "vclub" not in n
        and any(n.startswith(pfx) for pfx in _head_lr_prefixes)
    ]
    head_decay, head_no_decay = _split_decay(head_named)
    param_groups: List[Dict] = [
        {"params": head_decay, "weight_decay": weight_decay, "lr": head_lr},
        {"params": head_no_decay, "weight_decay": 0.0, "lr": head_lr},
    ]

    # 分割主干组(ImageNet 预训练)—— lr = head_lr × scale,与 blocks[6] 同档。
    seg_bb_named = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and "vclub" not in n
        and any(n.startswith(pfx) for pfx in SEG_TRAIN_BACKBONE)
    ]
    if seg_bb_named:
        seg_bb_lr = head_lr * backbone_lr_scale
        seg_bb_decay, seg_bb_no_decay = _split_decay(seg_bb_named)
        param_groups.append(
            {"params": seg_bb_decay, "weight_decay": weight_decay, "lr": seg_bb_lr}
        )
        param_groups.append(
            {"params": seg_bb_no_decay, "weight_decay": 0.0, "lr": seg_bb_lr}
        )

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
        "A": "Official Stage A: heads+seg warmup (cls_backbone/CBM frozen)",
        "B": "Official Stage B: heads+seg + LLRD unfreeze cls Top-2",
        "C": "Official Stage C: heads+seg low-lr calibration",
    }[stage]
    return param_groups, head_lr, stage_name

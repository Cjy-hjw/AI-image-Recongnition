"""官方三头 A/B/C 调度(`schedule_official.py`)+ JSONL 日志(`jsonl_logger.py`)的
关键数值/契约锁定测试。

**为什么需要**:官方数据四头训练是"只能提交一次、踩上去无法补救"的关键路径,而它
刻意**不复用** `stages.py`(后者 Stage1→2 边界 lr 100 倍跳变实测崩分割)。本文件把
新调度的三条硬约束钉进 CI:
  ① 相邻 epoch lr 比值 ≤ 5(把 100× 事故钉成不可能);
  ② 各 stage 的冻结/解冻集合(A/C 训四头 + **分割前向链** + vclub;B 额外解冻 cls Top-2);
  ③ 所有可训参数都进 optimizer(除 vclub 独立优化器)。

🔴 **2026-08-16 契约变更**:分割前向链由"A/B/C 全程冻结"改为"全程可训"。本文件
原有四个测试**刻意把"分割不训"钉进 CI**,故一并改为断言**新**契约(不是删掉,
是把断言方向翻过来)。变更理由:冻结的实测后果是 `pred_mask` 恒全零 → 官方
IoU≥0.5 匹配零命中 → 五项分类/分期零样本 → Sclin≈0(总纲 §6③)。

纯 CPU / 无网络 / pretrained=False,可直接进 CI。sklearn 须在 torch 前导入以避开
Windows OpenMP DLL 冲突(段错误 EXIT=-1073741819)—— 本文件不用 sklearn,但保持
与其它测试同序无害。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from src.models.loss import MultiTaskLoss
from src.models.multi_task_model import MultiTaskThyroidModel
from src.training.loop.bn_freeze import set_frozen_modules_eval
from src.training.schedule_official import (
    MAX_LR_JUMP_RATIO,
    SEG_TRAIN_BACKBONE,
    SEG_TRAIN_DECODER,
    SEG_TRAIN_PREFIXES,
    STAGE_A_END,
    STAGE_B_END,
    STAGE_C_END,
    STAGE_B_UNFREEZE,
    configure_official_stage,
    get_official_stage,
    official_head_lr,
)

LR_BASE = 0.0024
WEIGHT_DECAY = 5e-4
_HEAD_PFX = MultiTaskThyroidModel.OFFICIAL_HEAD_PREFIXES


def _is_allowed_trainable(name: str) -> bool:
    """A/C 阶段的合法可训名:四头 / 分割前向链 / vclub。"""
    return (
        name.startswith(_HEAD_PFX)
        or name.startswith(SEG_TRAIN_PREFIXES)
        or "vclub" in name
    )


@pytest.fixture(scope="module")
def model_and_criterion():
    model = MultiTaskThyroidModel(pretrained=False)
    criterion = MultiTaskLoss()
    return model, criterion


# ═════════════════════════════════════════════════════════════════════════════
# 阶段划分
# ═════════════════════════════════════════════════════════════════════════════


def test_stage_boundaries():
    assert get_official_stage(1) == "A"
    assert get_official_stage(STAGE_A_END) == "A"          # 10
    assert get_official_stage(STAGE_A_END + 1) == "B"      # 11
    assert get_official_stage(STAGE_B_END) == "B"          # 30
    assert get_official_stage(STAGE_B_END + 1) == "C"      # 31
    assert get_official_stage(STAGE_C_END) == "C"          # 35


# ═════════════════════════════════════════════════════════════════════════════
# lr 曲线 —— 本文件核心:相邻比值 ≤ 5
# ═════════════════════════════════════════════════════════════════════════════


def test_no_adjacent_lr_jump_exceeds_cap():
    """🔴 任意相邻 epoch 的 lr 比值 ≤ MAX_LR_JUMP_RATIO(=5)。

    这是把 stages.py 的 100× 崩塌根因钉成不可能的核心断言。全程 1..35 逐点核。
    """
    lrs = [official_head_lr(e, LR_BASE) for e in range(1, STAGE_C_END + 1)]
    assert all(x > 0 for x in lrs), "lr 必须恒正"
    for i in range(1, len(lrs)):
        ratio = max(lrs[i], lrs[i - 1]) / min(lrs[i], lrs[i - 1])
        assert ratio <= MAX_LR_JUMP_RATIO, (
            f"epoch {i}→{i+1} lr 跳变 {ratio:.2f}× 超上限 {MAX_LR_JUMP_RATIO}×"
            f"(lr {lrs[i-1]:.3e}→{lrs[i]:.3e})——重演 stages.py 100× 事故的风险"
        )


def test_lr_key_values():
    """A warmup 端点 / B 余弦端点 / C 固定值。"""
    # A: e1 = lr_base/10、e10 = lr_base
    assert official_head_lr(1, LR_BASE) == pytest.approx(LR_BASE * 0.1)
    assert official_head_lr(STAGE_A_END, LR_BASE) == pytest.approx(LR_BASE)
    # B: e11 起点回到 lr_base(与 e10 衔接比值 1.0)、e30 退到 lr_base×0.1
    assert official_head_lr(STAGE_A_END + 1, LR_BASE) == pytest.approx(LR_BASE)
    assert official_head_lr(STAGE_B_END, LR_BASE) == pytest.approx(LR_BASE * 0.1)
    # C: 固定 lr_base×0.05
    for e in range(STAGE_B_END + 1, STAGE_C_END + 1):
        assert official_head_lr(e, LR_BASE) == pytest.approx(LR_BASE * 0.05)


def test_stage_b_cosine_monotonic_decreasing():
    """Stage B(11-30)余弦退火严格单调下降。"""
    lrs = [official_head_lr(e, LR_BASE) for e in range(STAGE_A_END + 1, STAGE_B_END + 1)]
    assert all(a > b for a, b in zip(lrs, lrs[1:])), f"Stage B 应严格下降,got {lrs}"


def test_stage_a_warmup_ramps_up():
    """Stage A(1-10)线性 warmup 严格上升,末端达 lr_base。"""
    lrs = [official_head_lr(e, LR_BASE) for e in range(1, STAGE_A_END + 1)]
    assert all(a < b for a, b in zip(lrs, lrs[1:])), "warmup 段应严格上升"
    assert lrs[-1] == pytest.approx(LR_BASE)


# ═════════════════════════════════════════════════════════════════════════════
# 冻结集合 / 优化器覆盖
# ═════════════════════════════════════════════════════════════════════════════


def _trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_stage_a_heads_seg_and_vclub_trainable(model_and_criterion):
    """Stage A 可训集 = 四头 + 分割前向链 + vclub,cls_backbone/CBM 仍冻。

    🔴 2026-08-16 契约变更:本测试原名 `test_stage_a_only_heads_and_vclub_trainable`,
    断言"分割主干必须冻结"。现改为断言分割**确实可训** —— 冻结分割是
    Sclin≈0 的根因(总纲 §6③),那条旧断言曾把这个 bug 钉在 CI 里当成正确行为。
    """
    model, criterion = model_and_criterion
    configure_official_stage(model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY)
    trainable = _trainable(model)
    assert trainable, "Stage A 至少四头 + 分割可训"
    for n in trainable:
        assert _is_allowed_trainable(n), (
            f"Stage A 只应训四头 + 分割前向链 + vclub,却有 {n} 可训"
        )
    # 四头确实全可训
    assert any(n.startswith("location_head") for n in trainable)
    assert any(n.startswith("extension_head") for n in trainable)
    assert any(n.startswith("clinical_t_head") for n in trainable)
    assert any(n.startswith("tirads_head") for n in trainable)
    # 🔴 分割前向链每个前缀都确有参数可训(前缀拼错会静默退化成"仍然冻着")
    for pfx in SEG_TRAIN_PREFIXES:
        assert any(n.startswith(pfx) for n in trainable), (
            f"Stage A 分割前缀 {pfx} 未解冻 —— 拼写错误会让分割继续零梯度,"
            f"官方匹配零命中、Sclin≈0"
        )
    # 分类主干 / CBM 仍冻
    assert not any(n.startswith("cls_backbone") for n in trainable), "Stage A 应冻分类主干"
    assert not any(n.startswith("prior_extractor") for n in trainable), "Stage A 应冻物理先验"


def test_stage_b_unfreezes_exactly_top2_cls_blocks(model_and_criterion):
    model, criterion = model_and_criterion
    configure_official_stage(model, criterion, epoch=STAGE_A_END + 1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY)
    trainable = _trainable(model)
    cls_trainable = {n for n in trainable if n.startswith("cls_backbone")}
    assert cls_trainable, "Stage B 应解冻部分 cls_backbone"
    # 所有解冻的 cls 参数必属 top-2 block 前缀
    for n in cls_trainable:
        assert any(n.startswith(pfx) for pfx in STAGE_B_UNFREEZE), (
            f"Stage B 解冻了非 top-2 的 cls 参数 {n}"
        )
    # 两个 top-2 block 都确有参数被解冻(前缀没写错)
    for pfx in STAGE_B_UNFREEZE:
        assert any(n.startswith(pfx) for n in cls_trainable), f"{pfx} 未被解冻(前缀拼写?)"
    # 🔴 分割在 B 阶段同样可训(2026-08-16;原断言是"seg 主干仍冻")
    for pfx in SEG_TRAIN_PREFIXES:
        assert any(n.startswith(pfx) for n in trainable), f"Stage B 分割前缀 {pfx} 应可训"


def test_stage_c_back_to_heads_and_seg(model_and_criterion):
    """Stage C 回到 A 的可训集(四头 + 分割),cls Top-2 重新冻结。

    🔴 2026-08-16:原名 `test_stage_c_back_to_heads_only`,断言"只有四头 + vclub"。
    """
    model, criterion = model_and_criterion
    configure_official_stage(model, criterion, epoch=STAGE_B_END + 1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY)
    trainable = _trainable(model)
    for n in trainable:
        assert _is_allowed_trainable(n), f"Stage C 只应训四头 + 分割 + vclub,却有 {n}"
    assert not any(n.startswith("cls_backbone") for n in trainable), "Stage C 应重新冻结 cls Top-2"
    for pfx in SEG_TRAIN_PREFIXES:
        assert any(n.startswith(pfx) for n in trainable), f"Stage C 分割前缀 {pfx} 应仍可训"


@pytest.mark.parametrize("epoch", [1, 11, 31])
def test_optimizer_covers_all_trainable_except_vclub(model_and_criterion, epoch):
    """所有可训模型参数都进某个 param_group —— vclub 除外(独立优化器)。"""
    model, criterion = model_and_criterion
    param_groups, _lr, _name = configure_official_stage(
        model, criterion, epoch=epoch, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY
    )
    covered = {id(p) for g in param_groups for p in g["params"]}
    missing = [
        n for n, p in model.named_parameters()
        if p.requires_grad and "vclub" not in n and id(p) not in covered
    ]
    assert not missing, f"epoch {epoch}: 可训但未进 optimizer 的参数 {missing[:5]}"
    leaked = [n for n, p in model.named_parameters() if "vclub" in n and id(p) in covered]
    assert not leaked, f"epoch {epoch}: vclub 漏进主 optimizer(会被双重更新){leaked[:5]}"


def test_aucm_duals_bound_when_legacy_on_frozen_when_off(model_and_criterion):
    """freeze_legacy 决定 AUCM 对偶变量是否可训 + 是否进 optimizer。"""
    model, criterion = model_and_criterion
    duals = list(criterion.mal_loss_fn.parameters())
    assert duals, "AUCM 应有对偶参数"

    # 默认(legacy on):对偶可训且进组
    pg_on, _, _ = configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY, freeze_legacy=False
    )
    covered_on = {id(p) for g in pg_on for p in g["params"]}
    assert all(p.requires_grad for p in duals), "legacy on 时对偶应可训"
    assert all(id(p) in covered_on for p in duals), "legacy on 时对偶应进 optimizer"

    # freeze_legacy:对偶冻结且不进组
    pg_off, _, _ = configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY, freeze_legacy=True
    )
    covered_off = {id(p) for g in pg_off for p in g["params"]}
    assert not any(p.requires_grad for p in duals), "freeze_legacy 时对偶应冻结"
    assert not any(id(p) in covered_off for p in duals), "freeze_legacy 时对偶不应进 optimizer"


def test_stage_b_backbone_lr_smaller_than_head(model_and_criterion):
    """Stage B backbone(LLRD)组 lr 应远小于头 lr(= 头 lr × scale × η^i)。"""
    model, criterion = model_and_criterion
    param_groups, head_lr, _ = configure_official_stage(
        model, criterion, epoch=STAGE_A_END + 1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY,
        backbone_lr_scale=0.1, llrd_eta=0.75,
    )
    # 收集各组 lr(去重):应有头 lr 与两档 backbone lr。
    lrs = sorted({g["lr"] for g in param_groups if g["params"]})
    assert head_lr == pytest.approx(max(lrs)), "头 lr 应是最大组"
    backbone_lrs = [x for x in lrs if x < head_lr]
    assert backbone_lrs, "Stage B 应有更小的 backbone lr 组"
    assert max(backbone_lrs) == pytest.approx(head_lr * 0.1), "blocks[6] 应 = 头 lr×0.1"
    # 逐层 LLRD:blocks[5] = 头 lr×0.1×0.75
    assert min(backbone_lrs) == pytest.approx(head_lr * 0.1 * 0.75, rel=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# BN 冻结(冻结子模块置 eval,三头保持 train)
# ═════════════════════════════════════════════════════════════════════════════


def test_frozen_bn_eval_heads_and_seg_train(model_and_criterion):
    """set_frozen_modules_eval 后:冻结的 cls 主干 BN 置 eval;四头与**分割**保持 train。

    🔴 2026-08-16 契约翻转:原断言"冻结分割主干的 BN 应置 eval"。分割改为整块解冻后,
    正确表现是保持 **train** —— `bn_freeze.set_frozen_modules_eval` 的判据是"模块自身
    参数全部 requires_grad=False → eval()",整块解冻自动保持 train 态。
    若这里失败,多半是分割只被**部分**解冻(如解 conv 留 BN),会落进
    `tests/test_bn_freeze.py` 钉死的"兄弟 BN 仍被更新"病态。
    """
    model, criterion = model_and_criterion
    configure_official_stage(model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY)
    model.train()
    set_frozen_modules_eval(model)
    # 四头是全可训子模块 → 应保持 training(其内含 BatchNorm2d 需正常训练)
    assert model.location_head.training, "位置头应保持 train"
    assert model.extension_head.training, "侵犯头应保持 train"
    assert model.clinical_t_head.training, "临床T头应保持 train"
    # 🔴 分割前向链全可训 → 其归一化层应保持 train(BN/LN 均查)
    _NORMS = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm, torch.nn.GroupNorm)
    seg_norms = [m for m in model.seg_backbone.modules() if isinstance(m, _NORMS)]
    assert seg_norms, "seg_backbone 应含归一化层(convnext_nano 用 LayerNorm)"
    assert all(m.training for m in seg_norms), (
        "分割主干整块解冻后其归一化层应保持 train —— 若为 eval 说明只部分解冻"
    )
    for dec in (model.up_stages, model.dec_stages, model.seg_head):
        assert dec.training, "分割解码器应保持 train"
    # 冻结的分类主干 BN 应 eval(这条不变)
    cls_bns = [m for m in model.cls_backbone.modules()
               if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if cls_bns:
        assert all(not m.training for m in cls_bns), "Stage A 冻结的分类主干 BN 应置 eval"


# ═════════════════════════════════════════════════════════════════════════════
# 分割解冻契约(2026-08-16 新增 —— 解除 Sclin 闸门的核心改动)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("epoch", [1, STAGE_A_END + 1, STAGE_B_END + 1])  # A / B / C
def test_seg_trainable_and_in_optimizer_in_all_stages(model_and_criterion, epoch):
    """🔴 分割前向链在 A/B/C **三个 stage 都**可训,且都进 optimizer。

    两条缺一不可:
      ① 只 `requires_grad=True` 而漏建 param_group → 梯度跨 batch **永不清零、
         累积至溢出**(`src/models/AGENTS.md:62` 的 LoRA 同形事故 / QA_VP §1.1);
      ② 只进 optimizer 而 requires_grad=False → 梯度恒 None,等于没训。
    """
    model, criterion = model_and_criterion
    param_groups, _, _ = configure_official_stage(
        model, criterion, epoch=epoch, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY
    )
    covered = {id(p) for g in param_groups for p in g["params"]}
    for pfx in SEG_TRAIN_PREFIXES:
        named = [(n, p) for n, p in model.named_parameters() if n.startswith(pfx)]
        assert named, f"模型上找不到前缀 {pfx} 的参数(前缀写错?)"
        assert all(p.requires_grad for _, p in named), f"epoch {epoch}: {pfx} 应可训"
        uncovered = [n for n, p in named if id(p) not in covered]
        assert not uncovered, (
            f"epoch {epoch}: {pfx} 可训但未进 optimizer(梯度会累积不清零){uncovered[:3]}"
        )


@pytest.mark.parametrize("epoch", [1, STAGE_A_END + 1, STAGE_B_END + 1])
def test_seg_lr_tiers(model_and_criterion, epoch):
    """分割解码器 = head_lr(随机初始化);分割主干 = head_lr × backbone_lr_scale(ImageNet)。

    档位刻意与 Stage B 的 `blocks[6]` **数值重合**,不新增第三档 —— 相邻 epoch lr
    比值约束与 `test_stage_b_backbone_lr_smaller_than_head` 都按两档写死。
    """
    model, criterion = model_and_criterion
    scale = 0.1
    param_groups, head_lr, _ = configure_official_stage(
        model, criterion, epoch=epoch, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY,
        backbone_lr_scale=scale, llrd_eta=0.75,
    )

    def _lr_of(param_id):
        for g in param_groups:
            if any(id(p) == param_id for p in g["params"]):
                return g["lr"]
        return None

    for pfx in SEG_TRAIN_DECODER:
        for n, p in model.named_parameters():
            if n.startswith(pfx):
                assert _lr_of(id(p)) == pytest.approx(head_lr), (
                    f"{n} 应取满 head_lr(随机初始化的解码器,与四头同待遇)"
                )
                break
    for pfx in SEG_TRAIN_BACKBONE:
        for n, p in model.named_parameters():
            if n.startswith(pfx):
                assert _lr_of(id(p)) == pytest.approx(head_lr * scale), (
                    f"{n} 应取 head_lr×{scale}(ImageNet 预训练,护住已有表征)"
                )
                break


def test_seg_backbone_group_is_single_tier(model_and_criterion):
    """恰有一组含 `seg_backbone` 参数的 lr 值,且 == head_lr × scale。

    补 `test_stage_b_backbone_lr_smaller_than_head` 被静默削弱的语义:那条测的是
    "lr 值集合有两档",分割主干与 blocks[6] 同值时它照样通过,但**分不清**这一档
    是谁的。本条显式钉住 seg 组自己。
    """
    model, criterion = model_and_criterion
    scale = 0.1
    param_groups, head_lr, _ = configure_official_stage(
        model, criterion, epoch=STAGE_A_END + 1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY,
        backbone_lr_scale=scale, llrd_eta=0.75,
    )
    seg_ids = {id(p) for n, p in model.named_parameters() if n.startswith(SEG_TRAIN_BACKBONE)}
    seg_group_lrs = {
        g["lr"] for g in param_groups
        if g["params"] and any(id(p) in seg_ids for p in g["params"])
    }
    assert len(seg_group_lrs) == 1, f"seg_backbone 应只落在一个 lr 档,got {seg_group_lrs}"
    assert seg_group_lrs.pop() == pytest.approx(head_lr * scale)


def test_no_param_in_two_groups(model_and_criterion):
    """任一参数不得同时进两个 param_group —— AdamW 会对它做**双重更新**。

    分割解码器并入头组时若前缀集合有重叠(如 `SEG_TRAIN_DECODER` 与
    `OFFICIAL_HEAD_PREFIXES` 撞名),就会静默出现重复。
    """
    model, criterion = model_and_criterion
    for epoch in (1, STAGE_A_END + 1, STAGE_B_END + 1):
        param_groups, _, _ = configure_official_stage(
            model, criterion, epoch=epoch, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY
        )
        seen, dupes = set(), []
        for g in param_groups:
            for p in g["params"]:
                if id(p) in seen:
                    dupes.append(id(p))
                seen.add(id(p))
        assert not dupes, f"epoch {epoch}: {len(dupes)} 个参数进了多个 param_group(会被双重更新)"


# ═════════════════════════════════════════════════════════════════════════════
# JSONL 日志(V2.1 硬约束)
# ═════════════════════════════════════════════════════════════════════════════


def test_jsonl_logger_writes_required_fields(tmp_path):
    import json

    from src.training.jsonl_logger import JsonlLogger

    with JsonlLogger(log_dir=str(tmp_path), data_source="official/train_v1") as lg:
        rec = lg.log(epoch=1, step=0, phase="train", loss=2.345, lr=1e-3)
    # 必填字段齐全
    for k in ("timestamp", "epoch", "step", "phase", "mode", "data_source"):
        assert k in rec, f"缺必填字段 {k}"
    assert rec["phase"] == "train" and rec["mode"] == "training"
    # timestamp ISO 8601 带毫秒 + Z
    assert rec["timestamp"].endswith("Z") and "T" in rec["timestamp"]
    # 落盘可被逐行 JSON 解析
    path = os.path.join(str(tmp_path), "train.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    assert len(lines) == 1 and lines[0]["loss"] == pytest.approx(2.345)


def test_jsonl_logger_rejects_bad_phase_mode(tmp_path):
    from src.training.jsonl_logger import JsonlLogger

    with JsonlLogger(log_dir=str(tmp_path)) as lg:
        with pytest.raises(AssertionError):
            lg.log(epoch=1, step=0, phase="bogus")
        with pytest.raises(AssertionError):
            lg.log(epoch=1, step=0, phase="train", mode="bogus")


def test_jsonl_logger_extra_fields_merged_and_reserved_protected(tmp_path):
    """`extra=` 诊断字段并入行尾,但**不得覆盖** V2.1 规范字段。

    2026-08-14 新增(首次真训 loss 双峰无法定位来源 → 需要子损失分解)。
    覆盖保护是本测试的重点:静默覆盖 `loss` 会让官方评审读到错误的值,
    比直接崩溃阴 —— 故必须 assert 拦下而非"后写的赢"。
    """
    import json

    from src.training.jsonl_logger import JsonlLogger

    with JsonlLogger(log_dir=str(tmp_path)) as lg:
        rec = lg.log(epoch=1, step=0, phase="train", loss=2.0, lr=1e-3,
                     extra={"loss_seg": 1.25, "n_valid_tirads": 7})
        # 额外字段落盘
        assert rec["loss_seg"] == pytest.approx(1.25)
        assert rec["n_valid_tirads"] == 7
        # 规范字段未被污染
        assert rec["loss"] == pytest.approx(2.0)

        # 任一规范字段同名 → 必须 assert,不得静默覆盖
        for bad in ("loss", "epoch", "timestamp", "phase", "data_source"):
            with pytest.raises(AssertionError):
                lg.log(epoch=1, step=1, phase="train", extra={bad: "hijack"})

        # extra=None / {} 不改变行为(向后兼容)
        rec2 = lg.log(epoch=1, step=2, phase="train", loss=1.0, extra=None)
        assert "loss_seg" not in rec2

    path = os.path.join(str(tmp_path), "train.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    # 被 assert 拦下的那 5 行不应落盘(assert 在 write 之前)
    assert len(lines) == 2, f"劫持行不得落盘,实际 {len(lines)} 行"


def test_jsonl_logger_drops_nonfinite_extra(tmp_path):
    """NaN/Inf 诊断字段必须被丢弃 —— 否则写出非标准 JSON,整行不可解析。

    🔴 Gate2 实测(2026-08-14):`json.dumps` 默认 `allow_nan=True`,
    `{"loss_seg": float("nan")}` 写出的是**裸 `NaN`**,不是合法 JSON。
    Python 的 `json.loads` 宽容能读回,但严格解析器(Go/Java/多数评审脚本)拒收。
    V2.1「未能读取到指定日志文件…可能被标记为可疑对象」→ 合规红线。

    本测试用 `parse_constant` 强制严格模式,复现"严格解析器视角"。
    """
    import json

    from src.training.jsonl_logger import JsonlLogger

    def _strict(s):
        """拒绝 NaN/Infinity/-Infinity 的严格 JSON 解析(模拟官方评审脚本)。"""
        return json.loads(
            s, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"非法常量 {c}")))

    with JsonlLogger(log_dir=str(tmp_path)) as lg:
        rec = lg.log(epoch=1, step=0, phase="train", loss=1.0,
                     extra={"loss_seg": float("nan"), "loss_ete": float("inf"),
                            "loss_neg_inf": float("-inf"),
                            "loss_ok": 0.5, "n_valid_tirads": 3})
        # 非有限值被丢弃
        for k in ("loss_seg", "loss_ete", "loss_neg_inf"):
            assert k not in rec, f"{k} 是非有限值,必须丢弃(不得填 0 伪装成正常)"
        # 有限值与整数照常保留
        assert rec["loss_ok"] == pytest.approx(0.5)
        assert rec["n_valid_tirads"] == 3

    # 落盘内容必须能过严格 JSON 解析
    path = os.path.join(str(tmp_path), "train.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln for ln in f if ln.strip()]
    assert len(raw) == 1
    assert "NaN" not in raw[0] and "Infinity" not in raw[0], f"落盘含非法常量: {raw[0]}"
    _strict(raw[0])  # 抛异常即测试失败


# ---------------------------------------------------------------------------
# backlog #128:`build_official_head_config` 的 ETE 超参守卫
# ---------------------------------------------------------------------------
# 🔴 2026-08-20 Gate2 发现的口径裂缝(此前 `build_official_head_config` 全仓
# **零测试覆盖**,故没人挡得住):该函数按 `la_tau > 0` 决定造不造 log_prior,
# 而 `compute_extension_loss` 的守卫是 `la_tau != 0.0` —— 两者对**负值**判定
# 相反。`la_tau=-0.5` 会走进 LA 分支却拿到 log_prior=None,报出的
# "la_tau != 0 时必须提供 log_prior" 与真实病因(τ 不该为负)无关,极易误导排查。
# 且负 τ 数学上是**放大**不平衡(LA 符号方向见 extension.py 模块 docstring),
# 与本项目踩过两次的符号反转坑同源(#113 FAMO / #120 拉格朗日项)。


def _base_train_cfg(**overrides):
    """构造 `build_official_head_config` 所需的最小 train 段。"""
    cfg = {
        "ete_class_counts": [3906, 267, 67, 5],
        "ete_cb_beta": 0.9,
        "ete_focal_gamma": 0.0,
        "ete_la_tau": 0.0,
        "ete_qwk_weight": 0.0,
        "ete_dist_weight": 0.2,
    }
    cfg.update(overrides)
    return {"train": cfg}


@pytest.mark.parametrize("bad_tau", [-0.5, -1.0, -1e-6])
def test_negative_la_tau_rejected_at_config_build(bad_tau):
    """负 τ 必须在入口被拒,不得漏到损失函数里报无关错误。"""
    from scripts.train_official import build_official_head_config

    with pytest.raises(AssertionError, match=r"ete_la_tau"):
        build_official_head_config(_base_train_cfg(ete_la_tau=bad_tau))


@pytest.mark.parametrize(
    "la_tau,qwk_weight,expect_prior",
    [
        (0.0, 0.0, False),   # 默认:CB-Focal 老路径,不需要先验
        (0.5, 0.0, True),    # 仅 LA:CB-Focal 叠加 LA 的既有合法配置
        (0.5, 1.0, True),    # LA + SoftQWK:DR 主方案
    ],
)
def test_log_prior_built_exactly_when_needed(la_tau, qwk_weight, expect_prior):
    """log_prior 的构造条件与损失函数的消费条件必须一一对齐。

    漏造 → `assert log_prior is not None` 崩;多造 → 无害但说明条件写宽了。
    这三组合覆盖了 `--epochs 24` 那轮云端 A/B(2-①/2-②)会真实用到的全部配置。
    """
    from scripts.train_official import build_official_head_config

    cfg = build_official_head_config(
        _base_train_cfg(ete_la_tau=la_tau, ete_qwk_weight=qwk_weight)
    )
    assert (cfg["ete_log_prior"] is not None) is expect_prior, (
        f"la_tau={la_tau}, qwk_weight={qwk_weight} 时 log_prior 应"
        f"{'存在' if expect_prior else '为 None'}"
    )
    assert cfg["ete_la_tau"] == la_tau
    assert cfg["ete_qwk_weight"] == qwk_weight

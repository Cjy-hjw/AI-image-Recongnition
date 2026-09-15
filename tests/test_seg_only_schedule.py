# -*- coding: utf-8 -*-
"""seg-only 冻结模式契约测试(阶段2,2026-09-13)。

schedule_official.configure_official_stage 的 seg_only=True 分支锁定:
1. 🔴 requires_grad:仅 SEG_TRAIN_PREFIXES(decoder+seg_backbone)为 True,
   四头/cls_backbone/Adaptor/vclub 全 False(vclub 显式冻——不照抄 ete_only
   的"vclub 恒放行",单变量原则:分割损失实验不该牵动 vclub)。
2. lr = lr_base × _STAGE_C_LR_SCALE(低档微调),恒定不随 epoch 变。
3. param_groups **两档 lr、四组**(每档按 decay/no_decay 拆两组;decoder@head_lr /
   seg_backbone@head_lr×backbone_lr_scale),不是 ete_only 式两组——
   seg_backbone 必须拿到缩放 lr,不能是满 head_lr。
4. stage_name 标识 SEG-only(云端日志判据)。
5. seg_only 与 ete_only 互斥:同时传 True 时 raise ValueError。
6. 签名兼容:seg_only=False(默认)时与旧版逐位一致。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.loss import MultiTaskLoss  # noqa: E402
from src.models.multi_task_model import MultiTaskThyroidModel  # noqa: E402
from src.training.schedule_official import (  # noqa: E402
    SEG_TRAIN_BACKBONE,
    SEG_TRAIN_DECODER,
    _STAGE_C_LR_SCALE,
    configure_official_stage,
    official_head_lr,
)


@pytest.fixture(scope="module")
def model_and_criterion():
    model = MultiTaskThyroidModel(pretrained=False)
    criterion = MultiTaskLoss()
    return model, criterion


LR_BASE = 0.01


def test_seg_only_freezes_everything_but_seg_chain(model_and_criterion):
    model, criterion = model_and_criterion
    configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=5e-4,
        freeze_legacy=True, seg_only=True)
    seg_prefixes = SEG_TRAIN_DECODER + SEG_TRAIN_BACKBONE
    wrong_unfrozen = [
        n for n, p in model.named_parameters()
        if p.requires_grad and not any(n.startswith(pfx) for pfx in seg_prefixes)
    ]
    assert not wrong_unfrozen, f"seg_only 下不该可训的参数: {wrong_unfrozen[:8]}"
    seg_trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad and any(n.startswith(pfx) for pfx in seg_prefixes)]
    assert seg_trainable, "分割前向链必须可训"
    # vclub 显式冻(与 ete_only 的"恒放行"相反——单变量原则)。
    vclub_trainable = [n for n, p in model.named_parameters()
                        if "vclub" in n and p.requires_grad]
    assert not vclub_trainable, f"vclub 应被冻结: {vclub_trainable}"


def test_seg_only_lr_is_stage_c_scale_constant(model_and_criterion):
    model, criterion = model_and_criterion
    expect = LR_BASE * _STAGE_C_LR_SCALE
    lrs = []
    for ep in (1, 3, 6, 8, 15, 20):
        _, head_lr, _ = configure_official_stage(
            model, criterion, epoch=ep, lr_base=LR_BASE, weight_decay=5e-4,
            freeze_legacy=True, seg_only=True)
        lrs.append(head_lr)
    assert all(abs(lr - expect) < 1e-12 for lr in lrs), lrs


def test_seg_only_param_groups_two_tier_four_groups(model_and_criterion):
    """decoder 组 lr = head_lr;seg_backbone 组 lr = head_lr × backbone_lr_scale
    (不是 ete_only 式两组 —— backbone 必须拿到缩放 lr,不能是满 lr)。"""
    model, criterion = model_and_criterion
    scale = 0.1
    groups, head_lr, stage_name = configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=5e-4,
        freeze_legacy=True, seg_only=True, backbone_lr_scale=scale)
    assert "SEG-only" in stage_name
    lrs_seen = sorted({round(g["lr"], 10) for g in groups})
    expect_head = round(head_lr, 10)
    expect_bb = round(head_lr * scale, 10)
    assert expect_head in lrs_seen, lrs_seen
    assert expect_bb in lrs_seen, lrs_seen
    assert expect_head != expect_bb  # 两档必须不同,否则测不出"缩放生效"

    # 逐组核对参数来源:decoder 组只含 SEG_TRAIN_DECODER 前缀,backbone 组
    # 只含 SEG_TRAIN_BACKBONE 前缀。
    name_of = {id(p): n for n, p in model.named_parameters()}
    for g in groups:
        lr = round(g["lr"], 10)
        for p in g["params"]:
            n = name_of[id(p)]
            if lr == expect_head:
                assert any(n.startswith(pfx) for pfx in SEG_TRAIN_DECODER), n
            elif lr == expect_bb:
                assert any(n.startswith(pfx) for pfx in SEG_TRAIN_BACKBONE), n


def test_seg_only_optimizer_covers_exactly_trainable(model_and_criterion):
    model, criterion = model_and_criterion
    groups, _, _ = configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=5e-4,
        freeze_legacy=True, seg_only=True)
    in_opt = {id(p) for g in groups for p in g["params"]}
    trainable = {id(p) for n, p in model.named_parameters()
                 if p.requires_grad and "vclub" not in n}
    assert in_opt == trainable, (
        f"optimizer 覆盖与 requires_grad 不一致:多余 {in_opt - trainable}, "
        f"遗漏 {trainable - in_opt}")


def test_seg_only_mal_loss_fn_frozen(model_and_criterion):
    """seg_only 下 AUCM 对偶变量必须冻结(与 ete_only 的 freeze_legacy 可选不同,
    seg_only 恒冻——单变量:分割实验不该带上 malignancy 对偶更新)。"""
    model, criterion = model_and_criterion
    configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=5e-4,
        freeze_legacy=False, seg_only=True)
    duals = list(criterion.mal_loss_fn.parameters())
    assert duals and all(not p.requires_grad for p in duals)


def test_seg_only_and_ete_only_mutually_exclusive(model_and_criterion):
    model, criterion = model_and_criterion
    with pytest.raises(ValueError):
        configure_official_stage(
            model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=5e-4,
            ete_only=True, seg_only=True)


def test_seg_only_default_path_unchanged(model_and_criterion):
    """seg_only=False(默认)行为与旧版一致:epoch=1 Stage A,四头+分割链可训。"""
    model, criterion = model_and_criterion
    groups, head_lr, stage_name = configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=5e-4,
        freeze_legacy=True, seg_only=False)
    assert abs(head_lr - official_head_lr(1, LR_BASE)) < 1e-12
    assert "Stage A" in stage_name
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("location_head") for n in trainable)


def test_seg_only_epoch_cap_matches_gamma_zero_window():
    """SEG_ABORT_EPOCH 与 γ 退火起点的数值关系用断言钉死(3B 专审 P2-4:
    "常量同源防漂移"原是无机制支撑的主张——γ 起点是 stages.py 的字面量 20,
    改那边不会带动 SEG_ABORT_EPOCH)。断言:cap 恰在 γ=0 窗口末尾,cap+1
    γ>0——重排 γ 起点而不改 cap 时此测试变红。"""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from src.training.stages import compute_gamma_prior
    from scripts.train_official import SEG_ABORT_EPOCH
    assert compute_gamma_prior(SEG_ABORT_EPOCH) == 0.0
    assert compute_gamma_prior(SEG_ABORT_EPOCH + 1) > 0.0

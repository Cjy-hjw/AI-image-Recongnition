# -*- coding: utf-8 -*-
"""ETE-only 冻结模式契约测试(D10-15,2026-09-12)。

schedule_official.configure_official_stage 的 ete_only=True 分支锁定:
1. 🔴 requires_grad:仅 extension_head.* 为 True(其余一切——cls/seg backbone、
   分割解码器、另三头、Adaptor——全 False);vclub 仍 True(独立 optimizer 管)。
2. lr = lr_base × _STAGE_C_LR_SCALE(低档微调),恒定不随 epoch 变。
3. param_groups 只含 extension_head 参数(+ 未冻时的 AUCM 对偶);
   分割/位置/T/TI-RADS 的参数一个都不进 optimizer。
4. stage_name 标识 ETE-only(云端日志判据)。
5. 签名兼容:ete_only=False(默认)时与旧版逐位一致(既有调度测试已覆盖,
   此处只验不炸)。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.loss import MultiTaskLoss  # noqa: E402
from src.models.multi_task_model import MultiTaskThyroidModel  # noqa: E402
from src.training.schedule_official import (  # noqa: E402
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


def test_ete_only_freezes_everything_but_extension_head(model_and_criterion):
    model, criterion = model_and_criterion
    configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=5e-4,
        freeze_legacy=True, ete_only=True)
    wrong_unfrozen = [
        n for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("extension_head")
        and "vclub" not in n
    ]
    assert not wrong_unfrozen, f"ete_only 下不该可训的参数: {wrong_unfrozen[:8]}"
    ete_trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("extension_head")]
    assert ete_trainable, "extension_head 必须可训"
    # 实测 7 键 = local_proj.0/1(weight/bias) + classifier.0/3(weight/bias)+
    # classifier.4?——以 named_parameters 实数为准(BN 的 running 统计不进
    # named_parameters,云端 10 键是含 BN 统计的 state_dict 口径)。
    assert len(ete_trainable) == 7, ete_trainable


def test_ete_only_lr_is_stage_c_scale_constant(model_and_criterion):
    model, criterion = model_and_criterion
    expect = LR_BASE * _STAGE_C_LR_SCALE
    lrs = []
    for ep in (1, 3, 6, 8):
        _, head_lr, _ = configure_official_stage(
            model, criterion, epoch=ep, lr_base=LR_BASE, weight_decay=5e-4,
            freeze_legacy=True, ete_only=True)
        lrs.append(head_lr)
    assert all(abs(lr - expect) < 1e-12 for lr in lrs), lrs


def test_ete_only_param_groups_exclude_frozen(model_and_criterion):
    model, criterion = model_and_criterion
    groups, head_lr, stage_name = configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=5e-4,
        freeze_legacy=True, ete_only=True)
    assert "ETE-only" in stage_name
    in_opt = set()
    for g in groups:
        for p in g["params"]:
            in_opt.add(id(p))
    bad = [n for n, p in model.named_parameters()
           if id(p) in in_opt and not n.startswith("extension_head")
           and "vclub" not in n]
    assert not bad, f"optimizer 里混入冻结参数: {bad[:8]}"
    assert abs(head_lr - LR_BASE * _STAGE_C_LR_SCALE) < 1e-12


def test_ete_only_aucs_dual_follows_freeze_legacy(model_and_criterion):
    """freeze_legacy=False 时 AUCM 对偶变量可训且进 param_groups(守旧任务口径)。"""
    model, criterion = model_and_criterion
    groups, _, _ = configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=5e-4,
        freeze_legacy=False, ete_only=True)
    duals = [p for p in criterion.mal_loss_fn.parameters()]
    assert duals and all(p.requires_grad for p in duals)
    in_opt = {id(p) for g in groups for p in g["params"]}
    assert all(id(p) in in_opt for p in duals)


def test_ete_only_default_path_unchanged(model_and_criterion):
    """ete_only=False(默认)行为与旧版一致:epoch=1 Stage A,四头+分割链可训。"""
    model, criterion = model_and_criterion
    groups, head_lr, stage_name = configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=5e-4,
        freeze_legacy=True, ete_only=False)
    assert abs(head_lr - official_head_lr(1, LR_BASE)) < 1e-12
    assert "Stage A" in stage_name
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("seg_head") for n in trainable), "默认路径分割头应可训"
    assert any(n.startswith("location_head") for n in trainable)

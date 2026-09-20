# -*- coding: utf-8 -*-
"""深监督辅助头契约测试(plan 任务D 首攻扩展,2026-09-20 新架构线)。

覆盖(照 CC-DiceCE C4 模板,含变异必须 RED 项):
1. extra_dict 恒含 seg_deepsup 键(权重 0 = 0.0 张量)。
2. 数值分解:L(w) - L(0) = w × 深监督项;extra["seg_deepsup"] 与差值一致。
3. 与主头同式对拍:aux=logits 时 deepsup 项 = 主项 BCE+Dice(masked)。
4. 两路等权平均:aux2≠aux3 时 seg_deepsup = 各自同式值的均值。
5. valid_mask=0 样本零贡献。
6. 权重>0 但 outputs 缺 aux 键 → fail-loud ValueError。
7. legacy 路径(无 valid_mask)deep_sup_weight>0 → raise。
8. 权重 0 时 aux 键缺席不影响(零回归:旧 outputs dict 无 aux 键照跑)。
9. 变异杀测:aux 只算 BCE(丢 Dice)/ 两路取 max(非平均)——各注入一次,
   对拍值必须变(区分力)。
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.losses.components.segmentation import (  # noqa: E402
    compute_segmentation_loss,
)
from src.models.losses.segmentation import DiceLoss  # noqa: E402


def _batch(B=2, H=48, W=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, 1, H, W, generator=g)
    gt = torch.zeros(B, 1, H, W)
    return logits, gt


def _outputs_with_aux(logits, aux2=None, aux3=None):
    """构造带 aux 键的 outputs;aux 缺省=克隆主 logits(对拍主项用)。"""
    return {
        "mask": logits,
        "seg_aux2": logits.clone() if aux2 is None else aux2,
        "seg_aux3": logits.clone() if aux3 is None else aux3,
    }


def _same_formula_per_sample(aux_logits, gt):
    """与主头同式的 per-sample BCE+Dice(独立写,勿复用被测代码)。"""
    bce = F.binary_cross_entropy_with_logits(
        aux_logits, gt, reduction="none").mean(dim=[1, 2, 3])
    sig = torch.sigmoid(aux_logits)
    inter = (sig * gt).sum(dim=[1, 2, 3])
    union = sig.sum(dim=[1, 2, 3]) + gt.sum(dim=[1, 2, 3])
    dice = 1.0 - (2.0 * inter + 1e-8) / (union + 1e-8)
    return bce + dice


# ── 1. extra_dict 恒含键 ──────────────────────────────────────────────────
def test_extra_dict_always_has_deepsup_key():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=2)
    gt[0, 0, 5:15, 5:15] = 1
    targets = {"mask": gt, "valid_mask": torch.ones(2)}
    # 权重 0:键恒在、值 0.0,且 aux 键缺席(旧 outputs)不炸——零回归。
    _, extra = compute_segmentation_loss(bce, dice, {"mask": logits}, targets,
                                         deepsup_weight=0.0)
    assert "seg_deepsup" in extra and float(extra["seg_deepsup"]) == 0.0
    # 权重 >0:键在、值 >0
    _, extra_on = compute_segmentation_loss(
        bce, dice, _outputs_with_aux(logits), targets, deepsup_weight=0.5)
    assert float(extra_on["seg_deepsup"]) > 0


# ── 2. 数值分解 ────────────────────────────────────────────────────────────
def test_weight_additivity_decomposition():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=2, seed=7)
    gt[0, 0, 8:18, 8:18] = 1
    gt[1, 0, 20:40, 5:15] = 1
    targets = {"mask": gt, "valid_mask": torch.ones(2)}
    outs = _outputs_with_aux(logits)
    l0, e0 = compute_segmentation_loss(bce, dice, outs, targets,
                                       deepsup_weight=0.0)
    l1, e1 = compute_segmentation_loss(bce, dice, outs, targets,
                                       deepsup_weight=1.0)
    assert float(l1 - l0) == pytest.approx(float(e1["seg_deepsup"]), abs=1e-6)
    assert float(e0["seg_deepsup"]) == 0.0


# ── 3. aux=主 logits 时,深监督项 = 主项 masked BCE+Dice ──────────────────
def test_aux_equals_main_formula():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=2, seed=3)
    gt[0, 0, 5:20, 5:20] = 1
    gt[1, 0, 10:30, 25:40] = 1
    targets = {"mask": gt, "valid_mask": torch.ones(2)}
    # 主项(权重全 0,只有 BCE+Dice)
    l_main, _ = compute_segmentation_loss(
        bce, dice, {"mask": logits}, targets,
        boundary_weight=0.0, lovasz_weight=0.0, ccdice_weight=0.0,
        deepsup_weight=0.0)
    # 深监督项(aux=主 logits,两路同值)
    _, e = compute_segmentation_loss(
        bce, dice, _outputs_with_aux(logits), targets, deepsup_weight=1.0)
    assert float(e["seg_deepsup"]) == pytest.approx(float(l_main), abs=1e-6)


# ── 4. 两路等权平均 ───────────────────────────────────────────────────────
def test_two_aux_equal_weight_mean():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=1, seed=11)
    gt[0, 0, 8:24, 8:24] = 1
    aux2 = torch.randn(1, 1, 48, 48, generator=torch.Generator().manual_seed(21))
    aux3 = torch.randn(1, 1, 48, 48, generator=torch.Generator().manual_seed(22))
    targets = {"mask": gt, "valid_mask": torch.ones(1)}
    _, e = compute_segmentation_loss(
        bce, dice, _outputs_with_aux(logits, aux2, aux3), targets,
        deepsup_weight=1.0)
    ref2 = _same_formula_per_sample(aux2, gt)
    ref3 = _same_formula_per_sample(aux3, gt)
    assert float(e["seg_deepsup"]) == pytest.approx(
        float(0.5 * (ref2[0] + ref3[0])), abs=1e-5)


# ── 5. valid_mask 屏蔽 ─────────────────────────────────────────────────────
def test_valid_mask_zero_contributes_no_deepsup():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=2, seed=13)
    gt[0, 0, 8:18, 8:18] = 1
    gt[1, 0, 20:30, 20:30] = 1
    aux2 = torch.randn(2, 1, 48, 48, generator=torch.Generator().manual_seed(31))
    aux3 = torch.randn(2, 1, 48, 48, generator=torch.Generator().manual_seed(32))
    vm = torch.tensor([0.0, 1.0])                    # 样本0 屏蔽
    _, e = compute_segmentation_loss(
        bce, dice, _outputs_with_aux(logits, aux2, aux3),
        {"mask": gt, "valid_mask": vm}, deepsup_weight=1.0)
    # 屏蔽样本不进均值:seg_deepsup = 样本1 的两路平均
    ref2 = _same_formula_per_sample(aux2, gt)
    ref3 = _same_formula_per_sample(aux3, gt)
    assert float(e["seg_deepsup"]) == pytest.approx(
        float(0.5 * (ref2[1] + ref3[1])), abs=1e-5)


# ── 6. 权重>0 缺 aux 键 → fail-loud ───────────────────────────────────────
def test_missing_aux_keys_raises():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=1)
    targets = {"mask": gt, "valid_mask": torch.ones(1)}
    with pytest.raises(ValueError, match="deepsup"):
        compute_segmentation_loss(bce, dice, {"mask": logits}, targets,
                                  deepsup_weight=0.3)


# ── 7. legacy 路径拒绝 ─────────────────────────────────────────────────────
def test_legacy_path_raises_on_deepsup_weight():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=1)
    with pytest.raises(ValueError, match="deepsup"):
        compute_segmentation_loss(
            bce, dice, _outputs_with_aux(logits), {"mask": gt},
            deepsup_weight=0.1)


# ── 8. 梯度有限且非零 ─────────────────────────────────────────────────────
def test_grad_finite_and_nonzero():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=2, seed=17)
    gt[0, 0, 8:18, 8:18] = 1
    aux2 = torch.randn(2, 1, 48, 48, generator=torch.Generator().manual_seed(41))
    aux3 = torch.randn(2, 1, 48, 48, generator=torch.Generator().manual_seed(42))
    aux2.requires_grad_(True)
    aux3.requires_grad_(True)
    logits = logits.requires_grad_(True)
    targets = {"mask": gt, "valid_mask": torch.ones(2)}
    loss, e = compute_segmentation_loss(
        bce, dice, _outputs_with_aux(logits, aux2, aux3), targets,
        deepsup_weight=1.0)
    loss.backward()
    for t, name in ((aux2, "aux2"), (aux3, "aux3")):
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
        assert float(t.grad.abs().sum()) > 0, f"{name} 梯度恒零"
    # 主 logits 也应有梯度(主项贡献,非深监督项——但>0 即链路通)
    assert float(logits.grad.abs().sum()) > 0


# ── 9. 变异杀测(区分力)────────────────────────────────────────────────
def test_mutation_aux_only_bce_killed():
    """变异:aux 只算 BCE 丢 Dice → 对拍值必变。"""
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=1, seed=23)
    gt[0, 0, 8:24, 8:24] = 1
    aux2 = torch.randn(1, 1, 48, 48, generator=torch.Generator().manual_seed(51))
    aux3 = torch.randn(1, 1, 48, 48, generator=torch.Generator().manual_seed(52))
    targets = {"mask": gt, "valid_mask": torch.ones(1)}
    _, e = compute_segmentation_loss(
        bce, dice, _outputs_with_aux(logits, aux2, aux3), targets,
        deepsup_weight=1.0)
    full = float(e["seg_deepsup"])
    # 变异版:独立复算"只 BCE"的均值
    bce2 = F.binary_cross_entropy_with_logits(
        aux2, gt, reduction="none").mean(dim=[1, 2, 3])
    bce3 = F.binary_cross_entropy_with_logits(
        aux3, gt, reduction="none").mean(dim=[1, 2, 3])
    mutated = float(0.5 * (bce2[0] + bce3[0]))
    assert full != pytest.approx(mutated, abs=1e-4), \
        "变异(丢 Dice)与原实现同值 = 测试无区分力"


def test_mutation_max_instead_of_mean_killed():
    """变异:两路取 max(非平均)→ 对拍值必变。"""
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=1, seed=29)
    gt[0, 0, 8:24, 8:24] = 1
    aux2 = torch.randn(1, 1, 48, 48, generator=torch.Generator().manual_seed(61))
    aux3 = torch.randn(1, 1, 48, 48, generator=torch.Generator().manual_seed(62))
    targets = {"mask": gt, "valid_mask": torch.ones(1)}
    _, e = compute_segmentation_loss(
        bce, dice, _outputs_with_aux(logits, aux2, aux3), targets,
        deepsup_weight=1.0)
    mean_val = float(e["seg_deepsup"])
    r2 = _same_formula_per_sample(aux2, gt)
    r3 = _same_formula_per_sample(aux3, gt)
    maxed = float(torch.maximum(r2, r3)[0])
    assert mean_val != pytest.approx(maxed, abs=1e-4), \
        "变异(取max)与原实现同值 = 测试无区分力"

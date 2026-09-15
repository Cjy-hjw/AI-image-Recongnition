# -*- coding: utf-8 -*-
"""compute_segmentation_loss 的 Boundary/Lovász 扩展契约测试(阶段2,2026-09-13)。

覆盖:
1. 权重=0 时与旧路径逐位一致(零回归)。
2. 两项在 GT 全零样本上恒为 0,不炸(阴性病例)。
3. valid_mask=0 样本梯度精确 0(与 BCE/Dice 同口径的屏蔽)。
4. Lovász 方向性:perfect(=0)< half < worst,梯度符号逐类正确
   (GT=1 像素梯度<=0,GT=0 像素梯度>=0)——第一版 inclusive-cumsum 实现
   符号错(worst 得负分),已按 Berman 官方 exclusive-cumsum 公式改正。
5. Boundary 方向性:aligned < shifted(带符号距离场,GT 内负/外正)——
   第一版纯内部距离(内部为正)方向恰好反了,已改正。
6. legacy 路径(无 valid_mask)对非零权重显式 raise,不静默丢弃变量。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.losses.components.segmentation import (  # noqa: E402
    boundary_loss_per_sample,
    compute_segmentation_loss,
    lovasz_hinge_per_sample,
)
from src.models.losses.segmentation import DiceLoss  # noqa: E402


@pytest.fixture
def bce_dice():
    return torch.nn.BCEWithLogitsLoss(), DiceLoss()


def _mock_batch(seed=0, B=4, H=64, W=64):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(B, 1, H, W, generator=g, requires_grad=True)
    gt = (torch.rand(B, 1, H, W, generator=g) > 0.8).float()
    return pred, gt


def test_zero_weight_matches_legacy_exactly(bce_dice):
    """权重=0 时数值逐位一致(零回归)。2026-09-14 起 compute_segmentation_loss
    返回 (loss, extra_dict) 契约;此测试同时验 legacy 与 masked 两条路径
    的全 1 valid_mask 数值等价(非只比 f(x) 与 f(x,0,0) 的自证式对拍)。"""
    bce_fn, dice_fn = bce_dice
    pred, gt = _mock_batch()
    # legacy 路径(无 valid_mask)
    l_leg, extra_leg = compute_segmentation_loss(
        bce_fn, dice_fn, {"mask": pred.detach()}, {"mask": gt})
    # masked 路径(全 1)+ 默认权重
    targets = {"mask": gt, "valid_mask": torch.ones(pred.shape[0])}
    l_default, extra_default = compute_segmentation_loss(
        bce_fn, dice_fn, {"mask": pred.detach()}, targets)
    # masked 路径 + 显式 0 权重
    l_explicit_zero, _ = compute_segmentation_loss(
        bce_fn, dice_fn, {"mask": pred.detach()}, targets,
        boundary_weight=0.0, lovasz_weight=0.0)
    assert abs(float(l_default) - float(l_explicit_zero)) < 1e-12
    # legacy(DiceLoss 标量路径)与 masked(per-sample 均值)数学等价,
    # 全 1 valid_mask 下数值差应 <1e-6(Gate2 实测 0.0000%)
    assert abs(float(l_leg) - float(l_default)) < 1e-6
    # 权重=0 时分项恒 0.0
    assert float(extra_default["seg_boundary"]) == 0.0
    assert float(extra_default["seg_lovasz"]) == 0.0
    assert float(extra_leg["seg_boundary"]) == 0.0


def test_all_zero_gt_returns_finite_zero_contribution(bce_dice):
    """GT 全零(阴性病例)时两项恒为 0,总损失有限不炸。"""
    bce_fn, dice_fn = bce_dice
    pred, _ = _mock_batch()
    gt_zero = torch.zeros_like(pred.detach())
    targets = {"mask": gt_zero, "valid_mask": torch.ones(pred.shape[0])}
    loss, extra = compute_segmentation_loss(
        bce_fn, dice_fn, {"mask": pred.detach()}, targets,
        boundary_weight=1.0, lovasz_weight=1.0)
    assert torch.isfinite(loss)
    assert float(extra["seg_boundary"]) == 0.0
    assert float(extra["seg_lovasz"]) == 0.0
    # 直接核对逐样本项确为 0
    b_per = boundary_loss_per_sample(torch.sigmoid(pred.detach()), gt_zero)
    l_per = lovasz_hinge_per_sample(pred.detach(), gt_zero)
    assert torch.all(b_per == 0.0)
    assert torch.all(l_per == 0.0)


def test_nonzero_weight_actually_changes_loss(bce_dice):
    """非 0 权重必须真的改变 loss + 分项出口(Gate2 变异 M8 存活教训:接线被删
    时其余测试全绿,此测试钉住"接线在")。l(w)-l(0) == w*term 数值分解。"""
    bce_fn, dice_fn = bce_dice
    pred, gt = _mock_batch(seed=3)
    targets = {"mask": gt, "valid_mask": torch.ones(pred.shape[0])}
    l0, e0 = compute_segmentation_loss(
        bce_fn, dice_fn, {"mask": pred.detach()}, targets)
    lw, ew = compute_segmentation_loss(
        bce_fn, dice_fn, {"mask": pred.detach()}, targets, boundary_weight=0.5)
    b_term = float(boundary_loss_per_sample(torch.sigmoid(pred.detach()), gt).mean())
    assert abs((float(lw) - float(l0)) - 0.5 * b_term) < 1e-6
    assert float(ew["seg_boundary"]) == pytest.approx(b_term, abs=1e-6)
    assert float(e0["seg_boundary"]) == 0.0  # 权重 0 时分项出口也为 0(恒写出)


def test_valid_mask_zero_gets_precisely_zero_gradient(bce_dice):
    """valid_mask=0 样本对总损失梯度贡献精确为 0(两项都开时)。"""
    bce_fn, dice_fn = bce_dice
    pred, gt = _mock_batch(B=4)
    targets = {"mask": gt, "valid_mask": torch.tensor([1., 0., 1., 0.])}
    loss, _ = compute_segmentation_loss(
        bce_fn, dice_fn, {"mask": pred}, targets,
        boundary_weight=0.5, lovasz_weight=0.5)
    loss.backward()
    assert pred.grad[1].abs().sum() == 0.0
    assert pred.grad[3].abs().sum() == 0.0
    assert pred.grad[0].abs().sum() > 0.0
    assert pred.grad[2].abs().sum() > 0.0


def test_legacy_path_raises_on_nonzero_weight(bce_dice):
    """无 valid_mask 的 legacy 路径不支持新损失,显式 raise(防静默丢变量)。"""
    bce_fn, dice_fn = bce_dice
    pred, gt = _mock_batch()
    with pytest.raises(ValueError):
        compute_segmentation_loss(
            bce_fn, dice_fn, {"mask": pred.detach()}, {"mask": gt},
            boundary_weight=0.3)
    with pytest.raises(ValueError):
        compute_segmentation_loss(
            bce_fn, dice_fn, {"mask": pred.detach()}, {"mask": gt},
            lovasz_weight=0.3)


class TestLovaszDirection:
    """第一版 inclusive-cumsum 实现符号错(worst 得负分);已改正为 Berman 官方
    exclusive-cumsum 公式,本类钉死方向性,防回退。"""

    def _gt_and_logits(self):
        gt = (torch.rand(1, 1, 32, 32) > 0.8).float()
        perfect = (gt * 2 - 1) * 10.0
        return gt, perfect

    def test_perfect_prediction_is_zero(self):
        gt, perfect = self._gt_and_logits()
        loss = lovasz_hinge_per_sample(perfect, gt)
        assert float(loss[0]) == pytest.approx(0.0, abs=1e-6)

    def test_worst_prediction_exceeds_half(self):
        gt, perfect = self._gt_and_logits()
        worst = -perfect
        half = torch.zeros_like(perfect)
        l_worst = float(lovasz_hinge_per_sample(worst, gt)[0])
        l_half = float(lovasz_hinge_per_sample(half, gt)[0])
        l_perfect = float(lovasz_hinge_per_sample(perfect, gt)[0])
        assert l_worst > l_half > l_perfect

    def test_gradient_sign_per_class(self):
        """GT=1 处增 logit 应降 loss(梯度<=0);GT=0 处相反(梯度>=0)。"""
        gt, _ = self._gt_and_logits()
        z = torch.zeros(1, 1, 32, 32, requires_grad=True)
        lovasz_hinge_per_sample(z, gt).sum().backward()
        g_pos = z.grad[0, 0][gt[0, 0] == 1]
        g_neg = z.grad[0, 0][gt[0, 0] == 0]
        assert g_pos.max() <= 0.0
        assert g_neg.min() >= 0.0


class TestBoundaryDirection:
    """第一版纯内部距离(GT 内为正)方向恰好反了(对齐预测 loss 反而更大);
    已改为带符号距离场(GT 内负/外正),本类钉死方向性,防回退。"""

    def _gt_and_probs(self):
        gt = (torch.rand(1, 1, 32, 32) > 0.8).float()
        perfect_logits = (gt * 2 - 1) * 10.0
        return gt, perfect_logits

    def test_aligned_lower_than_shifted(self):
        gt, perfect_logits = self._gt_and_probs()
        p_aligned = torch.sigmoid(perfect_logits)
        p_shifted = torch.sigmoid(perfect_logits.roll(4, dims=2))
        l_aligned = float(boundary_loss_per_sample(p_aligned, gt)[0])
        l_shifted = float(boundary_loss_per_sample(p_shifted, gt)[0])
        assert l_aligned < l_shifted

    def test_gradient_sign_per_class(self):
        """logit 空间:GT 内像素梯度<=0(升概率降 loss),GT 外相反。"""
        gt, _ = self._gt_and_probs()
        z = torch.zeros(1, 1, 32, 32, requires_grad=True)
        p = torch.sigmoid(z)
        boundary_loss_per_sample(p, gt).sum().backward()
        g_in = z.grad[0, 0][gt[0, 0] == 1]
        g_out = z.grad[0, 0][gt[0, 0] == 0]
        assert g_in.max() <= 0.0
        assert g_out.min() >= 0.0

    def test_all_zero_gt_is_zero(self):
        gt = torch.zeros(2, 1, 16, 16)
        p = torch.rand(2, 1, 16, 16)
        loss = boundary_loss_per_sample(p, gt)
        assert torch.all(loss == 0.0)

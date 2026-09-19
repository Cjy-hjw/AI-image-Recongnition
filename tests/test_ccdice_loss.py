# -*- coding: utf-8 -*-
"""CC-DiceCE 契约测试(plan 任务 C1,2026-09-18 新架构线)。

覆盖(plan C4 预注册清单,含变异必须 RED 项):
1. 独立朴素参考实现逐值对拍:K=0/单实例/面积悬殊多实例/接触实例。
2. extra_dict 恒含 seg_ccdice 键(权重 0 = 0.0 张量,消费端无条件 update)。
3. 数值分解:L(w) - L(0) = w × 有效 CC 项(权重可加性)。
4. valid_mask=0 样本梯度精确 0。
5. K=0 回落全局(非 0)与主项同构;软 GT 计数;B=0 不炸。
6. float32 归约 + 梯度有限且非零。
7. 变异杀测:删实例平均改像素加权 / 邻近实例背景不排除 / K=0 归 0 ——
   各注入一次变异,断言对拍值变(区分力)。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.losses.components.segmentation import (  # noqa: E402
    _cc_dice_ce_one,
    cc_dice_ce_per_sample,
    compute_segmentation_loss,
)
from src.models.losses.segmentation import DiceLoss  # noqa: E402


# ── 朴素参考实现(独立写,勿复用被测代码路径)──────────────────────────────
def naive_cc_dice_ce_one_sample(pred_sig, gt, pad_floor=2):
    """单样本朴素版:逐实例逐像素 Python 循环,直接按定义算。

    pred_sig/gt: (1,H,W)/(1,H,W) numpy 可转;实例=GT>0.5 的 8 连通分量。
    返回 (loss, K)。K=0 时回落全局 BCE+Dice(与被测同契约)。
    """
    import numpy as np
    from scipy import ndimage

    p = pred_sig[0].detach().cpu().numpy().astype(np.float64)   # (H,W)
    g = gt[0].detach().cpu().numpy()                            # (H,W)
    if g.ndim == 3:                                             # 容 (1,H,W) 传入
        g = g[0]
    if p.ndim == 3:
        p = p[0]
    labeled, n = ndimage.label(g > 0.5)
    if n == 0:
        return 0.0, 0          # P2 修复后口径:K=0 → 0(监督归主项)

    h, w = g.shape[-2], g.shape[-1]
    pad = max(pad_floor, int(np.ceil(0.1 * max(h, w))))
    per_inst = []
    for k in range(1, n + 1):
        inst = labeled == k
        ys, xs = np.nonzero(inst)
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
        # ROI BCE:本实例前景为正;背景 = ROI 内非本实例且非其他实例
        other = np.zeros_like(inst)
        for k2 in range(1, n + 1):
            if k2 != k:
                other |= (labeled == k2)
        pos = inst[y0:y1, x0:x1]
        bg = ~(pos | other[y0:y1, x0:x1])
        pw = p[y0:y1, x0:x1]
        eps = 1e-8
        n_pos, n_neg = pos.sum(), bg.sum()
        # 直接算:前景 -log p,背景 -log(1-p)
        pos_term = -np.log(pw[pos] + eps).sum() if n_pos else 0.0
        neg_term = -np.log(1 - pw[bg] + eps).sum() if n_neg else 0.0
        bce = (pos_term + neg_term) / (n_pos + n_neg + eps)
        gw = pos.astype(np.float64)
        inter = (pw * gw).sum()
        union = pw.sum() + gw.sum()
        dice = 1 - (2 * inter + eps) / (union + eps)
        per_inst.append(bce + dice)
    return float(sum(per_inst) / len(per_inst)), n


def _batch(B=2, H=48, W=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, 1, H, W, generator=g)
    gt = torch.zeros(B, 1, H, W)
    return logits, gt


# ── 1. 参考实现对拍 ────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", ["single", "multi_gap", "multi_edge_touch", "k0"])
def test_matches_naive_reference(case):
    logits, gt = _batch(B=1, seed=3)
    if case == "single":
        gt[0, 0, 10:20, 10:20] = 1
    elif case == "multi_gap":          # 面积悬殊 + 远距两实例
        gt[0, 0, 5:10, 5:10] = 1
        gt[0, 0, 30:45, 30:45] = 1
    elif case == "multi_edge_touch":   # 边接触(8 连通下合并为单实例)
        gt[0, 0, 10:15, 10:15] = 1
        gt[0, 0, 15:20, 10:15] = 1
    else:                              # K=0 回落全局
        pass
    ours, diag = cc_dice_ce_per_sample(logits, gt)
    ref, k_ref = naive_cc_dice_ce_one_sample(
        torch.sigmoid(logits).float(), gt)
    assert float(ours[0]) == pytest.approx(ref, abs=1e-5), \
        (float(ours[0]), ref)
    assert diag[0] == (1 if case != "k0" else 0)


def test_multi_edge_touch_is_one_instance_8conn():
    """边接触 = 8 连通单实例(ndimage 默认结构)——K 语义锚定。
    (对角接触不算:对角像素在 8 连通结构里同属一个分量,但两个**矩形块**
    只有角点相对时 scipy 也会分——按实测:仅边接触合并。)"""
    _, gt = _batch(B=1)
    gt[0, 0, 10:15, 10:15] = 1
    gt[0, 0, 15:20, 10:15] = 1
    logits = torch.randn(1, 1, 48, 48, generator=torch.Generator().manual_seed(1))
    ours, diag = cc_dice_ce_per_sample(logits, gt)
    assert diag[0] == 1                # 单实例(接触合一),非 2


# ── 2. extra_dict 契约 ─────────────────────────────────────────────────────
def test_extra_dict_always_has_ccdice_key():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=2)
    gt[0, 0, 5:15, 5:15] = 1
    targets = {"mask": gt, "valid_mask": torch.ones(2)}
    _, extra = compute_segmentation_loss(bce, dice, {"mask": logits}, targets,
                                         ccdice_weight=0.0)
    assert "seg_ccdice" in extra and float(extra["seg_ccdice"]) == 0.0
    assert "n_cc_supervised" not in extra            # 权重 0 不记计数
    _, extra_on = compute_segmentation_loss(bce, dice, {"mask": logits}, targets,
                                            ccdice_weight=0.5)
    assert float(extra_on["seg_ccdice"]) > 0
    assert extra_on["n_cc_supervised"] == 1
    assert extra_on["n_cc_k0_fallback"] == 1


def test_legacy_path_raises_on_ccdice_weight():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    logits, gt = _batch(B=1)
    with pytest.raises(ValueError, match="ccdice"):
        compute_segmentation_loss(bce, dice, {"mask": logits}, {"mask": gt},
                                  ccdice_weight=0.1)


# ── 3. 数值分解 L(w)-L(0)=w×CC ────────────────────────────────────────────
def test_weight_additivity_decomposition():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(2, 1, 48, 48, generator=g)
    gt = torch.zeros(2, 1, 48, 48)
    gt[0, 0, 8:18, 8:18] = 1
    gt[1, 0, 20:40, 5:15] = 1
    gt[1, 0, 20:40, 30:44] = 1
    targets = {"mask": gt, "valid_mask": torch.ones(2)}
    l0, e0 = compute_segmentation_loss(bce, dice, {"mask": logits}, targets,
                                       ccdice_weight=0.0)
    l1, e1 = compute_segmentation_loss(bce, dice, {"mask": logits}, targets,
                                       ccdice_weight=1.0)
    # l1 - l0 == 1.0 × (CC 均值),且 extra 的 seg_ccdice 与差值一致
    assert float(l1 - l0) == pytest.approx(float(e1["seg_ccdice"]), abs=1e-6)
    assert float(e0["seg_ccdice"]) == 0.0


# ── 4. valid_mask 屏蔽 ─────────────────────────────────────────────────────
def test_valid_mask_zero_contributes_no_cc():
    bce, dice = torch.nn.BCEWithLogitsLoss(), DiceLoss()
    g = torch.Generator().manual_seed(11)
    logits = torch.randn(2, 1, 48, 48, generator=g).requires_grad_(True)
    gt = torch.zeros(2, 1, 48, 48)
    gt[0, 0, 8:18, 8:18] = 1
    gt[1, 0, 20:30, 20:30] = 1
    vm = torch.tensor([0.0, 1.0])                    # 样本0 屏蔽
    _, extra = compute_segmentation_loss(bce, dice, {"mask": logits},
                                         {"mask": gt, "valid_mask": vm},
                                         ccdice_weight=1.0)
    # 屏蔽样本不进 CC 均值分母:seg_ccdice = 样本1 的 CC 值
    ours_only, _ = cc_dice_ce_per_sample(logits.detach()[1:2], gt[1:2])
    assert float(extra["seg_ccdice"]) == pytest.approx(float(ours_only[0]), abs=1e-6)


# ── 5. K=0 / 软 GT / B=0 边界 ─────────────────────────────────────────────
def test_k0_returns_zero_supervision_in_main_term():
    """K=0 → 0.0(🔴 P2 修复后口径,2026-09-18 审查):阴性监督由主项
    BCE+Dice 全额承担,CC 外挂不重复计权;原"回落全局"版同一监督计两次
    且全局/局部混纲。此测试钉死新口径,防"顺手改回回落"的回归。
    boundary/lovasz 同口径(三项对无实例样本一致)。"""
    logits = torch.randn(1, 1, 32, 32, generator=torch.Generator().manual_seed(5))
    gt = torch.zeros(1, 1, 32, 32)
    ours, diag = cc_dice_ce_per_sample(logits, gt)
    assert float(ours[0]) == 0.0
    assert diag == (0, 0, 1)


def test_soft_gt_counted_and_thresholded():
    """软 GT:一律计 n_soft(P2 修复口径);二值化后仍有实例 → CC 生效;
    塌零 → 置 0。n_k_pos 只收硬 GT。"""
    logits = torch.randn(2, 1, 32, 32, generator=torch.Generator().manual_seed(9))
    gt = torch.zeros(2, 1, 32, 32)
    gt[0, 0, 10:20, 10:20] = 0.7                     # 软值>0.5 → CC 生效
    gt[1, 0, 10:20, 10:20] = 0.3                     # 软值<0.5 → 塌零置 0
    ours, diag = cc_dice_ce_per_sample(logits, gt)
    assert diag == (0, 2, 0), diag                   # 2 个 soft,0 个 k_pos
    assert float(ours[0]) > 0                        # 样本0 CC 真算(非 0)
    assert float(ours[1]) == 0.0                     # 样本1 塌零置 0


def test_batch_zero_returns_empty():
    ours, diag = cc_dice_ce_per_sample(torch.zeros(0, 1, 8, 8),
                                       torch.zeros(0, 1, 8, 8))
    assert ours.shape == (0,) and diag == (0, 0, 0)


# ── 6. 精度与梯度 ──────────────────────────────────────────────────────────
def test_bf16_autocast_finite_grad():
    """bf16 autocast 下梯度有限且非零(实例像素必被触及)。

    🔴 F.binary_cross_entropy 在 autocast 区内会 raise "unsafe to
    autocast"——这是 torch 的保护(BCE 要求输入过 sigmoid,数值域必须
    精确)。训练循环里本函数在 autocast 内被调用的路径**事先 sigmoid +
    .float() 上转**,局部算子已在 float32 域;此处按同口径在 autocast
    外先算 sigmoid/float,autocast 区内仅做窗口切片归约。
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA 不可用,本地 CPU 路径已被其它用例覆盖")
    logits = torch.randn(2, 1, 64, 64, device="cuda",
                         generator=torch.Generator(device='cuda').manual_seed(0)
                         ).requires_grad_(True)
    gt = torch.zeros(2, 1, 64, 64, device="cuda")
    gt[0, 0, 10:25, 10:25] = 1
    gt[1, 0, 40:50, 40:50] = 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # 复现训练路径:入口 logits.float() 使 autocast 不再下罩,
        # sigmoid/BCE 全程 float32(见被测函数 docstring autocast 条目)
        ours, _ = cc_dice_ce_per_sample(logits, gt)
    ours.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_cpu_grad_finite_and_nonzero():
    logits = torch.randn(1, 1, 48, 48,
                         generator=torch.Generator().manual_seed(13)).requires_grad_(True)
    gt = torch.zeros(1, 1, 48, 48)
    gt[0, 0, 15:30, 15:30] = 1
    ours, _ = cc_dice_ce_per_sample(logits, gt)
    ours.sum().backward()
    assert torch.isfinite(logits.grad).all()
    # 前景内像素必须收到非零梯度(实例等权 → 小前景梯度密度更高)
    assert logits.grad[0, 0, 18:27, 18:27].abs().sum() > 0


# ── 7. 变异杀测(区分力)───────────────────────────────────────────────────
def _sample_multi_inst():
    logits = torch.randn(1, 1, 48, 48,
                         generator=torch.Generator().manual_seed(17))
    gt = torch.zeros(1, 1, 48, 48)
    gt[0, 0, 5:8, 5:8] = 1                           # 小实例 9px
    gt[0, 0, 30:45, 30:45] = 1                       # 大实例 225px
    return logits, gt


def test_mutation_instance_mean_vs_pixel_weighted():
    """变异:实例等权 → 像素加权(大实例主导)。对拍值必须显著变化。"""
    logits, gt = _sample_multi_inst()
    ours, _ = cc_dice_ce_per_sample(logits, gt)
    # 朴素像素加权版:两实例 loss 按面积加权
    from scipy import ndimage
    import numpy as np
    lab, n = ndimage.label(gt[0, 0].numpy() > 0.5)
    p = torch.sigmoid(logits).float()[0, 0]
    per, areas = [], []
    for k in range(1, n + 1):
        m = torch.from_numpy((lab == k)).to(p.device)
        inter = (p * m.float()).sum(); union = p.sum() + m.float().sum()
        per.append(1 - (2 * inter + 1e-8) / (union + 1e-8))
        areas.append(float(m.sum()))
    weighted = sum(l_ * a for l_, a in zip(per, areas)) / sum(areas)
    assert abs(float(ours[0]) - float(weighted)) > 0.05, \
        "实例等权与像素加权无区分力(变异未被杀)"


def test_mutation_neighbor_bg_exclusion_matters():
    """变异:邻近实例背景不排除(其前景被当负背景)。构造贴合双实例 +
    预测盖住两实例,与朴素参考实现逐值对拍(排除语义在参考实现里同样成立)。"""
    logits = torch.randn(1, 1, 64, 64,
                         generator=torch.Generator().manual_seed(19))
    gt = torch.zeros(1, 1, 64, 64)
    gt[0, 0, 20:30, 20:30] = 1
    gt[0, 0, 20:30, 31:41] = 1                       # 邻接(1px 隔)双实例
    # 预测盖住两实例(邻接前景也高概率)——排除语义生效的场景
    logits_cover = logits.clone()
    logits_cover[0, 0, 20:30, 20:41] = 8.0
    ours_cover, _ = cc_dice_ce_per_sample(logits_cover, gt)
    ref, _ = naive_cc_dice_ce_one_sample(torch.sigmoid(logits_cover).float(), gt)
    assert float(ours_cover[0]) == pytest.approx(ref, abs=1e-5)


def test_mutation_k0_global_not_zero_killed():
    """变异杀测:K=0 改回"回落全局"(非 0)必须 RED——防重复计权回归
    (P2 修复的反向钉子;与 test_k0_returns_zero_supervision_in_main_term
    成对,一正一反)。"""
    logits = torch.randn(1, 1, 32, 32,
                         generator=torch.Generator().manual_seed(23))
    gt = torch.zeros(1, 1, 32, 32)
    ours, diag = cc_dice_ce_per_sample(logits, gt)
    assert float(ours[0]) == 0.0 and diag[2] == 1

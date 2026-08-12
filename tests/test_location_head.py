"""位置头契约测试（Phase2 阶段 2.1）。

守的是「错了不会崩、只会静默学反」的那几条：
  1. 因子分解的**拼接顺序**必须与 `LOCATION_LABELS` 一致 —— 顺序错了概率和仍为 1、
     JSON 仍合法，但左叶预测会填进右叶的槽。
  2. 7 维**严格**和为 1（官方 `LocationProbs` 字段要求）。
  3. 峡部样本的 level 分支必须屏蔽，且不能让 `cross_entropy` 因 -1 越界崩。
  4. CoordConv 必须真的打破平移等变性 —— 否则「左/右」在原理上就学不出来。
  5. 水平翻转必须联动翻标签。
"""

import pytest
import torch

from src.inference.prediction_json import LOCATION_LABELS, NUM_LOCATIONS
from src.models.blocks import CoordConv2d
from src.models.heads.location import (
    ISTHMUS_LOCATION_INDEX,
    LOCATION_TO_LEVEL,
    LOCATION_TO_SIDE,
    NUM_LEVELS,
    NUM_SIDES,
    SIDE_ISTHMUS,
    SIDE_LEFT,
    SIDE_RIGHT,
    LocationHead,
    build_level_validity,
    compose_location_probs,
    flip_location_index,
)
from src.models.losses.components.location import compute_location_loss

IN_CH = 352  # = MultiTaskThyroidModel.CLS_POOLED_DIM


# ── 1. 因子分解顺序 ────────────────────────────────────────────────────────


def _onehot(idx, n):
    v = torch.zeros(1, n)
    v[0, idx] = 1.0
    return v


@pytest.mark.parametrize(
    "side, level, expect_idx",
    [
        (SIDE_LEFT, 0, 0),   # 左上叶
        (SIDE_LEFT, 1, 1),   # 左中叶
        (SIDE_LEFT, 2, 2),   # 左下叶
        (SIDE_RIGHT, 0, 3),  # 右上叶
        (SIDE_RIGHT, 1, 4),  # 右中叶
        (SIDE_RIGHT, 2, 5),  # 右下叶
    ],
    ids=[LOCATION_LABELS[i] for i in range(6)],
)
def test_compose_puts_probability_in_the_right_slot(side, level, expect_idx):
    """确定性 one-hot 输入下，全部概率必须落在 LOCATION_LABELS 对应的那一格。

    这是本文件最重要的一条：顺序写错不会有任何异常表现（和仍为 1），
    只会让左右/上下系统性错位。
    """
    probs = compose_location_probs(_onehot(side, NUM_SIDES), _onehot(level, NUM_LEVELS))
    assert probs.shape == (1, NUM_LOCATIONS)
    assert probs[0, expect_idx] == pytest.approx(1.0, abs=1e-6), (
        f"({side},{level}) 应落在 idx={expect_idx}({LOCATION_LABELS[expect_idx]})，"
        f"实际分布={probs[0].tolist()}"
    )


def test_isthmus_probability_comes_straight_from_side_head():
    """峡部不参与 side×level 乘积，直接取侧别头的第 3 类。"""
    side = torch.tensor([[0.2, 0.3, 0.5]])
    level = torch.tensor([[0.1, 0.6, 0.3]])
    probs = compose_location_probs(side, level)
    assert probs[0, ISTHMUS_LOCATION_INDEX] == pytest.approx(0.5, abs=1e-6)


def test_probs_sum_to_one_exactly():
    """官方 LocationProbs 要求和为 1。这里是数学恒等式，不是靠再归一化。"""
    torch.manual_seed(0)
    side = torch.softmax(torch.randn(16, NUM_SIDES), dim=-1)
    level = torch.softmax(torch.randn(16, NUM_LEVELS), dim=-1)
    probs = compose_location_probs(side, level)
    assert torch.allclose(probs.sum(-1), torch.ones(16), atol=1e-6)
    assert (probs >= 0).all()


def test_factor_tables_are_consistent_with_labels():
    """映射表与 LOCATION_LABELS 的中文标签自洽（左/右/峡、上/中/下）。"""
    for idx, label in enumerate(LOCATION_LABELS):
        side, level = LOCATION_TO_SIDE[idx], LOCATION_TO_LEVEL[idx]
        if label == "峡部":
            assert side == SIDE_ISTHMUS and level == -1
            continue
        assert (side == SIDE_LEFT) == label.startswith("左")
        assert (side == SIDE_RIGHT) == label.startswith("右")
        assert level == {"上": 0, "中": 1, "下": 2}[label[1]]


# ── 2. 峡部与有效性 ────────────────────────────────────────────────────────


def test_level_validity_masks_isthmus():
    idx = torch.tensor([0, ISTHMUS_LOCATION_INDEX, 4])
    valid = torch.tensor([1.0, 1.0, 1.0])
    assert build_level_validity(idx, valid).tolist() == [1.0, 0.0, 1.0]


def test_level_validity_respects_location_validity():
    """位置本身无标注的样本，level 也必须无效（两个条件是与关系）。"""
    idx = torch.tensor([0, 1])
    assert build_level_validity(idx, torch.tensor([0.0, 1.0])).tolist() == [0.0, 1.0]


def test_flip_swaps_left_right_keeps_isthmus():
    idx = torch.arange(NUM_LOCATIONS)
    flipped = flip_location_index(idx)
    assert flipped.tolist() == [3, 4, 5, 0, 1, 2, 6]
    # 翻两次回到原样
    assert flip_location_index(flipped).tolist() == idx.tolist()
    # 层级不变，只换侧
    for i in range(6):
        assert LOCATION_TO_LEVEL[flipped[i].item()] == LOCATION_TO_LEVEL[i]
        assert LOCATION_TO_SIDE[flipped[i].item()] != LOCATION_TO_SIDE[i]


# ── 3. 损失 ────────────────────────────────────────────────────────────────


def test_loss_does_not_crash_on_isthmus_negative_level_target():
    """峡部 level 标签是 -1，未 clamp 会 IndexError。反证该保护是必需的。"""
    logits = torch.randn(2, 3)
    idx = torch.tensor([ISTHMUS_LOCATION_INDEX, 0])
    total, _ = compute_location_loss(logits, logits, idx, torch.ones(2))
    assert torch.isfinite(total)
    # 反证：直接把 -1 喂 cross_entropy 确实会崩（若哪天 torch 不再报错，本保护可复议）
    with pytest.raises((IndexError, RuntimeError)):
        torch.nn.functional.cross_entropy(logits, torch.tensor([-1, 0]))


def test_loss_rejects_2d_validity_that_would_broadcast_silently():
    """🔴 `[B,1]` validity 会隐式广播成 `[B,B]` 放大 loss 且**不报错**（QA_VP §2.2）。

    实测证据：同一批 3 样本，传 `[B]` 得 total=1.98、传 `[B,1]` 得 4.575 —— 数值错
    但训练照跑，是最难查的一类。故必须 fail-loud 而非依赖调用方自觉。
    """
    logits = torch.randn(3, 3)
    idx = torch.tensor([0, 1, 2])
    with pytest.raises(AssertionError, match=r"必须是 \[B\] 一维"):
        compute_location_loss(logits, logits, idx, torch.ones(3, 1))

    # 反证：确认广播坑真实存在（若哪天 torch 改成报错，本防护可复议）
    assert (torch.ones(3) * torch.ones(3, 1)).shape == (3, 3), (
        "[B]*[B,1] 不再广播成 [B,B]，本防护的前提已变，请复查"
    )


def test_loss_rejects_out_of_range_labels_both_directions():
    """标签越界两个方向都要 fail-loud。

    正负**不对称**是这条测试存在的理由：`idx=7` 本就会 IndexError，而 `idx=-1` 走
    python 负索引、静默取到末元素（恰是峡部的 -1），再被 clamp+validity 吞成"无监督"
    —— 上游若漏设 validity，这类样本会无声无息地丢掉监督。
    """
    logits = torch.randn(2, 3)
    for bad in (7, -1):
        with pytest.raises(AssertionError, match="位置标签越界"):
            compute_location_loss(
                logits, logits, torch.tensor([bad, 0]), torch.ones(2)
            )


def test_loss_zero_and_no_grad_when_all_invalid():
    """全无效样本 → 损失精确 0、梯度精确 0，且不产生 NaN（同 test_masked_loss 范式）。"""
    side = torch.randn(4, 3, requires_grad=True)
    level = torch.randn(4, 3, requires_grad=True)
    total, detail = compute_location_loss(
        side, level, torch.tensor([0, 1, 2, 3]), torch.zeros(4)
    )
    assert total.item() == 0.0
    assert torch.isfinite(total)
    total.backward()
    assert side.grad.abs().sum().item() == 0.0
    assert level.grad.abs().sum().item() == 0.0


def test_loss_level_branch_silent_when_batch_is_all_isthmus():
    """整批峡部 → level 分支无监督（loss 0），side 分支照常有值。"""
    side = torch.randn(3, 3, requires_grad=True)
    level = torch.randn(3, 3, requires_grad=True)
    idx = torch.full((3,), ISTHMUS_LOCATION_INDEX)
    total, detail = compute_location_loss(side, level, idx, torch.ones(3))
    assert detail["location_level"].item() == 0.0
    assert detail["location_side"].item() > 0.0
    total.backward()
    assert level.grad.abs().sum().item() == 0.0, "峡部不该给 level 分支任何梯度"
    assert side.grad.abs().sum().item() > 0.0


def test_loss_denominator_counts_only_valid_samples():
    """屏蔽样本不得稀释分母：3 个样本里只 1 个有效时，损失 == 该样本单独的损失。"""
    torch.manual_seed(0)
    side = torch.randn(3, 3)
    level = torch.randn(3, 3)
    idx = torch.tensor([0, 1, 2])
    v = torch.tensor([0.0, 1.0, 0.0])
    total_masked, _ = compute_location_loss(side, level, idx, v)
    total_single, _ = compute_location_loss(
        side[1:2], level[1:2], idx[1:2], torch.ones(1)
    )
    assert total_masked.item() == pytest.approx(total_single.item(), abs=1e-6)


# ── 4. CoordConv 与头 ──────────────────────────────────────────────────────


def test_coordconv_needs_nonlinearity_before_pooling():
    """🔴 钉死一条极易被无声破坏的设计约束：CoordConv 后必须先过非线性再池化。

    卷积是线性的，`GAP(conv([x,c])) = GAP(conv_x(x)) + GAP(conv_c(c))`：第二项是
    与输入无关的常数偏置，第一项平移不变 —— 所以**裸 CoordConv + GAP 对左右图案
    逐位相等**，坐标通道形同虚设。加 ReLU 后拆分不成立，坐标偏置调制哪些位置被
    激活，位置信息才进得了池化结果。

    本测试三段都必须成立，缺一则说明前提变了：
      A) 普通卷积 + GAP：左右不可分（基线，证明画布/图案构造本身没混入 padding 效应）
      B) 裸 CoordConv + GAP：左右**仍**不可分（反证 —— 这正是不能先池化的原因）
      C) CoordConv + ReLU + GAP：左右可分（这才是 LocationHead 的实际结构）

    图案避开边界：`padding=1` 时贴边图案两侧 padding 影响不对称，会让 A 段先失败、
    并把 padding 效应混进 C 段的差异里污染区分力（本测试第一版就踩了这个）。
    8 宽画布、kernel=3，图案放 col 2-3 / col 5-6，响应区 col 1-4 / col 4-7 全在界内。
    """
    torch.manual_seed(0)
    left = torch.zeros(1, 4, 8, 8)
    left[:, :, 3:5, 2:4] = 1.0
    right = torch.zeros(1, 4, 8, 8)
    right[:, :, 3:5, 5:7] = 1.0

    def gap(module, t):
        with torch.no_grad():
            return module(t).mean(dim=(2, 3))

    # A) 基线：普通卷积平移等变
    plain = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False).eval()
    assert torch.allclose(gap(plain, left), gap(plain, right), atol=1e-6), (
        "基线失效：普通卷积本应平移等变，说明画布构造混入了边界效应"
    )

    # B) 反证：裸 CoordConv + GAP 同样不可分
    coord = CoordConv2d(4, 8, kernel_size=3).eval()
    assert torch.allclose(gap(coord, left), gap(coord, right), atol=1e-6), (
        "裸 CoordConv+GAP 竟能区分左右 —— 与线性可拆分的推导矛盾，请复查本约束是否仍成立"
    )

    # C) 实际结构：加了非线性才可分
    with_relu = torch.nn.Sequential(coord, torch.nn.ReLU()).eval()
    assert not torch.allclose(gap(with_relu, left), gap(with_relu, right), atol=1e-4), (
        "CoordConv+ReLU 池化后左右仍不可分 —— 坐标通道没起作用，位置头学不出左/右叶"
    )


def test_location_head_can_distinguish_left_from_right():
    """端到端：整头（CoordConv→BN→ReLU→pool）对左右图案必须给出不同的侧别 logits。

    比上一条更贴近真实用法 —— 若有人重排 `LocationHead.spatial` 里的层序
    （例如把池化提前），这条会直接红。
    """
    torch.manual_seed(0)
    head = LocationHead(in_channels=8, hidden_channels=16).eval()
    left = torch.zeros(1, 8, 16, 16)
    left[:, :, 7:9, 3:6] = 1.0
    right = torch.zeros(1, 8, 16, 16)
    right[:, :, 7:9, 10:13] = 1.0
    with torch.no_grad():
        sl = head(left)["side_logits"]
        sr = head(right)["side_logits"]
    assert not torch.allclose(sl, sr, atol=1e-4), (
        f"整头对左右图案给出相同侧别 logits({sl.tolist()} vs {sr.tolist()}) —— "
        "spatial 分支的层序可能被改坏（池化必须在非线性之后）"
    )


def test_coordconv_output_shape_and_resolution_independence():
    conv = CoordConv2d(8, 16, kernel_size=3)
    for hw in (8, 16, 32):
        assert conv(torch.randn(2, 8, hw, hw)).shape == (2, 16, hw, hw)


def test_head_forward_shapes_and_normalization():
    torch.manual_seed(0)
    head = LocationHead(in_channels=IN_CH).eval()
    out = head(torch.randn(2, IN_CH, 16, 16))
    assert out["side_logits"].shape == (2, NUM_SIDES)
    assert out["level_logits"].shape == (2, NUM_LEVELS)
    assert out["location_probs"].shape == (2, NUM_LOCATIONS)
    assert torch.allclose(
        out["location_probs"].sum(-1), torch.ones(2), atol=1e-6
    )


def test_head_rejects_pooled_input():
    """传 pool 后的 [B,C] 必须 fail-loud —— CoordConv 需要 H/W。"""
    head = LocationHead(in_channels=IN_CH)
    with pytest.raises(AssertionError, match="空间特征图"):
        head(torch.randn(2, IN_CH))


def test_head_rejects_wrong_channel_count():
    """208 是 adaptor 输入通道，写错要立刻报，而不是等真训练。"""
    head = LocationHead(in_channels=IN_CH)
    with pytest.raises(AssertionError, match="通道失配"):
        head(torch.randn(2, 208, 16, 16))


def test_head_backward_reaches_all_parameters():
    """端到端反传：每个参数都要拿到梯度，防某分支被意外断开。"""
    head = LocationHead(in_channels=IN_CH)
    out = head(torch.randn(2, IN_CH, 16, 16))
    total, _ = compute_location_loss(
        out["side_logits"], out["level_logits"], torch.tensor([0, 4]), torch.ones(2)
    )
    total.backward()
    missing = [n for n, p in head.named_parameters() if p.grad is None]
    assert not missing, f"这些参数没收到梯度: {missing}"

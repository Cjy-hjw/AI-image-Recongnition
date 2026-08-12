"""临床 T 头（Phase2 阶段 2.3）单测。

判据强度参照 `tests/test_extension_head.py`：每条断言都要能被**写错的实现**推翻。
本文件里有区分力的几条（移除对应实现即失败，已逐条反证过）：
  - `test_t_conf_is_probability_of_positive_not_max_prob` —— 填成 max prob 即失败
  - `test_head_is_invariant_to_ete_logit_shift` —— 直接用 raw logits 即失败
  - `test_head_survives_zero_scale_without_nan` —— 去掉 log 的 clamp 即失败
  - `test_head_uses_no_batchnorm1d` —— 加回 BN1d 即失败（train + batch=1）
  - `test_cb_weights_silently_degenerate_at_clinical_t_counts` —— 钉死"别抄 β=0.99"
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heads.clinical_t import (
    NUM_CLIN_T_CLASSES,
    T_POSITIVE_INDEX,
    ClinicalTHead,
)
from src.models.losses.components.clinical_t import (
    compute_clinical_t_loss,
    inverse_frequency_weights,
)
from src.models.losses.components.extension import class_balanced_weights

IN_CH = 352
FEAT_HW = 16
CLIN_T_COUNTS = [2410, 1831]  # docs/训练与架构/训练数据定稿.md:87，阳性率 35.7%


def _head(**kw):
    torch.manual_seed(0)
    return ClinicalTHead(in_channels=IN_CH, **kw)


def _inputs(batch=2, scale=0.1, seed=0):
    torch.manual_seed(seed)
    feat = torch.randn(batch, IN_CH, FEAT_HW, FEAT_HW)
    nodule_scale = torch.full((batch, 1), float(scale))
    ete = torch.randn(batch, 4)
    return feat, nodule_scale, ete


# ── 输出契约 ──────────────────────────────────────────────────────────────


def test_head_forward_shapes_and_normalization():
    out = _head()(*_inputs(3))
    assert out["t_logits"].shape == (3, NUM_CLIN_T_CLASSES)
    assert out["t_probs"].shape == (3, NUM_CLIN_T_CLASSES)
    # t_conf 必须是 [B] 一维：[B,1] 与 [B] 相乘会隐式广播成 [B,B]（QA_VP §2.2）。
    assert out["t_conf"].shape == (3,), (
        f"t_conf 应为 [B] 一维，got {tuple(out['t_conf'].shape)}"
    )
    assert torch.allclose(out["t_probs"].sum(-1), torch.ones(3), atol=1e-6)
    # 概率一律 fp32（官方要读，低精度会把小概率抹掉）。
    assert out["t_probs"].dtype == torch.float32


def test_t_conf_is_probability_of_positive_not_max_prob():
    """🔴 `T_conf` = P(T=1)，**不是** max prob，也不是"对所报 T 值的置信度"。

    区分力：若实现写成 `t_probs.max(-1).values`，强阴性样本的 t_conf 会是 ≈1.0
    而非 ≈0.0，本测试立刻失败。口径依据见 `heads/clinical_t.py` 模块 docstring
    （与 `Malignancy.confidence` 的既有语义自洽）。

    做法：绕开随机初始化的头，直接把已知 logits 塞进 head 的输出构造逻辑 ——
    这里通过替换最后一层为常数输出来固定 logits。
    """
    head = _head().eval()
    # 把 classifier 末层换成"忽略输入、恒输出 [10, -10]"（强预测 T=0）。
    last = head.classifier[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.tensor([10.0, -10.0]))
    out = head(*_inputs(2))
    assert float(out["t_probs"][0, 0]) > 0.99, "构造失败：应是强 T=0 预测"
    assert float(out["t_conf"][0]) < 0.01, (
        f"t_conf={float(out['t_conf'][0]):.4f} —— 强阴性样本却给出高置信度，"
        " 说明 t_conf 取成了 max prob 或反了类别下标"
    )
    # 反向再确认一次，防"恒返回 0"式的假通过。
    with torch.no_grad():
        last.bias.copy_(torch.tensor([-10.0, 10.0]))
    out2 = head(*_inputs(2))
    assert float(out2["t_conf"][0]) > 0.99
    assert T_POSITIVE_INDEX == 1


# ── ETE 旁路：detach + softmax ────────────────────────────────────────────


def test_head_is_invariant_to_ete_logit_shift():
    """ETE 输入走 softmax，故 `logits` 与 `logits + c` 表示同一分布 → 输出必须相同。

    区分力：若实现直接把 raw logits 拼进 MLP，加常数 c=5 会显著改变输出。
    这同时验证了"softmax 只丢掉一个无意义的平移自由度"这一无损性主张。
    """
    head = _head().eval()
    feat, scale, ete = _inputs(2)
    with torch.no_grad():
        a = head(feat, scale, ete)["t_logits"]
        b = head(feat, scale, ete + 5.0)["t_logits"]
    assert torch.allclose(a, b, atol=1e-5), (
        f"ETE 平移后输出变了（最大差 {float((a - b).abs().max()):.4e}）"
        " —— 说明用的是 raw logits 而非 softmax"
    )


def test_head_does_not_backprop_into_ete_logits():
    """🔴 T 的梯度不得反灌侵犯头 —— ETE 是官方独立计分维度。

    含反证 C：改变 ete 必须改变输出，否则 A 只是"这条输入根本没接上"。
    """
    head = _head()
    feat, scale, ete = _inputs(2)
    feat.requires_grad_(True)
    ete.requires_grad_(True)
    head(feat, scale, ete)["t_logits"].sum().backward()
    assert ete.grad is None, "ETE logits 拿到了梯度 —— detach 丢了"
    assert feat.grad is not None and float(feat.grad.abs().sum()) > 0, (
        "stage4_spatial 没梯度 —— 上一条失去区分力"
    )
    head.eval()
    with torch.no_grad():
        o1 = head(feat, scale, ete)["t_logits"]
        o2 = head(feat, scale, ete + torch.tensor([[9.0, -9.0, 0.0, 0.0]]))["t_logits"]
    assert not torch.allclose(o1, o2, atol=1e-6), (
        "改变 ETE 分布不改变输出 —— ETE 旁路是死的，detach 断言零区分力"
    )


def test_head_does_not_backprop_into_nodule_scale():
    head = _head()
    feat, scale, ete = _inputs(2)
    scale.requires_grad_(True)
    head(feat, scale, ete)["t_logits"].sum().backward()
    assert scale.grad is None, "nodule_scale 拿到了梯度 —— 会回流分割主干"


# ── 尺度标量的 log 通道 ───────────────────────────────────────────────────


def test_head_survives_zero_scale_without_nan():
    """空 mask（阴性病例）→ scale 恰为 0 → `log 0 = -inf`。

    区分力：去掉 `clamp(min=1e-4)` 后本测试立刻产出 NaN 并失败。
    """
    head = _head().eval()
    feat, _, ete = _inputs(2)
    with torch.no_grad():
        out = head(feat, torch.zeros(2, 1), ete)
    assert torch.isfinite(out["t_logits"]).all(), "scale=0 产出非有限值 —— log clamp 丢了"
    assert torch.isfinite(out["t_probs"]).all()


def test_scale_input_has_log_channel_with_equal_spacing_per_ratio():
    """log 通道存在且语义正确：等倍数的 scale 对，在该通道上间距恒等。

    🔴 **这里必须用白盒 hook，黑盒判据零区分力**（已实证）：最初写的是"0.001 与
    0.01 必须给出不同输出"，但把 log 通道换成第二份线性 `s` 后该测试**照样全绿**
    —— `scale_proj` 的 Linear 能把 0.009 的线性差放大成任意可测的输出差。
    "输出不同"这件事根本碰不到 log 通道在不在。

    真正被 log 通道改变的是**表示的几何**：三对都是 2 倍关系，线性通道上的差
    相差 40 倍（0.001/0.01/0.04），log 通道上恒为 ln2≈0.6931。断言后者即可。
    """
    head = _head().eval()
    captured = []
    head.scale_proj.register_forward_pre_hook(lambda m, inp: captured.append(inp[0].detach()))
    feat, _, ete = _inputs(1)
    pairs = [(0.001, 0.002), (0.01, 0.02), (0.04, 0.08)]
    with torch.no_grad():
        for lo, hi in pairs:
            for s in (lo, hi):
                head(feat, torch.full((1, 1), s), ete)

    lin_gaps, log_gaps = [], []
    for i, (lo, hi) in enumerate(pairs):
        a, b = captured[2 * i], captured[2 * i + 1]
        assert a.shape == (1, 2), f"scale_proj 输入应为 [B,2]，got {tuple(a.shape)}"
        assert abs(float(a[0, 0]) - lo) < 1e-6, "第 0 通道应是线性 s 原值"
        lin_gaps.append(float(b[0, 0] - a[0, 0]))
        log_gaps.append(float(b[0, 1] - a[0, 1]))

    import math
    for g in log_gaps:
        assert abs(g - math.log(2)) < 1e-4, (
            f"第 1 通道在 2 倍关系上的间距应恒为 ln2≈0.6931，实测 {log_gaps}"
            " —— 该通道不是 log（换成第二份线性 s 时这里会变成 0.001/0.01/0.04）"
        )
    # 反证：线性通道确实**不**等距，否则上面那条可能只是"两列碰巧都是 log"。
    assert max(lin_gaps) / min(lin_gaps) > 30, (
        f"线性通道间距比应达 40 倍，实测 {lin_gaps} —— 反证失效，复议本测试"
    )


# ── batch=1 与归一化层选型 ────────────────────────────────────────────────


def test_head_uses_no_batchnorm1d():
    """🔴 BN1d 在 train 模式 batch=1 直接 ValueError（实测），而末批可能只剩 1 样本。

    两条断言互补：结构检查说明意图，train+batch=1 实跑才是真判据
    （有人换成 `BatchNorm1d` 的变体或自定义 BN 时，结构检查可能漏，实跑不会）。
    """
    head = _head()
    assert not any(isinstance(m, nn.BatchNorm1d) for m in head.modules()), (
        "临床 T 头里出现了 BatchNorm1d —— train 模式下 batch=1 会 ValueError，"
        " 需要归一化请用 LayerNorm"
    )
    head.train()
    out = head(*_inputs(1))  # train 模式、batch=1：这行就是判据
    assert out["t_logits"].shape == (1, NUM_CLIN_T_CLASSES)


def test_head_batch_one_eval_inference():
    """交付推理 batch 恒 1。"""
    head = _head().eval()
    with torch.no_grad():
        out = head(*_inputs(1))
    assert torch.isfinite(out["t_logits"]).all()
    assert abs(float(out["t_probs"].sum()) - 1.0) < 1e-6


# ── 形状契约 ─────────────────────────────────────────────────────────────


def test_head_rejects_wrong_shapes():
    head = _head()
    feat, scale, ete = _inputs(2)
    with pytest.raises(AssertionError, match="空间特征图"):
        head(feat.mean(dim=(2, 3)), scale, ete)  # 已池化的 [B,C]
    with pytest.raises(AssertionError, match="通道失配"):
        head(torch.randn(2, 208, FEAT_HW, FEAT_HW), scale, ete)  # 208 是 adaptor 输入通道
    with pytest.raises(AssertionError, match=r"nodule_scale 期望"):
        head(feat, scale.squeeze(1), ete)  # [B] 而非 [B,1]
    with pytest.raises(AssertionError, match="ete_logits 期望"):
        head(feat, scale, torch.randn(2, 3))
    with pytest.raises(AssertionError, match="batch 不一致"):
        head(feat, torch.full((3, 1), 0.1), ete)


def test_head_backward_reaches_all_parameters():
    """每个参数都要有梯度 —— 否则某条通路是死的（如 scale_proj 没接进 classifier）。"""
    head = _head()
    head(*_inputs(2))["t_logits"].sum().backward()
    dead = [n for n, p in head.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"这些参数没拿到梯度: {dead}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast 需要 CUDA")
def test_head_finite_under_bf16_autocast():
    head = _head().cuda().eval()
    feat, scale, ete = (t.cuda() for t in _inputs(2))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = head(feat, scale, ete)
    assert torch.isfinite(out["t_logits"]).all()
    assert torch.allclose(out["t_probs"].sum(-1).float(), torch.ones(2, device="cuda"), atol=1e-3)


# ── 类权重 ───────────────────────────────────────────────────────────────


def test_inverse_frequency_weights_at_official_counts():
    w = inverse_frequency_weights(CLIN_T_COUNTS)
    assert abs(float(w.mean()) - 1.0) < 1e-6, "必须归一化到均值 1"
    ratio = float(w[1] / w[0])
    assert abs(ratio - 2410 / 1831) < 1e-4, f"比值应等于逆频率 1.3162，got {ratio:.4f}"


def _cb_ratio(counts, beta):
    w = class_balanced_weights(counts, beta)
    return float(w[1] / w[0])


def test_cb_weights_silently_degenerate_at_clinical_t_counts():
    """🔴 把侵犯头的 β=0.99 抄到临床 T 上 = **权重比恒 1.0**，静默失效。

    钉住这个实测，是为了让日后"统一一下超参"的人必须先看到这个数：CB 的
    `1 − β^n` 在 n 千级时已饱和，两类 `E_n` 塌成同一常数。不报错、不告警。
    """
    assert abs(_cb_ratio(CLIN_T_COUNTS, 0.99) - 1.0) < 1e-6, (
        "β=0.99 在千级计数上竟然产生了非 1 的权重比 —— 前提变了，复议本注释与"
        " losses/components/clinical_t.py 的实测表"
    )
    # β 要拉到 0.9999 才勉强有效，且仍不及逆频率（1.2800 vs 1.3162）。
    ratio_9999 = _cb_ratio(CLIN_T_COUNTS, 0.9999)
    assert 1.27 < ratio_9999 < 1.29, f"β=0.9999 权重比实测应 ≈1.2800，got {ratio_9999:.4f}"
    assert ratio_9999 < 2410 / 1831, "CB 在此计数下仍弱于逆频率"
    # 反证：同一函数在 ETE 的长尾计数上**确实**有效（20.4:1），否则上面两条会被
    # 误读成"CB 实现坏了"，而真相是"CB 不适用于这个计数分布"。
    ete_ratio = float((lambda w: w[3] / w[0])(class_balanced_weights([3906, 267, 67, 5], 0.99)))
    assert 20.0 < ete_ratio < 21.0, f"CB 在 ETE 长尾上应给出 ≈20.4:1，got {ete_ratio:.4f}"


# ── 损失 ─────────────────────────────────────────────────────────────────


def test_loss_rejects_2d_validity_that_would_broadcast_silently():
    with pytest.raises(AssertionError, match="一维"):
        compute_clinical_t_loss(torch.randn(2, 2), torch.tensor([0, 1]), torch.ones(2, 1))


def test_loss_rejects_out_of_range_labels_both_directions():
    for bad in (torch.tensor([0, 2]), torch.tensor([0, -1])):
        with pytest.raises(AssertionError, match="越界"):
            compute_clinical_t_loss(torch.randn(2, 2), bad, torch.ones(2))


def test_loss_zero_and_no_grad_when_all_invalid():
    logits = torch.randn(3, 2, requires_grad=True)
    total, _ = compute_clinical_t_loss(logits, torch.zeros(3, dtype=torch.long), torch.zeros(3))
    total.backward()
    assert float(total) == 0.0
    assert float(logits.grad.abs().sum()) == 0.0
    assert bool(torch.isfinite(logits.grad).all())


def test_loss_denominator_counts_only_valid_samples():
    """分母只数有效样本：复制无效样本不得改变损失值。"""
    logits = torch.randn(2, 2)
    idx = torch.tensor([0, 1])
    base, _ = compute_clinical_t_loss(logits, idx, torch.tensor([1.0, 0.0]))
    padded, _ = compute_clinical_t_loss(
        torch.cat([logits, torch.randn(3, 2)]),
        torch.cat([idx, torch.tensor([1, 0, 1])]),
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert abs(float(base) - float(padded)) < 1e-6


def test_loss_weights_are_indexed_by_label_not_prediction():
    """类权重按**标签**取，不是按 argmax 取。

    构造：两个样本 logits 完全相同、标签相反。若按预测索引，两者权重会一样，
    总损失对调换权重不敏感；按标签索引则敏感。
    """
    logits = torch.zeros(2, 2)
    idx = torch.tensor([0, 1])
    a, _ = compute_clinical_t_loss(logits, idx, torch.ones(2),
                                   class_weights=torch.tensor([0.5, 1.5]))
    b, _ = compute_clinical_t_loss(logits, idx, torch.ones(2),
                                   class_weights=torch.tensor([1.5, 0.5]))
    # 两个样本对称，总损失相同 —— 换个只让一个样本有效的构造才有区分力。
    a1, _ = compute_clinical_t_loss(logits, idx, torch.tensor([1.0, 0.0]),
                                    class_weights=torch.tensor([0.5, 1.5]))
    b1, _ = compute_clinical_t_loss(logits, idx, torch.tensor([1.0, 0.0]),
                                    class_weights=torch.tensor([1.5, 0.5]))
    assert abs(float(a) - float(b)) < 1e-6, "对称构造下应相等（正常性检查）"
    assert float(a1) < float(b1), (
        f"只有标签 0 有效时，w[0]=0.5 应给出更小的损失（{float(a1):.4f} vs {float(b1):.4f}）"
        " —— 若按预测索引则两者相等"
    )


def test_loss_reduces_to_plain_ce_without_weights():
    logits = torch.randn(4, 2)
    idx = torch.tensor([0, 1, 1, 0])
    total, detail = compute_clinical_t_loss(logits, idx, torch.ones(4))
    expect = F.cross_entropy(logits.float(), idx)
    assert abs(float(total) - float(expect)) < 1e-6
    assert abs(float(detail["clin_t_ce"]) - float(expect)) < 1e-6


def test_loss_rejects_invalid_label_smoothing():
    with pytest.raises(AssertionError, match="label_smoothing"):
        compute_clinical_t_loss(torch.randn(2, 2), torch.tensor([0, 1]), torch.ones(2),
                                label_smoothing=1.0)


def _train_to_convergence(ls, weights=None, steps=400):
    """在可完美分离的数据上训到收敛，返回阳性样本的 P(T=1) 均值。

    固定两处 seed（数据 / 初始化），使不同配置之间可直接比较。
    """
    torch.manual_seed(0)
    X = torch.cat([torch.randn(200, 4) - 2.0, torch.randn(200, 4) + 2.0])
    y = torch.cat([torch.zeros(200, dtype=torch.long), torch.ones(200, dtype=torch.long)])
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        total, _ = compute_clinical_t_loss(
            net(X), y, torch.ones(400), class_weights=weights, label_smoothing=ls
        )
        total.backward()
        opt.step()
    return float(torch.softmax(net(X), -1)[200:, T_POSITIVE_INDEX].mean())


@pytest.mark.parametrize("ls,expect_ceiling", [(0.0, 1.0), (0.05, 0.975), (0.1, 0.95)])
def test_label_smoothing_caps_t_conf_at_one_minus_half_ls(ls, expect_ceiling):
    """🔴 LS 把 `P(T=1)` 的上界钳死在 `1 − ls/2`，而官方读的正是这个数。

    这不是理论推导后写下的期望值，是训到收敛的实测（可完美分离的数据，400 步）：
    ls=0.05 → 0.9750、ls=0.1 → 0.9500。钉住它，是为了让日后想开 LS 的人先看到
    交付置信度会被压缩多少。

    ⚠️ 该上界**只在 `class_weights is None` 时成立** —— 见下一个测试。
    """
    p_pos = _train_to_convergence(ls)
    assert abs(p_pos - expect_ceiling) < 0.02, (
        f"ls={ls} 下阳性样本 P(T=1)={p_pos:.4f}，期望 ≈{expect_ceiling}（= 1 − ls/2）"
    )


def test_label_smoothing_ceiling_breaks_when_class_weights_are_used():
    """🔴 `1 − ls/2` 上界在 `class_weights` 非 None 时**失效**（Gate2 查出，已复现）。

    根因：`F.cross_entropy` 的 `weight` 会按 `w[c]` 加权**整个平滑后的 soft target**，
    包括 `ls/C` 那个均匀项 —— 于是加权把上界推高。这是 PyTorch 的标准语义，不是 bug；
    钉住它是为了让"上界 1−ls/2"不被当成永远有效的护栏。

    实测（同一份数据与初始化，ls=0.1）：不加权 0.9500 / `w=[0.863,1.137]` 0.9615 /
    `w=[1,10]` 0.9947（几乎完全失效）。

    区分力：若哪天有人改实现把 `weight` 从 uniform 项解耦，加权后的值会回落到 0.95，
    本测试立刻变红 —— 那正是需要被注意到的行为变更，而不是"悄悄修好了"。
    """
    plain = _train_to_convergence(0.1, None)
    weighted = _train_to_convergence(0.1, inverse_frequency_weights(CLIN_T_COUNTS))
    extreme = _train_to_convergence(0.1, torch.tensor([1.0, 10.0]))
    assert abs(plain - 0.95) < 0.02, f"未加权基线应 ≈0.9500，got {plain:.4f}"
    assert weighted > plain + 0.005, (
        f"加权后 P(T=1)={weighted:.4f} 未高于未加权 {plain:.4f} —— "
        " weight 与 label_smoothing 的交互行为变了，复议本测试与损失 docstring"
    )
    assert extreme > 0.98, (
        f"极端权重 [1,10] 下 P(T=1)={extreme:.4f}，实测应 ≈0.9947（上界护栏几乎失效）"
    )


def test_loss_finite_under_extreme_logits():
    """±1000 的 logits 下损失与梯度仍有限（log_softmax 的稳定实现）。"""
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True)
    total, _ = compute_clinical_t_loss(logits, torch.tensor([1, 0]), torch.ones(2))
    total.backward()
    assert torch.isfinite(total) and bool(torch.isfinite(logits.grad).all())


# ── 端到端：参数级梯度隔离 ────────────────────────────────────────────────


def test_end_to_end_t_loss_leaves_seg_backbone_cbm_and_ete_head_untouched():
    """🔴 用**真** `MultiTaskThyroidModel` + 真 `ExtensionHead` 串起来跑。

    上面 head 级测试只断言"入参没拿到梯度"（接口契约）；QA_VP §1.1 要的是**参数级**
    证据。三条隔离各有各的通路：
      - `seg_backbone`：T 只吃 `nodule_scale_detached`，到不了分割主干（gate8 红线）。
      - `cbm_head`：T 不消费 margin，本就不该有梯度。
      - **`ExtensionHead` 的参数**：这条是本阶段特有的 —— ete_logits 在 T 头内 detach，
        T 的监督不得重塑 ETE 表征（ETE 是官方独立计分维度）。
      - 反证：`cls_backbone` **必须**有梯度，否则上面三条只是"头没接上"的空转。
    """
    from src.models.multi_task_model import MultiTaskThyroidModel
    from src.models.heads.extension import ExtensionHead

    torch.manual_seed(0)
    model = MultiTaskThyroidModel(pretrained=False, config={"data": {"img_size": 128}})
    ete_head = ExtensionHead(in_channels=model.CLS_POOLED_DIM)
    t_head = ClinicalTHead(in_channels=model.CLS_POOLED_DIM)

    outputs = model(torch.randn(2, 1, 128, 128), gamma_prior=0.0)
    ete_out = ete_head(
        outputs["stage4_spatial"], outputs["soft_mask_detached"], outputs["margin"]
    )
    t_out = t_head(
        outputs["stage4_spatial"],
        outputs["nodule_scale_detached"],
        ete_out["ete_logits"],
    )
    total, _ = compute_clinical_t_loss(
        t_out["t_logits"], torch.tensor([0, 1]), torch.ones(2)
    )
    total.backward()

    def grad_sum(module_or_prefix):
        if isinstance(module_or_prefix, str):
            return sum(
                float(p.grad.abs().sum())
                for n, p in model.named_parameters()
                if n.startswith(module_or_prefix) and p.grad is not None
            )
        return sum(
            float(p.grad.abs().sum())
            for p in module_or_prefix.parameters()
            if p.grad is not None
        )

    assert grad_sum("seg_backbone") == 0.0, (
        f"T 损失污染了 seg_backbone（{grad_sum('seg_backbone')}）—— gate8 前提被破坏"
    )
    assert grad_sum("cbm_head") == 0.0, f"T 损失污染了 cbm_head（{grad_sum('cbm_head')}）"
    assert grad_sum(ete_head) == 0.0, (
        f"T 损失污染了 ExtensionHead（{grad_sum(ete_head)}）"
        " —— ete_logits 的 detach 丢了，ETE 表征会被 T 任务拽偏"
    )
    assert grad_sum("cls_backbone") > 0.0, (
        "cls_backbone 没有梯度 —— stage4_spatial 通路断了，上面三条断言失去区分力"
    )

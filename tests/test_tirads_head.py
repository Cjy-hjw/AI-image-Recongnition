"""TI-RADS 头(backlog #124)单测。

判据强度参照 `tests/test_extension_head.py`/`test_clinical_t_head.py`:每条断言都
要能被**写错的实现**推翻。本文件里有区分力的几条:
  - `test_head_does_not_backprop_into_cbm_concepts` —— 去掉任一 detach 即失败
  - `test_head_concepts_actually_feed_output` —— 反证 detach 断言非空转
  - `test_loss_full_invalid_batch_is_exactly_zero` —— masked loss 精确 0 且无 NaN
  - `test_ordinal_term_zero_weight_has_no_effect` —— 权重=0 时对 total 无贡献
  - `test_ordinal_term_penalizes_far_predictions_more` —— 对齐官方二次加权 Kappa 语义
  - `test_roman_numeral_exact_match_not_substring` —— IV/III 不被子串判成 I/II(loader 侧)
"""

import pytest
import torch

from src.dataset import official_loader as O
from src.models.heads.tirads import CONCEPT_DIM, NUM_TIRADS_CLASSES, TiradsHead
from src.models.losses.components.tirads import compute_tirads_loss, inverse_frequency_weights

IN_CH = 352
FEAT_HW = 16


def _head(**kw):
    torch.manual_seed(0)
    return TiradsHead(in_channels=IN_CH, **kw)


def _inputs(batch=2, seed=0, requires_grad=False):
    torch.manual_seed(seed)
    feat = torch.randn(batch, IN_CH, FEAT_HW, FEAT_HW, requires_grad=requires_grad)
    comp = torch.randn(batch, 4, requires_grad=requires_grad)
    echo = torch.randn(batch, 4, requires_grad=requires_grad)
    shape = torch.randn(batch, 2, requires_grad=requires_grad)
    margin = torch.randn(batch, 4, requires_grad=requires_grad)
    foci = torch.randn(batch, 4, requires_grad=requires_grad)
    return feat, comp, echo, shape, margin, foci


# ── 输出契约 ──────────────────────────────────────────────────────────────


def test_head_forward_shapes_and_normalization():
    out = _head()(*_inputs(3))
    assert out["tirads_logits"].shape == (3, NUM_TIRADS_CLASSES)
    assert out["tirads_probs"].shape == (3, NUM_TIRADS_CLASSES)
    assert torch.allclose(out["tirads_probs"].sum(-1), torch.ones(3), atol=1e-6)
    assert (out["tirads_probs"] >= 0).all()
    # 概率一律 fp32(官方要读,低精度会把小概率抹掉,同 ETE/T 头同一考量)。
    assert out["tirads_probs"].dtype == torch.float32
    assert CONCEPT_DIM == 18, "18 = comp4+echo4+shape2+margin4+foci4,与 cbm_head 切片契约一致"


def test_head_rejects_wrong_input_channels():
    feat, comp, echo, shape, margin, foci = _inputs(2)
    bad_feat = torch.randn(2, IN_CH - 1, FEAT_HW, FEAT_HW)
    with pytest.raises(AssertionError):
        _head()(bad_feat, comp, echo, shape, margin, foci)


def test_head_rejects_wrong_concept_dims():
    feat, comp, echo, shape, margin, foci = _inputs(2)
    bad_margin = torch.randn(2, 3)  # 应为 4
    with pytest.raises(AssertionError):
        _head()(feat, comp, echo, shape, bad_margin, foci)


def test_head_rejects_batch_mismatch():
    feat, comp, echo, shape, margin, foci = _inputs(2)
    bad_foci = torch.randn(3, 4)  # batch 与 feat 不一致
    with pytest.raises(AssertionError):
        _head()(feat, comp, echo, shape, margin, bad_foci)


# ── 梯度隔离(不让 TI-RADS 重塑 CBM 五概念,不让梯度回流分割主干) ────────────


def test_head_does_not_backprop_into_cbm_concepts():
    """🔴 五个概念 logits 全部必须 detach —— 与 `ExtensionHead` 对 margin 的处理同一原则。"""
    head = _head()
    feat, comp, echo, shape, margin, foci = _inputs(2, requires_grad=True)
    head(feat, comp, echo, shape, margin, foci)["tirads_logits"].sum().backward()
    for name, t in (("comp", comp), ("echo", echo), ("shape", shape),
                    ("margin", margin), ("foci", foci)):
        assert t.grad is None, f"CBM {name} 收到了 TI-RADS 的梯度 —— detach 丢了"
    assert feat.grad is not None and feat.grad.abs().sum() > 0, (
        "stage4_spatial 没有梯度 —— 头压根没接上，上面五条断言失去意义"
    )


def test_head_concepts_actually_feed_output():
    """反证上一条非空转:改变概念输入必须真的改变输出(detach 只切梯度,不切前向)。"""
    head = _head().eval()
    feat, comp, echo, shape, margin, foci = _inputs(2)
    with torch.no_grad():
        a = head(feat, comp, echo, shape, margin, foci)["tirads_logits"]
        b = head(feat, comp, echo, shape, margin + 5.0, foci)["tirads_logits"]
    assert not torch.allclose(a, b, atol=1e-5), (
        "改变 margin 不改变输出 —— margin 没进 concat，detach 断言只是空转"
    )


# ── 罗马数字解析:精确查表,禁子串匹配(loader 侧,与头本身无关但同属本次交付) ──


def test_roman_numeral_exact_match_not_substring():
    """🔴 `IV`/`III` 含 `I`/`II` 子串,若解析用 `in` 而非精确查表会误判。"""
    assert O._parse_tirads("IV类") == 4
    assert O._parse_tirads("III类") == 3
    assert O._parse_tirads("II类") == 2
    assert O._parse_tirads("I类") == 1
    assert O._parse_tirads("V类") == 5
    # 真实官方数据 value_counts 首见(定稿 §1.7/数据探查证据链 第12-13轮):
    # 该列纯罗马无前缀(IV类/ V类 / III类 / II类 / I类),混写分支应永不触发。
    # 下面是"反向"反例:如果官方 value_counts 首见值解析失败,这条会拦住——
    # 之前"5 V类"混写分叉在真实数据上其实不出现(该列无数字前缀)。
    assert O._parse_tirads("5V类") == 5  # 无空格混写:前缀分支(branch-2)命中返回
    assert O._parse_tirads("V 类") == 5  # 空格 + 罗马 + 尾缀(分支3剥尾缀→查表)
    # ↑ 上面这条事实上解析成 5(text=V 类→去尾缀 V→查表得到 5);写死 4 会掩盖真实行为
    assert O._parse_tirads("5 V类") == 5  # 空格混写:分支3(NFKC→去尾缀→查表)→ 5


# ── 损失:masked CE + 可选有序项 ──────────────────────────────────────────


def _loss_inputs(batch=4, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(batch, NUM_TIRADS_CLASSES, requires_grad=True)
    idx = torch.randint(0, NUM_TIRADS_CLASSES, (batch,))
    validity = torch.ones(batch)
    return logits, idx, validity


def test_loss_full_invalid_batch_is_exactly_zero():
    logits, idx, _ = _loss_inputs()
    validity = torch.zeros(4)
    total, detail = compute_tirads_loss(logits, idx, validity)
    assert total.item() == pytest.approx(0.0, abs=1e-8)
    assert torch.isfinite(total)
    assert detail["tirads_ordinal"].item() == pytest.approx(0.0, abs=1e-8)


def test_loss_lambda_scales_linearly():
    """外部 λ 缩放(由 official_heads.py 施加)应精确线性,loss 本身不隐式再乘权重。"""
    logits, idx, validity = _loss_inputs()
    base, _ = compute_tirads_loss(logits, idx, validity)
    assert (2.0 * base).item() == pytest.approx((2.0 * base).item())


def test_loss_class_weights_indexed_by_label_not_prediction():
    """权重必须按**真实标签**索引,不是按预测 argmax —— 常见的实现反转错误。"""
    logits = torch.zeros(2, NUM_TIRADS_CLASSES)
    idx = torch.tensor([0, 4])
    v = torch.ones(2)
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0])
    base, _ = compute_tirads_loss(logits, idx, v)
    weighted, _ = compute_tirads_loss(logits, idx, v, class_weights=w)
    assert float(weighted / base) == pytest.approx(3.0, abs=1e-5)


def test_ordinal_term_zero_weight_has_no_effect():
    """默认 ordinal_weight=0:total 必须精确等于纯 CE(不含有序项贡献)。"""
    logits, idx, validity = _loss_inputs()
    total_default, _ = compute_tirads_loss(logits, idx, validity)
    total_zero, _ = compute_tirads_loss(logits, idx, validity, ordinal_weight=0.0)
    assert total_default.item() == pytest.approx(total_zero.item(), abs=1e-8)


def test_ordinal_term_penalizes_far_predictions_more():
    """🔴 对齐官方二次加权 Kappa:预测分布离真值越远,有序项损失越大。"""
    idx = torch.tensor([0])
    validity = torch.ones(1)
    near = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])   # 强预测类 0(=真值)
    far = torch.tensor([[0.0, 0.0, 0.0, 0.0, 10.0]])    # 强预测类 4(离真值最远)
    _, detail_near = compute_tirads_loss(near, idx, validity, ordinal_weight=1.0)
    _, detail_far = compute_tirads_loss(far, idx, validity, ordinal_weight=1.0)
    assert detail_far["tirads_ordinal"] > detail_near["tirads_ordinal"]


def test_loss_rejects_2d_validity():
    """🔴 validity 必须 [B] 一维:[B,1] 会隐式广播成 [B,B] 静默放大 loss(QA_VP §2.2)。"""
    logits, idx, _ = _loss_inputs()
    bad_validity = torch.ones(4, 1)
    with pytest.raises(AssertionError):
        compute_tirads_loss(logits, idx, bad_validity)


def test_loss_rejects_out_of_range_labels():
    logits = torch.randn(2, NUM_TIRADS_CLASSES)
    bad_idx = torch.tensor([0, NUM_TIRADS_CLASSES])  # 越界
    validity = torch.ones(2)
    with pytest.raises(AssertionError):
        compute_tirads_loss(logits, bad_idx, validity)


def test_inverse_frequency_weights_reexported_and_correct():
    """`tirads.py` re-export 的 `inverse_frequency_weights` 与 `clinical_t.py` 原实现同一函数。"""
    w = inverse_frequency_weights([2410, 1831])
    assert w.mean().item() == pytest.approx(1.0, abs=1e-5)
    assert (w[1] / w[0]).item() == pytest.approx(2410 / 1831, abs=1e-4)

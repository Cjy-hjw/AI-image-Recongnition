"""Phase 2 阶段 2.0 —— 三头接入接口的契约测试。

守四条契约（每条都对应一个「删掉就静默出错」的真实风险）：
  1. 形状：`stage4_spatial` 通道数是 CLS_POOLED_DIM(352) 而非 CLS_STAGE4_CHANNELS(208)。
     写错位置头第一层 in_channels 会在真训练时才崩。
  2. 梯度方向：`stage4_spatial` 必须**有**梯度（位置头要训 cls_backbone），
     两个 `*_detached` 必须**无**梯度。
  3. 🔴 隔离：从 `stage4_spatial` 反传，`seg_backbone` 梯度必须**精确 0** ——
     这是 `gate8_adaptor_isolation` 精确零断言的前提，也是 Stage2 崩塌四因之一
     （物理先验通路回流）不被三头重新打开的保证。
  4. FP16 求和溢出：`mask_features.nodule_scale_from_mask` 的 fp32 累加器不是防御性冗余，
     去掉它在 fp16 路径下会溢出成 inf（512²=262144 > fp16 上限 65504）。
"""

import pytest
import torch

from src.models.mask_features import nodule_scale_from_mask
from src.models.multi_task_model import ModelOutputs, MultiTaskThyroidModel

NEW_KEYS = ("stage4_spatial", "soft_mask_detached", "nodule_scale_detached")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = MultiTaskThyroidModel(pretrained=False)
    return m


@pytest.fixture(scope="module")
def train_outputs(model):
    """train 模式下的一次 forward，供梯度相关断言复用（建模型+512² forward 较慢）。"""
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    return model(torch.randn(2, 1, 512, 512))


def test_new_keys_declared_in_typed_dict():
    """键必须进 ModelOutputs —— test_phase_refactor 的穷举校验才能兜住漏填。"""
    for key in NEW_KEYS:
        assert key in ModelOutputs.__annotations__, f"{key} 未声明进 ModelOutputs"


def test_new_output_shapes(model, train_outputs):
    out = train_outputs
    assert out["stage4_spatial"].shape == (2, model.CLS_POOLED_DIM, 16, 16)
    # 反向锚点：若有人把 in_channels 误写成 CLS_STAGE4_CHANNELS，这条会提示原因。
    assert model.CLS_POOLED_DIM != model.CLS_STAGE4_CHANNELS, (
        "352(pool 维)与 208(adaptor_stage4 输入维)必须保持不同 —— "
        "二者一旦相等，本测试对『位置头 in_channels 取哪个』就失去区分力"
    )
    assert out["soft_mask_detached"].shape == (2, 1, 512, 512)
    assert out["nodule_scale_detached"].shape == (2, 1)


def test_gradient_direction_contract(train_outputs):
    """带 `_detached` 后缀 = 已断开，不带 = 保留梯度。命名即契约。"""
    assert train_outputs["stage4_spatial"].requires_grad is True
    assert train_outputs["soft_mask_detached"].requires_grad is False
    assert train_outputs["nodule_scale_detached"].requires_grad is False


def test_stage4_backward_leaves_seg_backbone_at_exactly_zero(model):
    """🔴 位置头梯度不得回流分割主干（gate8 精确零断言的前提）。

    用 `!= 0.0` 而非 `< 1e-7`：上游 `adaptor_stage3/4` 对 seg_features 是
    `.detach()` 硬断图，数学上精确为 0；且 `!=` 对 NaN 敏感（`nan != 0` 为 True），
    `< 1e-7` 式判断则会静默放过 NaN。同 gate8 的论证。
    """
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(2)
    out = model(torch.randn(2, 1, 512, 512))
    out["stage4_spatial"].sum().backward()

    cls_grad = sum(
        p.grad.abs().sum().item()
        for p in model.cls_backbone.parameters()
        if p.grad is not None
    )
    assert cls_grad > 0.0, "cls_backbone 收不到梯度 —— 位置头将无法训练主干 Top-2 层"

    leaked = [
        f"{name}: {p.grad.abs().sum().item()}"
        for name, p in model.seg_backbone.named_parameters()
        if p.grad is not None and p.grad.abs().sum().item() != 0.0
    ]
    assert not leaked, "stage4_spatial 反传泄漏进 seg_backbone: " + "; ".join(leaked)
    model.zero_grad(set_to_none=True)


@pytest.mark.parametrize(
    "fill, expected",
    [(0.0, 0.0), (1.0, 1.0), (0.25, 0.5)],  # sqrt(面积占比)
)
def test_nodule_scale_matches_analytic_solution(fill, expected):
    m = torch.full((1, 1, 512, 512), fill)
    got = nodule_scale_from_mask(m).item()
    assert got == pytest.approx(expected, abs=1e-6)


def test_nodule_scale_on_localized_block():
    """128² 方块 / 512² 画布 → 面积占比 1/16 → 等效直径 0.25。"""
    m = torch.zeros(1, 1, 512, 512)
    m[:, :, :128, :128] = 1.0
    got = nodule_scale_from_mask(m).item()
    assert got == pytest.approx(0.25, abs=1e-6)


def test_nodule_scale_survives_fp16_sum_overflow():
    """反证：朴素 fp16 求和会 inf，实现必须先升 fp32（QA_VP §1.3）。"""
    m16 = torch.ones(1, 1, 512, 512, dtype=torch.float16)
    assert m16.sum().item() == float("inf"), (
        "反证失效：fp16 全 1 求和未溢出，本测试对 `.float()` 已失去区分力 —— "
        "请改用更大分辨率或检查 torch 行为变更"
    )
    got = nodule_scale_from_mask(m16).item()
    assert got == pytest.approx(1.0, abs=1e-6)


def test_nodule_scale_is_detached_even_from_grad_bearing_mask():
    """输入带梯度时输出仍须断开（静态方法内部 detach，不依赖调用方自觉）。"""
    m = torch.rand(1, 1, 64, 64, requires_grad=True)
    assert nodule_scale_from_mask(m).requires_grad is False


@pytest.mark.parametrize(
    "bad_shape",
    [(2, 3, 64, 64), (2, 64, 64), (2, 1, 64)],
    ids=["C>1_静默算错", "3D_缺channel维", "3D_缺一维"],
)
def test_nodule_scale_rejects_wrong_input_shape(bad_shape):
    """C>1 时 `sum(dim=(2,3))` 返回 [B,C] 而分母仍是 H*W —— 语义错且**不崩**。

    静默错误比崩溃阴，必须 fail-loud（QA_VP §1.5 维度契约）。
    """
    with pytest.raises(AssertionError, match=r"期望 \[B,1,H,W\]"):
        nodule_scale_from_mask(torch.rand(*bad_shape))


def test_nodule_scale_returns_2d_not_1d():
    """返回 [B,1] 而非 [B] —— 下游 concat 依赖此契约。

    反向说明为什么这条值得单独测：`[B,1] * [B]` 会**静默**广播成 [B,B]
    （QA_VP §2.2 点名的坑），不报错但结果全错。
    """
    out = nodule_scale_from_mask(torch.rand(3, 1, 64, 64))
    assert out.shape == (3, 1)
    assert out.dim() == 2, "退化成 1D 会让下游 [B,1] 契约的 concat/乘法静默出错"
    # 坐实该坑真实存在（若哪天 torch 改成报错，这条会提醒我们防护可以放宽）
    assert (out * torch.rand(3)).shape == (3, 3), "广播坑前提变了，请复查下游假设"

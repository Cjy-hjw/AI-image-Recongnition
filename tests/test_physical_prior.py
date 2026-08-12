"""
Numerical stability and gradient health tests for decoupled modules:
  - DifferentiablePhysicalPriorExtractor
  - ConceptEmbeddingCBMHead

Run:
    pytest tests/test_model_stability.py
    # or
    python -m unittest tests/test_model_stability.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from src.models.physical_prior import DifferentiablePhysicalPriorExtractor
from src.models.cbm_head import ConceptEmbeddingCBMHead
from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.loss import MultiTaskLoss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE = 64  # small size for fast tests

def _make_extractor():
    return DifferentiablePhysicalPriorExtractor(img_size=IMG_SIZE, eps=1e-4)


def _make_cbm_head(in_features=128):
    return ConceptEmbeddingCBMHead(
        in_features=in_features, num_concepts=18, emb_dim=16, latent_dim=64
    )


def _assert_no_nan_inf(tensor, name="tensor"):
    """Assert tensor contains no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"


# ═════════════════════════════════════════════════════════════════════════════
# DifferentiablePhysicalPriorExtractor Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestDifferentiablePhysicalPriorExtractor:
    """Test suite for DifferentiablePhysicalPriorExtractor."""

    # ── Shape & Value Range Tests ────────────────────────────────────────────

    def test_output_shape(self):
        """forward(mask, img) must return [B, 3] regardless of batch size."""
        extractor = _make_extractor()
        B = 3
        mask = torch.rand(B, 1, IMG_SIZE, IMG_SIZE)
        img = torch.rand(B, 1, IMG_SIZE, IMG_SIZE)
        v_phys = extractor(mask, img)
        assert v_phys.shape == (B, 3), f"Expected shape ({B}, 3), got {v_phys.shape}"

    def test_output_value_ranges(self):
        """
        aspect_ratio ∈ [0.1, 10.0], echo_ratio ∈ [0.1, 10.0], circularity ∈ [0.0, 2.0].
        """
        extractor = _make_extractor()
        mask = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)
        img = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)
        v_phys = extractor(mask, img)
        ar = v_phys[:, 0]
        er = v_phys[:, 1]
        circ = v_phys[:, 2]
        assert (ar >= 0.1).all() and (ar <= 10.0).all(), f"aspect_ratio out of range: {ar}"
        assert (er >= 0.1).all() and (er <= 10.0).all(), f"echo_ratio out of range: {er}"
        assert (circ >= 0.0).all() and (circ <= 2.0).all(), f"circularity out of range: {circ}"

    def test_output_no_nan_inf(self):
        """Forward pass on random inputs must produce clean finite outputs."""
        extractor = _make_extractor()
        mask = torch.rand(4, 1, IMG_SIZE, IMG_SIZE)
        img = torch.rand(4, 1, IMG_SIZE, IMG_SIZE)
        v_phys = extractor(mask, img)
        _assert_no_nan_inf(v_phys, "v_phys (random)")

    # ── Extreme Input Tests ──────────────────────────────────────────────────

    def test_all_zero_mask(self):
        """All-zero mask must not cause NaN/Inf; physical features should default to ~1.0."""
        extractor = _make_extractor()
        mask = torch.zeros(2, 1, IMG_SIZE, IMG_SIZE)
        img = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)
        v_phys = extractor(mask, img)
        _assert_no_nan_inf(v_phys, "v_phys (zero mask)")
        # With no active mask, presence gate → 0, so features → baseline 1.0
        for i, name in enumerate(["aspect_ratio", "echo_ratio", "circularity"]):
            assert torch.allclose(v_phys[:, i], torch.ones(2), atol=0.1), \
                f"{name} should be ~1.0 under zero mask, got {v_phys[:, i]}"

    def test_all_one_mask(self):
        """All-one mask (full coverage) must produce finite, stable values."""
        extractor = _make_extractor()
        mask = torch.ones(2, 1, IMG_SIZE, IMG_SIZE)
        img = torch.rand(2, 1, IMG_SIZE, IMG_SIZE) * 0.5 + 0.1  # avoid black edges
        v_phys = extractor(mask, img)
        _assert_no_nan_inf(v_phys, "v_phys (all-one mask)")
        # aspect_ratio for a square uniform mask → ~1.0
        assert torch.allclose(v_phys[0, 0], torch.tensor(1.0), atol=0.2), \
            f"aspect_ratio for full-square mask should be ~1.0, got {v_phys[0, 0]}"

    def test_constant_weak_mask(self):
        """A constant very-low-value mask (0.01) must remain numerically stable."""
        extractor = _make_extractor()
        mask = torch.full((2, 1, IMG_SIZE, IMG_SIZE), 0.01)
        img = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)
        v_phys = extractor(mask, img)
        _assert_no_nan_inf(v_phys, "v_phys (weak mask 0.01)")

    def test_near_binary_mask(self):
        """Mask values extremely close to 0 or 1 (e.g. 0.001 and 0.999) must not overflow."""
        extractor = _make_extractor()
        # Half near-0, half near-1 creating a sharp boundary
        mask = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
        mask[:, :, IMG_SIZE//4:3*IMG_SIZE//4, IMG_SIZE//4:3*IMG_SIZE//4] = 0.999
        mask = mask + 0.001  # floor at 0.001
        mask = mask.clamp(0.0, 1.0)
        img = torch.rand(1, 1, IMG_SIZE, IMG_SIZE)
        v_phys = extractor(mask, img)
        _assert_no_nan_inf(v_phys, "v_phys (near-binary mask)")

    # ── Backward Propagation Tests ───────────────────────────────────────────

    def _run_backward_check(self, mask, img, label="default"):
        """
        Helper: run forward+backward on (mask, img), assert mask.grad has
        no NaN/Inf and max |grad| < 100.0.
        """
        extractor = _make_extractor()
        mask = mask.clone().detach().requires_grad_(True)
        img = img.detach()  # img doesn't need grad for these tests

        v_phys = extractor(mask, img)
        loss = v_phys.sum()
        loss.backward()

        assert mask.grad is not None, f"[{label}] mask.grad is None after backward()"
        _assert_no_nan_inf(mask.grad, f"mask.grad ({label})")
        max_abs_grad = mask.grad.abs().max().item()
        assert max_abs_grad < 100.0, \
            f"[{label}] max |mask.grad| = {max_abs_grad:.4f} exceeds 100.0"
        return max_abs_grad

    def test_backward_random_input(self):
        """Standard random input backward must produce healthy gradients."""
        mask = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)
        img = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)
        self._run_backward_check(mask, img, label="random")

    def test_backward_all_zero_mask(self):
        """Backward through all-zero mask must not produce NaN/Inf gradients; degenerate guard keeps |grad| < 10."""
        extractor = _make_extractor()
        mask = torch.zeros(2, 1, IMG_SIZE, IMG_SIZE, requires_grad=True)
        img = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)
        v_phys = extractor(mask, img)
        loss = v_phys.sum()
        loss.backward()
        assert mask.grad is not None, "mask.grad is None after backward()"
        _assert_no_nan_inf(mask.grad, "mask.grad (zero mask)")
        # Degenerate guard replaces mask with dummy 0.5, so gradient must be small
        max_abs_grad = mask.grad.abs().max().item()
        assert max_abs_grad < 10.0, \
            f"max |mask.grad| = {max_abs_grad:.4f} exceeds 10.0 (degenerate case)"

    def test_backward_all_one_mask(self):
        """Backward through all-one mask must not produce NaN/Inf gradients."""
        mask = torch.ones(2, 1, IMG_SIZE, IMG_SIZE)
        img = torch.rand(2, 1, IMG_SIZE, IMG_SIZE) * 0.5 + 0.1
        self._run_backward_check(mask, img, label="one mask")

    def test_backward_constant_weak_mask(self):
        """Backward through constant weak mask (0.01) must produce bounded gradients."""
        mask = torch.full((2, 1, IMG_SIZE, IMG_SIZE), 0.01)
        img = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)
        self._run_backward_check(mask, img, label="weak 0.01 mask")

    def test_backward_near_binary_mask(self):
        """Backward through near-binary mask must remain gradient-stable."""
        mask = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
        mask[:, :, IMG_SIZE//4:3*IMG_SIZE//4, IMG_SIZE//4:3*IMG_SIZE//4] = 0.999
        mask = mask + 0.001
        mask = mask.clamp(0.0, 1.0)
        img = torch.rand(1, 1, IMG_SIZE, IMG_SIZE)
        self._run_backward_check(mask, img, label="near-binary mask")

    def test_backward_fp16_safety(self):
        """
        Under FP16 input (AMP scenario), forward must still produce finite outputs
        when cast back to float32 inside the module.
        """
        extractor = _make_extractor()
        mask = torch.rand(2, 1, IMG_SIZE, IMG_SIZE, dtype=torch.float16).requires_grad_(True)
        img = torch.rand(2, 1, IMG_SIZE, IMG_SIZE, dtype=torch.float16)
        v_phys = extractor(mask, img)
        # Output should be returned in original dtype (fp16)
        assert v_phys.dtype == torch.float16, f"Expected fp16 output, got {v_phys.dtype}"
        _assert_no_nan_inf(v_phys.float(), "v_phys (fp16)")
        # backward
        loss = v_phys.float().sum()
        loss.backward()
        assert mask.grad is not None, "mask.grad is None under fp16"
        _assert_no_nan_inf(mask.grad.float(), "mask.grad (fp16)")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="FP16 autocast 仅 CUDA 支持(torch 2.1.0 CPU autocast 只接受 bfloat16);"
               "本测试复现 GPU FP16 梯度溢出场景,交付容器为 cuda12.1,CPU CI 非其合法运行环境",
    )
    def test_backward_fp16_degenerate_mask_autocast(self):
        """
        Under torch.cuda.amp.autocast (FP16/AMP), an all-zero / degenerate mask
        must produce gradients with max |grad| < 10.0 and no inf/NaN.
        This is the core safety test for the FP16 gradient overflow vulnerability.
        """
        extractor = _make_extractor()
        # All-zero mask is the degenerate case that previously caused inf gradients in FP16
        mask = torch.zeros(2, 1, IMG_SIZE, IMG_SIZE, requires_grad=True)
        img = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            v_phys = extractor(mask, img)
            _assert_no_nan_inf(v_phys.float(), "v_phys (autocast degenerate)")
            loss = v_phys.float().sum()

        loss.backward()
        assert mask.grad is not None, "mask.grad is None under autocast degenerate"
        grad = mask.grad.float()
        _assert_no_nan_inf(grad, "mask.grad (autocast degenerate)")
        max_abs_grad = grad.abs().max().item()
        assert max_abs_grad < 10.0, \
            f"max |mask.grad| = {max_abs_grad:.4f} exceeds 10.0 under AMP autocast (degenerate)"
        assert not torch.isinf(grad).any(), "mask.grad contains inf under AMP autocast"
        assert not torch.isnan(grad).any(), "mask.grad contains NaN under AMP autocast"

    # ── Buffer Contiguity / state_dict Round-Trip (2026-07-10 回归测试) ─────

    def test_coord_buffers_are_contiguous(self):
        """x_coords/y_coords 必须 contiguous，否则 load_state_dict() 会因目标
        tensor 内存别名而崩溃（torch.meshgrid indexing="ij" 默认返回 0-stride
        广播视图，直接 register_buffer 不加 .contiguous() 就会踩这个坑）。"""
        extractor = _make_extractor()
        assert extractor.x_coords.is_contiguous(), "x_coords 不是 contiguous"
        assert extractor.y_coords.is_contiguous(), "y_coords 不是 contiguous"

    def test_state_dict_roundtrip_load_does_not_crash(self):
        """回归测试（2026-07-10）：构造模型 → 取 state_dict → load_state_dict
        进一个新实例，必须成功不崩溃。此前非 contiguous 的 buffer 会导致
        PyTorch 报 "more than one element of the written-to tensor refers to
        a single memory location"，致使 predict.py/evaluate.py 的 build_model()
        对**任何**训练权重都无法真正加载（不仅仅是 LoRA 相关场景）。"""
        extractor = _make_extractor()
        state_dict = extractor.state_dict()

        fresh = _make_extractor()
        fresh.load_state_dict(state_dict, strict=True)  # 不能抛异常

        assert torch.equal(fresh.x_coords, extractor.x_coords)
        assert torch.equal(fresh.y_coords, extractor.y_coords)


# ═════════════════════════════════════════════════════════════════════════════
# e_prior → seg_backbone 端到端梯度耦合(2026-07-11 Gate2 review 非阻塞建议补齐)
#
# 背景:Gate8(scripts/verify/gates/gate8_adaptor_isolation.py)审计 TaskAlignmentAdaptor
# 的 cls→seg detach 隔离契约时,为了不被"物理先验"这条设计内合法通路(cbm_head 的
# `e_prior = prior_mlp(v_phys) * gamma_prior`,v_phys 来自**未 detach** 的
# seg_features 经 decoder 算出的 soft_mask)污染信号,显式用 gamma_prior=0.0 把这条
# 通路精确清零、隔离掉——这对 Gate8 自身是正确且必要的测试设计。
# 但这意味着 e_prior→seg_backbone 这条真实存在、训练时每一步都在发生的耦合
# (Gate2 review 实测:gamma_prior=1.0 默认值下 seg_backbone 梯度 abs().sum()≈14.2~30.6,
# 量级不可忽略),此前是**零端到端梯度测试覆盖**——`test_physical_prior.py` 里已有的
# 用例只测 DifferentiablePhysicalPriorExtractor 独立模块(合成 leaf 张量,不经过
# seg_backbone/decoder 完整链路),测不到这条路径本身是否真的按 gamma_prior 预期工作。
# 本节补齐:验证 gamma_prior 确实能精确控制这条通路的开关,而非仅凭代码走读推断。
# ═════════════════════════════════════════════════════════════════════════════

def _build_full_model_fixture():
    """与 gate8_adaptor_isolation.py / test_gate8_adaptor_isolation.py 保持一致的
    dummy 输入构造(见 scripts/verify/runner.py),确保结论可与 Gate8 的实测数字互相印证。"""
    torch.manual_seed(0)
    model = MultiTaskThyroidModel(backbone_name="efficientnet_b2", pretrained=False)
    criterion = MultiTaskLoss()

    x = torch.randn(2, 1, 512, 512)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    targets = {
        "mask": torch.randint(0, 2, (2, 1, 512, 512)).float(),
        "composition": torch.tensor([0, 3]),
        "echogenicity": torch.tensor([1, 2]),
        "shape": torch.tensor([0, 1]),
        "margin": torch.tensor([2, 0]),
        "foci": torch.randn(2, 4).sigmoid(),
        # 正负样本各 1 个,理由同 test_gate8_adaptor_isolation.py:避免 AUCMLossApprox
        # 单类 batch 退化捷径(*0.0 恒零损失)吞掉本测试要观测的梯度信号。
        "malignancy": torch.tensor([1, 0]),
    }
    return model, criterion, x, targets


def _seg_backbone_grad_abs_sum(model):
    total = 0.0
    any_grad = False
    for p in model.seg_backbone.parameters():
        if p.grad is not None:
            any_grad = True
            total += p.grad.abs().sum().item()
    return total, any_grad


class TestEPriorSegBackboneGradientCoupling:
    """e_prior(物理先验)→ seg_backbone 端到端梯度耦合:验证 gamma_prior 这个开关
    真的按预期控制这条合法通路,而不是代码写了但从没实测过是否生效。"""

    def test_gamma_prior_zero_gives_exact_zero_grad(self):
        """gamma_prior=0.0 且新建模型 adaptor.gamma 恒为 0(零初始化,见
        TaskAlignmentAdaptor 设计)时,loss_mal → seg_backbone 的两条已知路径
        (adaptor injection + e_prior)均被精确清零,seg_backbone 应收到精确 0 梯度
        (或从未参与图、grad 为 None,视具体 autograd 图连通性而定,两者都算通过)。"""
        model, criterion, x, targets = _build_full_model_fixture()
        model.train()
        model.zero_grad()

        out = model(x, gamma_prior=0.0)
        _, loss_dict = criterion(out, targets, update_famo_weights=False)
        loss_dict["malignancy"].backward()

        total, _ = _seg_backbone_grad_abs_sum(model)
        assert total == 0.0, (
            f"gamma_prior=0.0 时 seg_backbone 梯度应精确为 0,实际 abs().sum()={total}"
            "——说明 e_prior 通路没有被 gamma_prior 正确清零,该开关本身失效。"
        )

    def test_gamma_prior_one_gives_nonzero_finite_grad(self):
        """gamma_prior=1.0(默认值,训练时实际使用的值)下,e_prior 通路应该是
        真实连通、可微的——seg_backbone 必须收到非零、有限(无 NaN/Inf)的梯度。
        如果这里测出精确 0,说明 e_prior→seg_backbone 这条设计内耦合实际上是断的
        (比如某处意外多加了一次 .detach()),而不是 Gate8 场景里"有意隔离"的 0。"""
        model, criterion, x, targets = _build_full_model_fixture()
        model.train()
        model.zero_grad()

        out = model(x, gamma_prior=1.0)
        _, loss_dict = criterion(out, targets, update_famo_weights=False)
        loss_dict["malignancy"].backward()

        total, any_grad = _seg_backbone_grad_abs_sum(model)
        assert any_grad, "gamma_prior=1.0 时 seg_backbone 参数完全没有收到梯度(grad 全 None)"
        assert total > 0.0, (
            "gamma_prior=1.0 时 seg_backbone 梯度精确为 0——e_prior 这条设计内合法耦合"
            "实际上是断开的(可能被意外 detach),不是预期行为。"
        )
        for p in model.seg_backbone.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), "seg_backbone 梯度出现 NaN"
                assert not torch.isinf(p.grad).any(), "seg_backbone 梯度出现 Inf"
        # 量级健全性检查(非精确回归,只防真正的数值爆炸;Gate2 review 实测约
        # 14.2~30.6 量级,给足余量避免 batch/种子波动导致假失败)。
        assert total < 1e4, f"seg_backbone 梯度量级异常大(abs().sum()={total}),疑似数值不稳定"


# ═════════════════════════════════════════════════════════════════════════════

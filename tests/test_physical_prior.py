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
import torch.nn.functional as F

from src.models.physical_prior import DifferentiablePhysicalPriorExtractor
from src.models.cbm_head import ConceptEmbeddingCBMHead


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

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == "cuda")):
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


# ═════════════════════════════════════════════════════════════════════════════

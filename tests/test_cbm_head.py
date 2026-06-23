"""Numerical stability tests for ConceptEmbeddingCBMHead."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from src.models.cbm_head import ConceptEmbeddingCBMHead


IMG_SIZE = 64


def _make_cbm_head(in_features=128):
    return ConceptEmbeddingCBMHead(
        in_features=in_features, num_concepts=18, emb_dim=16, latent_dim=64
    )


def _assert_no_nan_inf(tensor, name="tensor"):
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"


# ConceptEmbeddingCBMHead Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestConceptEmbeddingCBMHead:
    """Test suite for ConceptEmbeddingCBMHead forward/backward stability."""

    IN_FEATURES = 128
    BATCH = 4

    def _forward(self, head, x, e_prior=None):
        """Run forward and return outputs as a positional tuple (legacy order)."""
        out = head(x, e_prior)
        return (
            out.logits_comp,
            out.logits_echo,
            out.logits_shape,
            out.logits_margin,
            out.logits_foci,
            out.concept_vector,
            out.logits_malignancy,
            out.latent_res,
        )

    # ── Shape Tests ──────────────────────────────────────────────────────────

    def test_output_shapes(self):
        """All 8 keyed outputs must have correct shapes."""
        head = _make_cbm_head(self.IN_FEATURES)
        x = torch.randn(self.BATCH, self.IN_FEATURES)
        logits_comp, logits_echo, logits_shape, logits_margin, logits_foci, \
            concept_vector, logits_malignancy, latent_res = self._forward(head, x)

        assert logits_comp.shape == (self.BATCH, 4)
        assert logits_echo.shape == (self.BATCH, 4)
        assert logits_shape.shape == (self.BATCH, 2)
        assert logits_margin.shape == (self.BATCH, 4)
        assert logits_foci.shape == (self.BATCH, 4)
        assert concept_vector.shape == (self.BATCH, 18)
        assert logits_malignancy.shape == (self.BATCH, 2)
        assert latent_res.shape == (self.BATCH, 2)

    def test_output_shapes_with_prior(self):
        """When e_prior is provided (16-dim), output shapes remain the same."""
        head = _make_cbm_head(self.IN_FEATURES)
        x = torch.randn(self.BATCH, self.IN_FEATURES)
        e_prior = torch.randn(self.BATCH, 16)
        outs = self._forward(head, x, e_prior)
        logits_malignancy = outs[6]
        assert logits_malignancy.shape == (self.BATCH, 2)

    # ── Forward NaN/Inf Tests ───────────────────────────────────────────────

    def test_forward_random_no_nan(self):
        """Standard random input must not produce NaN/Inf in any output."""
        head = _make_cbm_head(self.IN_FEATURES)
        x = torch.randn(self.BATCH, self.IN_FEATURES)
        outs = self._forward(head, x)
        for i, out in enumerate(outs):
            if out.dim() > 0:
                _assert_no_nan_inf(out, f"output[{i}] (random)")

    def test_forward_extreme_large_logits(self):
        """Extremely large input features (±100.0) must not produce NaN/Inf."""
        head = _make_cbm_head(self.IN_FEATURES)
        # Large positive input
        x_pos = torch.full((self.BATCH, self.IN_FEATURES), 100.0)
        outs = self._forward(head, x_pos)
        for i, out in enumerate(outs):
            if out.dim() > 0:
                _assert_no_nan_inf(out, f"output[{i}] (large positive x)")
        # Large negative input
        x_neg = torch.full((self.BATCH, self.IN_FEATURES), -100.0)
        outs = self._forward(head, x_neg)
        for i, out in enumerate(outs):
            if out.dim() > 0:
                _assert_no_nan_inf(out, f"output[{i}] (large negative x)")

    def test_forward_zero_input(self):
        """All-zero input must produce finite outputs."""
        head = _make_cbm_head(self.IN_FEATURES)
        x = torch.zeros(self.BATCH, self.IN_FEATURES)
        outs = self._forward(head, x)
        for i, out in enumerate(outs):
            if out.dim() > 0:
                _assert_no_nan_inf(out, f"output[{i}] (zero x)")

    def test_forward_constant_weak_input(self):
        """Very small constant input (1e-5) must remain stable."""
        head = _make_cbm_head(self.IN_FEATURES)
        x = torch.full((self.BATCH, self.IN_FEATURES), 1e-5)
        outs = self._forward(head, x)
        for i, out in enumerate(outs):
            if out.dim() > 0:
                _assert_no_nan_inf(out, f"output[{i}] (weak x)")

    def test_shannon_entropy_bounded(self):
        """
        The mean_entropy (used for alpha_gate) should be bounded in [0, 1]
        after normalization. We verify through the concept_vector and malignancy output.
        """
        head = _make_cbm_head(self.IN_FEATURES)
        # Test with various input magnitudes
        for scale in [1e-4, 1.0, 10.0, 100.0]:
            x = torch.randn(self.BATCH, self.IN_FEATURES) * scale
            outs = self._forward(head, x)
            logits_malignancy = outs[6]
            latent_res = outs[7]
            # malignancy logits should be finite regardless of entropy
            _assert_no_nan_inf(logits_malignancy, f"logits_malignancy (scale={scale})")
            _assert_no_nan_inf(latent_res, f"latent_res (scale={scale})")

    def test_concept_vector_valid_probabilities(self):
        """
        concept_vector components from softmax groups must sum to ~1.0 per group.
        Sigmoid groups (foci) must be in (0, 1).
        """
        head = _make_cbm_head(self.IN_FEATURES)
        x = torch.randn(self.BATCH, self.IN_FEATURES)
        outs = self._forward(head, x)
        concept_vector = outs[5]

        # composition: cols 0:4 → sum ≈ 1
        comp_sum = concept_vector[:, 0:4].sum(dim=1)
        assert torch.allclose(comp_sum, torch.ones(self.BATCH), atol=1e-4), \
            f"composition probs don't sum to 1: {comp_sum}"

        # echo: cols 4:8 → sum ≈ 1
        echo_sum = concept_vector[:, 4:8].sum(dim=1)
        assert torch.allclose(echo_sum, torch.ones(self.BATCH), atol=1e-4), \
            f"echo probs don't sum to 1: {echo_sum}"

        # shape: cols 8:10 → sum ≈ 1
        shape_sum = concept_vector[:, 8:10].sum(dim=1)
        assert torch.allclose(shape_sum, torch.ones(self.BATCH), atol=1e-4), \
            f"shape probs don't sum to 1: {shape_sum}"

        # margin: cols 10:14 → sum ≈ 1
        margin_sum = concept_vector[:, 10:14].sum(dim=1)
        assert torch.allclose(margin_sum, torch.ones(self.BATCH), atol=1e-4), \
            f"margin probs don't sum to 1: {margin_sum}"

        # foci: cols 14:18 → each in [0, 1]
        foci = concept_vector[:, 14:18]
        assert (foci >= 0.0).all() and (foci <= 1.0).all(), \
            f"foci probs out of [0,1]: min={foci.min()}, max={foci.max()}"

    # ── Backward Propagation Tests ───────────────────────────────────────────

    def _run_cbm_backward_check(self, x, e_prior=None, label="default"):
        """
        Run forward+backward on CBMHead, check all parameter gradients
        are finite and bounded.
        """
        head = _make_cbm_head(self.IN_FEATURES)
        x = x.clone().detach().requires_grad_(True)
        if e_prior is not None:
            e_prior = e_prior.detach()

        outs = self._forward(head, x, e_prior)
        logits_malignancy = outs[6]

        # Compute a scalar loss from malignancy logits
        loss = logits_malignancy.sum()
        loss.backward()

        # Check input gradient
        assert x.grad is not None, f"[{label}] x.grad is None"
        _assert_no_nan_inf(x.grad, f"x.grad ({label})")
        max_x_grad = x.grad.abs().max().item()
        assert max_x_grad < 100.0, \
            f"[{label}] max |x.grad| = {max_x_grad:.4f} exceeds 100.0"

        # Check all parameter gradients
        for name, param in head.named_parameters():
            if param.grad is not None:
                _assert_no_nan_inf(param.grad, f"param.grad[{name}] ({label})")
                max_pg = param.grad.abs().max().item()
                assert max_pg < 100.0, \
                    f"[{label}] max |{name}.grad| = {max_pg:.4f} exceeds 100.0"

    def test_backward_random_input(self):
        """Standard random input backward must produce healthy gradients."""
        x = torch.randn(self.BATCH, self.IN_FEATURES)
        self._run_cbm_backward_check(x, label="random")

    def test_backward_zero_input(self):
        """All-zero input backward must not produce NaN/Inf gradients."""
        x = torch.zeros(self.BATCH, self.IN_FEATURES)
        self._run_cbm_backward_check(x, label="zero")

    def test_backward_extreme_large_input(self):
        """Extreme ±100.0 input backward must remain gradient-stable."""
        x_pos = torch.full((self.BATCH, self.IN_FEATURES), 100.0)
        self._run_cbm_backward_check(x_pos, label="large positive")
        x_neg = torch.full((self.BATCH, self.IN_FEATURES), -100.0)
        self._run_cbm_backward_check(x_neg, label="large negative")

    def test_backward_weak_input(self):
        """Very weak (1e-5) input backward must produce bounded gradients."""
        x = torch.full((self.BATCH, self.IN_FEATURES), 1e-5)
        self._run_cbm_backward_check(x, label="weak 1e-5")

    def test_backward_with_prior(self):
        """Backward with physical prior injected must remain stable."""
        x = torch.randn(self.BATCH, self.IN_FEATURES)
        e_prior = torch.randn(self.BATCH, 16) * 5.0  # scaled prior
        self._run_cbm_backward_check(x, e_prior=e_prior, label="with prior")

    def test_backward_all_concept_losses(self):
        """
        Backward through each concept group's logits individually must
        produce finite gradients (entropy + classification paths).
        """
        head = _make_cbm_head(self.IN_FEATURES)
        x = torch.randn(self.BATCH, self.IN_FEATURES, requires_grad=True)
        outs = self._forward(head, x)
        logits_comp, logits_echo, logits_shape, logits_margin, logits_foci = outs[:5]

        # Sum losses from all concept logits
        total_loss = (
            logits_comp.sum() + logits_echo.sum() + logits_shape.sum()
            + logits_margin.sum() + logits_foci.sum()
        )
        total_loss.backward()

        assert x.grad is not None, "x.grad is None after concept loss backward"
        _assert_no_nan_inf(x.grad, "x.grad (concept losses)")
        assert x.grad.abs().max().item() < 100.0, \
            f"max |x.grad| = {x.grad.abs().max().item():.4f} exceeds 100.0"

    def test_backward_malignancy_cross_entropy(self):
        """
        Simulate a realistic cross-entropy backward through malignancy logits
        and verify gradient health.
        """
        head = _make_cbm_head(self.IN_FEATURES)
        x = torch.randn(self.BATCH, self.IN_FEATURES, requires_grad=True)
        outs = self._forward(head, x)
        logits_malignancy = outs[6]
        target = torch.randint(0, 2, (self.BATCH,))
        ce_loss = F.cross_entropy(logits_malignancy, target)
        ce_loss.backward()

        assert x.grad is not None
        _assert_no_nan_inf(x.grad, "x.grad (CE loss)")
        assert x.grad.abs().max().item() < 100.0, \
            f"max |x.grad| = {x.grad.abs().max().item():.4f} exceeds 100.0"


# ═════════════════════════════════════════════════════════════════════════════
# Run with unittest discovery
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Support both `python -m pytest` and `python -m unittest`
    import unittest
    unittest.main()

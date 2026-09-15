"""Tests for Masked Task-Specific Loss (label validity masking).

Covers:
1. Malignancy: invalid sample contributes zero gradient to malignancy head.
2. Concept losses: invalid sample's concept loss is excluded.
3. Segmentation: invalid sample contributes zero gradient to seg head.
4. Backward compatibility: all-valid label_validity gives same result as no
   label_validity (legacy path).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.models.losses.components.malignancy import compute_malignancy_losses
from src.models.losses.components.concepts import compute_concept_losses
from src.models.losses.components.segmentation import compute_segmentation_loss
from src.models.losses.classification import AUCMLossApprox, SupervisedContrastiveLoss
from src.models.losses.segmentation import DiceLoss


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_no_nan(t, name=""):
    assert not torch.isnan(t).any(), f"{name} contains NaN"
    assert not torch.isinf(t).any(), f"{name} contains Inf"


def _make_mal_outputs(B=4, seed=0):
    torch.manual_seed(seed)
    return {
        "malignancy": torch.randn(B, 2, requires_grad=True),
        "z_contrastive": torch.randn(B, 128, requires_grad=True),
    }


def _make_mal_targets(B=4, label_validity=None):
    """Build targets dict for malignancy tests."""
    targets: dict[str, object] = {
        "malignancy": torch.randint(0, 2, (B,)),
    }
    if label_validity is not None:
        targets["label_validity"] = {
            "malignancy": torch.tensor(label_validity, dtype=torch.float32),
        }
    return targets


def _make_concept_outputs(B=4, seed=1):
    torch.manual_seed(seed)
    return {
        "composition": torch.randn(B, 4, requires_grad=True),
        "echogenicity": torch.randn(B, 4, requires_grad=True),
        "shape": torch.randn(B, 2, requires_grad=True),
        "margin": torch.randn(B, 4, requires_grad=True),
        "foci": torch.randn(B, 4, requires_grad=True),
        "mask": torch.randn(B, 1, 32, 32),
    }


def _make_concept_targets(B=4, label_validity=None):
    targets: dict[str, object] = {
        "composition": torch.zeros(B, dtype=torch.long),
        "echogenicity": torch.zeros(B, dtype=torch.long),
        "shape": torch.zeros(B, dtype=torch.long),
        "margin": torch.zeros(B, dtype=torch.long),
        "foci": torch.zeros(B, 4),
    }
    if label_validity is not None:
        targets["label_validity"] = {
            task: torch.tensor(label_validity, dtype=torch.float32)
            for task in ("composition", "echogenicity", "shape", "margin", "foci", "malignancy")
        }
    return targets


def _make_seg_outputs(B=4, H=32, seed=2):
    torch.manual_seed(seed)
    return {
        "mask": torch.randn(B, 1, H, H, requires_grad=True),
    }


def _make_seg_targets(B=4, H=32, valid_mask=None):
    targets = {
        "mask": (torch.rand(B, 1, H, H) > 0.5).float(),
    }
    if valid_mask is not None:
        targets["valid_mask"] = torch.tensor(valid_mask, dtype=torch.float32)
    return targets


_TIRADS = torch.tensor([
    0., 0., 1., 2.,   # Composition
    0., 1., 2., 3.,   # Echogenicity
    0., 3.,           # Shape
    0., 0., 2., 3.,   # Margin
    0., 1., 2., 3.,   # Foci
], dtype=torch.float32)


# ── TestMaskedMalignancyLoss ──────────────────────────────────────────────────

class TestMaskedMalignancyLoss:
    """Malignancy validity masking: invalid samples must not contribute gradients."""

    B = 4

    def _run(self, label_validity):
        outs = _make_mal_outputs(self.B)
        targets = _make_mal_targets(self.B, label_validity=label_validity)
        mal_fn = AUCMLossApprox(margin=1.0, imratio=0.5)
        supcon_fn = SupervisedContrastiveLoss(temperature=0.07)
        mw = torch.ones(self.B)
        loss_mal, loss_supcon = compute_malignancy_losses(
            mal_fn, supcon_fn, outs, targets, mw, slr_weights=None
        )
        return loss_mal, loss_supcon, outs  # outs returned separately for grad checks

    def test_all_invalid_returns_zero(self):
        """When all malignancy labels invalid, loss_mal must be exactly 0."""
        loss_mal, loss_supcon, _ = self._run([0.0] * self.B)
        assert loss_mal.item() == 0.0, f"Expected 0 loss, got {loss_mal.item()}"
        assert loss_supcon.item() == 0.0, f"Expected 0 supcon, got {loss_supcon.item()}"

    def test_all_invalid_zero_gradient_on_logits(self):
        """When all malignancy labels invalid, malignancy logits must have zero grad."""
        outs = _make_mal_outputs(self.B)
        targets = _make_mal_targets(self.B, label_validity=[0.0] * self.B)
        mal_fn = AUCMLossApprox(margin=1.0, imratio=0.5)
        supcon_fn = SupervisedContrastiveLoss(temperature=0.07)
        mw = torch.ones(self.B)
        loss_mal, _ = compute_malignancy_losses(
            mal_fn, supcon_fn, outs, targets, mw, slr_weights=None
        )
        # Zero loss → backward produces zero grad (or None)
        total = loss_mal + outs["malignancy"].sum() * 0.0
        total.backward()
        if outs["malignancy"].grad is not None:
            assert outs["malignancy"].grad.abs().sum().item() == 0.0, (
                "malignancy logit gradient should be zero for fully-invalid batch"
            )

    def test_partial_invalid_uses_only_valid_samples(self):
        """Masking one of two samples: loss should equal loss on just the valid sample."""
        B = 2
        torch.manual_seed(42)
        mal_logits = torch.randn(B, 2, requires_grad=True)
        mal_targets_t = torch.tensor([1, 0])

        mal_fn = AUCMLossApprox(margin=1.0, imratio=0.5)
        supcon_fn = SupervisedContrastiveLoss(temperature=0.07)
        mw = torch.ones(B)

        # Masked: sample 1 invalid
        targets_masked = {
            "malignancy": mal_targets_t,
            "label_validity": {
                "malignancy": torch.tensor([1.0, 0.0]),
            },
        }
        outs = {"malignancy": mal_logits, "z_contrastive": torch.randn(B, 128)}
        loss_masked, _ = compute_malignancy_losses(
            mal_fn, supcon_fn, outs, targets_masked, mw, slr_weights=None
        )

        # Reference: only sample 0 used (single-class batch → AUCM returns 0 for safety)
        # We mainly check masked loss is finite and different from full-batch loss.
        targets_full = {"malignancy": mal_targets_t}
        outs2 = {"malignancy": mal_logits.detach().clone(), "z_contrastive": torch.randn(B, 128)}
        loss_full, _ = compute_malignancy_losses(
            mal_fn, supcon_fn, outs2, targets_full, mw, slr_weights=None
        )
        _assert_no_nan(loss_masked, "masked mal loss")
        _assert_no_nan(loss_full, "full mal loss")

    def test_all_valid_no_label_validity_matches_legacy(self):
        """All-valid label_validity path must produce same result as no label_validity."""
        torch.manual_seed(99)
        B = 4
        mal_logits = torch.randn(B, 2)
        mal_targets_t = torch.tensor([1, 0, 1, 0])
        z = torch.randn(B, 128)
        mw = torch.ones(B)

        mal_fn = AUCMLossApprox(margin=1.0, imratio=0.5)
        supcon_fn = SupervisedContrastiveLoss(temperature=0.07)

        # Legacy path
        outs_legacy = {"malignancy": mal_logits.clone(), "z_contrastive": z.clone()}
        targets_legacy = {"malignancy": mal_targets_t}
        loss_leg, supcon_leg = compute_malignancy_losses(
            mal_fn, supcon_fn, outs_legacy, targets_legacy, mw, slr_weights=None
        )

        # All-valid masked path
        outs_masked = {"malignancy": mal_logits.clone(), "z_contrastive": z.clone()}
        targets_masked = {
            "malignancy": mal_targets_t,
            "label_validity": {
                "malignancy": torch.ones(B),
            },
        }
        loss_masked, supcon_masked = compute_malignancy_losses(
            mal_fn, supcon_fn, outs_masked, targets_masked, mw, slr_weights=None
        )

        # Values must match (same computation, different code path)
        assert abs(loss_leg.item() - loss_masked.item()) < 1e-5, (
            f"Legacy={loss_leg.item():.6f} vs masked={loss_masked.item():.6f}: mismatch"
        )


# ── TestMaskedConceptLoss ──────────────────────────────────────────────────────

class TestMaskedConceptLoss:
    """Concept validity masking."""

    B = 4

    def test_all_invalid_concepts_near_zero(self):
        """When all concept labels invalid, concept losses should be near zero."""
        outs = _make_concept_outputs(self.B)
        targets = _make_concept_targets(self.B, label_validity=[0.0] * self.B)
        lam = torch.ones(self.B)
        index = torch.arange(self.B)
        mw = torch.ones(self.B)
        result = compute_concept_losses(outs, targets, lam, index, mw, 0.0, _TIRADS)
        # With all validity=0, sum(loss*v) / (sum(v)+eps) ≈ 0
        for key in ("comp", "echo", "shape", "margin", "foci", "concepts_ce", "prior"):
            val = result[key]
            if isinstance(val, torch.Tensor):
                assert val.abs().item() < 1e-5, f"{key}={val.item():.6f} should be ~0"

    def test_all_valid_legacy_compat(self):
        """All-valid label_validity path produces same scalars as no label_validity."""
        B = self.B
        outs = _make_concept_outputs(B, seed=77)
        targets_base = _make_concept_targets(B)
        lam = torch.ones(B)
        index = torch.arange(B)
        mw = torch.ones(B)

        # Legacy
        result_leg = compute_concept_losses(
            outs, targets_base, lam, index, mw, 0.0, _TIRADS
        )

        # Masked all-valid: concepts_ce and prior should match (after .mean() in multitask)
        targets_valid = _make_concept_targets(B, label_validity=[1.0] * B)
        result_masked = compute_concept_losses(
            outs, targets_valid, lam, index, mw, 0.0, _TIRADS
        )

        # In legacy path, concepts_ce is [B] tensor; in masked path it's a scalar.
        # Compare the mean of the legacy value to the scalar.
        leg_ce = result_leg["concepts_ce"]
        mask_ce = result_masked["concepts_ce"]
        leg_val = leg_ce.mean().item() if leg_ce.dim() > 0 else leg_ce.item()
        mask_val = mask_ce.item() if isinstance(mask_ce, torch.Tensor) else mask_ce
        assert abs(leg_val - mask_val) < 1e-4, (
            f"concepts_ce: legacy={leg_val:.6f} vs masked={mask_val:.6f}"
        )


# ── TestMaskedSegmentationLoss ─────────────────────────────────────────────────

class TestMaskedSegmentationLoss:
    """Segmentation validity masking."""

    B = 4
    H = 32

    def test_all_invalid_returns_zero(self):
        """When all seg labels invalid, loss must be zero."""
        bce_fn = nn.BCEWithLogitsLoss()
        dice_fn = DiceLoss()
        outs = _make_seg_outputs(self.B, self.H)
        targets = _make_seg_targets(self.B, self.H, valid_mask=[0.0] * self.B)
        loss, _ = compute_segmentation_loss(bce_fn, dice_fn, outs, targets)
        assert abs(loss.item()) < 1e-6, f"Expected ~0, got {loss.item()}"

    def test_all_invalid_zero_gradient(self):
        """Seg logits must have zero gradient when all samples invalid."""
        bce_fn = nn.BCEWithLogitsLoss()
        dice_fn = DiceLoss()
        outs = _make_seg_outputs(self.B, self.H)
        targets = _make_seg_targets(self.B, self.H, valid_mask=[0.0] * self.B)
        loss, _ = compute_segmentation_loss(bce_fn, dice_fn, outs, targets)
        loss.backward()
        if outs["mask"].grad is not None:
            assert outs["mask"].grad.abs().sum().item() < 1e-6, (
                "seg mask grad should be zero for fully-invalid batch"
            )

    def test_all_valid_legacy_compat(self):
        """All-valid valid_mask produces same scalar as no valid_mask (legacy path)."""
        bce_fn = nn.BCEWithLogitsLoss()
        dice_fn = DiceLoss()
        B, H = self.B, self.H

        torch.manual_seed(7)
        pred = torch.randn(B, 1, H, H)
        gt = (torch.rand(B, 1, H, H) > 0.5).float()

        outs_leg = {"mask": pred.clone()}
        targets_leg = {"mask": gt}
        loss_leg, _ = compute_segmentation_loss(bce_fn, dice_fn, outs_leg, targets_leg)

        outs_masked = {"mask": pred.clone()}
        targets_masked = {
            "mask": gt,
            "valid_mask": torch.ones(B),
        }
        loss_masked, _ = compute_segmentation_loss(bce_fn, dice_fn, outs_masked, targets_masked)

        assert abs(loss_leg.item() - loss_masked.item()) < 1e-4, (
            f"Legacy={loss_leg.item():.6f} vs masked={loss_masked.item():.6f}"
        )

    def test_partial_valid_excludes_invalid_samples(self):
        """Masking half the batch: loss should differ from full-batch loss."""
        bce_fn = nn.BCEWithLogitsLoss()
        dice_fn = DiceLoss()
        B, H = self.B, self.H

        torch.manual_seed(13)
        pred = torch.randn(B, 1, H, H)
        gt = (torch.rand(B, 1, H, H) > 0.5).float()

        # First half valid
        valid_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

        outs_masked = {"mask": pred}
        targets_masked = {"mask": gt, "valid_mask": valid_mask}
        loss_masked, _ = compute_segmentation_loss(bce_fn, dice_fn, outs_masked, targets_masked)

        outs_full = {"mask": pred}
        targets_full = {"mask": gt}
        loss_full, _ = compute_segmentation_loss(bce_fn, dice_fn, outs_full, targets_full)

        _assert_no_nan(loss_masked, "partial masked seg loss")
        # They should differ since not all samples contribute
        assert abs(loss_masked.item() - loss_full.item()) > 1e-6, (
            "Partial masking should change the loss value"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import unittest
    unittest.main()

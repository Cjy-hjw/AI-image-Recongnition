import torch
import torch.nn.functional as F


def compute_segmentation_loss(bce_fn, dice_fn, outputs, targets):
    """Compute BCE + Dice segmentation loss with NaN sentinel.

    When targets contains "valid_mask" (a [B] float tensor), applies
    per-sample segmentation validity masking with sum / (valid_count + eps).
    Samples with valid_mask=0 contribute zero gradient to the seg head.

    Falls back to original scalar BCE+Dice path when "valid_mask" is absent.
    """
    pred = outputs["mask"]
    gt = targets["mask"]

    # ── Masked path (per-sample valid_mask) ──────────────────────────────────
    if "valid_mask" in targets:
        v = targets["valid_mask"]  # [B] float tensor
        _eps = 1e-8

        # Per-sample BCE: reduce over H×W, keep batch dim
        bce_per = F.binary_cross_entropy_with_logits(
            pred, gt, reduction="none"
        ).mean(dim=[1, 2, 3])  # [B]

        # Per-sample Dice
        pred_sig = torch.sigmoid(pred)
        inter = (pred_sig * gt).sum(dim=[1, 2, 3])      # [B]
        union = pred_sig.sum(dim=[1, 2, 3]) + gt.sum(dim=[1, 2, 3])  # [B]
        dice_per = 1.0 - (2.0 * inter + _eps) / (union + _eps)       # [B]

        loss_per = bce_per + dice_per  # [B]
        loss_seg = (loss_per * v).sum() / (v.sum() + _eps)
    else:
        # ── Legacy path — behaviour unchanged ──────────────────────────────
        loss_bce_seg = bce_fn(pred, gt)
        loss_dice_seg = dice_fn(pred, gt)
        loss_seg = loss_bce_seg + loss_dice_seg

    if torch.isnan(loss_seg):
        print(
            f"  [NaN Guard] loss_seg is NaN! "
            f"mask range=[{pred.min().item():.4f}, {pred.max().item():.4f}]",
            flush=True,
        )
    return loss_seg

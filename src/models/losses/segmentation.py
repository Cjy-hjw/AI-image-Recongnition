import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Sorensen-Dice loss for binary image segmentation."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred shape: [B, 1, H, W], logits — compute sample-level mDice then average
        pred_prob = torch.sigmoid(pred)
        B = pred_prob.size(0)
        pred_flat = pred_prob.view(B, -1)
        target_flat = target.view(B, -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice_coeff = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice_coeff).mean()

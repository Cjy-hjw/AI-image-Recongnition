import torch


def compute_segmentation_loss(bce_fn, dice_fn, outputs, targets):
    """Compute BCE + Dice segmentation loss with NaN sentinel."""
    loss_bce_seg = bce_fn(outputs["mask"], targets["mask"])
    loss_dice_seg = dice_fn(outputs["mask"], targets["mask"])
    loss_seg = loss_bce_seg + loss_dice_seg
    if torch.isnan(loss_seg):
        print(
            f"  [NaN Guard] loss_seg is NaN! bce={loss_bce_seg.item()}, dice={loss_dice_seg.item()}, "
            f"mask range=[{outputs['mask'].min().item():.4f}, {outputs['mask'].max().item():.4f}]",
            flush=True,
        )
    return loss_seg

import torch


def compute_malignancy_losses(
    mal_loss_fn, supcon_loss_fn, outputs, targets, mask_weight, slr_weights
):
    """Compute malignancy AUCM loss and supervised contrastive loss."""
    mal_logits = outputs["malignancy"]
    mal_targets = targets["malignancy"]

    loss_mal = mal_loss_fn(mal_logits, mal_targets) * mask_weight
    loss_supcon = supcon_loss_fn(outputs["z_contrastive"], mal_targets)

    if slr_weights is not None:
        loss_mal = loss_mal * slr_weights

    loss_mal = loss_mal.mean()
    if loss_supcon.dim() > 0:
        loss_supcon = loss_supcon.mean()

    if torch.isnan(loss_mal):
        print(
            f"  [NaN Guard] loss_mal is NaN! mal_logits range=[{mal_logits.min().item():.4f}, {mal_logits.max().item():.4f}]",
            flush=True,
        )
    if torch.isnan(loss_supcon):
        print("  [NaN Guard] loss_supcon is NaN!", flush=True)

    return loss_mal, loss_supcon

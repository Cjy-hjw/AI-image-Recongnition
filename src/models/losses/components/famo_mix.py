import torch


def apply_famo_weighting(
    famo,
    loss_seg,
    loss_concepts_ce,
    loss_prior,
    loss_mal,
    loss_supcon,
    update_famo_weights,
    is_training,
):
    """Apply FAMO 5-channel dynamic loss weighting."""
    if torch.isnan(loss_concepts_ce):
        print("  [NaN Guard] loss_concepts_ce is NaN!", flush=True)
    if torch.isnan(loss_prior):
        print("  [NaN Guard] loss_prior is NaN!", flush=True)

    losses_tensor = torch.stack(
        [loss_seg, loss_concepts_ce, loss_prior, loss_mal, loss_supcon]
    )
    total_loss, weights = famo.get_weighted_loss(
        losses_tensor, update_weights=update_famo_weights and is_training
    )
    if torch.isnan(total_loss):
        print(
            f"  [NaN Guard] total_loss is NaN! seg={loss_seg.item():.4f}, "
            f"concepts_ce={loss_concepts_ce.item():.4f}, prior={loss_prior.item():.4f}, "
            f"mal={loss_mal.item():.4f}, supcon={loss_supcon.item():.4f}",
            flush=True,
        )
        total_loss = loss_seg + loss_concepts_ce + loss_prior + loss_mal + loss_supcon

    return total_loss

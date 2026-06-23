import torch


def compute_regularizers(
    total_loss, outputs, lambda_latent, delta_budget, spin_l1_weight
):
    """Compute latent residual L2 (Lagrangian) constraint and SPIN L1 penalty.

    Returns (total_loss, loss_l2).
    """
    latent_res = outputs["latent_res"]
    loss_l2 = torch.mean(latent_res**2)
    loss_lagrangian = lambda_latent * (loss_l2 - delta_budget)

    total_loss = total_loss + loss_lagrangian + spin_l1_weight * outputs["spin_l1"]

    return total_loss, loss_l2

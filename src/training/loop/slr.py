import torch
import torch.nn.functional as F
import numpy as np


def compute_slr_weights(
    outputs,
    targets,
    mix_info,
    running_losses_history,
    slr_beta,
    slr_threshold,
):
    """
    计算 SLR (Sample Loss Reweighting) 权重，用于鲁棒标签校正。
    
    Returns:
        slr_weights: 样本权重张量
        sample_losses: 原始样本损失（用于更新历史）
    """
    with torch.no_grad():
        mal_logits = outputs["malignancy"]
        mal_targets = targets["malignancy"]
        log_prob = F.log_softmax(mal_logits, dim=1)

        if mix_info is not None:
            index = mix_info["index"]
            lam = mix_info["lam"]
            loss_mal_raw_orig = F.nll_loss(
                log_prob, mal_targets, reduction="none"
            )
            loss_mal_raw_perm = F.nll_loss(
                log_prob, mal_targets[index], reduction="none"
            )
            sample_losses = (
                lam * loss_mal_raw_orig + (1.0 - lam) * loss_mal_raw_perm
            )
        else:
            sample_losses = F.nll_loss(
                log_prob, mal_targets, reduction="none"
            )

    for sl in sample_losses.cpu().numpy():
        running_losses_history.append(sl)

    if len(running_losses_history) > 1000:
        running_losses_history[:] = running_losses_history[-1000:]

    mean_loss = np.mean(running_losses_history)
    std_loss = np.std(running_losses_history) + 1e-6

    slr_weights = torch.ones_like(sample_losses)
    anomaly_mask = sample_losses > (mean_loss + slr_threshold * std_loss)
    slr_weights[anomaly_mask] = 1.0 - slr_beta

    return slr_weights, sample_losses

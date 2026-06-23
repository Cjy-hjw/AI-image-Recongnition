import torch

from src.dataset.augmentation import apply_mixup_or_cutmix
from src.training.loop.slr import compute_slr_weights
from src.training.loop.vclub import vclub_learning_step, compute_vclub_mi_bound


def train_batch(
    model,
    images,
    masks,
    labels,
    optimizer,
    vclub_optimizer,
    criterion,
    scaler,
    device,
    config,
    running_losses_history,
    epoch,
    stage,
    update_famo_weights,
    prev_z_latent,
    prev_concept_vector,
    slr_beta,
    slr_threshold,
    label_smoothing,
    spin_l1_weight,
):
    """
    单个批次的训练逻辑。

    Returns:
        loss: 总损失
        loss_dict: 损失字典
        prev_z_latent: 缓存的 detached z_latent（用于下一批次 vCLUB）
        prev_concept_vector: 缓存的 detached concept_vector（用于下一批次 vCLUB）
    """
    images = images.to(device)
    masks = masks.to(device)

    targets = {
        "mask": masks,
        "composition": labels["composition"].to(device),
        "echogenicity": labels["echogenicity"].to(device),
        "shape": labels["shape"].to(device),
        "margin": labels["margin"].to(device),
        "foci": labels["foci"].to(device),
        "malignancy": labels["malignancy"].to(device),
    }

    # Apply MixUp / CutMix
    images, masks, targets, mix_info = apply_mixup_or_cutmix(
        images, masks, targets, alpha=0.2, prob=0.3
    )
    targets["mask"] = masks

    # vCLUB 独立优化：使用前一批次的 detached 输出，在 autocast 外执行独立 backward
    vclub_learning_step(model, vclub_optimizer, prev_z_latent, prev_concept_vector)

    optimizer.zero_grad()

    gamma_prior = max(0.0, min(1.0, (epoch - 20) / 20.0))
    amp_enabled = config["train"].get("amp", True)
    with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=torch.bfloat16):
        outputs = model(images, gamma_prior=gamma_prior)

        # SLR 鲁棒标签校正
        if slr_beta < 1.0:
            slr_weights, sample_losses = compute_slr_weights(
                outputs,
                targets,
                mix_info,
                running_losses_history,
                slr_beta,
                slr_threshold,
            )

            loss, loss_dict = criterion(
                outputs,
                targets,
                slr_weights=slr_weights,
                mix_info=mix_info,
                update_famo_weights=update_famo_weights,
                spin_l1_weight=spin_l1_weight,
                label_smoothing=label_smoothing,
                enable_mal_loss=(stage > 1),
            )
        else:
            loss, loss_dict = criterion(
                outputs,
                targets,
                mix_info=mix_info,
                update_famo_weights=update_famo_weights,
                spin_l1_weight=spin_l1_weight,
                label_smoothing=label_smoothing,
                enable_mal_loss=(stage > 1),
            )

        # vCLUB 互信息极小化约束
        vclub_mi_bound = compute_vclub_mi_bound(model, outputs)
        loss = loss + 0.02 * vclub_mi_bound

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    if hasattr(criterion, "mal_loss_fn"):
        criterion.mal_loss_fn.step_dual_variable("pre")

    params_to_clip = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
    scaler.step(optimizer)

    if hasattr(criterion, "mal_loss_fn"):
        criterion.mal_loss_fn.step_dual_variable("post")

    scaler.update()

    # Cache detached outputs for next batch's vCLUB learning loss
    prev_z_latent = outputs["z_latent"].detach()
    prev_concept_vector = outputs["concept_vector"].detach()

    return loss, loss_dict, prev_z_latent, prev_concept_vector

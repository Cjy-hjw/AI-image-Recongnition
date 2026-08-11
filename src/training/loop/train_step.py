import torch

from src.dataset.augmentation import apply_mixup_or_cutmix
from src.training.loop.slr import compute_slr_weights
from src.training.loop.vclub import vclub_learning_step, compute_vclub_mi_bound
from src.training.stages import compute_gamma_prior


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

    # Label validity mask (per-sample per-task): 1=valid, 0=absent
    # If labels contain valid_* keys (new datasets), build label_validity dict.
    # If absent (old data / no valid_* keys), skip → loss components use legacy path.
    _validity_tasks = ("composition", "echogenicity", "shape", "margin", "foci", "malignancy")
    if f"valid_{_validity_tasks[0]}" in labels:
        _B = masks.shape[0]

        def _to_validity_tensor(key):
            v = labels.get(key)
            if v is None:
                return torch.ones(_B, dtype=torch.float32, device=device)
            if isinstance(v, torch.Tensor):
                return v.float().to(device)
            # scalar (int/float) shared across batch
            return torch.full((_B,), float(v), dtype=torch.float32, device=device)

        targets["label_validity"] = {
            task: _to_validity_tensor(f"valid_{task}") for task in _validity_tasks
        }
        # Segmentation validity
        targets["valid_mask"] = _to_validity_tensor("valid_mask")

    # Apply MixUp / CutMix
    images, masks, targets, mix_info = apply_mixup_or_cutmix(
        images, masks, targets, alpha=0.2, prob=0.3
    )
    targets["mask"] = masks
    # label_validity is NOT mixed — validity flags are batch-level metadata,
    # not feature-space signals. We keep the original batch's validity unchanged.
    # (A mixed sample still has the same label availability as its primary sample.)

    # vCLUB 独立优化：使用前一批次的 detached 输出，在 autocast 外执行独立 backward
    vclub_learning_step(model, vclub_optimizer, prev_z_latent, prev_concept_vector)

    optimizer.zero_grad()

    gamma_prior = compute_gamma_prior(epoch)
    amp_enabled = config["train"].get("amp", True)
    with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=torch.bfloat16):
        outputs = model(images, gamma_prior=gamma_prior)

        # SLR 鲁棒标签校正
        if slr_beta < 1.0:
            slr_weights, _sample_losses = compute_slr_weights(
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

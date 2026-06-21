import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.dataset.augmentation import apply_mixup_or_cutmix


def train_epoch(
    model,
    loader,
    optimizer,
    vclub_optimizer,
    criterion,
    scaler,
    device,
    config,
    running_losses_history,
    epoch,
):
    # Sync epoch for time-annealed physical prior gating
    if hasattr(model, "set_epoch"):
        model.set_epoch(epoch)

    # Determine current training stage for FAMO control
    from src.training.stages import get_stage_number

    stage = get_stage_number(epoch)
    update_famo_weights = stage != 3

    criterion.train()

    # Cache for vCLUB learning loss (computed from previous batch's detached outputs)
    prev_z_latent = None
    prev_concept_vector = None

    model.train()
    # 自动将所有完全冻结的子模块（其所有 requires_grad 参数为 False）设为 eval 模式，防止其 BatchNorm 更新运行统计数据
    for m in model.modules():
        params = list(m.parameters())
        if len(params) > 0 and all(not p.requires_grad for p in params):
            m.eval()

    running_loss = 0.0

    seg_loss = 0.0
    mal_loss = 0.0

    slr_beta = config["train"].get("slr_beta", 0.99)
    slr_threshold = config["train"].get("slr_threshold", 2.0)
    label_smoothing = config["train"].get("label_smoothing", 0.0)

    for batch_idx, (images, masks, labels) in enumerate(loader):
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
        if prev_z_latent is not None:
            vclub_optimizer.zero_grad()
            vclub_learning_loss = model.vclub.learning_loss(
                prev_z_latent, prev_concept_vector
            )
            vclub_learning_loss.backward()
            vclub_optimizer.step()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config["train"].get("amp", True)):
            outputs = model(images)

            # SLR 鲁棒标签校正
            if slr_beta < 1.0:
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

                spin_l1_weight = 0.005 if epoch > 80 else 0.0
                loss, loss_dict = criterion(
                    outputs,
                    targets,
                    slr_weights=slr_weights,
                    mix_info=mix_info,
                    update_famo_weights=update_famo_weights,
                    spin_l1_weight=spin_l1_weight,
                    label_smoothing=label_smoothing,
                    stage=stage,
                )
            else:
                spin_l1_weight = 0.005 if epoch > 80 else 0.0
                loss, loss_dict = criterion(
                    outputs,
                    targets,
                    mix_info=mix_info,
                    update_famo_weights=update_famo_weights,
                    spin_l1_weight=spin_l1_weight,
                    label_smoothing=label_smoothing,
                    stage=stage,
                )

            # vCLUB 互信息极小化约束：全程启用，软正交隔离隐表征与属性概念
            # 对输入 cbm_head 的 z_latent 分支进行 detach 计算条件概念向量，并 detach 该向量防止其梯度回传
            z_latent_detached = outputs["z_latent"].detach()
            (
                _,
                _,
                _,
                _,
                _,
                concept_vector_decouple,
                _,
                _,
                _,
            ) = model.cbm_head(z_latent_detached)
            concept_vector_decouple = concept_vector_decouple.detach()

            # 计算互信息上界时，第一个 z_latent 参数必须保留计算图和梯度，以将退耦约束导流回 encoder (backbone)
            vclub_mi_bound = model.vclub(outputs["z_latent"], concept_vector_decouple)
            loss = loss + 0.02 * vclub_mi_bound
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # 对偶乘子 alpha 的梯度反向 (Gradient Sign Flip for maximizing AUC-Margin dual variable)
        if criterion.mal_loss_fn.alpha.grad is not None:
            criterion.mal_loss_fn.alpha.grad.data.mul_(-1.0)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)

        # 强制对偶乘子 alpha 非负边界约束 projection
        with torch.no_grad():
            criterion.mal_loss_fn.alpha.clamp_(min=0.0)

        scaler.update()

        # Cache detached outputs for next batch's vCLUB learning loss
        prev_z_latent = outputs["z_latent"].detach()
        prev_concept_vector = outputs["concept_vector"].detach()

        running_loss += loss.item()
        seg_loss += loss_dict["seg"].item()
        mal_loss += loss_dict["malignancy"].item()

    avg_loss = running_loss / len(loader)
    return avg_loss, seg_loss / len(loader), mal_loss / len(loader)


def validate_epoch(model, loader, criterion, device, epoch=1):
    model.eval()
    criterion.eval()
    val_loss = 0.0
    val_loss_l2 = 0.0
    seg_dice = 0.0
    mal_correct = 0
    total_samples = 0

    from src.training.stages import get_stage_number

    stage = get_stage_number(epoch)

    with torch.no_grad():
        for batch_idx, (images, masks, labels) in enumerate(loader):
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

            outputs = model(images)
            loss, loss_dict = criterion(
                outputs, targets, update_famo_weights=False, stage=stage
            )

            val_loss += loss.item()
            val_loss_l2 += loss_dict["latent_l2"].item()
            batch_size = images.size(0)
            total_samples += batch_size

            # Seg Dice with diagnostic
            pred_prob = torch.sigmoid(outputs["mask"])
            pred_mask = (pred_prob > 0.5).float()

            # Diagnostic: log if predictions are degenerate (all-zeros or all-ones)
            if batch_idx == 0:
                pred_mean = pred_prob.mean().item()
                pred_std = pred_prob.std().item()
                print(
                    f"  [Val Diag] pred_prob: mean={pred_mean:.4f}, std={pred_std:.4f}, min={pred_prob.min().item():.4f}, max={pred_prob.max().item():.4f}",
                    flush=True,
                )

            intersection = (pred_mask * targets["mask"]).sum(dim=(2, 3))
            union = pred_mask.sum(dim=(2, 3)) + targets["mask"].sum(dim=(2, 3))
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            seg_dice += dice.sum().item()

            # Classification
            pred_mal = torch.argmax(outputs["malignancy"], dim=1)
            mal_correct += (pred_mal == targets["malignancy"]).sum().item()

    avg_loss = val_loss / len(loader)
    avg_loss_l2 = val_loss_l2 / len(loader)
    avg_dice = seg_dice / total_samples
    avg_acc = mal_correct / total_samples
    return avg_loss, avg_dice, avg_acc, avg_loss_l2

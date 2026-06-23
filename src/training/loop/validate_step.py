import torch


def validate_batch(
    model, images, masks, labels, criterion, device, stage, batch_idx, epoch
):
    """
    单个批次的验证逻辑。

    Returns:
        loss: 损失
        loss_dict: 损失字典
        dice_sum: Dice 系数总和
        mal_correct: 恶性分类正确数
        batch_size: 批次大小
        mal_probs: 恶性概率 (CPU tensor)
        mal_targets: 恶性标签 (CPU tensor)
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

    # 验证期固定 gamma_prior=1.0(与最终部署一致),使各 epoch 的 val 指标在相同
    # 物理先验强度下计算,保证选模/早停判据的跨 epoch 可比性。训练期仍按 epoch 退火。
    outputs = model(images, gamma_prior=1.0)
    loss, loss_dict = criterion(
        outputs, targets, update_famo_weights=False, enable_mal_loss=(stage > 1)
    )

    batch_size = images.size(0)

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
    dice_sum = dice.sum().item()

    # Classification
    pred_mal = torch.argmax(outputs["malignancy"], dim=1)
    mal_correct = (pred_mal == targets["malignancy"]).sum().item()
    mal_probs = torch.softmax(outputs["malignancy"].float(), dim=1)[:, 1].detach().cpu()
    mal_targets = targets["malignancy"].detach().cpu()

    return loss, loss_dict, dice_sum, mal_correct, batch_size, mal_probs, mal_targets

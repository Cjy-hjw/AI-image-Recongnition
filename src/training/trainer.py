import torch
from sklearn.metrics import roc_auc_score

from src.training.loop.bn_freeze import set_frozen_modules_eval
from src.training.loop.train_step import train_batch
from src.training.loop.validate_step import validate_batch


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
    set_frozen_modules_eval(model)

    running_loss = 0.0

    seg_loss = 0.0
    mal_loss = 0.0

    slr_beta = config["train"].get("slr_beta", 0.99)
    slr_threshold = config["train"].get("slr_threshold", 2.0)
    label_smoothing = config["train"].get("label_smoothing", 0.0)

    spin_l1_weight = 0.005 if epoch > 80 else 0.0
    for batch_idx, (images, masks, labels) in enumerate(loader):
        loss, loss_dict, prev_z_latent, prev_concept_vector = train_batch(
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
        )

        running_loss += loss.item()
        seg_loss += loss_dict["seg"].item()
        mal_loss += loss_dict["malignancy"].item()

    avg_loss = running_loss / len(loader)
    return avg_loss, seg_loss / len(loader), mal_loss / len(loader)


def validate_epoch(model, loader, criterion, device, epoch=1, official_metrics_out=None):
    """验证一个 epoch。

    Args:
        official_metrics_out: 可选 dict。传入时,官方口径指标(病灶级聚合 + 官方筛选)
            会**原地写入**该 dict,键见 `metrics_official.compute_official_metrics`。
            为什么用出参而不是加返回值:`train.py:337` 按 5 元组解包本函数,加宽
            返回值会直接破坏 legacy 训练入口(lane2b 实测该处是唯一调用点)。
            三头键缺席的数据集(legacy)→ 写入全 None 的指标 dict。
    """
    model.eval()
    criterion.eval()
    val_loss = 0.0
    val_loss_l2 = 0.0
    seg_dice = 0.0
    mal_correct = 0
    total_samples = 0
    all_mal_probs: list[float] = []
    all_mal_targets: list[int] = []
    official_records: list[dict] = []

    from src.training.stages import get_stage_number

    stage = get_stage_number(epoch)

    with torch.no_grad():
        for batch_idx, (images, masks, labels) in enumerate(loader):
            (
                loss,
                loss_dict,
                dice_sum,
                mal_correct_batch,
                batch_size,
                mal_probs,
                mal_targets,
                batch_records,
            ) = validate_batch(
                model, images, masks, labels, criterion, device, stage, batch_idx, epoch
            )

            val_loss += loss.item()
            val_loss_l2 += loss_dict["latent_l2"].item()
            total_samples += batch_size
            seg_dice += dice_sum
            mal_correct += mal_correct_batch
            all_mal_probs.extend(mal_probs.tolist())
            all_mal_targets.extend(mal_targets.tolist())
            if official_metrics_out is not None:
                official_records.extend(batch_records)

    if official_metrics_out is not None:
        from src.training.metrics_official import compute_official_metrics

        official_metrics_out.update(compute_official_metrics(official_records))

    avg_loss = val_loss / len(loader)
    avg_loss_l2 = val_loss_l2 / len(loader)
    avg_dice = seg_dice / total_samples
    avg_acc = mal_correct / total_samples

    if stage == 1:
        val_auc = None
    elif len(set(all_mal_targets)) >= 2:
        val_auc = float(roc_auc_score(all_mal_targets, all_mal_probs))
    else:
        val_auc = None
        print(
            "[Val AUC Warning] skipped: val set has single malignancy class",
            flush=True,
        )

    return avg_loss, avg_dice, avg_acc, avg_loss_l2, val_auc

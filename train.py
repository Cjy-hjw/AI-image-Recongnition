import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset.loader import ThyroidDataset
from src.dataset.augmentation import get_train_transforms
from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.loss import MultiTaskLoss
from src.training import (
    configure_stage_parameters,
    train_epoch,
    validate_epoch,
    train_authenticity_classifier,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/thyroid_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument(
        "--skip-auth",
        action="store_true",
        help="Skip authenticity classifier training when weights exist",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        config["environment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    train_dataset = ThyroidDataset(
        data_dir=config["data"]["train_path"],
        img_size=config["data"]["img_size"],
        transform=get_train_transforms(),
    )
    val_dataset = ThyroidDataset(
        data_dir=config["data"]["val_path"], img_size=config["data"]["img_size"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["environment"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["environment"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(
        f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples."
    )

    auth_weights_path = "checkpoints/authenticity_model.pth"
    if args.skip_auth and os.path.exists(auth_weights_path):
        print(
            f"--> Skipping authenticity training; found existing weights at {auth_weights_path}"
        )
    elif args.skip_auth:
        print(
            f"[WARNING] --skip-auth set but {auth_weights_path} not found; training authenticity classifier."
        )
        train_authenticity_classifier(
            train_dataset, val_dataset, device, auth_weights_path
        )
    else:
        train_authenticity_classifier(
            train_dataset, val_dataset, device, auth_weights_path
        )

    # Initialize dual-backbone model (ConvNeXt-Nano seg + EfficientNet-B2 cls + TaskAlignmentAdaptor + ConceptEmbeddingCBMHead)
    model = MultiTaskThyroidModel(
        backbone_name=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        config=config,
    ).to(device)

    # Initialize Criterion
    criterion = MultiTaskLoss().to(device)
    vclub_optimizer = optim.Adam(model.vclub.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=config["train"]["amp"])
    best_val_score = -1.0
    os.makedirs("checkpoints", exist_ok=True)

    if args.resume:
        if os.path.exists(args.resume):
            model.load_state_dict(
                torch.load(args.resume, map_location=device), strict=False
            )
            print(f"--> Loaded resume weights from {args.resume}")
        else:
            print(
                f"--> [WARNING] Resume weights not found at {args.resume}. Starting from scratch."
            )

    print("Starting training loop with Stage-wise Progressive Training...")
    running_losses_history = []

    # 全局学习率调度
    lr_base = config["train"]["lr"]
    weight_decay = config["train"].get("weight_decay", 5e-4)
    optimizer = None

    # Early Stopping
    early_stopping_patience = config["train"].get("early_stopping_patience", 5)
    w_dice = config["train"].get("checkpoint_w_dice", 0.30)
    w_auc = config["train"].get("checkpoint_w_auc", 0.70)
    stage3_es_patience = config["train"].get("stage3_early_stopping_patience", 0)
    patience_counter = 0

    for epoch in range(args.start_epoch, config["train"]["epochs"] + 1):
        # 1. 配置当前阶段参数（LR 调度由 stages.py 统一返回）
        param_groups, stage_lr, stage_name = configure_stage_parameters(
            model, criterion, epoch, lr_base, weight_decay
        )

        # 2. 创建或重建优化器：阶段边界时必须重建以反映冻结参数变化
        # prev_stage 取上一 epoch 的真实阶段(对续训同样成立),不再用 None 特例,
        # 使「optimizer 重建」与「Stage3 重置」都能在续训(start_epoch=101)时正确触发。
        from src.training.stages import get_stage_number

        current_stage = get_stage_number(epoch)
        prev_stage = get_stage_number(epoch - 1)
        stage_changed = prev_stage != current_stage

        if optimizer is None or stage_changed:
            optimizer = optim.AdamW(param_groups, lr=stage_lr)
            print(f"  [Optimizer rebuilt] Stage {current_stage}, lr={stage_lr:.6f}")
        else:
            for g in optimizer.param_groups:
                g["lr"] = stage_lr

        # 3. 运行单 Epoch 训练
        train_loss, train_seg, train_mal = train_epoch(
            model,
            train_loader,
            optimizer,
            vclub_optimizer,
            criterion,
            scaler,
            device,
            config,
            running_losses_history,
            epoch,
        )
        val_loss, val_dice, val_acc, val_loss_l2, val_auc = validate_epoch(
            model, val_loader, criterion, device, epoch
        )

        # 3.4 动态更新拉格朗日乘子
        lambda_latent = criterion.lambda_latent.item()
        lambda_latent = max(1e-4, lambda_latent + 1e-3 * (val_loss_l2 - 0.1))
        criterion.lambda_latent.copy_(
            torch.tensor(lambda_latent, device=criterion.lambda_latent.device)
        )

        # Stage 3 起点(由 get_stage_number 边界派生,避免硬编码 epoch==101):
        # 进入 Stage 3 的首个 epoch 重置选模/早停跟踪器。
        # 用 prev_stage(上一 epoch 真实阶段)判定,续训(start_epoch=101)也能正确触发。
        if current_stage == 3 and prev_stage != 3:
            best_val_score = -1.0
            patience_counter = 0
            print(
                "--> Stage 3 started: reset checkpoint tracker and early-stopping counter"
            )

        if current_stage == 1:
            val_score = val_dice
        elif val_auc is not None:
            val_score = w_dice * val_dice + w_auc * val_auc
        else:
            val_score = val_dice

        if current_stage == 1:
            val_auc_str = "N/A (Stage 1)"
        elif val_auc is not None:
            val_auc_str = f"{val_auc:.4f}"
        else:
            val_auc_str = "N/A"

        print(
            f"Epoch {epoch:02d}/{config['train']['epochs']:02d} | [{stage_name}] | "
            f"Train Loss: {train_loss:.4f} (Seg: {train_seg:.3f}, Mal: {train_mal:.3f}) | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.3f} | Val Mal Acc: {val_acc:.3f} | "
            f"Val AUC: {val_auc_str} | Val Score: {val_score:.4f} | "
            f"Latent L2: {val_loss_l2:.4f} | Lambda: {lambda_latent:.4f}"
        )

        # 3.5 早期熔断安全锁 (Early Abort Check): 在第 10 轮若 Val Dice 仍然接近零（低于 0.05），说明模型不收敛发生崩塌，立刻停机排查
        if epoch == 10 and val_dice < 0.05:
            raise RuntimeError(
                f"Early abort triggered at epoch 10 due to non-convergence (Val Dice: {val_dice:.3f} < 0.05). Check gradients or parameters."
            )

        val_auc_log = f"{val_auc:.4f}" if val_auc is not None else "N/A"
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(
                f"--> Saved new best checkpoint (Score: {val_score:.4f}, Dice: {val_dice:.4f}, AUC: {val_auc_log}) to checkpoints/best_model.pth"
            )
        else:
            patience_counter += 1
            active_patience = (
                stage3_es_patience if current_stage == 3 else early_stopping_patience
            )
            if active_patience > 0 and patience_counter >= active_patience:
                print(
                    f"--> Early stopping triggered at epoch {epoch} "
                    f"(patience={active_patience}, best={best_val_score:.4f})"
                )
                break

        # 确保最后一轮训练结束时保存最终模型
        if epoch == config["train"]["epochs"]:
            torch.save(model.state_dict(), "checkpoints/final_model.pth")
            print(f"--> Saved final epoch checkpoint to checkpoints/final_model.pth")
    print("Training finished successfully.")


if __name__ == "__main__":
    main()

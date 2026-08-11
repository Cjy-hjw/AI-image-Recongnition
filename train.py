import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset.loader import ThyroidDataset, MultiSourceThyroidDataset
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
    parser.add_argument(
        "--mix-thyroidxl",
        type=str,
        default=None,
        metavar="THYROIDXL_SEG_DIR",
        help="启用 TN3K + ThyroidXL 分割混训(补 A0 证实的跨域失效,见 docs/项目现状总纲 §6)。"
             "值 = format_thyroidxl.py 产出的 data/thyroidxl_seg 目录。"
             "不传则维持纯 TN3K 单源(默认,不破坏现有行为)。",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        config["environment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # ── 训练数据源:纯 TN3K(默认) 或 TN3K+ThyroidXL 混训(--mix-thyroidxl)──────────
    # 混训动机(A0 零训练实测):TN3K 分割主干单域过拟合,跨域 ThyroidXL 仅 18% Dice。
    # 契约红线(sonnet 对抗审):
    #   ① TN3K 源级 label_mask 屏蔽 5 概念+malignancy(概念是 mask 反推伪标签 split_tn3k.py),
    #      仅留 mask:1——否则伪标签当真监督污染概念头。
    #   ② ThyroidXL 源级补 5 概念=0 双保险(逐样本 meta 已写全,源级再兜底防漏键落 loader 默认1);
    #      mask/malignancy 用 format_thyroidxl.py 逐样本 meta 真值(无 mask 样本 valid_mask=0 零梯度)。
    #   ③ 采样器 num_samples=len(mds)、replacement=True——若用默认 shuffle 或 num_samples 过小,
    #      ThyroidXL(n=9864)每样本每 epoch 抽不到 0.5 次,配比意图落空。
    train_transform = get_train_transforms()
    if args.mix_thyroidxl is not None:  # 空串/坏路径进此分支,由下方空源 raise 兜住,不静默落回纯 TN3K
        _tn3k_src = {
            "name": "tn3k", "path": config["data"]["train_path"],
            "label_mask": {"malignancy": 0, "composition": 0, "echogenicity": 0,
                           "shape": 0, "margin": 0, "foci": 0, "mask": 1},
            "ratio": 1.0,
        }
        _xl_src = {
            "name": "thyroidxl", "path": os.path.join(args.mix_thyroidxl, "train"),
            "label_mask": {"composition": 0, "echogenicity": 0, "shape": 0,
                           "margin": 0, "foci": 0},  # mask/malignancy 用逐样本 meta 真值
            "ratio": 0.6,
        }
        train_dataset = MultiSourceThyroidDataset(
            sources=[_tn3k_src, _xl_src],
            img_size=config["data"]["img_size"], transform=train_transform,
        )
        _n_tn = len(train_dataset.datasets[0])
        _n_xl = len(train_dataset.datasets[1])
        # 中4(Gate2):空源静默降级红线——ThyroidDataset 对不存在/空目录只跳过(loader.py:25),
        # MultiSource 对空源 per_sample_w=0,sampler 不崩但 XL 权重全 0 → 实际纯 TN3K,
        # 配比意图静默落空(print 仍报 62.5:37.5)。混训必须两源都非空,否则显式拒绝。
        if _n_tn == 0 or _n_xl == 0:
            raise RuntimeError(
                f"[混训] 数据源为空(TN3K={_n_tn} ThyroidXL={_n_xl}):核对 "
                f"--mix-thyroidxl={args.mix_thyroidxl!r}(应含 train/ 子目录,先跑 "
                f"scripts/data/format_thyroidxl.py 生成)与 config train_path。"
                "拒绝静默降级为纯 TN3K。")
        print(f"[混训] TN3K={_n_tn} + ThyroidXL={_n_xl} = {len(train_dataset)} "
              f"(抽样占比≈62.5%:37.5%,补 A0 证实的跨域失效)")
    else:
        train_dataset = ThyroidDataset(
            data_dir=config["data"]["train_path"],
            img_size=config["data"]["img_size"],
            transform=train_transform,
        )
    # val 严格纯 TN3K(高4 护栏):TN3K val Dice 是「混训毁没毁主域」的唯一探针,
    # 绝不混 ThyroidXL 稀释退化信号;epoch10 熔断 + 早停据此天然守护 TN3K 98%。
    val_dataset = ThyroidDataset(
        data_dir=config["data"]["val_path"], img_size=config["data"]["img_size"]
    )

    # 混训用 WeightedRandomSampler(按源 ratio 配比),与 shuffle 互斥(同传 PyTorch 报错)。
    if args.mix_thyroidxl is not None:  # 与上方数据集分支判据一致(空串已被空源 raise 拦截)
        _sampler = WeightedRandomSampler(
            train_dataset.get_sample_weights(),
            # num_samples=len(mds):按源权重质量(TN3K 1.0 : XL 0.6)抽样。单 TN3K 样本每 epoch
            # 期望被抽 ~1.2 次、XL 每 epoch 去重可见 ~4700 张不同跨域图(0.8 draw/样本)。
            # 用 len(mds) 而非更小值:num_samples 过小会让 XL(n=9864)每样本抽不到 0.5 次。
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["data"]["batch_size"],
            sampler=_sampler,
            num_workers=config["environment"]["num_workers"],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
    else:
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
        # 恢复续训态:FAMO 动态权重 + criterion(AUCM 对偶参数/lambda_latent) + best 分数。
        # 不含 optimizer 动量——optimizer 按 stage 边界重建(见下方循环),续训通常落在
        # stage 边界,故动量持久化非必需,此处为已知简化。
        state_path = "checkpoints/training_state.pth"
        if os.path.exists(state_path):
            ts = torch.load(state_path, map_location=device)
            criterion.load_state_dict(ts["criterion"])
            criterion.famo.load_state_dict(ts["famo"])
            # 2026-08-11 R6: FAMO 方向已修复(见 src/models/famo.py),旧 checkpoint 的
            # w_logits 是"反方向"产物(给强任务加权),语义颠倒不可继承。强制重置为均权重
            # 从零重启调度——旧 best=0.5653 那批本就是错方向产物,不值得继承。
            criterion.famo.w_logits = torch.zeros(criterion.famo.num_tasks)
            criterion.famo.prev_loss = torch.zeros(criterion.famo.num_tasks)
            print(
                "--> [R6] FAMO w_logits/prev_loss reset to zero "
                "(direction fix invalidates old reversed logits)."
            )
            best_val_score = ts.get("best_val_score", best_val_score)
            print(
                f"--> Restored training state from epoch {ts.get('epoch')} "
                f"(best_val_score={best_val_score:.4f}; FAMO + AUCM dual params + lambda_latent)"
            )
        else:
            print(
                "--> [WARNING] training_state.pth not found; FAMO/criterion start fresh."
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
            # 续训态与 best_model 同步落盘,严格配对。best_model.pth 保持纯
            # model.state_dict 以兼容 predict/evaluate/SHA256;FAMO 动态权重与
            # criterion(AUCM 对偶参数 a/b/alpha + lambda_latent)单列于此。
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_score": best_val_score,
                    "criterion": criterion.state_dict(),
                    "famo": criterion.famo.state_dict(),
                },
                "checkpoints/training_state.pth",
            )
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
            print("--> Saved final epoch checkpoint to checkpoints/final_model.pth")
    print("Training finished successfully.")


if __name__ == "__main__":
    main()

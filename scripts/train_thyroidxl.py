"""ThyroidXL LP-FT-LoRA 训练入口。

两阶段训练：
  Stage LP（线性探测）：冻结双主干，仅训任务头，lr=1e-3，最多 lp_epochs（默认10）
  Stage FT（LoRA 微调）：LoRA A/B 解冻，主干原始权重保持冻结，lr=1e-4，ft_epochs（默认40）

评估：患者级 AUC（GroupKFold by patient_id，Max 聚合），图像级 Dice。
"""
import argparse
import json
import os
import random
import warnings
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from src.dataset.multiset_adapter import ThyroidXLAdapter, AdaptedThyroidDataset
from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.lora_utils import get_lora_params


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/public_datasets/ThyroidXL")
    p.add_argument("--config", default="configs/thyroid_config.yaml")
    p.add_argument("--out", default="checkpoints/thyroidxl_best.pth")
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lp-epochs", type=int, default=10, help="Linear Probing阶段epoch数")
    p.add_argument("--ft-epochs", type=int, default=40, help="LoRA Fine-Tuning阶段epoch数")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="LP阶段头部lr")
    p.add_argument("--lora-lr", type=float, default=1e-4, help="FT阶段LoRA lr（固定，不衰减）")
    p.add_argument("--val-fold", type=int, default=0, help="GroupKFold 验证fold索引(0-4)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="冒烟测试：只用少量数据跑3epoch")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(args, img_size: int):
    adapter = ThyroidXLAdapter(args.data_root)
    dataset = AdaptedThyroidDataset(adapter, img_size=img_size)

    samples = adapter.samples
    patient_ids = [s["patient_id"] for s in samples]
    groups = np.array(patient_ids)
    indices = np.arange(len(samples))

    # 患者级 GroupKFold（防泄漏）
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(indices, groups=groups))
    train_idx, val_idx = splits[args.val_fold % len(splits)]

    # 患者级泄漏自检
    train_pids = set(patient_ids[i] for i in train_idx)
    val_pids = set(patient_ids[i] for i in val_idx)
    assert len(train_pids & val_pids) == 0, "患者泄漏！train/val 有重叠 patient_id"

    if args.smoke:
        train_idx = train_idx[:min(50, len(train_idx))]
        val_idx = val_idx[:min(20, len(val_idx))]

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    val_patients = [patient_ids[i] for i in val_idx]
    return train_loader, val_loader, val_patients


# ─────────────────────────────────────────────────────────────────────────────
def build_head_optimizer(model, lr: float, weight_decay: float = 1e-4):
    """LP 阶段：只训 backbone 以外的头部参数（backbone 已冻结）。"""
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "cls_backbone" not in n and "seg_backbone" not in n]
    return optim.AdamW(head_params, lr=lr, weight_decay=weight_decay)


def build_lora_optimizer(model, lora_lr: float, head_lr: float = 3e-4, weight_decay: float = 1e-4):
    """FT 阶段：头部 + LoRA A/B，各自独立 lr。"""
    lora_params = get_lora_params(model)
    lora_ids = {id(p) for p in lora_params}
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and id(p) not in lora_ids
                   and "cls_backbone" not in n and "seg_backbone" not in n]
    groups = [
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        {"params": lora_params, "lr": lora_lr, "weight_decay": 0.0},
    ]
    return optim.AdamW(groups)


# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, device, config):
    model.train()
    ce = nn.CrossEntropyLoss()
    dice_loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for imgs, masks, labels in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        mal_labels = labels["malignancy"].long().to(device)
        valid_mal = labels.get("valid_malignancy", torch.ones_like(mal_labels)).to(device)

        optimizer.zero_grad()
        with autocast():
            out = model(imgs)
            # malignancy head
            loss = torch.tensor(0.0, device=device)
            if valid_mal.any():
                mal_logits = out["mal_logits"][valid_mal.bool()]
                loss = loss + ce(mal_logits, mal_labels[valid_mal.bool()])
            # segmentation head
            seg_logits = out.get("seg_logits")
            if seg_logits is not None:
                loss = loss + 0.5 * dice_loss_fn(seg_logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, val_patients, device):
    """患者级 AUC（Max 聚合）+ 图像级 Dice。"""
    model.eval()
    patient_probs: dict[str, list] = defaultdict(list)
    patient_labels: dict[str, int] = {}
    dice_scores = []

    for i, (imgs, masks, labels) in enumerate(loader):
        imgs = imgs.to(device)
        out = model(imgs)
        mal_probs = torch.softmax(out["mal_logits"], dim=1)[:, 1].cpu().numpy()
        mal_labels = labels["malignancy"].numpy()
        batch_patients = val_patients[i * loader.batch_size: (i + 1) * loader.batch_size]

        for pid, prob, lbl in zip(batch_patients, mal_probs, mal_labels):
            patient_probs[pid].append(float(prob))
            patient_labels[pid] = int(lbl)

        seg_logits = out.get("seg_logits")
        if seg_logits is not None:
            pred = (torch.sigmoid(seg_logits.cpu()) > 0.5).float()
            intersection = (pred * masks).sum(dim=(1, 2, 3))
            union = pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice_scores.extend(((2 * intersection + 1e-6) / (union + 1e-6)).tolist())

    # Max 聚合
    pids = list(patient_labels.keys())
    y_true = [patient_labels[p] for p in pids]
    y_score = [max(patient_probs[p]) for p in pids]

    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    dice = float(np.mean(dice_scores)) if dice_scores else float("nan")
    return auc, dice


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)
    warnings.filterwarnings("ignore", category=UserWarning)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config["data"]["img_size"]

    print(f"Device: {device} | img_size: {img_size}")

    train_loader, val_loader, val_patients = build_dataloaders(args, img_size)
    print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # 构建模型（use_lora 在 __init__ 里 apply_lora + 冻结骨干原始权重）
    model = MultiTaskThyroidModel(
        config=config,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    scaler = GradScaler()
    best_auc = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total_epochs = args.lp_epochs + args.ft_epochs
    if args.smoke:
        total_epochs = 3
        args.lp_epochs = min(args.lp_epochs, 2)

    for epoch in range(1, total_epochs + 1):
        is_lp = epoch <= args.lp_epochs
        stage_name = "LP" if is_lp else "FT-LoRA"

        # 切换 optimizer
        if epoch == 1:
            optimizer = build_head_optimizer(model, lr=args.lr)
        elif epoch == args.lp_epochs + 1:
            # 切到 FT 阶段：激活 LoRA 参数（骨干原始权重保持冻结）
            for n, p in model.named_parameters():
                if "lora_A" in n or "lora_B" in n:
                    p.requires_grad = True
            optimizer = build_lora_optimizer(model, lora_lr=args.lora_lr)
            print(f"[Epoch {epoch}] 切换到 FT-LoRA 阶段")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, config)
        auc, dice = evaluate(model, val_loader, val_patients, device)

        print(f"[Epoch {epoch:3d}/{total_epochs} {stage_name}] "
              f"loss={train_loss:.4f}  patient-AUC={auc:.4f}  Dice={dice:.4f}")

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            torch.save({
                "epoch": epoch,
                "stage": stage_name,
                "state_dict": model.state_dict(),
                "patient_val_auc": auc,
                "dice": dice,
            }, args.out)
            print(f"  → saved best (AUC={best_auc:.4f})")

    print(f"\n训练完成。Best patient-AUC: {best_auc:.4f} → {args.out}")


if __name__ == "__main__":
    main()

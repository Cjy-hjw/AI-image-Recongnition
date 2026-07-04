"""Malignancy 分支预训练脚本(官方数据到位前的抢跑基线)。

用真实病理金标准(ThyUS2Path, figshare 27021604)预训 MultiTaskThyroidModel 的
malignancy 通路(cls_backbone + adaptors + cbm_head + 经物理先验回流的 seg 主干)。

设计取舍(为何不复用 MultiTaskLoss/train_batch):
- 本任务**只有 malignancy 标签**,seg/concept 全无标注。走完整多任务循环会拖入
  FAMO 5 通道调度、vCLUB 互信息、SLR、AUCM 对偶变量等一堆为多任务设计的机制,
  在单任务纯预训下既无必要又增加不稳定性与调试面。
- 故这里只前向模型、只对 outputs["malignancy"] 算带 class-weight 的 CE,直接反向。
  模型 forward 仍会算出 mask/concept(无法跳过),但我们不对其反向,梯度只经
  malignancy logits 回流,co-adapt cls 主干 + CBM 头 + 物理先验路径。
- AMP 用 bf16(与 train_step 一致,bf16 无需 GradScaler,规避 AMP/AUCM 不兼容坑)。

产物:checkpoints/malignancy_pretrain_best.pth(按 val AUC 存最优),供官方数据到位后
加载做初始化微调。

用法(A800):
    python scripts/pretrain_malignancy.py \
        --data-root data/27021604_extract --epochs 30 --batch-size 32 --img-size 512
冒烟自检(3 epoch,少量数据):
    python scripts/pretrain_malignancy.py --epochs 3 --smoke
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# 允许脚本在仓库根目录直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.pathology_dataset import (
    PathologyDataset,
    build_pathology_samples,
    patient_grouped_split,
)
from src.dataset.augmentation.jpeg_augment import get_train_transforms_with_jpeg
from src.models.multi_task_model import MultiTaskThyroidModel


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_weight(train_ds: PathologyDataset, device) -> torch.Tensor:
    """按训练集良恶性频率反比构造 CE 权重(缓解类不平衡),归一化到均值 1。"""
    counts = train_ds.class_counts()  # {0: n_ben, 1: n_mal}
    n0 = counts.get(0, 1)
    n1 = counts.get(1, 1)
    total = n0 + n1
    # 反频率权重
    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)
    w = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    return w


@torch.no_grad()
def evaluate(model, loader, device) -> tuple:
    """返回 (val_loss, acc, auc)。auc 在验证集单类时为 None。"""
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    all_probs, all_tgts = [], []
    for imgs, _masks, labels in loader:
        imgs = imgs.to(device)
        tgt = labels["malignancy"].to(device)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            out = model(imgs)
            logits = out["malignancy"].float()
            loss = ce(logits, tgt)
        loss_sum += loss.item() * imgs.size(0)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        correct += (pred == tgt).sum().item()
        total += imgs.size(0)
        all_probs.extend(prob.tolist())
        all_tgts.extend(tgt.tolist())
    acc = correct / max(1, total)
    val_loss = loss_sum / max(1, total)
    auc = float(roc_auc_score(all_tgts, all_probs)) if len(set(all_tgts)) >= 2 else None
    return val_loss, acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/27021604_extract")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/malignancy_pretrain_best.pth")
    ap.add_argument("--jpeg-stage", default="mid", choices=["early", "mid", "late"])
    ap.add_argument("--no-pretrained", action="store_true",
                    help="不加载本地 backbone 权重(无 checkpoints 时用,随机初始化)")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式:每类各截断少量患者、关早停,仅验证管道跑通")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pretrain] device={device} img_size={args.img_size} bs={args.batch_size}", flush=True)

    # ── 数据 ──────────────────────────────────────────────────────────────
    samples = build_pathology_samples(args.data_root)
    train_s, val_s = patient_grouped_split(samples, val_frac=args.val_frac, seed=args.seed)

    if args.smoke:
        # 冒烟:各集按类各取一半,**保证两类都在**,否则单类切片会使 AUC 恒 None、
        # 判别信号缺失,冒烟变假绿(Gate2 审查修复)。
        def _take_both(pool, n):
            pos = [s for s in pool if s["label"] == 1][: n // 2]
            neg = [s for s in pool if s["label"] == 0][: n // 2]
            return pos + neg
        train_s = _take_both(train_s, 64)
        val_s = _take_both(val_s, 32)
        args.epochs = min(args.epochs, 3)

    train_tf = get_train_transforms_with_jpeg(stage=args.jpeg_stage, jpeg_prob=0.7)
    train_ds = PathologyDataset(train_s, img_size=args.img_size, transform=train_tf)
    val_ds = PathologyDataset(val_s, img_size=args.img_size, transform=None)

    print(f"[pretrain] train={len(train_ds)} val={len(val_ds)} "
          f"train_cls={train_ds.class_counts()} val_cls={val_ds.class_counts()}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    # ── 模型 ──────────────────────────────────────────────────────────────
    config = {"data": {"img_size": args.img_size}}
    model = MultiTaskThyroidModel(
        backbone_name="efficientnet_b2",
        pretrained=(not args.no_pretrained),
        config=config,
    ).to(device)

    class_w = build_class_weight(train_ds, device)
    print(f"[pretrain] CE class_weight (benign,malignant) = {class_w.tolist()}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_auc = -1.0

    # ── 训练循环 ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for imgs, _masks, labels in train_loader:
            imgs = imgs.to(device)
            tgt = labels["malignancy"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                out = model(imgs)
                logits = out["malignancy"].float()
                loss = criterion(logits, tgt)

            if torch.isnan(loss):
                print(f"  [NaN Guard] epoch{epoch} loss=NaN,跳过该 batch", flush=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            run_loss += loss.item() * imgs.size(0)
            seen += imgs.size(0)

        scheduler.step()
        train_loss = run_loss / max(1, seen)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, device)
        auc_str = f"{val_auc:.4f}" if val_auc is not None else "N/A(单类)"
        print(
            f"[epoch {epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={auc_str} "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        # 早停短路(rules.md §3):Epoch>=3 且 AUC 仍近随机则中止(冒烟模式豁免)
        if (not args.smoke) and epoch >= 3 and val_auc is not None and val_auc < 0.55:
            print(f"[abort] epoch{epoch} val_auc={val_auc:.4f} < 0.55,判为不收敛,中止", flush=True)
            break

        if val_auc is not None and val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_auc": val_auc,
                 "img_size": args.img_size},
                args.out,
            )
            print(f"  ↳ 保存 best (val_auc={val_auc:.4f}) → {args.out}", flush=True)

    print(f"[done] best_val_auc={best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()

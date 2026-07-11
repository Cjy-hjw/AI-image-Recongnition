"""ThyUS2Path(figshare 27021604)LP-FT-LoRA 训练入口 —— MIL vs LoRA 对拍实验。

背景：malignancy MIL 路线（pretrain_malignancy_mil.py）在同一份 27021604 数据上
卡在患者级 AUC=0.5653（Phase1冻结backbone），Phase2解冻震荡未突破。ThyroidXL 上
验证过的 LP-FT-LoRA 管道（同架构：EfficientNet-B2/ConvNeXt-Nano 双backbone +
LoRA r=8 alpha=16）却测得 patient-AUC=0.8514（真实 predict.py 链路）。

本脚本把 LP-FT-LoRA 管道原样搬到 27021604 数据上做对拍：同架构、同患者级
GroupKFold/patient_grouped_split评估协议，唯一变量是数据源。目的是回答
"MIL 卡在 0.57 的根因是弱监督bag框架本身，还是这份数据集本身的天花板"——
若这里也能到 0.7+，说明是框架问题；若仍卡 0.55~0.60，说明是数据天花板。

与 train_thyroidxl.py 的差异（不复用其代码,遵循项目"新实验独立脚本"惯例）：
  - 数据源换成 build_pathology_samples + patient_grouped_split（单帧+患者级
    histo_label广播，非per-nodule标注），无分割mask标注 → 不训分割头、不算Dice。
  - 训练/评估单位仍是"帧"，但评估时按 patient_id 做 Max 聚合（与 MIL 评估协议
    对齐，可直接和 0.5653 比较）。

两阶段：LP（冻结双主干，仅训malignancy头，lr=1e-3）→ FT-LoRA（解冻LoRA A/B，
lr=1e-4，主干原始权重全程冻结）。
"""
import argparse
import os
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from src.dataset.pathology_dataset import build_pathology_samples, patient_grouped_split, PathologyDataset
from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.lora_utils import get_lora_params
from src.training.loop.bn_freeze import set_frozen_modules_eval


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/27021604_extract")
    p.add_argument("--config", default="configs/thyroid_config.yaml")
    p.add_argument("--out", default="checkpoints/thyus2path_lora_best.pth")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lp-epochs", type=int, default=10, help="Linear Probing阶段epoch数")
    p.add_argument("--ft-epochs", type=int, default=40, help="LoRA Fine-Tuning阶段epoch数")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="LP阶段头部lr")
    p.add_argument("--lora-lr", type=float, default=1e-4, help="FT阶段LoRA lr（固定，不衰减）")
    p.add_argument("--val-frac", type=float, default=0.2)
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
    samples = build_pathology_samples(args.data_root)
    train_samples, val_samples = patient_grouped_split(samples, val_frac=args.val_frac, seed=args.seed)

    # 患者级泄漏自检（patient_grouped_split 设计上已无泄漏，此处是双保险，
    # 对齐 train_thyroidxl.py / duipai_probe_vs_head.py 的既定写法）
    train_pids = {s["patient_key"] for s in train_samples}
    val_pids = {s["patient_key"] for s in val_samples}
    assert len(train_pids & val_pids) == 0, "患者泄漏！train/val 有重叠 patient_key"

    if args.smoke:
        train_samples = train_samples[:min(50, len(train_samples))]
        val_samples = val_samples[:min(20, len(val_samples))]

    train_ds = PathologyDataset(train_samples, img_size=img_size)
    val_ds = PathologyDataset(val_samples, img_size=img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    val_patients = [s["patient_key"] for s in val_samples]
    return train_loader, val_loader, val_patients


# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(model, head_lr: float, weight_decay: float = 1e-4):
    """创建单一 optimizer，LP 阶段只含头部参数。
    FT 阶段通过 add_param_group 追加 LoRA 组，保留头部 Adam 状态。
    """
    lora_ids = {id(p) for p in get_lora_params(model)}
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and id(p) not in lora_ids
                   and "cls_backbone" not in n and "seg_backbone" not in n]
    return optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)


# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    # 2026-07-10 血泪教训（详见 src/training/AGENTS.md）：冻结 backbone 的
    # BatchNorm running_mean/running_var 只受 module.training 控制,与
    # requires_grad 无关——漏调此函数会致数值病态。LP/FT 每 epoch 必调。
    set_frozen_modules_eval(model)
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    for imgs, _masks, labels in loader:
        imgs = imgs.to(device)
        mal_labels = labels["malignancy"].long().to(device)

        optimizer.zero_grad()
        # bf16 AMP（无 GradScaler），对齐 train_thyroidxl.py 既定规范：fp16 在
        # FT-LoRA 阶段会被 backbone LoRA 扰动顶穿 65504 天花板，bf16 指数范围
        # 同 fp32，根除溢出。
        with autocast(dtype=torch.bfloat16):
            out = model(imgs)
            loss = ce(out["malignancy"], mal_labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if torch.isfinite(loss):
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, val_patients, device):
    """患者级 AUC（Max 聚合）——与 MIL 评估协议对齐，可直接对比 0.5653。"""
    model.eval()
    patient_probs: dict[str, list] = defaultdict(list)
    patient_labels: dict[str, int] = {}

    for i, (imgs, _masks, labels) in enumerate(loader):
        imgs = imgs.to(device)
        out = model(imgs)
        mal_probs = torch.softmax(out["malignancy"], dim=1)[:, 1].cpu().numpy()
        mal_labels = labels["malignancy"].numpy()
        batch_patients = val_patients[i * loader.batch_size: (i + 1) * loader.batch_size]

        for pid, prob, lbl in zip(batch_patients, mal_probs, mal_labels):
            patient_probs[pid].append(float(prob))
            patient_labels[pid] = int(lbl)

    pids = list(patient_labels.keys())
    y_true = [patient_labels[p] for p in pids]
    y_score = [max(patient_probs[p]) for p in pids]

    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return auc


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
    print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples "
          f"| Val patients: {len(set(val_patients))}")

    model = MultiTaskThyroidModel(
        config=config,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    best_auc = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total_epochs = args.lp_epochs + args.ft_epochs
    if args.smoke:
        total_epochs = 3
        args.lp_epochs = min(args.lp_epochs, 2)

    for epoch in range(1, total_epochs + 1):
        is_lp = epoch <= args.lp_epochs
        stage_name = "LP" if is_lp else "FT-LoRA"

        # 两个 if 独立（非 elif）：--lp-epochs 0 时 epoch==1 与 epoch==lp_epochs+1(=1)
        # 重合，必须都执行（同 train_thyroidxl.py §1.2 fix，同一坑不重踩）。
        if epoch == 1:
            optimizer = build_optimizer(model, head_lr=args.lr)
        if epoch == args.lp_epochs + 1:
            for n, p in model.named_parameters():
                if "lora_A" in n or "lora_B" in n:
                    p.requires_grad = True
            backbone_trainable = [n for n, p in model.named_parameters()
                                  if p.requires_grad and "lora_A" not in n and "lora_B" not in n
                                  and ("cls_backbone" in n or "seg_backbone" in n)]
            assert len(backbone_trainable) == 0, f"骨干原始权重意外解冻：{backbone_trainable[:3]}"
            lora_params = get_lora_params(model)
            optimizer.add_param_group({"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.0})
            print(f"[Epoch {epoch}] 切换到 FT-LoRA 阶段（add_param_group，头部Adam状态保留）")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        auc = evaluate(model, val_loader, val_patients, device)

        print(f"[Epoch {epoch:3d}/{total_epochs} {stage_name}] "
              f"loss={train_loss:.4f}  patient-AUC={auc:.4f}")

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            torch.save({
                "epoch": epoch,
                "stage": stage_name,
                "state_dict": model.state_dict(),
                "patient_val_auc": auc,
            }, args.out)
            print(f"  → saved best (AUC={best_auc:.4f})")

    print(f"\n训练完成。Best patient-AUC: {best_auc:.4f} → {args.out}")
    print(f"对拍参照：MIL Phase1(冻结backbone)=0.5653 | ThyroidXL同架构真实链路=0.8514")


if __name__ == "__main__":
    main()

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
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset

from src.dataset.multiset_adapter import ThyroidXLAdapter, AdaptedThyroidDataset
from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.lora_utils import get_lora_params
from src.training.loop.bn_freeze import set_frozen_modules_eval

# ThyroidXL train 实测前景占比：mean=0.0790, median=0.0508（结节远小于背景）
# pos_weight = bg:fg ≈ 11.66，用于抵消 BCE 在类别不均衡下学到"全图判正/判负"trivial解的问题
SEG_POS_WEIGHT = 11.66


def seg_combined_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE(带pos_weight) + Differentiable Dice 组合损失，防止分割头收敛到trivial常量解。"""
    pos_weight = torch.tensor([SEG_POS_WEIGHT], device=pred_logits.device)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)
    prob = torch.sigmoid(pred_logits)
    flat_p = prob.flatten(1)
    flat_t = target.flatten(1)
    inter = (flat_p * flat_t).sum(1)
    dice = 1 - (2 * inter + 1) / (flat_p.sum(1) + flat_t.sum(1) + 1)
    return 0.5 * bce + 0.5 * dice.mean()


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
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(model, head_lr: float, weight_decay: float = 1e-4):
    """创建单一 optimizer，LP 阶段只含头部参数。
    FT 阶段通过 add_param_group 追加 LoRA 组，保留头部 Adam 状态（§1.2 fix）。
    """
    lora_ids = {id(p) for p in get_lora_params(model)}
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and id(p) not in lora_ids
                   and "cls_backbone" not in n and "seg_backbone" not in n]
    return optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)


# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, config):
    model.train()
    # §3 BATCHNORM STATE LOCKING（src/models/AGENTS.md）：backbone 原始权重全程
    # requires_grad=False，但 model.train() 递归下去，其内部 BatchNorm 仍处于训练
    # 模式——running_mean/running_var 只受 module.training 控制，与 requires_grad
    # 无关，会持续随每个 batch 的真实激活值漂移。50 epoch 下来把本就临界的通道
    # （如 blocks.0.1.bn1 预训练时 running_var 已达 3.5e-42）冲向数值病态区间，
    # 经 cls_backbone 逐层复合放大，导致 CPU/GPU、LoRA/合并 forward 相对误差炸到
    # 70%~1240%（Opus 诊断 + 本地 A800 实测锁定，checkpoint 因此报废重训）。
    # train.py 主管线 trainer.py 已用 set_frozen_modules_eval() 锁定，此前
    # train_thyroidxl.py 漏调——LP/FT 两阶段每个 epoch 都要补上。
    set_frozen_modules_eval(model)
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    for imgs, masks, labels in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        mal_labels = labels["malignancy"].long().to(device)
        valid_mal = labels.get("valid_malignancy", torch.ones_like(mal_labels)).to(device)

        optimizer.zero_grad()
        # bf16 AMP（无 GradScaler），对齐项目既定规范（见 src/dataset/AGENTS.md pretrain_malignancy）。
        # 根因锁定：fp16(autocast默认)在 FT-LoRA 阶段被 backbone LoRA 扰动顶穿 65504 天花板，
        # GradScaler scale 从 65536 一路减半仍失守 → loss 爆 1e99。bf16 指数范围同 fp32，根除溢出。
        with autocast(dtype=torch.bfloat16):
            out = model(imgs)
            # malignancy head
            loss = torch.tensor(0.0, device=device)
            if valid_mal.any():
                mal_logits = out["malignancy"][valid_mal.bool()]
                loss = loss + ce(mal_logits, mal_labels[valid_mal.bool()])
            # segmentation head：clamp 仍保留为数值安全边界（QA_VP.md §1.3）+ 压制 logit 单调爬升
            seg_logits = out.get("mask")
            if seg_logits is not None:
                seg_logits_c = seg_logits.clamp(-15, 15)
                loss = loss + 0.5 * seg_combined_loss(seg_logits_c, masks)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if torch.isfinite(loss):
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
        mal_probs = torch.softmax(out["malignancy"], dim=1)[:, 1].cpu().numpy()
        mal_labels = labels["malignancy"].numpy()
        batch_patients = val_patients[i * loader.batch_size: (i + 1) * loader.batch_size]

        for pid, prob, lbl in zip(batch_patients, mal_probs, mal_labels):
            patient_probs[pid].append(float(prob))
            patient_labels[pid] = int(lbl)

        seg_logits = out.get("mask")
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

    best_auc = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total_epochs = args.lp_epochs + args.ft_epochs
    if args.smoke:
        total_epochs = 3
        args.lp_epochs = min(args.lp_epochs, 2)

    for epoch in range(1, total_epochs + 1):
        is_lp = epoch <= args.lp_epochs
        stage_name = "LP" if is_lp else "FT-LoRA"

        # 切换 optimizer / 阶段。两个 if 独立（非 elif）：--lp-epochs 0 时
        # epoch==1 与 epoch==lp_epochs+1(=1) 重合，必须都执行——否则 elif 只建头部
        # optimizer、LoRA 解冻分支被跳过，训练全程静默不动 LoRA 参数且无报错。
        if epoch == 1:
            optimizer = build_optimizer(model, head_lr=args.lr)
        if epoch == args.lp_epochs + 1:
            # FT 阶段：解冻 LoRA 参数，add_param_group 保留头部 Adam 状态（§1.2 fix）
            for n, p in model.named_parameters():
                if "lora_A" in n or "lora_B" in n:
                    p.requires_grad = True
            # §1.1 断言：骨干非LoRA参数仍冻结
            backbone_trainable = [n for n, p in model.named_parameters()
                                  if p.requires_grad and "lora_A" not in n and "lora_B" not in n
                                  and ("cls_backbone" in n or "seg_backbone" in n)]
            assert len(backbone_trainable) == 0, f"骨干原始权重意外解冻：{backbone_trainable[:3]}"
            lora_params = get_lora_params(model)
            optimizer.add_param_group({"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.0})
            print(f"[Epoch {epoch}] 切换到 FT-LoRA 阶段（add_param_group，头部Adam状态保留）")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
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

"""MIL(多示例学习)患者级 malignancy 预训练——阶段1 & 阶段2。

背景(见 memory/patient-leakage-and-probe-trap.md):单帧强监督在患者级弱标签下
无泄漏 AUC 卡 0.50(梯度被背景帧污染,已对拍证实)。本脚本用门控注意力 MIL:
一个患者的全部帧 = 一个 bag,注意力自动挑病灶帧,患者级预测+评估。

阶段1(默认):冻结 ImageNet backbone,只训 mil_pooling + classifier。
    目标:无泄漏患者级 AUC 从 0.50 突破到 0.65-0.70,验证 MIL 路线。
    结果:best_patient_val_auc=0.5653(epoch 15 早停)。

阶段2(--unfreeze-backbone):从阶段1 checkpoint 热启动,LLRD 解冻 backbone。
    目标:patient_val_auc > 0.65-0.70。
    --resume-from 指定阶段1 checkpoint,--backbone-lr-scale 控制 backbone 顶层
    lr 相对 head lr 的倍率(默认 0.1),--llrd-eta 控制层间衰减(默认 0.75)。

与 pretrain_malignancy.py(单帧 CE 基线,保留)相互独立,互不干扰。

用法(A800):
    # 阶段1
    python scripts/pretrain_malignancy_mil.py --data-root data/27021604_extract \\
        --epochs 30 --lr 1e-3 --max-frames-per-bag 32
    # 阶段2
    python scripts/pretrain_malignancy_mil.py --data-root data/27021604_extract \\
        --epochs 30 --lr 1e-3 --unfreeze-backbone \\
        --resume-from checkpoints/malignancy_mil_best.pth \\
        --backbone-lr-scale 0.1 --llrd-eta 0.75
冒烟:
    python scripts/pretrain_malignancy_mil.py --smoke
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.pathology_dataset import build_pathology_samples, patient_grouped_split
from src.dataset.pathology_mil_dataset import build_patient_bags, PathologyMILDataset
from src.dataset.augmentation.jpeg_augment import get_train_transforms_with_jpeg
from src.models.mil_pooling import ThyroidMILNet, EFFNET_B2_FEATURE_DIM


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_weight(bags, device) -> torch.Tensor:
    """按训练集**患者级**良恶性频率反比构造 CE 权重(归一化均值 1)。"""
    counts = {0: 0, 1: 0}
    for b in bags:
        counts[int(b["label"])] += 1
    n0 = max(1, counts[0])
    n1 = max(1, counts[1])
    total = n0 + n1
    w = torch.tensor([total / (2.0 * n0), total / (2.0 * n1)],
                     dtype=torch.float32, device=device)
    return w


def _mil_collate(batch):
    """batch_size 可 >1:返回 list-of-(bag_tensor, label, pkey),逐 bag 处理(变长)。"""
    return batch


@torch.no_grad()
def evaluate_mil(model, loader, device) -> tuple:
    """患者级评估:每 bag 一个预测 → roc_auc_score(患者标签, 患者概率)。

    返回 (val_loss, patient_acc, patient_auc)。auc 单类时为 None。
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    all_probs, all_tgts = [], []
    for batch in loader:
        for bag, label, _pkey in batch:
            bag = bag.to(device)
            tgt = torch.tensor([label], device=device)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                logits, _w = model(bag)          # [1, 2]
                loss = ce(logits.float(), tgt)
            loss_sum += loss.item()
            prob = torch.softmax(logits.float(), dim=1)[0, 1].item()
            pred = int(logits.float().argmax(dim=1).item())
            correct += int(pred == label)
            total += 1
            all_probs.append(prob)
            all_tgts.append(int(label))
    val_loss = loss_sum / max(1, total)
    acc = correct / max(1, total)
    auc = float(roc_auc_score(all_tgts, all_probs)) if len(set(all_tgts)) >= 2 else None
    return val_loss, acc, auc


def _assert_no_patient_leak(train_bags, val_bags):
    """无泄漏自检(memory/patient-leakage 铁律):train/val 患者集合必须不相交。"""
    tr = {b["patient_key"] for b in train_bags}
    va = {b["patient_key"] for b in val_bags}
    overlap = tr & va
    assert not overlap, f"患者泄漏!train/val 交集 {len(overlap)} 例: {list(overlap)[:5]}"
    return len(tr), len(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/27021604_extract")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4, help="每 batch 患者(bag)数")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3, help="MIL 头 lr(阶段1 backbone 冻结)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/malignancy_mil_best.pth")
    ap.add_argument("--jpeg-stage", default="mid", choices=["early", "mid", "late"])
    ap.add_argument("--max-frames-per-bag", type=int, default=32)
    ap.add_argument("--mil-attn-dim", type=int, default=128)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="不加载 backbone ImageNet 权重(随机初始化)")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟:各类少量患者、3 epoch、关早停,仅验证管道")
    # ── Phase2 参数 ──────────────────────────────────────────────────────────
    ap.add_argument("--resume-from", default=None,
                    help="Phase2: 从指定 checkpoint 热启动(路径)")
    ap.add_argument("--unfreeze-backbone", action="store_true",
                    help="Phase2: 解冻 backbone，启用 LLRD 分层学习率")
    ap.add_argument("--backbone-lr-scale", type=float, default=0.1,
                    help="Phase2: backbone 顶层 lr = lr * backbone_lr_scale(默认0.1)")
    ap.add_argument("--llrd-eta", type=float, default=0.75,
                    help="Phase2: LLRD 层间衰减系数 η(默认0.75)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[mil] device={device} img_size={args.img_size} bs={args.batch_size}", flush=True)

    # ── 数据:复用无泄漏患者级切分,再封装成 bag ──────────────────────────
    samples = build_pathology_samples(args.data_root)
    train_s, val_s = patient_grouped_split(samples, val_frac=args.val_frac, seed=args.seed)
    train_bags = build_patient_bags(train_s)
    val_bags = build_patient_bags(val_s)

    if args.smoke:
        def _take_both(bags, n):
            pos = [b for b in bags if b["label"] == 1][: n // 2]
            neg = [b for b in bags if b["label"] == 0][: n // 2]
            return pos + neg
        train_bags = _take_both(train_bags, 16)
        val_bags = _take_both(val_bags, 8)
        args.epochs = min(args.epochs, 3)

    n_tr, n_va = _assert_no_patient_leak(train_bags, val_bags)
    print(f"[mil] train_patients={n_tr} val_patients={n_va} (无泄漏自检通过)", flush=True)

    train_tf = get_train_transforms_with_jpeg(stage=args.jpeg_stage, jpeg_prob=0.7)
    train_ds = PathologyMILDataset(train_bags, img_size=args.img_size, transform=train_tf,
                                   max_frames_per_bag=args.max_frames_per_bag, seed=args.seed,
                                   random_frames=True)
    val_ds = PathologyMILDataset(val_bags, img_size=args.img_size, transform=None,
                                 max_frames_per_bag=args.max_frames_per_bag, seed=args.seed,
                                 random_frames=False)  # 固定前N帧,确保评估可复现
    print(f"[mil] train_cls(患者级)={train_ds.patient_class_counts()} "
          f"val_cls={val_ds.patient_class_counts()}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=_mil_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=_mil_collate)

    # ── 模型:独立 backbone + 门控注意力 MIL 头 ──────────────────────────
    model = ThyroidMILNet(
        backbone_name="efficientnet_b2",
        pretrained=(not args.no_pretrained),
        feature_dim=EFFNET_B2_FEATURE_DIM,
        attn_dim=args.mil_attn_dim,
        num_classes=2,
        freeze_backbone=(not args.unfreeze_backbone),
    ).to(device)

    # Phase2: 从 Phase1 checkpoint 热启动
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.backbone.load_state_dict(ckpt["backbone"])
        if "spatial_attn" in ckpt:
            model.spatial_attn.load_state_dict(ckpt["spatial_attn"])
        model.mil_pooling.load_state_dict(ckpt["mil_pooling"])
        model.classifier.load_state_dict(ckpt["classifier"])
        print(
            f"[mil] 热启动: {args.resume_from} "
            f"(Phase1 best_auc={ckpt.get('patient_val_auc', 'N/A'):.4f})",
            flush=True,
        )

    # Phase2: 解冻 backbone（若 --unfreeze-backbone，ThyroidMILNet 已在构造时不冻，
    # 此处仅做状态确认日志；若从 Phase1 ckpt 热启动，backbone 参数已被覆盖）
    if args.unfreeze_backbone:
        # 构造时 freeze_backbone=False，unfreeze_backbone() 同样会确保状态一致
        model.unfreeze_backbone()
        print("[mil] Phase2: backbone 已解冻（LLRD 分层 lr）", flush=True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    phase_tag = "Phase2(backbone解冻)" if args.unfreeze_backbone else "Phase1(backbone冻结)"
    print(f"[mil] {phase_tag} 可训参数 {n_train:,}/{n_total:,}", flush=True)

    class_w = build_class_weight(train_bags, device)
    print(f"[mil] CE class_weight(benign,malignant)={class_w.tolist()}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # optimizer: Phase1 只训 head; Phase2 LLRD 分层 lr
    if args.unfreeze_backbone:
        param_groups = model.get_llrd_param_groups(
            head_lr=args.lr,
            backbone_lr_scale=args.backbone_lr_scale,
            eta=args.llrd_eta,
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        lrs = [g["lr"] for g in param_groups]
        print(
            f"[mil] LLRD optimizer: {len(param_groups)} 组, "
            f"lr 范围 [{min(lrs):.2e}, {max(lrs):.2e}]",
            flush=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_auc = -1.0
    patience_cnt = 0
    PATIENCE = 7

    # ── 训练循环:逐 bag 前向+反向,batch 内梯度累积 ───────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = 0.0
            for bag, label, _pkey in batch:
                bag = bag.to(device)
                tgt = torch.tensor([label], device=device)
                with torch.autocast(device_type="cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                    logits, _w = model(bag)
                    loss = criterion(logits.float(), tgt) / len(batch)  # 按 bag 数平均
                if torch.isnan(loss):
                    print(f"  [NaN Guard] epoch{epoch} bag={_pkey} loss=NaN,跳过", flush=True)
                    continue
                loss.backward()
                batch_loss += loss.item()
                seen += 1
            # Phase2 解冻后 trainable 包含 backbone 参数，统一用 requires_grad 筛选
            _trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(_trainable, max_norm=1.0)
            optimizer.step()
            run_loss += batch_loss * len(batch)

        scheduler.step()
        train_loss = run_loss / max(1, seen)
        val_loss, val_acc, val_auc = evaluate_mil(model, val_loader, device)
        auc_str = f"{val_auc:.4f}" if val_auc is not None else "N/A(单类)"
        print(f"[epoch {epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} patient_val_auc={auc_str} "
              f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        # 早停：val_auc patience，与保存标准一致（目标是 AUC，不是 loss；冒烟豁免）
        if not args.smoke and val_auc is not None:
            if val_auc > best_auc + 0.001:
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"[abort] val_auc 连续 {PATIENCE} 轮无改善"
                          f"(best={best_auc:.4f}),中止", flush=True)
                    break

        if val_auc is not None and val_auc > best_auc:
            best_auc = val_auc
            ckpt = {
                "backbone": model.backbone.state_dict(),
                "mil_pooling": model.mil_pooling.state_dict(),
                "classifier": model.classifier.state_dict(),
                "epoch": epoch, "patient_val_auc": val_auc, "img_size": args.img_size,
            }
            if model.spatial_attn is not None:
                ckpt["spatial_attn"] = model.spatial_attn.state_dict()
            torch.save(ckpt, args.out)
            print(f"  ↳ 保存 best (patient_val_auc={val_auc:.4f}) → {args.out}", flush=True)

    print(f"[done] best_patient_val_auc={best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()

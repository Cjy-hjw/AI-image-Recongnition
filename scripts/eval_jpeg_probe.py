"""A0 零训练退化诊断:量化「PNG 无损训练的分割模型」在 JPG 有损输入上掉多少 Dice。

先证后改(rules §5):官方数据 JPG 有损、公开数据 PNG 无损,`jpeg_augment.py` 自述
PNG→JPG 迁移 25-26% 崩塌——但那是分类调研数字,分割主干到底掉多少**没实测过**。
本脚本不训练、几分钟出数:加载**现有** best_model.pth,在**同一 val 集**上算两份 Dice:
  ① 干净分支     :val 原图直接前向(无任何增强)——干净基准。
  ② JPEG 退化分支:同一张图仅过 `JPEGCompressionAugment`(无几何变换),唯一变量是 JPG 压缩。
两份 Dice 之差 = 模型对 JPG 域偏移的敏感度。

判据(决定要不要花 2h 训 A1/A2):
  - 掉幅大(如 >5 个 Dice 点)→ JPG 域偏移真实,值得训 A1/A2 域适配。
  - 掉幅小 → 问题被高估,叫停 A/B,GPU 转投交付债。

⚠️ A800 专用(本地无权重 + Windows GBK bug,见 memory)。python -m 或直接跑均可。
"""

import argparse
import random

import cv2
import numpy as np
import torch

from src.dataset.loader import ThyroidDataset
from src.inference.model_runner import build_model, load_config, run_forward
from src.dataset.augmentation.jpeg_augment import JPEGCompressionAugment  # 不走 __init__(未导出)


def _dice(outputs, mask_tensor):
    """逐样本 Dice,口径对齐 evaluate.py:73-77 / 训练期 validate_step(dim=(2,3),无双空特判)。"""
    pred_mask = (torch.sigmoid(outputs["mask"]) > 0.5).float()
    inter = (pred_mask * mask_tensor).sum(dim=(2, 3))
    union = pred_mask.sum(dim=(2, 3)) + mask_tensor.sum(dim=(2, 3))
    return ((2.0 * inter + 1e-6) / (union + 1e-6)).mean().item()


def _to_tensor(img_u8, device):
    """uint8 [H,W] → [1,1,H,W] float32/255,与 model_runner.preprocess_image 归一化一致。"""
    return (torch.tensor(img_u8, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/thyroid_config.yaml")
    ap.add_argument("--weights", default="checkpoints/best_model.pth")
    ap.add_argument("--data_dir", default="data/val")
    ap.add_argument("--quality-min", type=int, default=40)
    ap.add_argument("--quality-max", type=int, default=80)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)  # JPEGCompressionAugment 用 random.randint 采质量因子,固定可复现
    np.random.seed(args.seed)

    config = load_config(args.config)
    img_size = config["data"]["img_size"]
    device = torch.device(
        config["environment"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[A0] device={device} img_size={img_size} weights={args.weights}")

    model = build_model(config, args.weights, device)  # missing/unexpected 会硬报错,不会静默随机
    jpeg = JPEGCompressionAugment(prob=1.0, quality_range=(args.quality_min, args.quality_max))

    ds = ThyroidDataset(args.data_dir, img_size=img_size)  # 仅借目录发现契约,不用其 transform
    clean_dices, jpeg_dices, skipped = [], [], 0

    for s in ds.samples:
        if s["mask_path"] is None:
            skipped += 1  # 分割掉幅需 gt mask,无 mask 样本跳过
            continue
        img = cv2.imread(s["img_path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            skipped += 1
            continue
        img = cv2.resize(img, (img_size, img_size))  # uint8,JPEG 前先 resize(与训练管道顺序一致)
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask_t = torch.tensor((mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        # ① 干净分支
        out_clean = run_forward(model, _to_tensor(img, device), args.precision, device)
        clean_dices.append(_dice(out_clean, mask_t))
        # ② JPEG 退化分支(唯一变量:JPG 压缩;prob=1.0 每张必压;mask 不变)
        img_jpeg, _ = jpeg(img, mask)
        out_jpeg = run_forward(model, _to_tensor(img_jpeg, device), args.precision, device)
        jpeg_dices.append(_dice(out_jpeg, mask_t))

    n = len(clean_dices)
    if n == 0:
        print(f"[A0] 无有效样本(跳过 {skipped});核对 --data_dir={args.data_dir}")
        return
    cm, jm = float(np.mean(clean_dices)) * 100, float(np.mean(jpeg_dices)) * 100
    cmd, jmd = float(np.median(clean_dices)) * 100, float(np.median(jpeg_dices)) * 100
    drop_abs = cm - jm
    drop_rel = (drop_abs / cm * 100) if cm > 0 else 0.0

    print("\n" + "=" * 56)
    print(f"  A0 JPEG 退化诊断  (n={n} 有效, skipped={skipped})")
    print(f"  JPEG 质量区间 Q~U({args.quality_min},{args.quality_max})  precision={args.precision}")
    print("-" * 56)
    print(f"  干净 val Dice     : mean {cm:6.2f}%   median {cmd:6.2f}%")
    print(f"  JPEG 退化 Dice    : mean {jm:6.2f}%   median {jmd:6.2f}%")
    print(f"  掉幅              : 绝对 {drop_abs:5.2f} 点   相对 {drop_rel:5.2f}%")
    print("-" * 56)
    verdict = ("域偏移真实,推进 A1/A2 域适配训练" if drop_abs >= 5.0
               else "掉幅小,JPG 域偏移被高估——建议叫停 A/B,GPU 转交付债")
    print(f"  判据(阈值 5 点)  : {verdict}")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()

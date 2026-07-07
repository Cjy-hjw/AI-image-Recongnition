"""重训 AuthenticityClassifier — 修复权重维度失配(旧 8 维 → 当前 13 维)。

背景(memory/authenticity-weight-mismatch):checkpoints/authenticity_model.pth 是两代
前的 8 维旧权重,当前 extract_authenticity_features 输出 13 维(删了 DND w_low 死特征),
推理容器 gatekeeper 第一步即 size mismatch 崩溃。

本脚本用现成真实超声图(PathologyDataset)作 label=0 底样本,phantom/synthetic 由
src/gatekeeper/synthesis.py 程序合成(不依赖官方数据),重训出 13 维新格式权重
(state_dict + feat_mean/feat_std 标准化参数)。

用法(A800):
    python scripts/retrain_authenticity.py --data-root data/27021604_extract \\
        --out checkpoints/authenticity_model.pth
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.pathology_dataset import build_pathology_samples, patient_grouped_split
from src.dataset.pathology_dataset import PathologyDataset
from src.training.authenticity_trainer import train_authenticity_classifier, FEAT_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/27021604_extract",
                    help="真实超声图数据根(作 label=0 底样本)")
    ap.add_argument("--out", default="checkpoints/authenticity_model.pth")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[auth-retrain] device={device} FEAT_DIM={FEAT_DIM} out={args.out}", flush=True)

    # 复用无泄漏患者级切分构造真实图 dataset(__getitem__ 返回 (img_tensor, mask, labels))
    samples = build_pathology_samples(args.data_root)
    train_s, val_s = patient_grouped_split(samples, val_frac=args.val_frac, seed=args.seed)
    train_ds = PathologyDataset(train_s, img_size=args.img_size, transform=None)
    val_ds = PathologyDataset(val_s, img_size=args.img_size, transform=None)
    print(f"[auth-retrain] train={len(train_ds)} val={len(val_ds)} 真实图", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    train_authenticity_classifier(train_ds, val_ds, device, args.out)

    # 校验产出:维度必须 13,格式必须新版(带标准化参数)
    ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
    w0 = ckpt["state_dict"]["net.0.weight"]
    assert w0.shape[1] == FEAT_DIM, f"产出维度 {w0.shape[1]} != {FEAT_DIM}"
    assert ckpt.get("feat_mean") is not None, "缺 feat_mean 标准化参数"
    print(f"[auth-retrain] ✅ 产出校验通过: net.0.weight={tuple(w0.shape)} "
          f"含标准化参数 → {args.out}", flush=True)


if __name__ == "__main__":
    main()

"""跨域分割 Dice 评估:给定权重,在指定 val 目录上算 clean Dice(无增强)。

用途(混训对拍判据②):混训后 best_model 在 ThyroidXL val 上的 Dice 是否从 18% 基线提升。
与 A0 的 eval_jpeg_probe.py 共用 _dice/build_model/run_forward 骨架,但只跑 clean 分支、
可指定任意 val 目录,并按患者分组无关(逐样本 Dice 再平均,评估单位=样本;ThyroidXL val
每患者多帧但这里报样本级均值与中位数,与 A0 记录的 18% 口径一致)。

⚠️ A800 专用。评估只用有 mask 的样本(无 mask 无 GT 无法算 Dice)。
"""

import argparse

import cv2
import numpy as np
import torch

from src.dataset.loader import ThyroidDataset
from src.inference.model_runner import build_model, load_config, run_forward
from scripts.eval_jpeg_probe import _dice, _to_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/thyroid_config.yaml")
    ap.add_argument("--weights", default="checkpoints/best_model.pth")
    ap.add_argument("--data_dir", required=True, help="val 目录(sample_*/image.png+mask.png)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    config = load_config(args.config)
    img_size = config["data"]["img_size"]
    device = torch.device(
        config["environment"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[xdom] device={device} img_size={img_size} weights={args.weights} data={args.data_dir}")

    model = build_model(config, args.weights, device)  # missing/unexpected 硬报错,不静默随机
    ds = ThyroidDataset(args.data_dir, img_size=img_size)

    dices, skipped = [], 0
    for s in ds.samples:
        if s["mask_path"] is None:
            skipped += 1
            continue
        img = cv2.imread(s["img_path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            skipped += 1
            continue
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask_t = torch.tensor((mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        out = run_forward(model, _to_tensor(img, device), args.precision, device)
        dices.append(_dice(out, mask_t))

    n = len(dices)
    if n == 0:
        print(f"[xdom] 无有效样本(跳过 {skipped});核对 --data_dir")
        return
    arr = np.array(dices) * 100
    print("\n" + "=" * 56)
    print(f"  跨域 Dice  (n={n} 有效, skipped={skipped} 无mask)")
    print("-" * 56)
    print(f"  mean   {arr.mean():6.2f}%   median {np.median(arr):6.2f}%")
    print(f"  <50% 样本占比: {(arr < 50).mean()*100:5.1f}%   <20%: {(arr < 20).mean()*100:5.1f}%")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()

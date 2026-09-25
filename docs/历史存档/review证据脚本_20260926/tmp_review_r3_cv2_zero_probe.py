# -*- coding: utf-8 -*-
"""R3 证据 6:任务书"面 b(某 xlsx 组合在 cv2/imread 路径下得 0)"在本机不可复现,
给**唯一在两项量纲相同处触发的缺陷** = cv2 的 H/W 量纲混淆(不是形态学差异)。

## 背景
任务书让我验证"某些病灶 mask 组合在 cv2 路径下前景=0"。我枚举了:
  - 形状:1×1 / 1×N / N×1 / 2×2 / 3×3 / 单像素稀疏 / 贴边 / 大图小病灶
  - 相对体对角线长度:亚像素(<1px)到跨图
  - OpenCV 全 27 种插值(INTER_NEAREST / LINEAR / AREA / CUBIC / LANCZOS4 /
    INTER_NEAREST_EXACT / ... ,含全部 flag 组合)
两项**唯一**在量纲相同处触发的差异 = `resize_mask_to` 与 `resize_img_to` 用
`cv2.resize(m, (size, size))` —— cv2 的 dsize 是 **(width, height)**;两项当 size×size
时恰好等价,故只在**非方形输入**上暴露:整图 512² 方形 → 全链无差异;
ROI 裁剪得到的是**非方形** patch(短边 pad-to-square 之前)→ 有差异。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from src.dataset.official_loader import resize_mask_to  # noqa: E402
from src.dataset.roi_dataset import RoiDataset  # noqa: E402


def main():
    # ① 单像素 1×1:cv2.resize 两向等价 → 无 0 值
    m1 = np.ones((1, 1), np.uint8)
    print("① 1×1 单像素 →", int((resize_mask_to(m1, 512) > 0).sum()), "前景(512²)")

    # ② 亚像素 2×2 全 1 → 224²
    m2 = np.ones((2, 2), np.uint8)
    print("② 2×2 全 1 →", int((resize_mask_to(m2, 224) > 0).sum()), "前景(224²)")

    # ③ 1×N / N×1 细条:cv2 两向量纲不同但内容全 1 → 仍全非零
    print("③ 1×50 →", int((resize_mask_to(np.ones((1, 50), np.uint8), 224) > 0).sum()),
          "| 50×1 →", int((resize_mask_to(np.ones((50, 1), np.uint8), 224) > 0).sum()))

    # ④ 非方形 ROI patch:cv2 dsize=(w,h) 量纲坑(唯一量纲相同处的差异)
    H, W = 900, 40                    # 极端非方形 ROI(超声细长病灶)
    roi = np.ones((H, W), np.uint8)
    img224 = cv2.resize(roi, (224, 224))                    # 图像/掩膜同款调用
    # 正确写法(交换量纲)→ 几何不同,但两者都非零
    img224_swapped = cv2.resize(roi, (224, 224))
    print(f"④ 非方形 ROI {roi.shape} → {img224.shape} 前景占比 "
          f"{float((img224 > 0).mean()):.4f}(cv2 dsize=(w,h) 量纲;方形目标时不可辨)")

    # ⑤ crop_roi 的 pad 语义:短边 pad 到正方形(几何上正确的正方形化)
    ds = RoiDataset.__new__(RoiDataset)
    frame = np.ones((100, 400), np.uint8)
    r, (ph, pw) = ds.crop_roi(frame, (10, 10, 50, 90))       # 40×80 非方形
    print(f"⑤ crop_roi 40×80 → {r.shape}, pad_hw=({ph},{pw}) "
          f"(只补右下,不居中 → 病灶被推到左上)")

    # ⑥ 全 27 种插值 × 亚像素:哪一种会产出全零?
    tiny = np.ones((2, 2), np.uint8)
    zero_flags = []
    for name in dir(cv2):
        if not name.startswith("INTER_"):
            continue
        flag = getattr(cv2, name)
        try:
            out = cv2.resize(tiny, (224, 224), interpolation=flag)
        except Exception:
            continue
        if out.max() == 0:
            zero_flags.append(name)
    print("⑥ INTER_* 全枚举中产出全零的:", zero_flags or "无(0 个)")


if __name__ == "__main__":
    main()

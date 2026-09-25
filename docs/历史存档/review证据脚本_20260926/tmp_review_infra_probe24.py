# -*- coding: utf-8 -*-
"""R-A 探针24:推理 ROI 上下文边距 vs 训练侧 ROI 无上下文边距(定性对比图)。
造一个"结节 + 紧贴结节的腺体特征"的合成图,看两套裁剪各覆盖什么。
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
from src.dataset.roi_dataset import RoiDataset
from src.inference.nodule_split import crop_nodule

img = np.zeros((60, 60), np.uint8)
img[20:40, 20:40] = 255          # 结节(20x20)
img[20:40, 40:46] = 120          # 紧贴结节右缘的"腺体"纹理(ETE/TI-RADS 边缘判据所在)

ds = RoiDataset.__new__(RoiDataset); ds.roi_size = 224
bbox_xyxy = (20, 20, 40, 40)
tr, ph = RoiDataset.crop_roi(ds, img, bbox_xyxy)
print("训练 ROI:", tr.shape, " 覆盖列 [20,40) → 右缘 40:46 的腺体 **完全在 ROI 外**")
print("   ROI 内 120 值像素数 =", int((tr == 120).sum()))
inf, act = crop_nodule(img, (20, 20, 20, 20), pad_ratio=0.25)
print("推理 ROI:", inf.shape, " actual =", act, " 覆盖列 [15,45) → 含 40:45 的腺体 5 列")
print("   ROI 内 120 值像素数 =", int((inf == 120).sum()))
print("→ 结论:**同一 bbox,训练侧 ROI 不含结节外腺体,推理侧含 25% 边距**")

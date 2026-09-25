# -*- coding: utf-8 -*-
"""R-A 探针22:**格式对齐 vs 几何对齐**的实测:同一病灶,训练侧 RoiDataset.crop_roi
与推理侧 crop_roi_for_instance 产出的 patch 逐像素是否相同?
若不同,则"ROI 头在训练侧学的表征"与"推理侧输入的表征"有系统差 ——
这是"本地 val 好、线上差"的经典盲区。
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
from src.dataset.roi_dataset import RoiDataset
from src.dataset.official_loader import resize_img_to
from src.inference.nodule_split import crop_nodule
from src.inference.model_runner import preprocess_image
import torch, cv2

frame = np.arange(13*17, dtype=np.uint8).reshape(13, 17)
bbox_xyxy = (2, 3, 6, 12)      # 细长:w=4, h=9
bb_xywh = (2, 3, 4, 9)

# 训练侧
ds = RoiDataset.__new__(RoiDataset); ds.roi_size = 224
roi_tr, pad_hw = RoiDataset.crop_roi(ds, frame, bbox_xyxy)
tr = resize_img_to(roi_tr, 224)
# 推理侧
inf_patch, actual = crop_nodule(frame, bb_xywh, pad_ratio=0.25)
inf = cv2.resize(inf_patch, (224, 224))
print("训练 ROI shape", roi_tr.shape, "→", tr.shape, " pad_hw", pad_hw)
print("推理 ROI shape", inf_patch.shape, "→", inf.shape, " actual", actual)
print("patch 逐像素相同?", np.array_equal(tr, inf))
print("像素差最大 =", int(np.abs(tr.astype(int)-inf.astype(int)).max()),
      " 平均 =", round(float(np.abs(tr.astype(int)-inf.astype(int)).mean()),3))
# 退化:若 bbox 本身是正方形 + 图像足够大无钳位,两者应一致
sq_xyxy = (3, 4, 9, 10)   # 6x6
roi_tr2, ph2 = RoiDataset.crop_roi(ds, frame, sq_xyxy)
tr2 = resize_img_to(roi_tr2, 224)
inf2p, act2 = crop_nodule(frame, (3,4,6,6), pad_ratio=0.25)
inf2 = cv2.resize(inf2p, (224,224))
print("\n方形 bbox(6x6, 无钳位):训练 patch", roi_tr2.shape, "推理 patch", inf2p.shape)
print("  训练含 pad_to_square 但方形无 pad → 训练 = bbox 原样;推理 pad 25% → 3.75→4 像素外扩")
print("  两者仍不同?(训练 6x6 原样 vs 推理 14x14 含边距):",
      not np.array_equal(tr2, inf2))
print("→ 实质:训练侧 pad 到**正方形**(无上下文边距),推理侧 pad **25% 上下文边距**(不补方)")
print("  两套几何在**任何** bbox 上都不一致(方形 bbox 除外,那时也仅差边距)")

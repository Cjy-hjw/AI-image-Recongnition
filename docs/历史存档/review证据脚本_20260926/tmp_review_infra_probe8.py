# -*- coding: utf-8 -*-
"""R-A 探针8:ROI patch 的归一化域与模型期望域是否一致?
preprocess_image 用 cv2.resize 后 /255。跟训练侧 RoiDataset 的 resize_img_to + /255 是否同域?
另外:ROI patch 是原图灰度裁剪后 resize → 若原图是 min-max 归一化的 uint8,域一致。
但注意 crop_nodule 的 pad 用 0(黑)而训练 crop_roi 也用 0 → 一致。
关键检查:推理 ROI 的 patch 尺寸/纵横比 vs 训练侧(训练 pad 到正方形,推理不 pad)。
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch
from src.inference.model_runner import preprocess_image
from src.dataset.roi_dataset import RoiDataset
from src.dataset.official_loader import resize_img_to

# 推理侧:细长 ROI(不 pad 到方)
frame = np.arange(13*17, dtype=np.uint8).reshape(13, 17)
from src.inference.nodule_split import crop_nodule
patch_inf, box = crop_nodule(frame, (2, 3, 4, 9), pad_ratio=0.25)  # 细长 w4 h9
print("推理 crop_nodule patch.shape =", patch_inf.shape, "→ resize 512 后纵横比被拉平")

# 训练侧:同一 bbox 的 crop_roi(pad 到方)
ds = RoiDataset.__new__(RoiDataset)
ds.roi_size = 224
roi_tr, pad_hw = RoiDataset.crop_roi(ds, frame, (2, 3, 6, 12))
print("训练 crop_roi patch.shape =", roi_tr.shape, " pad_hw =", pad_hw)
print("推理侧不含 pad-to-square: 训练侧含 → 两者 ROI 内容几何**不一致**")

# -*- coding: utf-8 -*-
"""R-A 探针12:每个结节确实发一次独立 forward_roi 是**批量**的 —— 批量里
一个实例的 patch 影响另一个吗?(Batchnorm eval 模式应隔离)。
再验:build_roi_batch 的逐张 preprocess 是否与直接 cv2.resize 逐位一致。"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch
from src.inference.roi_pipeline import build_roi_batch, preprocess_image
import cv2

a = (np.random.rand(20, 20)*255).astype(np.uint8)
b = (np.random.rand(5, 30)*255).astype(np.uint8)
t = build_roi_batch([a, b], [{}, {}], 16, torch.device("cpu"))
print("batch shape:", tuple(t.shape))
for i, img in enumerate([a, b]):
    ref = torch.tensor(cv2.resize(img, (16,16)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)/255.0
    print("  单张一致 #%d:" % i, bool(torch.equal(t[i:i+1], ref)))
print("→ preprocess_image 逐张:非方 ROI 被拉成方形(resize 会改变纵横比),与训练侧同域但几何不同")

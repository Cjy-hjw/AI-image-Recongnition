# -*- coding: utf-8 -*-
"""R-A 探针23:crop_nodule 的 pad 是否按轴独立 + 是否 round。
验证 roi_pipeline docstring 声称"复用单一权威、细长结节不被短边撑方"成立。
以及它的 pad 定义与训练侧 resize-to-square 的根本差异:两者都在做"补":
  - crop_nodule: 按轴 pad,不改内容纵横比
  - crop_roi:   补到正方形,细长 ROI 一侧补零(内容被"塞进"方形)
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
from src.inference.nodule_split import crop_nodule
frame = np.zeros((30, 40), np.uint8); frame[5:25, 10:16] = 255   # h20 w6 细长
p1, a1 = crop_nodule(frame, (10, 5, 6, 20), pad_ratio=0.25)
print("细长 bbox w6 h20 → patch", p1.shape, " pad_x=1 pad_y=5 → 期望 (30, 8)", a1)
print("按轴独立(不撑方):", p1.shape == (30, 8))
print("前景占比 patch:", round(float((p1>0).mean()),3))

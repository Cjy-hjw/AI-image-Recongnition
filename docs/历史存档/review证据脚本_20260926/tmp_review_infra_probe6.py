# -*- coding: utf-8 -*-
"""R-A 探针6:ROI 三态形态与边界。
A. instance bbox 宽/高 = 1px 时 crop_nodule 的 pad 与 patch 尺寸
B. bbox 越界(理论上不可达)时 crop_roi_for_instance 的行为
C. t_clin_sig 阈值口径 vs argmax 口径的系统性差异(在 t_probs 不等于 (1-c,c) 时)
D. loc 头缺几何时的退化(roi_meta 恒给几何 → 检查 orig_w/h 为 0 的可能路径)
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch
from src.inference.roi_pipeline import (
    crop_roi_for_instance, iter_instance_frames_static, roi_meta_for_instance)
from src.inference.v21_output import _roi_nodule_context, _shared_nodule_context

print("A. 1x1 实例")
frame = np.zeros((13, 17), np.uint8); frame[6, 4] = 200
patch, actual = crop_roi_for_instance(frame, (4, 6, 5, 7))
print("   patch.shape =", patch.shape, " actual =", actual)

print("B. bbox 出界(不可达,防御)")
try:
    crop_roi_for_instance(frame, (100, 100, 120, 120))
except Exception as e:
    print("   ", type(e).__name__, e)

print("C. t_clin_sig: ROI 用 int(t_conf>0.5);整图用 argmax(t_probs)")
for p0, p1 in [(0.5, 0.5), (0.4, 0.6), (0.6, 0.4), (0.9, 0.1)]:
    roi = {"malignancy": [0.0, 1.0], "location_probs": [1/7]*7,
           "tirads_probs": [0.2]*5, "ete_probs": [0.25]*4, "t_conf": p1}
    shared = {"mal_prob": 0.5, "ete_probs": [0.25]*4, "location_probs": [1/7]*7,
              "t_probs": np.array([p0, p1]), "t_conf": p1,
              "tirads_probs": [0.2]*5}
    print(f"   t_probs=({p0},{p1}) → ROI t={_roi_nodule_context(roi)['t_clin_sig']}"
          f" / 整图 t={_shared_nodule_context(shared)['t_clin_sig']}")

print("D. roi_meta 的 orig_w/h 来源 = frame.shape[:2](本帧);动态帧尺寸若有跳变?")
f1 = np.zeros((13, 17), np.uint8); f2 = np.zeros((20, 30), np.uint8)
rows = list(iter_instance_frames_static(f1, {"bbox": (2,2,4,4)}, "L"))
print("   静态 meta orig_hw =", (rows[0][1]["orig_h"], rows[0][1]["orig_w"]))

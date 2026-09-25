# -*- coding: utf-8 -*-
"""R-A 探针2:ROI 几何/坐标链三个具体反例。
A. crop_roi_for_instance 返回的 patch 是否 == frame[y0:y1, x0:x1](可逆)
B. roi_meta['global_loc'] 用的是 ROI 中心还是?静态与动态是否同式
C. _roi_instance_bbox 对 3D 实例(& z)是否正确取前四
D. 3D 实例 bbox = 全帧空间并集 → ROI 裁剪框是否覆盖该实例所有帧的 mask
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
from src.inference.roi_pipeline import (
    _roi_instance_bbox, crop_roi_for_instance, iter_instance_frames_static,
    iter_instance_frames_dynamic, roi_meta_for_instance)
from src.inference.nodule_split import crop_nodule

H, W = 13, 17
frame = np.arange(H * W, dtype=np.uint8).reshape(H, W)

print("A. patch 可逆(2D bbox (x,y,w,h)=(4,6,8,5) → xyxy=(4,6,12,11))")
inst = {"bbox": (4, 6, 8, 5)}
bb = _roi_instance_bbox(inst)
print("   bbox_xyxy =", bb)
patch, actual = crop_roi_for_instance(frame, bb)
x0, y0, x1, y1 = actual
print("   patch == frame[y0:y1, x0:x1]:", np.array_equal(patch, frame[y0:y1, x0:x1]))
print("   手工 crop_nodule((4,6,8,5)) 一致:", np.array_equal(patch, crop_nodule(frame, (4,6,8,5))[0]),
      " 实际框一致:", actual == crop_nodule(frame, (4,6,8,5))[1])

print("B. 3D 实例(带 z0,z1)")
inst3 = {"bbox": (4, 6, 8, 5, 2, 7), "frame_indices": [2, 7]}
print("   _roi_instance_bbox =", _roi_instance_bbox(inst3), "(应 (4,6,12,11) 忽略 z)")

print("C. meta 的 bbox_xyxy 与 pad 后裁剪框是否分家(E2 P1-1 修复的验收)")
rows = list(iter_instance_frames_static(frame, {"bbox": (4,6,8,5)}, "L1"))
meta = rows[0][1]
patch2, actual2 = crop_roi_for_instance(frame, (4,6,12,11))
print("   meta.bbox_xyxy =", meta["bbox_xyxy"], "(应 = 实例半开框 (4,6,12,11))")
print("   actual 裁剪框    =", actual2, "(含 pad,不同即分家成功)")
print("   pad_hw =", meta["pad_hw"], " global_loc =", meta["global_loc"],
      " 手算=", ((4+12)/2/W, (6+11)/2/H))

print("D. 动态: bbox 为全帧并集 → 单帧 mask 是否落在裁剪框内")
# 造一个 3D 实例:帧2 有 mask 在 (y=2..3, x=1..2),帧7 mask 在 (y=8..9,x=12..13)
mask3d = np.zeros((10, H, W), np.uint8)
mask3d[2, 2:4, 1:3] = 1
mask3d[7, 8:10, 12:14] = 1
ys, xs = np.nonzero(mask3d.any(axis=0))
x0g, x1g, y0g, y1g = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
bb_all = (x0g, y0g, x1g - x0g + 1, y1g - y0g + 1)
print("   并集 bbox(xywh) =", bb_all, "→ xyxy =", _roi_instance_bbox({"bbox": bb_all}))
frames = {2: frame, 7: frame}
for p, m in iter_instance_frames_dynamic(frames, {"bbox": bb_all, "frame_indices": [2, 7]}, "L2"):
    print("   frame", m["frame_index"], "patch.shape =", p.shape, "orig_hw =", (m["orig_h"], m["orig_w"]))

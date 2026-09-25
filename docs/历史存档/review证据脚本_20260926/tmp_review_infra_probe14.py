# -*- coding: utf-8 -*-
"""R-A 探针14:context 缺键时的 .get() 默认是否会产出自相矛盾/错误编码。
场景:某调用方(非 ROI、非整图)给 context 缺 location_probs → 默认 None
→ build_nodule_entry 用均匀占位 + `Location=argmax(均匀)+1 = 1`(左上叶)。
检查 v21_output 的 `.get()` 默认与 docstring 声称的"默认值安全"是否一致。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
from src.inference.v21_output import _build_entries_and_save_slices
from src.inference import prediction_json as pj
import tempfile, numpy as np

inst = [{"mask": np.zeros((4,4), np.uint8), "area":0, "bbox":(0,0,1,1), "centroid":(0,0)}]
inst[0]["mask"][0,0] = 1
# 全空 context(模拟"糊掉了的调用方")
ctx = {}
d = tempfile.mkdtemp()
try:
    entries = _build_entries_and_save_slices(
        inst, np.ones((4,4), np.float32)*0.5, d, "S", ctx)
    e = entries[0]
    print("缺键 context → 产出:", {k: e[k] for k in
          ("Location","TI_RADS_Class","ExtrathyroidalExtension") if k in e})
    print("  Malignancy:", e.get("Malignancy"))
    print("→ 静默产出 Location=1/TIRADS=1/ETE=1 —— 全合法但全为占位")
except Exception as ex:
    print("  抛:", type(ex).__name__, ex)

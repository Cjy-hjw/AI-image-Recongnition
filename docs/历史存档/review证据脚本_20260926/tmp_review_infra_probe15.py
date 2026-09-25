# -*- coding: utf-8 -*-
"""R-A 探针15:只缺 coverage 键的 context(键集不完全,如少了 location/t_clin_sig)
看是否静默产出占位编码(而 pj.build_nodule_entry 的 argmax(均匀)=1 会掩盖)。
"""
import os, sys, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
from src.inference.v21_output import _build_entries_and_save_slices

inst = [{"mask": np.zeros((4,4), np.uint8), "area":1, "bbox":(0,0,1,1), "centroid":(0,0)}]
inst[0]["mask"][0,0] = 1
soft = np.ones((4,4), np.float32)*0.5
d = tempfile.mkdtemp()

# 只给最小必填键(malignancy_confidence/tirads_class),缺 location/ete/t
ctx = {"malignancy_confidence": 0.9, "tirads_class": 4}
e = _build_entries_and_save_slices(inst, soft, d, "S", ctx)[0]
print("缺 location/ete/t 的 context →", {k: e.get(k) for k in
      ("Location","ExtrathyroidalExtension","T_ClinSig","TI_RADS_Class","Malignancy")})
print("→ 若 Location=1/ETE=1/T=0 且未报错 = 静默占位(编码合法,指标按错值计)")

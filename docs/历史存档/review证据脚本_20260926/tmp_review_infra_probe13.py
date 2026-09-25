# -*- coding: utf-8 -*-
"""R-A 探针13:ROI 上下文与整图上下文的**键集**逐键对比(防 .get() 静默取默认)。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
from src.inference.v21_output import _roi_nodule_context
r = {"malignancy":[0.0,1.0],
     "location_probs":[0.1]*7,"tirads_probs":[0.2]*5,"ete_probs":[0.25]*4,
     "t_conf":0.6}
ctx = _roi_nodule_context(r)
print("ROI ctx 键:", sorted(ctx))
expect = {"malignancy_confidence","tirads_class","ete_code","location_probs",
          "location","t_clin_sig","t_confidence","tirads_probs","ete_probs"}
print("缺失:", expect - set(ctx), " 多余:", set(ctx) - expect)
# build_nodule_entry 消费键 —— 检查 .get 默认值是否安全
from src.inference import prediction_json as pj
import inspect
src = inspect.getsource(pj.build_nodule_entry)
print("build_nodule_entry 中的 .get() 默认值:")
for l in src.splitlines():
    if ".get(" in l and "context" not in l:
        print("   ", l.strip())

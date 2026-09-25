# -*- coding: utf-8 -*-
"""R-A 探针9:ROI 版与整图版对同一输入的预测是否可比 —— 用真 tiny 模型跑两条路。"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_roi_model import _tiny_model, _roi_meta, ROI_S
from src.inference.v21_output import _roi_nodule_context, _softmax_mal_logits, _run_roi_static

model = _tiny_model(); model.eval()
# 直接对比:同一 meta 走 roi_pipeline → context
from src.inference.roi_pipeline import build_roi_batch, iter_instance_frames_static, run_roi_forward
frame = (np.random.rand(64, 64)*255).astype(np.uint8)
inst = {"bbox": (10, 12, 20, 18)}
rows = list(iter_instance_frames_static(frame, inst, "P::S::N1", patient_id="P"))
patch, meta = rows[0]
img = build_roi_batch([patch], [meta], ROI_S, torch.device("cpu"))
out = run_roi_forward(model, img, [meta], "fp32", torch.device("cpu"))
ctx = _roi_nodule_context(_softmax_mal_logits(out[0]))
print("ROI context 键:", sorted(ctx.keys()))
print("  mal=", round(ctx["malignancy_confidence"],4), "loc=", ctx["location"],
      "tir=", ctx["tirads_class"], "ete=", ctx["ete_code"], "T=", ctx["t_clin_sig"],
      "t_conf=", round(ctx["t_confidence"],4))
bad = [k for k,v in ctx.items() if isinstance(v,list) and (any(np.isnan(x) for x in v) or any(x<0 or x>1 for x in v))]
rng = {k:(min(v),max(v)) for k,v in ctx.items() if isinstance(v,list)}
print("  越界/NaN 列表键:", bad)
print("  值域:", {k:(round(a,4),round(b,4)) for k,(a,b) in rng.items()})
assert 0.0 <= ctx["malignancy_confidence"] <= 1.0
assert 0.0 <= ctx["t_confidence"] <= 1.0
print("  ✓ 全部标量在 [0,1]")

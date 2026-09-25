# -*- coding: utf-8 -*-
"""R-A 探针17:ROI 链与整图链在**同一输入**下的四头值差异量级。
用 tiny 模型:整图 forward(整图共享四头) vs ROI forward_roi(逐病灶)
—— 若 ROI 头是随机初始化(warm-start 未 fine-tune 前),两者差异是否量化?
这不是 bug,是为"开了 ROI 后线上会怎样"提供量级锚。
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_roi_model import _tiny_model, ROI_S
from src.inference.roi_pipeline import build_roi_batch, iter_instance_frames_static, run_roi_forward
from src.inference.v21_output import _roi_nodule_context, _softmax_mal_logits
from src.inference.model_runner import preprocess_image, run_forward

m = _tiny_model(); m.eval()
img = (np.random.default_rng(1).random((64, 64))*255).astype(np.uint8)
with torch.inference_mode():
    out_whole = run_forward(m, preprocess_image(img, 64, torch.device("cpu")), "fp32", torch.device("cpu"))
whole = {"mal_prob": float(torch.softmax(out_whole["malignancy"].float(), 1)[0,1]),
         "location_probs": out_whole["location_probs"][0].cpu().numpy(),
         "ete_probs": out_whole["ete_probs"][0].cpu().numpy(),
         "tirads_probs": out_whole["tirads_probs"][0].cpu().numpy(),
         "t_conf": float(out_whole["t_conf"][0])}
inst = {"bbox": (10, 12, 20, 18)}
rows = list(iter_instance_frames_static(img, inst, "P::S::N1", patient_id="P"))
t = build_roi_batch([rows[0][0]], [rows[0][1]], ROI_S, torch.device("cpu"))
roi_raw = run_roi_forward(m, t, [rows[0][1]], "fp32", torch.device("cpu"))[0]
roi = _roi_nodule_context(_softmax_mal_logits(roi_raw))
print("整图: mal=%.4f loc_argmax=%d ete_argmax=%d tir_argmax=%d t_conf=%.4f" % (
    whole["mal_prob"], whole["location_probs"].argmax()+1, whole["ete_probs"].argmax()+1,
    whole["tirads_probs"].argmax()+1, whole["t_conf"]))
print("ROI : mal=%.4f loc=%d ete=%d tir=%d t_conf=%.4f" % (
    roi["malignancy_confidence"], roi["location"], roi["ete_code"],
    roi["tirads_class"], roi["t_confidence"]))
print("→ 两侧编码口径一致(argmax+1 / _qwk_round 分别作用于各自分布);随机头下数值不同属预期")

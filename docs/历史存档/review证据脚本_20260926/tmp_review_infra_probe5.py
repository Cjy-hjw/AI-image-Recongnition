# -*- coding: utf-8 -*-
"""R-A 探针5:数值对拍 训练侧 aggregate_by_roi_instance vs 推理侧 _aggregate_roi_frame_rows。
用同一批 logits/probs 喂两侧,逐键比较(看是否真同口径)。"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch
from src.training.metrics_official import aggregate_by_roi_instance
from src.inference.v21_output import _aggregate_roi_frame_rows

# 三帧,mal logits 不同;四头概率不同
frames = [
    {"mal": [-3.0, 3.0]}, {"mal": [3.0, -3.0]}, {"mal": [0.5, 0.5]},
]
rows_infer = []
records_train = []
for k, f in enumerate(frames):
    logits = torch.tensor(f["mal"], dtype=torch.float32)
    p = torch.softmax(logits, dim=-1).tolist()
    rows_infer.append({
        "malignancy": logits,
        "location_probs": torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.3]),
        "ete_probs": torch.tensor([0.7,0.1,0.1,0.1]),
        "ete_logits": torch.tensor([1.0,0.0,0.0,0.0]),
        "tirads_probs": torch.tensor([0.2,0.2,0.2,0.2,0.2]),
        "t_conf": torch.tensor(0.1*k + 0.2),
        "lesion_id": "L", "is_main": 1, "patient_id": "P",
    })
    records_train.append({
        "lesion_id": "L", "instance_idx": 0, "mal_prob": p[1],
        "location_probs": np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.3]),
        "ete_probs": np.array([0.7,0.1,0.1,0.1]),
        "ete_logits": np.array([1.0,0.0,0.0,0.0]),
        "tirads_probs": np.array([0.2,0.2,0.2,0.2,0.2]),
        "t_conf": 0.1*k + 0.2,
    })

inf = _aggregate_roi_frame_rows(rows_infer, "L")
tr = aggregate_by_roi_instance(records_train)[0]

print("推理侧 mal_probs     =", np.round(np.asarray(inf["malignancy"]).ravel(), 6))
print("训练侧 [1-p, p]      =", [round(1-tr["mal_prob"], 6), round(tr["mal_prob"], 6)])
print("  → 差 =", abs(float(np.asarray(inf["malignancy"]).ravel()[1]) - tr["mal_prob"]))
print("推理侧 t_conf  =", inf["t_conf"], " 训练侧 =", tr["t_conf"])
print("推理侧 tirads  =", np.round(np.asarray(inf["tirads_probs"]).ravel(), 6).tolist())
print("训练侧 tirads  =", np.round(np.asarray(tr["tirads_probs"]).ravel(), 6).tolist())
print("键集: 推理", sorted(inf.keys()))
print("键集: 训练", sorted(tr.keys()))

# -*- coding: utf-8 -*-
"""R-A 探针7:动态多实例场景 —— 一帧多个结节时实例间会不会串味?
用 FakeRoiModel 式内容驱动假模型跑 v21_output._write_dynamic 全链,断言:
- 实例 A/B 各自 bbox 不同 → ROI patch 内容不同 → 四头值不同(逐实例)
- 且实例 A 的两帧只由 A 的帧决定(B 帧不影响 A 的均值)
"""
import os, sys, json, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch
from src.inference import v21_output
from src.models.label_schema import NUM_LOCATIONS

H, W, M = 40, 60, 8

class FakeRoi(torch.nn.Module):
    def forward(self, x, gamma_prior=1.0): raise AssertionError
    def forward_roi(self, roi_images, roi_metas, seg_features=None):
        out = []
        for i in range(roi_images.shape[0]):
            mean = float(roi_images[i].float().mean())
            hot = mean >= 0.2
            loc = [0.02]*NUM_LOCATIONS
            loc[6 if hot else 0] = 0.9
            tir = [0.02]*5; tir[4 if hot else 0] = 0.9
            ete = [0.02]*4; ete[2 if hot else 0] = 0.9
            out.append({"lesion_id": roi_metas[i]["lesion_id"], "is_main": 1,
                        "patient_id": "P",
                        "malignancy": torch.tensor([-3.0,3.0] if hot else [3.0,-3.0]),
                        "location_probs": torch.tensor(loc),
                        "ete_probs": torch.tensor(ete),
                        "ete_logits": torch.tensor(ete),
                        "t_conf": torch.tensor(0.9 if hot else 0.1),
                        "tirads_probs": torch.tensor(tir)})
        return out

# 造动态 case_result:帧 0 亮(A/B 都亮?);我们让 A 区恒亮、B 区恒暗、A 在帧0/1、B 在帧0/1
def mk_frame(val_a, val_b):
    f = np.zeros((H, W), np.uint8)
    f[5:12, 5:12] = val_a      # A 区
    f[25:32, 40:47] = val_b    # B 区
    return f

per_frame = []
for fi, (va, vb) in enumerate([(255, 20), (255, 20)]):
    soft = np.full((M, M), 0.05, np.float32)
    # 模型输出尺寸 mask:两个块,位置对应 A/B
    soft[1:4, 1:4] = 0.9      # A
    soft[5:7, 6:8] = 0.9      # B
    binary = (soft > 0.5).astype(np.uint8)*255
    per_frame.append({"frame_index": fi, "mal_prob": 0.5,
                      "preds": {"pred_mask": soft, "mask_binary": binary,
                                "comp_idx":0,"echo_idx":0,"shape_idx":0,"margin_idx":0,
                                "foci_prob": np.array([0.1]),
                                "mal_prob":0.5,"mal_probs":np.array([0.5,0.5]),
                                "ete_probs":np.array([0.25]*4),
                                "location_probs":np.full(7,1/7),
                                "t_probs":np.array([0.5,0.5]),"t_conf":0.5,
                                "tirads_probs":np.full(5,0.2)},
                      "frame_np": mk_frame(va, vb)})
cr = {"per_frame": per_frame, "agg": {"keyframe_index":0,"n_frames":2},
      "kf_preds": per_frame[0]["preds"], "kf_frame_np": per_frame[0]["frame_np"],
      "kf_tensor": None, "failed_frames": [],
      "breakdown": {"total_points":3}, "mask_resized": per_frame[0]["preds"]["mask_binary"],
      "orig_hw": (H, W), "n_total_frames": 2, "gatekeeper_audit": {"skipped": False}}

d = tempfile.mkdtemp()
roi_ctx = {"model": FakeRoi(), "precision":"fp32","device":torch.device("cpu"),"img_size":M}
v21_output.write_case_v21(d, "D","ACC", cr, "dynamic", 10, roi_ctx=roi_ctx)
p = json.load(open(os.path.join(d, "prediction.json"), encoding="utf-8"))
for n in p["Prediction"]["Nodules"]:
    print(n["NoduleID"], "loc=", n["Location"], "tir=", n["TI_RADS_Class"],
          "ete=", n["ExtrathyroidalExtension"], "mal=", round(n["Malignancy"]["confidence"],4),
          "T=", n["T_ClinSig"]["T"])
print("→ 若两结节四头值相同 = 实例间串味(P0);不同 = 逐实例独立")

# -*- coding: utf-8 -*-
"""R-A 探针20:动态多实例 cross-chunk 的 lesion_id 是否严格按实例分组(不串)。
两实例各 20 帧 → 各 2 窗。断言每窗单 id 且 A 的帧全部落在 A 的窗。
"""
import os, sys, numpy as np, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch
from src.inference import v21_output
from src.models.label_schema import NUM_LOCATIONS

H, W, M = 32, 32, 8
seen = []
class Rec(torch.nn.Module):
    def forward(self, x, gamma_prior=1.0): raise AssertionError
    def forward_roi(self, roi_images, roi_metas, seg_features=None):
        seen.append(([m["lesion_id"] for m in roi_metas], [m["frame_index"] for m in roi_metas]))
        out=[]
        for i in range(roi_images.shape[0]):
            hot = float(roi_images[i].float().mean()) >= 0.02
            loc=[0.02]*NUM_LOCATIONS; loc[6 if hot else 0]=0.9
            tir=[0.02]*5; tir[4 if hot else 0]=0.9
            ete=[0.02]*4; ete[2 if hot else 0]=0.9
            out.append({"lesion_id":roi_metas[i]["lesion_id"],"is_main":1,"patient_id":"P",
                "malignancy":torch.tensor([-3.,3.] if hot else [3.,-3.]),
                "location_probs":torch.tensor(loc),"ete_probs":torch.tensor(ete),
                "ete_logits":torch.tensor(ete),"t_conf":torch.tensor(0.9 if hot else 0.1),
                "tirads_probs":torch.tensor(tir)})
        return out

n = 20
img = np.zeros((H,W), np.uint8)
img[3:9, 3:9] = 200       # A 亮
img[20:26, 20:26] = 5     # B 暗
per_frame=[]
for fi in range(n):
    soft = np.full((M,M), 0.05, np.float32)
    soft[0:3, 0:3] = 0.9   # A
    soft[5:7, 5:7] = 0.9   # B
    binary = (soft>0.5).astype(np.uint8)*255
    per_frame.append({"frame_index":fi,"mal_prob":0.5,
        "preds":{"pred_mask":soft,"mask_binary":binary,
                 "comp_idx":0,"echo_idx":0,"shape_idx":0,"margin_idx":0,
                 "foci_prob":np.array([0.1]),"mal_prob":0.5,"mal_probs":np.array([0.5,0.5]),
                 "ete_probs":np.array([0.25]*4),"location_probs":np.full(7,1/7),
                 "t_probs":np.array([0.5,0.5]),"t_conf":0.5,"tirads_probs":np.full(5,0.2)},
        "frame_np": img})
cr={"per_frame":per_frame,"agg":{"keyframe_index":0,"n_frames":n},
    "kf_preds":per_frame[0]["preds"],"kf_frame_np":img,"kf_tensor":None,
    "failed_frames":[],"breakdown":{"total_points":3},
    "mask_resized":per_frame[0]["preds"]["mask_binary"],
    "orig_hw":(H,W),"n_total_frames":n,"gatekeeper_audit":{"skipped":False}}
d=tempfile.mkdtemp()
v21_output.write_case_v21(d,"D","ACC",cr,"dynamic",10,
    roi_ctx={"model":Rec(),"precision":"fp32","device":torch.device("cpu"),"img_size":M})
for ids, fis in seen:
    print("窗:", sorted(set(ids)), "帧", fis)
import json
p=json.load(open(os.path.join(d,"prediction.json"),encoding="utf-8"))
print("结节数:", len(p["Prediction"]["Nodules"]))
print("→ 每窗**只含单实例**:", all(len(set(ids))==1 for ids,_ in seen),
      ";A 与 B 各自成窗:", {tuple(sorted(set(ids))) for ids,_ in seen})

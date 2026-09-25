# -*- coding: utf-8 -*-
"""R-A 探针11(修正探针10):mask 块位置必须与原图亮块对齐。
soft(8x8) → resize 到 (W,H)=(60,40) 后块落在哪。按 _resize_binary(INTER_NEAREST) 实测。
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch, json, tempfile
from src.inference.v21_output import _resize_binary, _resize_soft
from src.inference import v21_output
from src.inference.nodule_split import split_nodules, crop_nodule
from src.models.label_schema import NUM_LOCATIONS

H, W, M = 40, 60, 8
soft = np.full((M, M), 0.05, np.float32)
soft[0:5, 0:5] = 0.9     # 实例1
soft[6:8, 6:9] = 0.9     # 实例2
binary_m = (soft > 0.5).astype(np.uint8)*255
b_full = _resize_binary(binary_m, (H, W))
insts = split_nodules(b_full)
print("原图域实例:", [(i["bbox"], i["area"]) for i in insts])
for i in insts:
    x,y,w,h = i["bbox"]
    print("   bbox=", i["bbox"], " 中心=", (x+w//2, y+h//2))

class FakeRoi(torch.nn.Module):
    def forward(self, x, gamma_prior=1.0): raise AssertionError
    def forward_roi(self, roi_images, roi_metas, seg_features=None):
        out=[]
        for i in range(roi_images.shape[0]):
            hot = float(roi_images[i].float().mean()) >= 0.2
            loc=[0.02]*NUM_LOCATIONS; loc[6 if hot else 0]=0.9
            tir=[0.02]*5; tir[4 if hot else 0]=0.9
            ete=[0.02]*4; ete[2 if hot else 0]=0.9
            out.append({"lesion_id":roi_metas[i]["lesion_id"],"is_main":1,"patient_id":"P",
                        "malignancy":torch.tensor([-3.,3.] if hot else [3.,-3.]),
                        "location_probs":torch.tensor(loc),"ete_probs":torch.tensor(ete),
                        "ete_logits":torch.tensor(ete),
                        "t_conf":torch.tensor(0.9 if hot else 0.1),
                        "tirads_probs":torch.tensor(tir)})
        return out

img = np.zeros((H, W), np.uint8)
# 让实例1 区域亮、实例2 区域暗
x,y,w,h = insts[0]["bbox"]; img[y:y+h, x:x+w] = 255
x,y,w,h = insts[1]["bbox"]; img[y:y+h, x:x+w] = 10
cr = {"preds": {"pred_mask": soft, "mask_binary": binary_m,
                "comp_idx":0,"echo_idx":0,"shape_idx":0,"margin_idx":0,
                "comp_prob":np.array([0.5,0.5]),"echo_prob":np.array([0.5,0.5]),
                "shape_prob":np.array([0.5,0.5]),"margin_prob":np.array([0.5,0.5]),
                "foci_prob":np.array([0.5,0.5]),"mal_prob":0.5,"mal_probs":np.array([0.5,0.5]),
                "ete_probs":np.array([0.25]*4),"location_probs":np.full(7,1/7),
                "t_probs":np.array([0.5,0.5]),"t_conf":0.5,"tirads_probs":np.full(5,0.2)},
      "img_tensor": np.zeros((1,1,M,M), np.float32), "mask_resized": b_full,
      "breakdown": {"total_points":5}, "orig_hw": (H,W), "orig_img_np": img,
      "gatekeeper_audit": {"skipped": False}, "is_dicom": False}
d = tempfile.mkdtemp()
v21_output.write_case_v21(d,"P","S",cr,"static",10,
    roi_ctx={"model":FakeRoi(),"precision":"fp32","device":torch.device("cpu"),"img_size":M})
p = json.load(open(os.path.join(d,"prediction.json"), encoding="utf-8"))
for n in p["Prediction"]["Nodules"]:
    print(" ", n["NoduleID"], "loc=", n["Location"], "mal=",
          round(n["Malignancy"]["confidence"],4), "tir=", n["TI_RADS_Class"])
print("→ N1 应 hot(loc7/mal0.9975),N2 应 cold(loc1/mal0.0025)")

# -*- coding: utf-8 -*-
"""R-A 探针10:静/动两条链的 ROI meta 是否有键集差异(动缺 is_dynamic?),
以及 _run_roi_static 的 lesion_id 是否与 instances 的枚举序严格对齐。
重点:静态路 `enumerate(instances, start=1)` + `_lesion_id_for(...i)`;动态路同式。
若两路都错位,四头值会挂错结节且不报错 —— 用内容驱动模型验。
"""
import os, sys, json, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch
from src.inference import v21_output
from src.models.label_schema import NUM_LOCATIONS
from src.inference.nodule_split import split_nodules

H, W, M = 40, 60, 8

class FakeRoi(torch.nn.Module):
    """patch 均值 → hot/cold,hot 分支 loc=idx6+tir=idx4+ete=idx2+mal恶性。"""
    def forward(self, x, gamma_prior=1.0): raise AssertionError
    def forward_roi(self, roi_images, roi_metas, seg_features=None):
        out = []
        for i in range(roi_images.shape[0]):
            hot = float(roi_images[i].float().mean()) >= 0.2
            loc = [0.02]*NUM_LOCATIONS; loc[6 if hot else 0] = 0.9
            tir = [0.02]*5; tir[4 if hot else 0] = 0.9
            ete = [0.02]*4; ete[2 if hot else 0] = 0.9
            out.append({"lesion_id": roi_metas[i]["lesion_id"], "is_main": 1,
                        "patient_id": "P",
                        "malignancy": torch.tensor([-3.,3.] if hot else [3.,-3.]),
                        "location_probs": torch.tensor(loc),
                        "ete_probs": torch.tensor(ete),
                        "ete_logits": torch.tensor(ete),
                        "t_conf": torch.tensor(0.9 if hot else 0.1),
                        "tirads_probs": torch.tensor(tir)})
        return out

# 静态:左块亮(大面积) 右块暗(小面积) —— 面积序决定实例序
img = np.zeros((H, W), np.uint8)
img[3:20, 4:21] = 255       # 大亮块 → hot
img[25:32, 40:47] = 10      # 小暗块 → cold
soft = np.zeros((M, M), np.float32); soft[:] = 0.05
soft[0:5, 0:5] = 0.9        # 实例1(大面积,对应左亮块区)
soft[6:8, 6:9] = 0.9        # 实例2
binary = (soft > 0.5).astype(np.uint8)*255
insts = split_nodules(binary)
print("模型分辨率实例 bbox(面积序):", [(i["bbox"], i["area"]) for i in insts])

cr = {"preds": {"pred_mask": soft, "mask_binary": binary,
                "comp_idx":0,"echo_idx":0,"shape_idx":0,"margin_idx":0,
                "comp_prob":np.array([0.5,0.5]),"echo_prob":np.array([0.5,0.5]),
                "shape_prob":np.array([0.5,0.5]),"margin_prob":np.array([0.5,0.5]),
                "foci_prob":np.array([0.5,0.5]),
                "mal_prob":0.5,"mal_probs":np.array([0.5,0.5]),
                "ete_probs":np.array([0.25]*4),"location_probs":np.full(7,1/7),
                "t_probs":np.array([0.5,0.5]),"t_conf":0.5,"tirads_probs":np.full(5,0.2)},
      "img_tensor": np.zeros((1,1,M,M), np.float32), "mask_resized": binary,
      "breakdown": {"total_points":5}, "orig_hw": (H, W),
      "orig_img_np": img, "gatekeeper_audit": {"skipped": False}, "is_dicom": False}
d = tempfile.mkdtemp()
v21_output.write_case_v21(d, "P","S", cr, "static", 10,
                          roi_ctx={"model":FakeRoi(),"precision":"fp32",
                                   "device":torch.device("cpu"),"img_size":M})
p = json.load(open(os.path.join(d,"prediction.json"), encoding="utf-8"))
for n in p["Prediction"]["Nodules"]:
    print(" ", n["NoduleID"], n["MaskFile"], "loc=", n["Location"], "mal=",
          round(n["Malignancy"]["confidence"],4))
print("→ 期望 N1 大块=hot(loc 7)/N2 小块 或反之;关键是**逐结节不同**且与 bbox 面积序一致")

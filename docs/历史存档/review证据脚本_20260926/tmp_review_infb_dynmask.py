"""反例探针:动态 ROI 路径下 ETE band 的 roi_mask 是否退化"""
import sys, numpy as np, torch
sys.path.insert(0,'.')
from src.inference.v21_output import _run_roi_dynamic

class ProbeModel(torch.nn.Module):
    def forward_roi(self, roi_images, roi_metas, seg_features=None):
        # 真实 forward_roi 走 m.get("roi_mask") → 缺键 → 全零
        zeros = []
        for m in roi_metas:
            mm = m.get("roi_mask")
            if mm is None:
                zeros.append(True)
            else:
                zeros.append(bool(torch.as_tensor(mm).max()==0))
        print(f"  [probe] K={len(roi_metas)} roi_mask 缺键/全零 = {zeros}")
        out=[]
        for m in roi_metas:
            out.append({"lesion_id":m["lesion_id"],"is_main":1,"patient_id":"P",
                "malignancy":torch.tensor([0.0,1.0]),
                "location_probs":torch.full((7,),1/7),
                "ete_probs":torch.full((4,),0.25),"ete_logits":torch.zeros(4),
                "t_conf":torch.tensor(0.5),"tirads_probs":torch.full((5,),0.2)})
        return {"roi_outputs":out}

# 构造一个实例跨 3 帧、每帧都有前景的 3D 实例
inst = {"bbox": (2,2,4,4), "frame_indices":[0,1,2], "mask_sparse": None}
frame_by_index = {i: np.full((10,10), 255, np.uint8) for i in range(3)}
ctx = {"model":ProbeModel(),"precision":"fp32","device":torch.device("cpu"),"img_size":8}
try:
    res = _run_roi_dynamic(frame_by_index, [inst], "P","S", ctx)
    print("contexts:", res)
except Exception as e:
    print("raise:", type(e).__name__, e)

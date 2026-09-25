"""反例:两处 _to_np 在 bf16 张量输入下的行为对比(整图路径 vs ROI 路径)"""
import sys, torch, numpy as np
sys.path.insert(0,'.')
from src.inference.v21_output import _to_np as v21_to_np
from src.inference.postprocess import _to_np as pp_to_np
from src.inference.v21_output import _softmax_mal_logits, _roi_nodule_context, _aggregate_roi_frame_rows

t_bf16 = torch.randn(4, dtype=torch.bfloat16)
t_fp16 = torch.randn(4, dtype=torch.float16)
for name, fn in (("v21_output._to_np", v21_to_np), ("postprocess._to_np", pp_to_np)):
    for tn, t in (("bf16", t_bf16), ("fp16", t_fp16)):
        try:
            r = fn(t); print(f"{name}({tn}) OK -> {r[:2]}")
        except Exception as e:
            print(f"{name}({tn}) FAIL {type(e).__name__}: {e}")

# 若模型在 autocast(bf16) 下返回 bf16 t_conf, 真实路径会怎样
class BF16Model(torch.nn.Module):
    def forward_roi(self, roi_images, roi_metas, seg_features=None):
        out=[]
        for m in roi_metas:
            out.append({"lesion_id":m["lesion_id"],"is_main":1,"patient_id":"P",
              "malignancy":torch.tensor([0.,1.],dtype=torch.bfloat16),
              "location_probs":torch.full((7,),1/7,dtype=torch.bfloat16),
              "ete_probs":torch.full((4,),.25,dtype=torch.bfloat16),
              "ete_logits":torch.zeros(4,dtype=torch.bfloat16),
              "t_conf":torch.tensor([0.5],dtype=torch.bfloat16),
              "tirads_probs":torch.full((5,),.2,dtype=torch.bfloat16)})
        return {"roi_outputs":out}
from src.inference.v21_output import _run_roi_dynamic
inst={"bbox":(2,2,4,4),"frame_indices":[0]}
fb={0:np.full((10,10),255,np.uint8)}
ctx={"model":BF16Model(),"precision":"bf16","device":torch.device("cpu"),"img_size":8}
try:
    print("bf16 ROI dynamic ->", _run_roi_dynamic(fb,[inst],"P","S",ctx))
except Exception as e:
    print("bf16 ROI dynamic FAIL:", type(e).__name__, e)

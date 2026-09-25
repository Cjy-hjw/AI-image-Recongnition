"""反例探针:ROI 路径是否给 forward_roi 装 roi_mask / 是否吞 dtype"""
import sys, numpy as np, torch
sys.path.insert(0,'.')
from src.inference.v21_output import _roi_nodule_context, _softmax_mal_logits, _aggregate_roi_frame_rows
from src.inference.roi_pipeline import roi_meta_for_instance

inst={"bbox":(10,20,30,40)}
m=roi_meta_for_instance(inst,"lid",(100,200),bbox_xyxy=(10,20,40,60))
print("meta keys:",sorted(m.keys()))
print("has roi_mask:", "roi_mask" in m)
print("meta:",{k:v for k,v in m.items() if k!='roi_mask'})

# 头输出 dtype 探针(模拟 fp16 autocast 张量)
def mk(dt):
    return {"location_probs":torch.softmax(torch.randn(7,dtype=dt).float(),-1).to(dt),
            "ete_probs":torch.softmax(torch.randn(4,dtype=dt).float(),-1).to(dt),
            "ete_logits":torch.randn(4,dtype=dt),
            "tirads_probs":torch.softmax(torch.randn(5,dtype=dt).float(),-1).to(dt),
            "malignancy":torch.randn(2,dtype=dt),
            "t_conf":torch.rand(1,dtype=dt)}
for dt in (torch.float32,torch.float16,torch.bfloat16):
    ctx=_roi_nodule_context(_softmax_mal_logits(mk(dt)))
    print(str(dt), {k:ctx[k] for k in ("malignancy_confidence","t_confidence","tirads_class","ete_code","location")})

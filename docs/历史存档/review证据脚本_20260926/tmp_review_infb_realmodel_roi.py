"""真实模型 forward_roi 在推理侧 meta(无 roi_mask)下的行为对比"""
import sys, os, numpy as np, torch
sys.path.insert(0,'.')
os.environ.pop("PRODUCTION",None)
from src.inference.model_runner import build_model, load_config
from src.inference.roi_pipeline import roi_meta_for_instance

cfg=load_config('configs/thyroid_config.yaml')
dev=torch.device('cpu')
m=build_model(cfg,'checkpoints/does_not_exist.pth',dev)   # 随机初始化,只看结构行为
m.eval()

H,W=200,160
frame=np.full((H,W),128,np.uint8)
img=torch.rand(1,1,*map(int,[cfg['data']['img_size']]*2))
# 三条 meta:① 推理侧真实 meta(无 roi_mask) ② 位掩膜 GT-like ③ 缺 mask
def mk(name, extra):
    d=roi_meta_for_instance({"bbox":(40,40,40,30)},"lid",(H,W),bbox_xyxy=(40,40,80,70))
    d["lesion_id"]=name; d.update(extra); return d
m_nomask=mk("infer_nomask",{})
m_zero  =mk("zero_mask",{"roi_mask":torch.zeros(1,32,32)})
# ③ 正确 ROI 域 mask:病灶填满 ROI
gm=torch.zeros(1,32,32); gm[:,8:24,8:24]=1.0
m_gt    =mk("gt_like",{"roi_mask":gm})
mes=[m_nomask,m_zero,m_gt]
with torch.inference_mode():
    out=m.forward_roi(img, mes)
for name,o in zip([x['lesion_id'] for x in mes], out["roi_outputs"]):
    print(name, "ete_probs=",[round(float(x),4) for x in o["ete_probs"]],
          "ete_logits_max=",round(float(o["ete_logits"].abs().max()),4))
print("roi_band_ratio=", [round(float(x),6) for x in out["roi_band_ratio"]])
print("roi_d_mm=", out["roi_d_mm"].flatten().tolist())

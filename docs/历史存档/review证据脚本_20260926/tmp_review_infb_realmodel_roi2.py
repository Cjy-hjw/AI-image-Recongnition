import sys, os, numpy as np, torch
sys.path.insert(0,'.')
os.environ.pop("PRODUCTION",None)
from src.inference.model_runner import build_model, load_config
from src.inference.roi_pipeline import roi_meta_for_instance
cfg=load_config('configs/thyroid_config.yaml')
m=build_model(cfg,'nope.pth',torch.device('cpu')).eval()
H,W=200,160
S=int(cfg['data']['img_size'])
img=torch.rand(3,1,S,S)
def mk(name, extra):
    d=roi_meta_for_instance({"bbox":(40,40,40,30)},name,(H,W),bbox_xyxy=(40,40,80,70))
    d.update(extra); return d
gm=torch.zeros(1,32,32); gm[:,8:24,8:24]=1.0
mes=[mk("infer_nomask",{}), mk("zero_mask",{"roi_mask":torch.zeros(1,32,32)}), mk("gt_like",{"roi_mask":gm})]
with torch.inference_mode():
    out=m.forward_roi(img, mes)
for name,o in zip([x['lesion_id'] for x in mes], out["roi_outputs"]):
    print(name,"ete_probs",[round(float(x),4) for x in o["ete_probs"]],"loc",[round(float(x),3) for x in o["location_probs"]][:3],"t",float(o["t_conf"]))
print("band_ratio",[round(float(x),6) for x in out["roi_band_ratio"]])
print("d_mm",out["roi_d_mm"].flatten().tolist())

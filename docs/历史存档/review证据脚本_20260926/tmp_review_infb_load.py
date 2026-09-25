import sys, torch, os
sys.path.insert(0,'.')
os.environ.pop("PRODUCTION",None)
from src.inference.model_runner import build_model, load_config
import glob
# 找一个最小的 ckpt
cands=[p for p in glob.glob('checkpoints/*.pth')]
cands.sort(key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0)
print("cands:", [(p, os.path.getsize(p)//1024//1024) for p in cands[:6]])
cfg=load_config('configs/thyroid_config.yaml')
dev=torch.device('cpu')
p=cands[0]
m=build_model(cfg, p, dev)
print("has forward_roi:", hasattr(m,'forward_roi'))
print("ROI_HEAD_PREFIXES:", m.ROI_HEAD_PREFIXES)

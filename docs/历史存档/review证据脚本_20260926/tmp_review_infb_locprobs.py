import sys, torch, numpy as np
sys.path.insert(0,'.')
from src.models.heads.location import compose_location_probs
worst=0.0; worst_case=None
for seed in range(200):
    g=torch.Generator().manual_seed(seed)
    for scale in (1.0, 10.0, 40.0):
        side=torch.softmax(torch.randn(3,generator=g)*scale,dim=-1)
        level=torch.softmax(torch.randn(3,generator=g)*scale,dim=-1)
        p=compose_location_probs(side.unsqueeze(0), level.unsqueeze(0))[0]
        d=abs(float(p.sum())-1.0)
        if d>worst: worst, worst_case = d,(seed,scale)
print("compose_location_probs max |sum-1| =",worst,"case",worst_case,"-> 容差1e-6:", "PASS" if worst<1e-6 else "FAIL")
from src.inference.prediction_json import normalize_location_probs
s=[float(x) for x in p]; n=normalize_location_probs(s)
print("归一化后 |sum-1| =",abs(sum(n)-1.0))

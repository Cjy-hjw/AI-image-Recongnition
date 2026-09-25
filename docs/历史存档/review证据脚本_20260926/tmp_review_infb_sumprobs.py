import numpy as np, sys
sys.path.insert(0,'.')
import torch
for dt,name in [(torch.float32,'fp32'),(torch.float16,'fp16'),(torch.bfloat16,'bf16')]:
    x=torch.randn(20,3,dtype=dt)
    a=torch.softmax(x.float(),dim=-1)
    b=torch.softmax(x.float(),dim=-1)
    p=torch.cat([(a[...,:2].unsqueeze(-1)*b.unsqueeze(-2)).reshape(20,6), a[...,2:3]],dim=-1)
    print(name,'sum min/max',float(p.sum(-1).min()),float(p.sum(-1).max()))

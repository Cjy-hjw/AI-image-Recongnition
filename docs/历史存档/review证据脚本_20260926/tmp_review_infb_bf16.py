import sys, torch, numpy as np
sys.path.insert(0,'.')
from src.inference.v21_output import _to_np
print("torch", torch.__version__, "has bfloat16 numpy:", hasattr(torch.Tensor,'numpy'))
t=torch.randn(4,dtype=torch.bfloat16)
try:
    print(_to_np(t))
except Exception as e:
    print("DIRECT numpy FAIL:", type(e).__name__, e)
print("float() first:", _to_np(t.float()))
# 对照 _shared_nodule_context 整图路径怎么走
from src.inference.postprocess import _to_np as pp_to_np
try:
    print("postprocess._to_np(bf16):", pp_to_np(t))
except Exception as e:
    print("pp DIRECT FAIL:", type(e).__name__, e)

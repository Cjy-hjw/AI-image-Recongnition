"""反例:ROI 打开的 prediction.json 逐字段契约核 + 对照整图路径"""
import sys, os, json, numpy as np, torch
sys.path.insert(0,'.')
from src.inference.nodule_split import crop_nodule
from src.inference.model_runner import preprocess_image
from src.inference import prediction_json as pj

# 用真实 forward_roi 输出跑 _roi_nodule_context:检查各字段值域/长度
from src.inference.v21_output import (_roi_contexts_from_outputs, _shared_nodule_context)
from src.models.label_schema import LOCATION_LABELS, ETE_CODES

def out(seed):
    g=torch.Generator().manual_seed(seed)
    return {"lesion_id":"L","is_main":1,"patient_id":"P",
      "malignancy":torch.randn(2,generator=g),          # 裸 logits(真模型行为)
      "location_probs":torch.softmax(torch.randn(7,generator=g).float(),-1),
      "ete_probs":torch.softmax(torch.randn(4,generator=g).float(),-1),
      "ete_logits":torch.randn(4,generator=g),
      "t_conf":torch.rand(1,generator=g),
      "tirads_probs":torch.softmax(torch.randn(5,generator=g).float(),-1)}

# 大 logits 反例(置信度越界?未 softmax 时曾被报 P0-1)
for scale in (1.0, 8.0, 40.0):
    ctx=_roi_contexts_from_outputs([out(int(scale))])[0]
    print(f"scale={scale}: mal_conf={ctx['malignancy_confidence']:.6f} t_conf={ctx['t_confidence']:.6f} "
          f"tirads={ctx['tirads_class']} ete={ctx['ete_code']} loc={ctx['location']} "
          f"locprobs_sum={sum(ctx['location_probs']):.9f} "
          f"loc_len={len(ctx['location_probs'])} ete_len={len(ctx['ete_probs'])} tir_len={len(ctx['tirads_probs'])}")

# 与整图路径键集/编码口径对拍
def shared(seed):
    g=torch.Generator().manual_seed(seed)
    return {"mal_prob":0.5,"ete_probs":torch.softmax(torch.randn(4,generator=g),-1),
            "location_probs":torch.softmax(torch.randn(7,generator=g),-1),
            "t_probs":torch.softmax(torch.randn(2,generator=g),-1),
            "t_conf":0.5,"tirads_probs":torch.softmax(torch.randn(5,generator=g),-1)}
s=_shared_nodule_context(shared(7)); r=_roi_contexts_from_outputs([out(7)])[0]
print("key set equal:", set(s)==set(r), "| shared-only:", set(s)-set(r), "| roi-only:", set(r)-set(s))
print("shared keys:", sorted(s))

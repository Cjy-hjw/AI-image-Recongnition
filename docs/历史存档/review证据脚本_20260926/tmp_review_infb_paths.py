import sys, traceback, numpy as np, torch
sys.path.insert(0,'.')
from src.inference.case_pipeline import diagnose_static_case
from src.service.runner import _process_case, _roi_heads_on
# 1) 默认路径(旧 fixture 无 orig_img_np) ROI 分流开启时的行为
import inspect
from src.inference import v21_output as V

# 构造最小 case_result:走 _write_static 体内(不跑真模型),看 fail-loud 分支
case_result = {
  "preds": {
    "pred_mask": np.zeros((32,32),np.float32), "mask_binary": np.zeros((32,32),np.uint8),
    "ete_probs":[.25]*4,"location_probs":[1/7]*7,"t_probs":[.5,.5],"t_conf":.5,
    "tirads_probs":[.2]*5,"mal_prob":.5,
  },
  "orig_hw": (64,64),
}
roi_ctx={"model":None,"precision":"fp32","device":torch.device("cpu"),"img_size":512}
try:
    V._write_static("tmp_review_infb_out","PID1","SID",case_result,1,None,roi_ctx=roi_ctx)
    print("static ROI: no raise")
except Exception as e:
    print("static ROI raise:", type(e).__name__, str(e)[:120])

# 2) roi_ctx 非 None 但实例为 0:看看 _run_roi_static 返回什么
print("zero-instance path ->", V._run_roi_static(np.zeros((64,64),np.uint8), [], "P","S", roi_ctx))

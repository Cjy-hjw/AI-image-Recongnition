"""核 T-临床显著性打点/ROI 打点是否污染 prediction 目录白名单(写 stdout 不落盘)"""
import sys, os, tempfile, numpy as np, torch
sys.path.insert(0,'.')
from src.inference import v21_output as V
import io, contextlib
# 单帧实例(触发 print 打点)
class M(torch.nn.Module):
    def forward_roi(self, imgs, metas, seg_features=None):
        return {"roi_outputs":[{"lesion_id":m["lesion_id"],"is_main":1,"patient_id":"P",
            "malignancy":torch.tensor([0.,1.]),
            "location_probs":torch.full((7,),1/7),
            "ete_probs":torch.full((4,),.25),"ete_logits":torch.zeros(4),
            "t_conf":torch.tensor(.5),"tirads_probs":torch.full((5,),.2)} for m in metas]}
outdir=tempfile.mkdtemp()
fb={0:np.full((16,16),255,np.uint8)}
inst={"bbox":(2,2,4,4),"frame_indices":[0]}
ctx={"model":M(),"precision":"fp32","device":torch.device("cpu"),"img_size":8}
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    V._run_roi_dynamic(fb,[inst],"P","S",ctx)
print("stdout 打点:",buf.getvalue().strip()[:120])
print("目录内容(应只有 stdout,无文件):",os.listdir(outdir))

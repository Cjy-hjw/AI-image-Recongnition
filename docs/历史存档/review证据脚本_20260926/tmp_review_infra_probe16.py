# -*- coding: utf-8 -*-
"""R-A 探针16:诊断镜像链的 ROI env 盲区(具体反例)。
AICOMP_ROI_HEADS=1 下跑 scripts.diagnose_detection.run_static_inference,
case_result 会带 roi_heads=True,但下一步 post(case_result) 走的是
run_static_inference_from_result(整图共享链,不落盘、不调 forward_roi)
→ 诊断结果与开关关时逐位相同,而打点声称"ROI 已开"。
用 monkeypatch 的最小真模型验证:同一输入,env 开/关两跑,结果逐位比较。
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_roi_model import _tiny_model
from scripts.diagnose_detection import run_static_inference, run_static_inference_from_result

m = _tiny_model(); m.eval()
rng = np.random.default_rng(0)
img_path = None
# 造一个临时 png
import cv2, tempfile
d = tempfile.mkdtemp()
img_path = os.path.join(d, "t.png")
cv2.imwrite(img_path, (rng.random((48, 64))*255).astype(np.uint8))
cfg = {"data": {"img_size": 64}}

res = {}
for val in (None, "1"):
    if val is None: os.environ.pop("AICOMP_ROI_HEADS", None)
    else: os.environ["AICOMP_ROI_HEADS"] = val
    os.environ.pop("AICOMP_STATIC_TILING", None)
    cr, post = run_static_inference(m, cfg, None, "fp32", torch.device("cpu"), img_path)
    insts = post(cr)
    res[val] = (cr.get("roi_heads"), [tuple(i.get("bbox", ())) for i in insts],
                [round(float(i.get("confidence", -1)),8) for i in insts])
print("env 关 :", res[None])
print("env 开 :", res["1"])
print("→ 开关开时诊断侧多出 roi_heads 标记但实例/置信度逐位相同:",
      res[None][1:] == res["1"][1:])
print("   标记 ", res["1"][0], "= True,而实跑链**未走 ROI**(post 是整图共享镜像)")

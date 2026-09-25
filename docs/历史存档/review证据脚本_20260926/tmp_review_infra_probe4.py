# -*- coding: utf-8 -*-
"""R-A 探针4:训练侧 mal_prob 到底是"概率域逐帧均值"还是"logits 均值再 softmax"?
以及 instance_idx 的来源(=模型输出侧哪一维)。"""
import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
from src.training import collect_roi_records as cr
src = inspect.getsource(cr)
import re
for m in re.finditer(r"instance_idx", src):
    s = max(0, m.start()-300); print(src[s:m.end()+200]); print("-----")

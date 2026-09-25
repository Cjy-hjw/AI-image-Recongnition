# -*- coding: utf-8 -*-
"""R-A 探针3:动态 ROI 帧均值 vs 训练侧 aggregate_by_roi_instance 口径对拍。

要点:
1. 训练侧 `metrics_official.aggregate_by_roi_instance` 的均值定义
2. 推理侧 `_aggregate_roi_frame_rows` 的:mal 逐帧 softmax 后均值(概率域),其余逐帧均值
3. 训练侧 `aggregate_by_lesion` 是否也是概率域均值(同口径?)
"""
import os, sys, inspect, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
import torch

from src.training import metrics_official as mo
src = inspect.getsource(mo.aggregate_by_roi_instance) if hasattr(mo, "aggregate_by_roi_instance") else "NO FUNC"
print("=== metrics_official.aggregate_by_roi_instance ===")
print(src[:2200] if src != "NO FUNC" else [n for n in dir(mo) if "agg" in n])

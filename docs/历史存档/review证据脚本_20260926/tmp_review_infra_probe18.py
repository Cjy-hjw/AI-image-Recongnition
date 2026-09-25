# -*- coding: utf-8 -*-
"""R-A 探针18:case_result 层的新键是否影响**关路** case_result 键集。
runner 关时不注入 orig_img_np(仅开时);但 diagnose_static_case 开时加 roi_heads。
检查运行诊断镜像时开/关的键集差 —— 是否有键在关路被意外注入。
"""
import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # noqa
from src.inference import case_pipeline
src = inspect.getsource(case_pipeline.diagnose_static_case)
import re
print("case_pipeline 中所有 result[ 写入:")
for m in re.finditer(r'result\[[^\]]+\]', src):
    print("   ", m.group(0))
print()
from src.service import runner
src2 = inspect.getsource(runner._run_static_case)
print("runner._run_static_case 中 result[ 写入:")
for m in re.finditer(r'result\[[^\]]+\]', src2):
    print("   ", m.group(0))

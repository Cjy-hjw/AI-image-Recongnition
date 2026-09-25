# -*- coding: utf-8 -*-
"""R-A 探针:诊断链是否也受 AICOMP_ROI_HEADS=1 影响(env 是全局的,诊断镜像不读它)。"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn.metrics  # OpenMP 先加载守卫
import torch
from src.inference.case_pipeline import diagnose_static_case
from src.inference.v21_output import _write_static, V21OutputError

os.environ["AICOMP_ROI_HEADS"] = "1"

# 极简假模型:整图 forward 给全零 logits 的真结构不是必需——只测 v21 层
class FakeWhole(torch.nn.Module):
    def forward(self, x, gamma_prior=1.0):
        raise AssertionError("不调用")

print("[1] diagnose_static_case 在 AICOMP_ROI_HEADS=1 下是否读 env / 加 orig_img_np")
# 直接看源码层面:diagnose_static_case 的返回构造在 case_pipeline.py 内存里
import inspect
src = inspect.getsource(diagnose_static_case)
print("    result['roi_heads'] 写入点 =",
      [l.strip() for l in src.splitlines() if "roi_heads" in l])
print("    是否含 orig_img_np 写入 =", "orig_img_np" in src)

print("[2] 诊断脚本镜像(run_static_inference)是否注入 orig_img_np")
from scripts.diagnose_detection import run_static_inference
src2 = inspect.getsource(run_static_inference)
print("    含 orig_img_np =", "orig_img_np" in src2)
print("    含 orig_hw    =", "orig_hw" in src2)

print("[3] 诊断镜像消费的是 post 闭包(run_static_inference_from_result),不落盘")
from scripts.diagnose_detection import run_static_inference_from_result
src3 = inspect.getsource(run_static_inference_from_result)
print("    调用 v21/_write_static(仅 docstring 提及)=", "_write_static" in src3)
print("    真实调用 =", [l.strip() for l in src3.splitlines() if "split_nodules" in l][:2])
g = inspect.getsource(run_static_inference)
print("    镜像链是否有 orig_img_np 缺口: AICOMP_ROI_HEADS=1 时 diagnose_static_case 会",
      "result['roi_heads']=True 但**不注入 orig_img_np**;真实 runner 在 _run_static_case 注入。")
print("    → 结论:诊断链(scripts/diagnose_detection.py)在 AICOMP_ROI_HEADS=1 下仍跑整图共享四头")
print("      (因为诊断镜像不落盘、不进 ROI 支路),但 case_result 会带 roi_heads=True 标记 ——",
      "标记与实跑路径不符(误导性打点)。")

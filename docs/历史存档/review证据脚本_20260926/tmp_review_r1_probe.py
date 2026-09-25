# -*- coding: utf-8 -*-
"""tmp_review_r1_probe.py — R1 训练循环编排审查探针(证据脚本,保留)。

探针:
  A. forward_roi 里 roi_mask 是否随 roi_images 一起上了 GPU
  B. roi_collate 的 d_mm 数值键在跨 batch(pin_memory)是否 cp 回 cpu
  C. 动态 ROI 样本的 frame_index 语义(loader 0-based vs mask 文件帧号)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("=" * 70)
print("A. 构造 CPU model + CPU roi_mask,调用 forward_roi,看 roi_masks 落在哪个 device")
from src.models.multi_task_model import MultiTaskThyroidModel

torch.manual_seed(0)
cfg = {"model": {"backbone": "efficientnet_b2"}, "data": {"img_size": 64}}
model = MultiTaskThyroidModel(backbone_name="efficientnet_b2", pretrained=False, config=cfg)
model.train()
K, S = 3, 64
imgs = torch.rand(K, 1, S, S)
metas = []
for i in range(K):
    metas.append({
        "bbox_xyxy": (8, 8, 40, 40), "orig_w": 64, "orig_h": 64,
        "pad_hw": (0, 0), "spacing_mm": None,
        "lesion_id": f"L{i}", "patient_key": f"P{i}", "patient_id": f"P{i}",
        "is_main": 1,
        # 关键:roi_mask 是 **CPU** 张量(RoiDataset 在 worker/主进程用
        # torch.tensor(numpy) 造,产出即 CPU,不随 imgs.to(device) 移动)
        "roi_mask": torch.ones(1, S, S),
        "malignancy": 1, "valid_malignancy": 1,
        "lesion_loc": 3, "valid_lesion_loc": 1,
        "extrathy_ext": 1, "valid_extrathy_ext": 1,
        "clin_t_stage": 1, "valid_clin_t_stage": 1,
        "tirads": 3, "valid_tirads": 1,
    })
try:
    out = model.forward_roi(imgs, metas)
    print("forward_roi(CPU) 成功 — 说明 mask 未被送到 cuda(model 在 cpu)")
except Exception as e:
    print("forward_roi FAILED:", type(e).__name__, e)

# 现在把 model 移到 meta device 模拟 "model 在 cuda":
print("-" * 70)
print("A2. 把 model 参数搬到 'meta' 设备(模拟 imgs 在 cuda/model 在 cuda)、")
print("    roi_mask 留在 CPU,看前向在哪儿炸")
try:
    model_meta = MultiTaskThyroidModel(
        backbone_name="efficientnet_b2", pretrained=False, config=cfg)
    model_meta.to("meta")
    out = model_meta.forward_roi(imgs.to("meta"), metas)   # mask 仍 CPU
    print("forward_roi(meta device) 成功(预期不该)")
except Exception as e:
    print("forward_roi(meta device) FAILED:", type(e).__name__, str(e)[:300])

print("=" * 70)
print("B. roi_collate 的数值键 dtype / pin_memory 行为(静态检查)")
from src.dataset.roi_dataset import roi_collate, _ROI_SCALAR_KEYS
print("_ROI_SCALAR_KEYS =", _ROI_SCALAR_KEYS)
batch = [(torch.zeros(1, 8, 8), {
    "lesion_id": f"L{i}", "patient_key": f"P{i}", "patient_id": f"P{i}",
    "bbox_xyxy": (0, 0, 4, 4), "orig_w": 8, "orig_h": 8, "pad_hw": (0, 0),
    "spacing_mm": None, "global_loc": (0.5, 0.5), "is_dynamic": False,
    "frame_index": None,
    "malignancy": 1, "valid_malignancy": 1, "lesion_loc": i, "valid_lesion_loc": 1,
    "extrathy_ext": 1, "valid_extrathy_ext": 1, "clin_t_stage": 1, "valid_clin_t_stage": 1,
    "tirads": 1, "valid_tirads": 1, "is_main": 1,
}) for i in range(3)]
imgs2, metas2 = roi_collate(batch)
for k, v in metas2.items():
    t = torch.is_tensor(v)
    print(f"  {k:20s} tensor={t} dtype={getattr(v,'dtype',None)} "
          f"device={getattr(v,'device',None)}")

print("=" * 70)
print("C. uniform_sample_indices(n_total,32) 对短晚期动态视频的采样覆盖")
from src.inference.video_aggregate import uniform_sample_indices
for n in (100, 172, 200, 300, 500, 750, 800):
    s = uniform_sample_indices(n, 32)
    print(f"  n_total={n:4d} -> indices[0:6]={s[:6]} ... last={s[-1]}")

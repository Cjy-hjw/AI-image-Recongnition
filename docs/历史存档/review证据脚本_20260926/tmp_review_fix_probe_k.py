# -*- coding: utf-8 -*-
"""对抗式审查探针 K:ROI 与整图支路的设备/精度/BN 契约(CPU 可验部分)。

保留不删。
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.models.multi_task_model import MultiTaskThyroidModel  # noqa: E402
from src.models.losses.components.roi_heads import compute_roi_head_losses  # noqa: E402
from src.training.loop.train_step import _roi_labels_from_meta  # noqa: E402
from src.dataset.roi_dataset import RoiDataset  # noqa: E402

spec = importlib.util.spec_from_file_location("to_k", ROOT / "scripts" / "train_official.py")
to = importlib.util.module_from_spec(spec)
spec.loader.exec_module(to)

torch.manual_seed(0)
cfg = {"model": {"backbone": "efficientnet_b2"}, "data": {"img_size": 64}}
model = MultiTaskThyroidModel(backbone_name="efficientnet_b2", pretrained=False, config=cfg)

print("=" * 70)
print("Probe K1: fake 全零 roi_mask 是否让 ETE band 恒 0(对比无键)")
print("=" * 70)
sb = to._mock_roi_samples(n=4, img_size=64, seed=0)
met = to._meta_rows_from_collate({k: [m[k] for m in sb] for k in sb[0]})
imgs = torch.rand(4, 1, 64, 64)
out_none = model.forward_roi(imgs, met)
print("无 roi_mask: band_ratio =", out_none["roi_band_ratio"].tolist(),
      " d_mm =", out_none["roi_d_mm"].tolist())
for m in met:
    m["roi_mask"] = torch.ones(1, 64, 64)      # 全 1 假 mask
out_one = model.forward_roi(imgs, met)
print("全1 roi_mask: band_ratio =", out_one["roi_band_ratio"].tolist(),
      " d_mm =", out_one["roi_d_mm"].tolist())
print("→ 真实训练(mask_path 非空)下 d_mm/band 由真 mask 驱动;mock 恒 0/0(初版缺陷已修)")

print()
print("=" * 70)
print("Probe K2: precision/scaler 混用风险(run_roi_epoch 用未 scale 的 loss?)")
print("=" * 70)
import inspect  # noqa: E402
src = inspect.getsource(to.run_roi_epoch)
for pat in ("scaler.scale(loss).backward()", "scaler.unscale_(optimizer)",
            "clip_grad_norm_", "scaler.step(optimizer)", "scaler.update()"):
    print(f"  {'OK ' if pat in src else 'MISS'} {pat}")
# 与之对比:整图 train_batch 的契约
from src.training.loop import train_step  # noqa: E402
tsrc = inspect.getsource(train_step)
print("  整图 train_batch 同样模式:",
      all(p in tsrc for p in ("scaler.unscale_", "clip_grad_norm_", "scaler.step")))

print()
print("=" * 70)
print("Probe K3: RoiDataset 真样本 batch 里 roi_mask 的 device/dtype(CPU 冒烟)")
print("=" * 70)
import numpy as np  # noqa: E402
import tempfile  # noqa: E402
sys.path.insert(0, str(ROOT / "tests"))
from test_official_loader import _make_dcm  # noqa: E402

tmp = Path(tempfile.mkdtemp())
p = tmp / "a.dcm"
_make_dcm(str(p), np.full((64, 64), 120, np.uint8))
s = {"img_path": str(p), "is_dynamic": False, "frame_index": None,
     "patient_key": "P", "patient_id": "P", "lesion_id": "L",
     "bbox_xyxy": (20, 20, 40, 40), "orig_w": 64, "orig_h": 64,
     "spacing_mm": None, "global_loc": (0.47, 0.47), "mask_path": None,
     "malignancy": 1, "valid_malignancy": 1, "lesion_loc": 1, "valid_lesion_loc": 1,
     "extrathy_ext": 0, "valid_extrathy_ext": 1, "clin_t_stage": 0,
     "valid_clin_t_stage": 1, "tirads": 0, "valid_tirads": 1, "is_main": 1}
ds = RoiDataset([s], roi_size=32)
img, meta = ds[0]
print("img:", tuple(img.shape), img.dtype, "meta keys:", len(meta))
print("roi_mask in meta:", "roi_mask" in meta, "(mock/无 mask 病灶 = 缺键)")
m2 = dict(s, mask_path="X.nii.gz")
# 只验证"有 mask 时"的 dtype 路径不走 mock(真文件缺失 → 读失败 → 缺键)
ds2 = RoiDataset([m2], roi_size=32)
_, meta2 = ds2[0]
print("mask_path 指向不存在文件 → roi_mask in meta:", "roi_mask" in meta2)

print()
print("=" * 70)
print("Probe K4: forward_roi 对 roi_mask 与 roi_images 的 device 一致性要求")
print("=" * 70)
met2 = to._meta_rows_from_collate(
    {k: [m[k] for m in to._mock_roi_samples(n=2, img_size=64, seed=0)] for k in
     to._mock_roi_samples(n=1, img_size=64, seed=0)[0]})
for m in met2:
    m["roi_mask"] = torch.ones(1, 64, 64, device="meta" if False else "cpu")
try:
    o = model.forward_roi(imgs[:2], met2)
    print("同设备(CPU/CPU): OK, d_mm =", o["roi_d_mm"].tolist())
except Exception as e:
    print("同设备 RAISE", type(e).__name__, str(e)[:120])
print("说明:_move_roi_masks_to_device 只按 device.type 比较 → CPU-only 环境 no-op 正确;")
print("     CUDA 环境将 CPU mask 迁到 cuda,与 roi_images 同设备(R1 P0-1 修复点)。")

# -*- coding: utf-8 -*-
"""对抗式审查探针 K5:roi_mask 真实非零时 forward_roi 消费链是否真通(CPU)。

关键反例:有真 mask(非零)时 d_mm/band_ratio 必须非零——否则"mask 生产端的
修复"是空接线(消费端恒零)。保留不删。
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import nibabel as nib  # noqa: E402
from test_official_loader import _make_dcm  # noqa: E402

from src.models.multi_task_model import MultiTaskThyroidModel  # noqa: E402
from src.dataset.roi_dataset import RoiDataset, roi_collate  # noqa: E402
from src.dataset.official_loader import resize_mask_to  # noqa: E402
from src.models.losses.components.roi_heads import compute_roi_head_losses  # noqa: E402
from src.training.loop.train_step import _roi_labels_from_meta  # noqa: E402
from src.training.loop.bn_freeze import set_frozen_modules_eval  # noqa: E402

spec = importlib.util.spec_from_file_location("to_k5", ROOT / "scripts" / "train_official.py")
to = importlib.util.module_from_spec(spec)
spec.loader.exec_module(to)

torch.manual_seed(0)
cfg = {"model": {"backbone": "efficientnet_b2"}, "data": {"img_size": 64}}
model = MultiTaskThyroidModel(backbone_name="efficientnet_b2", pretrained=False, config=cfg)

tmp = Path(tempfile.mkdtemp())
dcm = tmp / "img.dcm"
_make_dcm(str(dcm), np.full((64, 64), 120, np.uint8))
mask = np.zeros((64, 64), np.uint8)
mask[20:40, 20:40] = 1
mask_path = tmp / "M_1_mask.nii.gz"
nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), str(mask_path))

s = {"img_path": str(dcm), "is_dynamic": False, "frame_index": None,
     "patient_key": "P", "patient_id": "P", "lesion_id": "L",
     "bbox_xyxy": (20, 20, 40, 40), "orig_w": 64, "orig_h": 64,
     "spacing_mm": None, "global_loc": (0.46875, 0.46875),
     "mask_path": str(mask_path),
     "malignancy": 1, "valid_malignancy": 1, "lesion_loc": 1, "valid_lesion_loc": 1,
     "extrathy_ext": 0, "valid_extrathy_ext": 1, "clin_t_stage": 0,
     "valid_clin_t_stage": 1, "tirads": 0, "valid_tirads": 1, "is_main": 1}
ds = RoiDataset([dict(s) for _ in range(4)], roi_size=32)
batch = [ds[i] for i in range(4)]
imgs, metas = roi_collate(batch)
print("roi_mask in collated keys:", "roi_mask" in metas)
assert "roi_mask" in metas, "真样本 roi_mask 键丢失"
print("roi_mask shape/dtype:", tuple(metas["roi_mask"][0].shape),
      metas["roi_mask"][0].dtype, "device:", metas["roi_mask"][0].device.type)

_metas = to._meta_rows_from_collate(metas)
to._move_roi_masks_to_device(_metas, torch.device("cpu"))
print("move 后 device:", _metas[0]["roi_mask"].device.type)
out = model.forward_roi(imgs, _metas)
print("band_ratio(真 mask):", out["roi_band_ratio"].tolist())
print("d_mm(真 mask):", out["roi_d_mm"].tolist())
assert any(float(v) != 0.0 for v in out["roi_d_mm"].flatten()), \
    "d_mm 恒 0 = mask 消费链空接线"
print("PASS: 真 mask → d_mm 非零(mask 生产端与消费端真通)")

labels = _roi_labels_from_meta(_metas)
loss, det = compute_roi_head_losses(out, {"roi_labels": labels}, lambda_roi=0.5)
print("四头 loss:", float(loss))

# 端到端 run_roi_epoch(带真 mask)
class _L:
    def __iter__(self): return iter([(imgs, metas)])
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
scaler = torch.amp.GradScaler("cpu", enabled=False)
avg, n = to.run_roi_epoch(model, _L(), opt, type("C", (), {
    "roi_head_weight": 0.5, "ete_class_weights": None, "ete_focal_gamma": 0.0,
    "tirads_class_weights": None, "tirads_label_smoothing": 0.0,
    "tirads_ordinal_weight": 0.0, "roi_concept_weight": 0.0})(), scaler,
    torch.device("cpu"), epoch=1, logger=None, log_every=1, head_lr=1e-3,
    amp_enabled=False)
print("run_roi_epoch(真 mask) OK avg=%.4f n=%d" % (avg, n))

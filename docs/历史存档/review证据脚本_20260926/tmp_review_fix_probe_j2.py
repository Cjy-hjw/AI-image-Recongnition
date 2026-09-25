# -*- coding: utf-8 -*-
"""对抗式审查探针 J2:整图 epoch + ROI epoch 串行(真 train_batch 路径)。

保留不删。
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.models.multi_task_model import MultiTaskThyroidModel  # noqa: E402
from src.models.loss import MultiTaskLoss  # noqa: E402
from src.training.schedule_official import configure_official_stage  # noqa: E402
from src.training.loop.bn_freeze import set_frozen_modules_eval  # noqa: E402
from src.dataset.roi_dataset import roi_collate  # noqa: E402

spec = importlib.util.spec_from_file_location("to_probe", ROOT / "scripts" / "train_official.py")
to = importlib.util.module_from_spec(spec)
spec.loader.exec_module(to)

N_SEG_CLASSES = None


class _Loader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


torch.manual_seed(0)
cfg = {"model": {"backbone": "efficientnet_b2"},
       "data": {"img_size": 64},
       "train": {"slr_beta": 0.99, "slr_threshold": 2.0, "label_smoothing": 0.0,
                 "amp": False}}
model = MultiTaskThyroidModel(backbone_name="efficientnet_b2", pretrained=False, config=cfg)
crit = MultiTaskLoss()
crit.roi_head_weight = 0.5
groups, hl, name = configure_official_stage(
    model, crit, epoch=1, lr_base=1e-3, weight_decay=0.0, roi_heads=True,
    freeze_legacy=True)
opt = torch.optim.AdamW([p for g in groups for p in g["params"]], lr=1e-3)
vclub_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
scaler = torch.amp.GradScaler("cpu", enabled=False)

# 整图 mock loader:images[B,1,H,W], masks[B,1,H,W], labels(dict 的 batch)
imgs = torch.rand(2, 1, 64, 64)
masks = torch.zeros(2, 1, 64, 64)
labels = {
    "label": torch.tensor([1, 0]), "label_mask": torch.tensor([1.0, 1.0]),
    "valid_lesion_loc": torch.tensor([1.0, 1.0]), "lesion_loc": torch.tensor([1, 2]),
    "valid_extrathy_ext": torch.tensor([1.0, 1.0]), "extrathy_ext": torch.tensor([0, 1]),
    "valid_clin_t_stage": torch.tensor([1.0, 1.0]), "clin_t_stage": torch.tensor([0, 1]),
    "valid_tirads": torch.tensor([1.0, 1.0]), "tirads": torch.tensor([1, 2]),
    # 四概念:整图 legacy 概念损失无条件消费(无 valid_ 键时走 legacy 路径)
    "composition": torch.tensor([0, 1]), "echogenicity": torch.tensor([0, 1]),
    "shape": torch.tensor([0, 1]), "margin": torch.tensor([0, 1]),
    "foci": torch.tensor([0, 0]),
    "malignancy": torch.tensor([1, 0]),
}
whole_loader = _Loader([(imgs, masks, labels)])

sb = to._mock_roi_samples(n=6, img_size=64, seed=0)
rows = [(torch.rand(1, 64, 64), m) for m in sb]
imgs2, metas = roi_collate(rows)
roi_loader = _Loader([(imgs2, metas)])

print("=" * 70)
print("Probe J2: 整图 run_official_epoch → run_roi_epoch 串行(真 train_batch)")
print("=" * 70)
try:
    avg = to.run_official_epoch(model, whole_loader, opt, vclub_opt, crit, scaler,
                                torch.device("cpu"), cfg, [], 1, True, None, 1, hl)
    print("整图 epoch OK avg=", avg)
except Exception as e:
    print("整图 epoch RAISE", type(e).__name__, str(e)[:300])

# 进 ROI epoch 前记录 legacy 模块 training 态
before = {n: m.training for n, m in model.named_modules()}
n_frozen = sum(1 for v in before.values() if not v)
print("整图 epoch 后 frozen 模块数:", n_frozen)
try:
    avg2, nb = to.run_roi_epoch(model, roi_loader, opt, crit, scaler,
                                torch.device("cpu"), epoch=1, logger=None,
                                log_every=1, head_lr=hl, amp_enabled=False)
    print("ROI epoch OK avg=%s n=%s" % (avg2, nb))
except Exception as e:
    print("ROI epoch RAISE", type(e).__name__, str(e)[:300])

# 再跑一次整图 epoch:看 legacy 路径是否被 ROI 的 model.train() 打乱
try:
    avg3 = to.run_official_epoch(model, whole_loader, opt, vclub_opt, crit, scaler,
                                 torch.device("cpu"), cfg, [], 2, True, None, 1, hl)
    print("第二轮整图 epoch OK avg=", avg3)
except Exception as e:
    print("第二轮整图 RAISE", type(e).__name__, str(e)[:300])

# ROI 头梯度是否真进 optimizer(权重变化)
print()
print("=" * 70)
print("Probe J3: 两轮 ROI epoch(真 optimizer)→ ROI 头权重变化 + legacy 权重不动")
print("=" * 70)
w_before = {n: p.detach().clone() for n, p in model.named_parameters()}
for _ in range(2):
    to.run_roi_epoch(model, roi_loader, opt, crit, scaler, torch.device("cpu"),
                     epoch=1, logger=None, log_every=1, head_lr=hl, amp_enabled=False)
roi_chg = [n for n, p in model.named_parameters()
           if n.startswith(MultiTaskThyroidModel.ROI_HEAD_PREFIXES)
           and not torch.allclose(w_before[n], p.detach())]
legacy_chg = [n for n, p in model.named_parameters()
              if not n.startswith(MultiTaskThyroidModel.ROI_HEAD_PREFIXES)
              and not torch.allclose(w_before[n], p.detach())]
print("ROI 头变化数:", len(roi_chg))
print("非 ROI(整图支路)参数变化数(同两轮内):", len(legacy_chg))
print("示例 legacy 变化:", legacy_chg[:3])

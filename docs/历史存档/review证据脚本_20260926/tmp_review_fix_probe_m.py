# -*- coding: utf-8 -*-
"""对抗式审查探针 M:训练循环与 ROI 支路的状态机交互(R1 修复面)。

保留不删(主会话复核依据)。覆盖:
  M1 _collect_cliff_ious 是否自管 eval/还原(与 validate_roi_epoch 的逐模块还原叠加)
  M2 run_roi_epoch 的 model.train()+set_frozen_modules_eval 之后,整图 legacy 路径是否仍正确
  M3 validate_roi_epoch 在 model.training=False 入参下是否逐位还原(不强行置 train)
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

spec = importlib.util.spec_from_file_location("to_m", ROOT / "scripts" / "train_official.py")
to = importlib.util.module_from_spec(spec)
spec.loader.exec_module(to)

import inspect  # noqa: E402
print("=" * 70)
print("M1: _collect_cliff_ious 的 training 态自管")
print("=" * 70)
src = inspect.getsource(to._collect_cliff_ious)
print(src[:1400])

print()
print("=" * 70)
print("M3: validate_roi_epoch 在 eval 入参下逐位还原(不强行置 train)")
print("=" * 70)
from src.models.multi_task_model import MultiTaskThyroidModel  # noqa: E402


class _L:
    def __iter__(self):
        return iter([(torch.rand(2, 1, 64, 64),
                      {"lesion_id": ["a", "b"], "bbox_xyxy": [(0, 0, 8, 8)] * 2,
                       "orig_w": [64, 64], "orig_h": [64, 64],
                       "global_loc": [(0.5, 0.5)] * 2, "is_main": torch.tensor([1, 1]),
                       "malignancy": torch.tensor([1, 0]),
                       "valid_malignancy": torch.tensor([1, 1]),
                       "lesion_loc": torch.tensor([1, 2]),
                       "valid_lesion_loc": torch.tensor([1, 1]),
                       "extrathy_ext": torch.tensor([0, 1]),
                       "valid_extrathy_ext": torch.tensor([1, 1]),
                       "clin_t_stage": torch.tensor([0, 1]),
                       "valid_clin_t_stage": torch.tensor([1, 1]),
                       "tirads": torch.tensor([0, 1]), "valid_tirads": torch.tensor([1, 1]),
                       "patient_key": ["P", "P"]})])


model = MultiTaskThyroidModel(backbone_name="efficientnet_b2", pretrained=False,
                              config={"model": {"backbone": "efficientnet_b2"},
                                      "data": {"img_size": 64}})
# 入参:eval 态(顶层 False)
model.eval()
before = {n: m.training for n, m in model.named_modules()}
out = to.validate_roi_epoch(model, _L(), torch.device("cpu"))
after = {n: m.training for n, m in model.named_modules()}
print("入参 model.training =", model.training, "(期望保持 False)")
print("eval 入参下逐位还原:", before == after, "| roi_n_records =", out["roi_n_records"])
assert before == after
assert model.training is False, "validate_roi_epoch 不应强行置 train"

print()
print("=" * 70)
print("M2: run_roi_epoch 的 train()/set_frozen_modules_eval 与 legacy 路径一致性")
print("=" * 70)
from src.training.loop.bn_freeze import set_frozen_modules_eval  # noqa: E402
crit = _C = type("C", (), {
    "roi_head_weight": 0.5, "ete_class_weights": None, "ete_focal_gamma": 0.0,
    "tirads_class_weights": None, "tirads_label_smoothing": 0.0,
    "tirads_ordinal_weight": 0.0, "roi_concept_weight": 0.0})()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
scaler = torch.amp.GradScaler("cpu", enabled=False)
model.eval()                       # 先置于 eval
to.run_roi_epoch(model, _L(), opt, crit, scaler, torch.device("cpu"), epoch=1,
                 logger=None, log_every=1, head_lr=1e-3, amp_enabled=False)
print("run_roi_epoch 后 model.training =", model.training, "(与整图 run_official_epoch 同款:置 train)")
modes = {n: m.training for n, m in model.named_modules()}
print("frozen 模块被置 eval 的数量:", sum(1 for v in modes.values() if not v))
# 与整图路径逐位对拍:清空后用 set_frozen_modules_eval
model.train()
for m in model.modules():
    m.training = True
set_frozen_modules_eval(model)
modes2 = {n: m.training for n, m in model.named_modules()}
print("两条路径 training 态逐位一致:", modes2 == modes)
for k in list(modes)[:0]:
    pass
diff = [k for k in modes if modes[k] != modes2[k]]
print("差异模块数:", len(diff), diff[:5])

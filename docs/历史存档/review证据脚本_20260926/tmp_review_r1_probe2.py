# -*- coding: utf-8 -*-
"""tmp_review_r1_probe2.py — R1 训练循环编排审查探针 2(证据脚本,保留)。

目的(不触云、不跑全量):
  D. run_roi_epoch 的 optimizer.zero_grad() 时序:实证"ROI 支路 step 之后
     整图支路累积的梯度是否被丢弃"(静态读代码证据 + 动态最小复现)。
  E. 整图 run_official_epoch 的 train_batch 每 batch 内部 zero_grad → ROI 头
     从整图支路拿到的梯度在 ROI epoch 的第一次 zero_grad 就被清零。
  F. ROI val loader 的 shuffle 默认值(是否依赖 train_ds 决定顺序)。
  G. amp=bf16 时 GradScaler(enabled=True) 的行为。
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


print("=" * 78)
print("G. GradScaler 在 bf16 autocast 下 enabled=True 的行为")
sc = torch.amp.GradScaler("cpu", enabled=True)
w = torch.nn.Parameter(torch.tensor([1.0]))
opt = torch.optim.SGD([w], lr=1.0)
with torch.autocast(device_type="cpu", enabled=True, dtype=torch.bfloat16):
    loss = (w * 2.0).sum()          # bf16 autocast 不缩放梯度
sc.scale(loss).backward()
print("  未 unscale 前 w.grad =", w.grad.item(), "(=scale 倍数)")
sc.unscale_(opt)
print("  unscale_ 后  w.grad =", w.grad.item())
sc.step(opt)
sc.update()
print("  step 后 w =", w.item(), "→ 正常更新(scale/unscale 成对,非 no-op 但也无副作用)")

print("=" * 78)
print("D/E. 最小复现:'两个支路共享 optimizer,各自 zero_grad' 的梯度命运")
p_whole = torch.nn.Parameter(torch.tensor([0.0]))   # 整图独有参数
p_roi = torch.nn.Parameter(torch.tensor([0.0]))     # ROI 独有参数
opt = torch.optim.SGD([p_whole, p_roi], lr=1.0)

# --- 整图 epoch 语义:每 batch 内部 zero_grad → backward → step(accumulate 0)---
def whole_batch():
    opt.zero_grad()
    (p_whole * 3.0).sum().backward()
    # 真实 train_batch:/workspace src/training/loop/train_step.py:286 optimizer.zero_grad()
    #                 :334 scaler.scale(loss).backward()
    #                 :342 scaler.step(optimizer)  ← 这里就 step 了
opt.zero_grad()
(p_whole * 3.0).sum().backward()
opt.step()
print("  整图 epoch 结束:p_whole.grad =", None if p_whole.grad is None
      else p_whole.grad.item(), "(train_batch 已 step,梯度已消费)")

# --- ROI epoch 第一个 batch:zero_grad 先执行 -----------------------------
opt.zero_grad()
print("  ROI 第一个 batch 的 optimizer.zero_grad() 后 p_whole.grad =",
      None if p_whole.grad is None else p_whole.grad.item(),
      "→ 整图支路残留梯度在此清零(ROI 头若从整图支路拿梯度也被清零)")
(p_roi * 5.0).sum().backward()
opt.step()
print("  该 batch step 后 p_whole =", p_whole.item(), "p_roi =", p_roi.item())
print("  若 design 声称'epoch 内两支路梯度叠加到同一 optimizer':")
print("    整图支路 1 次 step 已单独消费其梯度;ROI 支路每 batch 都先 zero_grad。")
print("    => 不存在'累积'——两条支路各自 step,互不叠加(实现是串行 SGD,不是累积)。")

print("=" * 78)
print("F. ROI val loader shuffle 默认值")
from src.dataset.roi_dataset import RoiDataset  # noqa: F401
import inspect
from torch.utils.data import DataLoader
sig = inspect.signature(DataLoader.__init__)
print("  DataLoader.shuffle 默认 =", sig.parameters["shuffle"].default)
src = Path(ROOT / "scripts/train_official.py").read_text(encoding="utf-8")
for tag in ["roi_train_loader = DataLoader(", "roi_val_loader = DataLoader("]:
    i = src.find(tag)
    print("  ---", tag)
    print("   ", "\n    ".join(src[i:i + 280].splitlines()))

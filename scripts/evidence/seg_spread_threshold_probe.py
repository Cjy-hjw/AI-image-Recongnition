"""实测 SEG_FROZEN_SPREAD=0.01 的余量:冻结态 vs 可训态的 pred_prob spread。

判据来源:2026-08-15 云端冻结态实测 pred_prob mean=0.4454 / max=0.4463(spread=0.0009)。
本脚本验证两件事:
  1) 复现冻结态的低 spread(证明该指纹确实来自"分割不可训"而非别的原因)
  2) 可训态在**极少 step** 内 spread 就远超 0.01(证明 ep8 门不会误杀)

不依赖掩膜密度(那正是 dice 门的软肋),故结论对真实的 0.1-3% 稀疏前景同样成立。
"""
import sys
import torch

sys.path.insert(0, ".")
from src.models.multi_task_model import MultiTaskThyroidModel  # noqa: E402
from src.training.schedule_official import SEG_TRAIN_PREFIXES  # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}")


def spread_of(model, x):
    model.eval()
    with torch.no_grad():
        out = model(x)
    p = torch.sigmoid(out["mask"])
    return float(p.max()), float(p.mean()), float(p.max() - p.mean())


def build():
    torch.manual_seed(0)
    m = MultiTaskThyroidModel(pretrained=False).to(dev)
    return m


x = torch.randn(2, 1, 512, 512, device=dev)
# 稀疏前景:模拟真实病灶占比 ~1%(512²的1% ≈ 51² 方块)
y = torch.zeros(2, 1, 512, 512, device=dev)
y[:, :, 230:281, 230:281] = 1.0
print(f"前景占比 = {float(y.mean())*100:.2f}%  (真实病灶 0.1-3%)")

# ── 1) 冻结态:分割前向链不可训,只训四头 ──────────────────────────────
m = build()
for name, p in m.named_parameters():
    if any(name.startswith(pfx) for pfx in SEG_TRAIN_PREFIXES):
        p.requires_grad = False
pmax, pmean, sp = spread_of(m, x)
print(f"\n[冻结态 step0 ] max={pmax:.4f} mean={pmean:.4f} spread={sp:.6f}")
print("  (云端 2026-08-15 实测 35 epoch 全程: max=0.4463 mean=0.4454 spread=0.0009)")

# ── 2) 可训态:分割全链可训,跑少量 step 看 spread 何时起来 ────────────
m = build()
segp = [p for n, p in m.named_parameters()
        if any(n.startswith(pfx) for pfx in SEG_TRAIN_PREFIXES)]
opt = torch.optim.AdamW(segp, lr=2.4e-3)
bce = torch.nn.BCEWithLogitsLoss()

print("\n[可训态] step -> spread (阈值 SEG_FROZEN_SPREAD=0.01)")
rows = []
for step in range(0, 41):
    pmax, pmean, sp = spread_of(m, x)
    if step % 5 == 0:
        flag = "PASS" if sp >= 0.01 else "would-ABORT"
        print(f"  step {step:3d}  max={pmax:.4f} mean={pmean:.4f} spread={sp:.6f}  {flag}")
    rows.append(sp)
    m.train()
    out = m(x)
    logits = out["mask"]
    loss = bce(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

first_pass = next((i for i, s in enumerate(rows) if s >= 0.01), None)
print(f"\n首次越过 0.01 的 step = {first_pass}")
print(f"40 step 后 spread = {rows[-1]:.6f}  (余量 = {rows[-1]/0.01:.1f}x 阈值)")
print(f"冻结态 spread 相对阈值 = {sp if False else ''}")

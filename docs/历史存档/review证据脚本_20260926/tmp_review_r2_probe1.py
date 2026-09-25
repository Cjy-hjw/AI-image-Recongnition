# -*- coding: utf-8 -*-
"""R2 review probe (part 1): concept CE math / masking / grad contract.

Evidence for tmp_review_r2_report.md. Keep this file (do not delete).
Run: D:\\program\\Python\\anaconda\\envs\\Demo_D\\python.exe tmp_review_r2_probe1.py
"""
import torch
import torch.nn.functional as F

from src.models.losses.components.concepts_roi import compute_roi_concept_losses
from src.models.losses.components.roi_heads import compute_roi_head_losses

torch.manual_seed(0)


def mk_logits(K, dims=(4, 4, 2, 4)):
    return {
        "logits_comp": torch.randn(K, dims[0], requires_grad=True),
        "logits_echo": torch.randn(K, dims[1], requires_grad=True),
        "logits_shape": torch.randn(K, dims[2], requires_grad=True),
        "logits_margin": torch.randn(K, dims[3], requires_grad=True),
        "logits_foci": torch.randn(K, 4, requires_grad=True),
    }


def labels_for(K, valid_flags, gts=None):
    out = []
    for i in range(K):
        v = valid_flags[i]
        g = (gts[i] if gts else (0, 0, 0, 0))
        out.append({
            "gt_composition": g[0], "valid_composition": v,
            "gt_echogenicity": g[1], "valid_echogenicity": v,
            "gt_shape": g[2], "valid_shape": v,
            "gt_margin": g[3], "valid_margin": v,
            "is_main": 1,
        })
    return out


# ── P1: gradient direction check ───────────────────────────────────────────
print("=== P1 gradient direction ===")
K = 6
concepts = mk_logits(K)
labels = labels_for(K, [1, 1, 1, 1, 1, 1], gts=[(0,)*4]*K)
total, detail = compute_roi_concept_losses({"roi_concepts": concepts}, labels,
                                           lambda_concept=1.0)
total.backward()
g = concepts["logits_comp"].grad
print("comp grad row0:", [round(x, 6) for x in g[0].tolist()])
# analytic: 4 concepts summed, each dloss/dz = (softmax - onehot)/v.sum()
lp = F.log_softmax(concepts["logits_comp"].detach().float(), dim=-1)
sm = lp.exp()
grad_analytic = (sm - F.one_hot(torch.zeros(K, dtype=torch.long), 4).float()) / K
print("comp grad analytic row0:", [round(x, 6) for x in grad_analytic[0].tolist()])
print("max|diff|:", float((g - grad_analytic).abs().max()))
print("detail keys:", sorted(detail.keys()))
print("roi_concept (4-concept sum):", float(detail["roi_concept"]),
      "per-concept:", [float(detail[f"roi_concept_{s}"]) for s in
                       ("comp", "echo", "shape", "margin")])
print("foci grad is None:", concepts["logits_foci"].grad is None)

# ── P2: valid=0 sample gets exactly zero grad ─────────────────────────────
print("\n=== P2 invalid sample zero-grad ===")
torch.manual_seed(1)
K = 4
concepts = mk_logits(K)
labels = labels_for(K, [1, 0, 1, 0])
total, _ = compute_roi_concept_losses({"roi_concepts": concepts}, labels,
                                     lambda_concept=1.0)
total.backward()
g = concepts["logits_comp"].grad
print("grad row1 (invalid) all-zero:", bool((g[1] == 0).all()),
      "row3 (invalid) all-zero:", bool((g[3] == 0).all()))
print("grad row0 (valid) nonzero:", bool(g[0].abs().sum() > 0),
      "row2 (valid) nonzero:", bool(g[2].abs().sum() > 0))
# 对照: 若"缺失样本先算 CE 再乘 v"的写法是否等价 —— 显式重建
lp = F.log_softmax(concepts["logits_comp"].detach().float(), dim=-1)
print("invalid sample placeholder label=0 的 CE 值(未屏蔽时):",
      float(-lp[1, 0]))

# ── P3: k_ref / all-invalid / NaN ─────────────────────────────────────────
print("\n=== P3 k_ref denominator ===")
for nv in (1, 2, 3, 4, 5, 8):
    torch.manual_seed(2)
    K = 8
    c = mk_logits(K)
    flags = [1] * nv + [0] * (K - nv)
    t, _ = compute_roi_concept_losses({"roi_concepts": c},
                                      labels_for(K, flags), lambda_concept=1.0)
    row_sum = float((-F.log_softmax(c["logits_comp"].detach(), -1)[0, 0]))
    print(f"  n_valid={nv}: total/row0 = {float(t)/row_sum:.4f} "
          f"(expect min(n,4)/{min(max(nv,0),4) if nv else 0} ... "
          f"n<4 → 1/4)")
print("  --- all invalid ---")
K = 5
c = mk_logits(K)
t, d = compute_roi_concept_losses({"roi_concepts": c},
                                  labels_for(K, [0] * K), lambda_concept=1.0)
print("  all-invalid total (expect python 0.0 / exact 0):", t)
print("  detail roi_concept:", float(d["roi_concept"]),
      "counts:", {k: v for k, v in d.items() if k.startswith("concept_n")})

# ── P4: NaN logits behaviour ──────────────────────────────────────────────
print("\n=== P4 NaN logits ===")
torch.manual_seed(3)
K = 3
c = mk_logits(K)
c["logits_comp"] = c["logits_comp"].detach().clone()
c["logits_comp"][0, 1] = float("nan")      # valid sample 的 logits 含 NaN
t, d = compute_roi_concept_losses({"roi_concepts": c},
                                  labels_for(K, [1, 1, 1]), lambda_concept=1.0)
print("total with NaN logits on a VALID sample:", float(t))
print("roi_concept:", float(d["roi_concept"]))
# NaN on an INVALID sample
c2 = mk_logits(K)
c2["logits_comp"] = c2["logits_comp"].detach().clone()
c2["logits_comp"][0, 1] = float("nan")
c2["logits_comp"].requires_grad_(True)
t2, d2 = compute_roi_concept_losses({"roi_concepts": c2},
                                    labels_for(K, [0, 1, 1]), lambda_concept=1.0)
print("NaN on an INVALID sample → total:", float(t2),
      "roi_concept:", float(d2["roi_concept"]))
# NaN logits 且该概念全 invalid(其它概念正常)→ 该键 detail 是否 NaN
c3 = mk_logits(K)
c3["logits_comp"] = c3["logits_comp"].detach().clone()
c3["logits_comp"][0, 1] = float("nan")
c3["logits_comp"].requires_grad_(True)
l3 = labels_for(K, [1, 1, 1])
for r in l3:
    r["valid_composition"] = 0
t3, d3 = compute_roi_concept_losses({"roi_concepts": c3}, l3,
                                    lambda_concept=1.0)
print("comp all-invalid + NaN logits → roi_concept_comp:", float(d3["roi_concept_comp"]),
      "| roi_concept(sum):", float(d3["roi_concept"]), "| total:", float(t3))

# ── P5: detail key types / _roi_detail_extra ──────────────────────────────
print("\n=== P5 detail key types ===")
torch.manual_seed(4)
c = mk_logits(4)
_, d = compute_roi_concept_losses({"roi_concepts": c}, labels_for(4, [1, 1, 0, 1]),
                                  lambda_concept=1.0)
for k, v in d.items():
    print(f"  {k}: type={type(v).__name__} value={v if not torch.is_tensor(v) else float(v)}")
# 模拟 _roi_detail_extra(逐键 float(v.item() if hasattr(v,'item') else v))
out = {}
for k, v in d.items():
    try:
        out[f"loss_{k}"] = round(float(v.item() if hasattr(v, "item") else v), 6)
    except (TypeError, ValueError, RuntimeError):
        continue
print("  _roi_detail_extra json keys:", sorted(out.keys()))
print("  has concept keys in json:", [k for k in out if "concept" in k])

# ── P6: lambda_concept=0 zero-regression & no-key fail-loud ───────────────
print("\n=== P6 zero-weight & fail-loud ===")
t0, d0 = compute_roi_concept_losses({}, [], lambda_concept=0.0)
print("lambda=0 empty outputs →", t0, "keys:", sorted(d0.keys()))
try:
    compute_roi_concept_losses({"roi_outputs": []}, labels_for(2, [1, 1]),
                               lambda_concept=1.0)
    print("narrow dict: NO RAISE  ← bug")
except ValueError as e:
    print("narrow dict raises ValueError OK:", str(e)[:60], "...")

# 与四头编排联调:lambda_roi=1 且 concept=0 → 逐位等于不传 concept
print("\n=== P6b lambda_roi=1, concept=0 bitwise regression ===")
outs = {"roi_outputs": [
    {"location_probs": torch.rand(7, requires_grad=True),
     "ete_logits": torch.randn(4, requires_grad=True),
     "t_conf": torch.rand(1, requires_grad=True),
     "tirads_probs": torch.rand(5, requires_grad=True)} for _ in range(3)]}
tgs = {"roi_labels": [
    {"gt_lesion_loc": 0, "valid_lesion_loc": 1, "gt_extrathy_ext": 0,
     "valid_extrathy_ext": 1, "gt_clin_t_stage": 0, "valid_clin_t_stage": 1,
     "gt_tirads": 0, "valid_tirads": 1, "is_main": 1} for _ in range(3)]}
ta, da = compute_roi_head_losses(outs, tgs, lambda_roi=1.0)
tb, db = compute_roi_head_losses(outs, tgs, lambda_roi=1.0, concept_weight=0.0)
print("bitwise total equal:", float(ta) == float(tb))
print("bitwise detail equal:", all(float(da[k]) == float(db[k]) for k in da))
print("keys with concept:", [k for k in da if "concept" in k])

print("\nDONE probe1")

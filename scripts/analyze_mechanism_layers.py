"""unclassified 静态漏检的分层分析(plan 任务 D · 人工分层检查,2026-09-18)。

## 为什么存在

A 轮机制分账(arch_push/detection_diag_20260918_045543):静态漏检 198 个里
96 个(48%)落 unclassified——五机制判据(阈值 0.10/0.15/0.35/0.45/0.5 与
IoU/质心)都不命中,机制分散无多数派,任务 D 优先级表落到"人工分层检查"。

本脚本把 unclassified(及对照:全部可分类漏检)按**连续证据轴**切片,看这批
"判据之外"的漏检聚在哪——三段分层(预注册):

1. **表征段**:GT 内 soft_max<0.10(等于 mech5 判据但 soft_valid=False 的
   采集缺失不算)→ 偏"模型没看见",药方在表征/训练侧;
2. **阈值段**:soft_p95≥0.35(等于 mech1 的软证据半边)→ 偏"概率给了但
   没到阈值/被后处理删",药方在阈值/后处理;
3. **形状段**:0.1<max_iou<0.35(有重叠但远不够)→ 偏欠分割/边缘收缩,
   药方在损失/深监督。

三段互斥(max_iou 分段与 soft 分段可交叉,报告里给交叉表);一段占比过半
即给出首攻方向建议,分散则维持 r5。

## 数据契约

消费 diagnose_detection --mechanism-audit 的 sidecar v2 行(gts[].soft_max/
soft_p95/soft_valid/max_iou/pred_ious/mechanism_primary/area_ratio/is_main)。
v1 行(无 schema_version)跳过并计数。纯离线读 JSONL,无 torch 无 GPU。

## 判读预注册(写死,防事后挑段)

- 表征段(unclassified 内 soft_max<0.10)>40% → 表征路线评估(合并
  mech5 主桶 24.2% 后同向);
- 形状段(max_iou∈(0.1,0.35))>40% → 深监督/形状先验单独立项;
- 阈值段(soft_p95≥0.35)>40% → CC-DiceCE 考虑线(机制1+5 合并口径);
- 无过半段 → 维持 r5 保守收尾(证据不足不硬冲)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 与 detection_audit 同阈值(单一权威:本脚本只消费不重判,阈值仅用于分层
# 切片的连续轴边界,与机制判据数值一致以便对照)。
MECH5_SOFT_MAX = 0.10
MECH1_SOFT_P95 = 0.35
NEAR_MISS_LO = 0.10     # 形状段下界:与 mech2 粘连门槛同值(0.1)
NEAR_MISS_HI = 0.35     # 形状段上界:与 mech1 p95 同值(0.35)


def load_missed_gts(sidecar_path: str) -> tuple[list, dict]:
    """读 sidecar → 全部漏检 GT 行(带池标签),附跳过计数。"""
    missed, skipped_v1, n_rows = [], 0, 0
    with open(sidecar_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            r = json.loads(line)
            if "schema_version" not in r:
                skipped_v1 += 1
                continue
            pool = "dynamic" if r.get("is_dynamic") else "static"
            for g in r.get("gts", []):
                if g.get("matched"):
                    continue
                g = dict(g)
                g["_pool"] = pool
                g["_case"] = r.get("case_key", "")
                missed.append(g)
    stats = {"n_rows": n_rows, "skipped_v1_rows": skipped_v1}
    return missed, stats


def _layer_of(g: dict) -> str | None:
    """unclassified GT 的三段分层(互斥优先级:表征 > 阈值 > 形状)。

    优先级理由:表征段(soft_max<0.10)是最硬的证据——模型输出层面就没
    响应,任何 IoU 证据都是噪声;阈值段次之(概率给了 p95≥0.35 但判据
    因有保留覆盖未命中 mech1——覆盖来源不同);形状段兜底(有重叠但
    两头都不沾)。三段都不落 = 真分散。
    """
    if g.get("soft_valid") and g.get("soft_max") is not None \
            and g["soft_max"] < MECH5_SOFT_MAX:
        return "repr"          # 表征段
    if g.get("soft_valid") and g.get("soft_p95") is not None \
            and g["soft_p95"] >= MECH1_SOFT_P95:
        return "threshold"     # 阈值段
    mi = g.get("max_iou")
    if mi is not None and NEAR_MISS_LO < mi < NEAR_MISS_HI:
        return "shape"         # 形状段
    return "dispersed"         # 真分散(三段都不落)


def _hist(values: list, edges: list) -> dict:
    """值直方图(边界标签 → 计数);values 可含 None(单列 invalid)。"""
    h: dict = {}
    n_invalid = 0
    for v in values:
        if v is None or not np.isfinite(v):
            n_invalid += 1
            continue
        placed = False
        for i in range(len(edges) - 1):
            # 最后一段右闭,其余左闭右开(与 GT-iou 桶风格一致)
            if edges[i] <= v < edges[i + 1] or \
                    (i == len(edges) - 2 and v == edges[i + 1]):
                h[str(edges[i])] = h.get(str(edges[i]), 0) + 1
                placed = True
                break
        if not placed and v >= edges[-1]:
            h[f">={edges[-1]}"] = h.get(f">={edges[-1]}", 0) + 1
    h["invalid"] = n_invalid
    return h


def analyze(missed: list) -> dict:
    """主分析:unclassified 三段分层 + 全漏检对照分布 + 交叉表。"""
    unclassified = [g for g in missed
                    if g.get("mechanism_primary") == "unclassified"]
    by_pool = {"static": {}, "dynamic": {}}
    out = {"n_missed_total": len(missed),
           "n_unclassified": len(unclassified)}

    for pool in ("static", "dynamic"):
        uc = [g for g in unclassified if g["_pool"] == pool]
        allm = [g for g in missed if g["_pool"] == pool]
        layers = [_layer_of(g) for g in uc]
        layer_counts = {k: layers.count(k)
                        for k in ("repr", "threshold", "shape", "dispersed")}
        n_uc = len(uc)
        # 分层占比(分母=该池 unclassified;空时分母记 0 占比全 0)
        layer_ratios = ({k: (v / n_uc if n_uc else 0.0)
                         for k, v in layer_counts.items()})

        # 对照:全部可分类漏检(带 mechanism_primary 桶)的同一分层
        # ——看 unclassified 是否真"判据之外",还是与某桶同分布
        cls_layers = {}
        for g in allm:
            p = g.get("mechanism_primary", "?")
            cls_layers.setdefault(p, []).append(_layer_of(g))

        # 连续轴直方图(unclassified 池内)
        sm = [g.get("soft_max") for g in uc]
        sp = [g.get("soft_p95") for g in uc]
        mi = [g.get("max_iou") for g in uc]
        ar = [g.get("area_ratio") for g in uc]
        by_pool[pool] = {
            "n_miss": len(allm),
            "n_unclassified": n_uc,
            "layer_counts": layer_counts,
            "layer_ratios": layer_ratios,
            # unclassified 的连续轴分布(判读三段的原始证据)
            "soft_max_hist": _hist(sm, [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0]),
            "soft_p95_hist": _hist(sp, [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]),
            "max_iou_hist": _hist(mi, [0.0, 0.10, 0.35, 0.50, 1.0]),
            "area_ratio_hist": _hist(ar, [0.0, 0.0005, 0.001, 0.002, 0.005,
                                          0.01, 0.05, 1.0]),
            # 交叉表:机制主桶 × 三段分层(全漏检;看 unclassified 与
            # mech5/mech1 的分层是否同源)
            "cross_primary_layer": {
                p: {k: lst.count(k)
                    for k in ("repr", "threshold", "shape", "dispersed")}
                for p, lst in sorted(cls_layers.items())
            },
            # 主病灶/次病灶拆分(unclassified 内):次病灶漏检不受分类
            # 评分消费,首攻只对主病灶段负责
            "unclassified_is_main": {
                "main": sum(1 for g in uc if g.get("is_main")),
                "secondary": sum(1 for g in uc if not g.get("is_main")),
            },
        }
    out["pools"] = by_pool

    # 预注册判读(静态池为权威;动态池只参考)
    s = by_pool["static"]
    out["verdict_static"] = {
        "repr_line": s["layer_ratios"]["repr"] > 0.40,
        "shape_line": s["layer_ratios"]["shape"] > 0.40,
        "threshold_line": s["layer_ratios"]["threshold"] > 0.40,
        "dispersed_majority": s["layer_ratios"]["dispersed"] >= 0.50,
        "note": "预注册:任一线过 40% 即定向;dispersed≥50% 维持 r5;"
                "其余组合按三段占比排序取最大者报人工裁决",
    }
    return out


def main():
    ap = argparse.ArgumentParser(
        description="unclassified 静态漏检分层分析(plan 任务 D 人工分层检查)")
    ap.add_argument("--sidecar", required=True,
                    help="diagnose_detection --mechanism-audit 的 cases.jsonl")
    ap.add_argument("--out", default=None, help="分析 JSON 落盘路径")
    args = ap.parse_args()

    missed, stats = load_missed_gts(args.sidecar)
    print(f"[load] sidecar 行={stats['n_rows']} 跳过 v1 行="
          f"{stats['skipped_v1_rows']} 漏检 GT={len(missed)}")
    if stats["n_rows"] == 0:
        raise SystemExit("sidecar 为空——路径错或轮次不对")

    result = analyze(missed)

    # 控制台输出(全部贴回给主会话)
    print("===== unclassified 分层 =====")
    for pool in ("static", "dynamic"):
        p = result["pools"][pool]
        print(f"[{pool}] 漏检={p['n_miss']} unclassified={p['n_unclassified']}")
        if p["n_unclassified"] == 0:
            continue
        print(f"  分层: 表征={p['layer_counts']['repr']} "
              f"({p['layer_ratios']['repr']:.1%}) / "
              f"阈值={p['layer_counts']['threshold']} "
              f"({p['layer_ratios']['threshold']:.1%}) / "
              f"形状={p['layer_counts']['shape']} "
              f"({p['layer_ratios']['shape']:.1%}) / "
              f"分散={p['layer_counts']['dispersed']} "
              f"({p['layer_ratios']['dispersed']:.1%})")
        print(f"  主/次: {p['unclassified_is_main']}")
        print(f"  soft_max 分布: {p['soft_max_hist']}")
        print(f"  soft_p95 分布: {p['soft_p95_hist']}")
        print(f"  max_iou 分布: {p['max_iou_hist']}")
        print(f"  area_ratio 分布: {p['area_ratio_hist']}")
        print(f"  交叉(主桶×分层): {p['cross_primary_layer']}")
    v = result["verdict_static"]
    print("[verdict] 预注册线: "
          f"表征线={v['repr_line']} 阈值线={v['threshold_line']} "
          f"形状线={v['shape_line']} 分散主导={v['dispersed_majority']}")
    print(f"[verdict] {v['note']}")

    out_path = args.out or (
        os.path.splitext(args.sidecar)[0] + "_mech_layers.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()

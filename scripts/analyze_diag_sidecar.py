"""次病灶匹配率归因分析(零训练四件套②,2026-09-10)。

纯 sidecar 后处理、零推理:输入 `detection_diag_{ts}.cases.jsonl`
(`diagnose_detection.py` 既有产物,09-08 全量诊断那份即可复用),
按 is_main 分桶输出主/次病灶的匹配率与 max-IoU 桶分布,
回答 Opus 路线的一个关键归因问题:

    次病灶匹配率 0.577 vs 主病灶 0.869(29pp 缺口)——
    是 min_area 丢小实例(表现为 miss 桶,IoU=0),
    还是连通域合并成大实例(表现为低 IoU 桶 0.1~0.35,IoU 被稀释),
    还是概率被 0.5 阈值抹掉(与 miss 桶同形态,须扫参区分)?

用法:
    python scripts/analyze_diag_sidecar.py --sidecar logs/detection_diag_XXX.cases.jsonl

判定表(输出已按此组织,结论直接可读):
    miss 桶(IoU=0)占比高      → 小实例被 min_area/阈值抹掉 → 件套①④的活
    低 IoU 桶(0.1,0.35)占比高 → 预测与 GT 形状/位置偏差 → 分割质量(640)的活
    临界带 [0.35,0.5) 占比高   → 差一点过线 → 件套①阈值下调的第一目标
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

# 🔴 桶界与 detection_match.IOU_BUCKETS/AREA_BUCKETS 数值**不同**(B路审查P2):
# 本脚本的面积桶(4 桶,按临床面积占比直觉)与上游(5 桶)刻意不同口径,
# 直方图不能与 run_diagnosis 报告的 gt_iou_hist 直接对照——对拍时以本表为准。
# 曾经的注释声称"与上游一致"是错的(复制即漂移源);上游若改桶不影响本表,
# 但若需要与上游逐桶对拍,须 import detection_match 的常量重写此处。
IOU_BUCKETS = [
    (0.0, 0.0, "miss"),           # IoU 恰为 0(无任何重叠候选)
    (0.0, 0.1, "(0,0.1)"),
    (0.1, 0.35, "[0.1,0.35)"),    # 低 IoU:形状/位置偏差或连通域合并稀释
    (0.35, 0.5, "[0.35,0.5)"),    # 临界带:差一点过匹配线
    (0.5, 1.01, "[0.5,1]"),       # 已匹配
]

AREA_BUCKETS = [
    (0.0, 0.001, "<0.1%"),
    (0.001, 0.005, "0.1-0.5%"),
    (0.005, 0.03, "0.5-3%"),
    (0.03, 1.01, ">3%"),
]

# 与上游 detection_match._bucket 的 "invalid" 桶(A-F1 修复)对齐:NaN/负值/
# 越界归 invalid 而非落进最后的命中桶(NaN 在 1.01 上界下会虚增匹配表现)。
_INVALID = "invalid"


def _bucket(v: float, buckets) -> str:
    # NaN/非有限值/越界 → invalid(与上游 A-F1 修复方向一致,不虚增命中桶)。
    if v != v or v < 0.0 or v > buckets[-1][1]:
        return _INVALID
    for lo, hi, name in buckets:
        if name == "miss":
            if v == 0.0:
                return name
        elif lo <= v < hi:
            return name
    return buckets[-1][2]


def analyze(sidecar_path: str) -> dict:
    """读 sidecar → 主/次分桶统计。

    🔴 只接受 run_diagnosis 的 sidecar(detection_diag_*.cases.jsonl)。
    sweep sidecar(sweep_*.cases.jsonl)的 gts 行含 combo 字段且 GT 按组合
    重复出现,喂进来会得出全 miss 假报告(B路审查 P1 的另一半)——入口即拦。
    """
    stats = {
        "main": _empty(), "secondary": _empty(),
    }
    n_anomaly = 0
    with open(sidecar_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "combo" in r:
                raise ValueError(
                    "这是 sweep sidecar(GT 按组合重复、gts 行字段不同),"
                    "本脚本只接受 run_diagnosis 的 detection_diag_*.cases.jsonl")
            if r.get("anomaly"):
                n_anomaly += 1
                continue
            for g in r.get("gts", []):
                key = "main" if g.get("is_main") else "secondary"
                s = stats[key]
                s["n_gt"] += 1
                max_iou = float(g.get("max_iou", 0.0))
                area_ratio = float(g.get("area_ratio", 0.0))
                b_iou = _bucket(max_iou, IOU_BUCKETS)
                b_area = _bucket(area_ratio, AREA_BUCKETS)
                s["iou_hist"][b_iou] = s["iou_hist"].get(b_iou, 0) + 1
                s["iou_area_cross"].setdefault(
                    b_iou, {})[b_area] = s["iou_area_cross"].get(
                    b_iou, {}).get(b_area, 0) + 1
                matched = bool(g.get("matched"))
                s["n_matched"] += int(matched)
    out = {"n_anomaly_cases": n_anomaly, "buckets": stats}
    for key, s in stats.items():
        s["match_rate"] = (s["n_matched"] / s["n_gt"]) if s["n_gt"] else None
    return out


def _empty():
    return {"n_gt": 0, "n_matched": 0, "iou_hist": {}, "iou_area_cross": {}}


def main():
    ap = argparse.ArgumentParser(
        description="次病灶匹配率归因:主/次分桶 + IoU×面积交叉表(纯 sidecar 分析)")
    ap.add_argument("--sidecar", required=True,
                    help="detection_diag_*.cases.jsonl 路径")
    args = ap.parse_args()

    report = analyze(args.sidecar)
    print(f"anomaly cases(已剔除): {report['n_anomaly_cases']}")
    for key in ("main", "secondary"):
        s = report["buckets"][key]
        rate = f"{s['match_rate']:.4f}" if s["match_rate"] is not None else "n/a"
        print(f"\n== {key} == n_gt={s['n_gt']} 匹配率={rate}")
        total = s["n_gt"] or 1
        for name in [b[2] for b in IOU_BUCKETS] + [_INVALID]:
            n = s["iou_hist"].get(name, 0)
            print(f"  {name:12s} {n:5d} ({n / total:.1%})")
        print("  IoU×面积交叉(行=IoU桶,列=面积桶):")
        print(f"  {'':12s}" + "".join(f"{a:>9s}" for a in
                                      [b[2] for b in AREA_BUCKETS]))
        for name in [b[2] for b in IOU_BUCKETS]:
            row = s["iou_area_cross"].get(name, {})
            print(f"  {name:12s}" + "".join(
                f"{row.get(a, 0):>9d}" for a in [b[2] for b in AREA_BUCKETS]))

    # 自动判读(只按占比最高桶给提示,结论仍由人核对)
    print("\n== 判读提示 ==")
    sec = report["buckets"]["secondary"]
    if sec["n_gt"]:
        hist = sec["iou_hist"]
        miss = hist.get("miss", 0) / sec["n_gt"]
        low = hist.get("[0.1,0.35)", 0) / sec["n_gt"]
        crit = hist.get("[0.35,0.5)", 0) / sec["n_gt"]
        print(f"次病灶 miss 占比 {miss:.1%} | 低IoU {low:.1%} | 临界带 {crit:.1%}")
        if miss >= 0.15:
            print("→ miss 为主:min_area/阈值抹掉小实例 —— 件套①(阈值)④(min_area)的第一目标")
        if low >= 0.15:
            print("→ 低IoU 为主:形状/位置偏差或连通域合并稀释 —— 分割质量(640)目标")
        if crit >= 0.10:
            print("→ 临界带可观:阈值下调可直接转匹配 —— 件套①第一目标")


if __name__ == "__main__":
    main()

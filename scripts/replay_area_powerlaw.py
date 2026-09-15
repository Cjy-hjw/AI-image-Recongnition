"""S0 手段3(面积幂律置信度衰减)—— 离线重打分,纯回放 sidecar,零跑分。

用法(云端,workspace 树,不进推理闭包,不影响 verify_closure 41 文件清单):
    cd /2026aicompetition/workspace && \
    PYTHONPATH=/2026aicompetition/workspace/pylibs python -u \
    scripts/replay_area_powerlaw.py \
    --sidecar '/2026aicompetition/workspace/logs/detection_diag_*.cases.jsonl'

只读 sidecar,不改任何文件、不跑模型、不花跑分名额。

## 做什么

对静态池每个预测实例重打分 `s' = s · min(1, (Area/Area_thr)^α)`
(`Area` = `preds[].area` 绝对像素,`s` = `preds[].confidence`),在
α×Area_thr 网格上重算 AP,与基线(原始 conf)相减得 ΔAP。

## ⚠️ 单位陷阱(写死在此,防误判)

`preds[].area` 是**绝对像素**;生产 `nodule_split.py` 的面积判据一律用
**相对占比 area_ratio**(超声图从 ~400×300 到 ~1024×768,面积跨度 6.5×,
绝对阈值在大小图上语义不一致)。本脚本用绝对 px **只能定量级、定 α 形状**
——S0 是**软筛选不是判死门**:第一段全负也不判死,只降手段3优先级;
硬判死与上线参数等 S1 补录 `hw` 后用 area_ratio 复核(两段均 ΔAP≤+0.005
才判死)。见 plan stage4-posthoc-final.md S0 节(Opus P1-1 修订)。

## ⚠️ 只算静态池

动态 `area` 是全帧累计体素(`nodule_split` 刻意选的口径),与静态单帧像素
**不同源**,混进一张网格调出来的 α 必然是错的;动态侧靶子也小(G2:61 个
FP 里仅 10 个 >0.6)。手段3 先做静态专用,与 `_ENV_MIN_AREA_RATIO`
「只作用于静态」的既有先例一致。

## 输出三张表(判读门:两口径 ΔAP 同号>0 且小 TP 不沉底)

1. 网格 ΔAP:全局混排口径 + 逐例平均口径各算一遍(G3 实测两口径差 9pt,
   只在一种口径下为正的一律不稳定);
2. 小 TP 排位位移:面积<Area_thr 且 matched=True 的静态 TP 在全池排序中的
   max ΔRank(正=被压下去;2 个静态小 TP 被推到大量 FP 之后即回调 α);
3. 被压沉的高分 FP 计数:conf>0.6 的小面积 FP 中,重打分后落到
   **全部 TP 之后**(s' < 静态池 TP 最小 conf)的个数——期望盖住 G2 的 70 个。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 复用既有 step-wise AP(n_gt 分母口径,与全部既有诊断同源)——禁手写 AP。
from scripts.detection_match import average_precision  # noqa: E402

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
AREA_THRS = [50, 100, 150, 200, 300]
HI_CONF = 0.6  # G2 判据里的"高分"线


def load_rows(pattern: str) -> list:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"没找到 sidecar: {pattern}")
        sys.exit(1)
    print("sidecar 文件:", files)
    rows = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    print("总行数:", len(rows))
    return rows


def rescore(conf: float, area: float, area_thr: float, alpha: float) -> float:
    return float(conf) * min(1.0, (float(area) / area_thr) ** alpha)


def ap_global(pool: list, scores) -> float:
    """全局混排口径:全池实例一起排序,分母 = 全池 n_gt 总和。"""
    labels, confs, n_gt = [], [], 0
    for r, sc in zip(pool, scores):
        for p, s in zip(r["preds"], sc):
            labels.append(int(bool(p["matched"])))
            confs.append(s)
        n_gt += int(r["n_gt"])
    return average_precision(labels, confs, n_gt)


def ap_percase(pool: list, scores) -> float:
    """逐例平均口径:每例自己算 AP(例内排序)再算术平均。

    排除条件 = 无预测且无 GT 的空例(与 09-15 云端统计 G3 的 by_case 写法
    同口径:`if r['n_pred'] or r['n_gt']`)。**有 GT 无预测的漏检例计入且记
    AP=0**(漏检必须受罚);`n_gt=0 但有 preds` 的例也计入、恒为 0.0——
    它对 ΔAP 无影响(基线与重打分同值),但会拉低 base_percase 绝对值,
    判读时勿把 base_percase 当"模型逐例 AP"引用(审查 Finding 2)。
    """
    vals = []
    for r, sc in zip(pool, scores):
        if not r["preds"] and not r["n_gt"]:
            continue
        labels = [int(bool(p["matched"])) for p in r["preds"]]
        vals.append(average_precision(labels, list(sc), int(r["n_gt"])))
    return sum(vals) / len(vals) if vals else 0.0


def _ranks(scores: list) -> list:
    """0-based 排位(降序,同分按原下标稳定破平)。"""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    r = [0] * len(scores)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def flat_scores(pool: list, score_fn) -> list:
    """每 case 的 preds 分数列表(score_fn(p) 或 None=原始 conf)。"""
    return [[(score_fn(p) if score_fn else float(p["confidence"]))
             for p in r["preds"]] for r in pool]


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--sidecar", required=True,
                     help="cases.jsonl 路径,可用 glob 通配")
    args = ap_.parse_args()

    rows = load_rows(args.sidecar)
    pool = [r for r in rows
            if not r.get("is_dynamic") and not r.get("anomaly")]
    print(f"静态正常行: {len(pool)}(动态/anomaly 行已排除)")

    # ── 基线 ──
    base_global = ap_global(pool, flat_scores(pool, None))
    base_percase = ap_percase(pool, flat_scores(pool, None))
    tp_confs = [float(p["confidence"]) for r in pool
                for p in r["preds"] if p["matched"]]
    n_gt = sum(int(r["n_gt"]) for r in pool)
    n_pred = sum(len(r["preds"]) for r in pool)
    n_tp = len(tp_confs)
    min_tp = min(tp_confs) if tp_confs else float("nan")
    print(f"\n[基线] n_gt={n_gt} n_pred={n_pred} n_tp={n_tp} "
          f"TP_min_conf={min_tp:.4f}")
    print(f"[基线] AP 全局混排={base_global:.6f}  逐例平均={base_percase:.6f}")
    # 🔴 无 TP 的池:沉底判定的基准不存在,网格照出但表3 记 -1(标记无效)
    if n_tp == 0:
        print("[基线] 警告: 静态池零 TP——表3 沉底计数无基准,记 -1")

    # 小 TP 清单(位移监控对象)
    small_tps_all = [(p["area"], float(p["confidence"])) for r in pool
                     for p in r["preds"] if p["matched"]
                     and float(p["area"]) < max(AREA_THRS)]
    print(f"[小TP] area<{max(AREA_THRS)} 的静态 TP 共 {len(small_tps_all)} 个:")
    for a, c in sorted(small_tps_all):
        print(f"    area={a} conf={c:.4f}")

    # 全池展平(排位/沉底判定用),与网格无关的部分只算一次
    flat = [(float(p["confidence"]), float(p["area"]), bool(p["matched"]))
            for r in pool for p in r["preds"]]
    base_rank = _ranks([c for c, _, _ in flat])

    # ── 网格 ──
    print("\n[表1+2+3] alpha  thr  dAP全局   dAP逐例   "
          "小TP_n  小TP_maxΔRank  高分小FP  沉底FP")
    any_positive = False
    for alpha in ALPHAS:
        for thr in AREA_THRS:
            sc = [[rescore(p["confidence"], p["area"], thr, alpha)
                   for p in r["preds"]] for r in pool]
            d_glob = ap_global(pool, sc) - base_global
            d_case = ap_percase(pool, sc) - base_percase

            # 小 TP 排位位移(全池展平排序,重打分只动分数)
            new_scores = [rescore(c, a, thr, alpha) for c, a, _ in flat]
            new_rank = _ranks(new_scores)
            small_tp_idx = [i for i, (_, a, m) in enumerate(flat)
                            if m and a < thr]
            max_dr = max((new_rank[i] - base_rank[i] for i in small_tp_idx),
                         default=0)

            # 高分小 FP 沉底计数(落到全部 TP 之后)。
            # 🔴 基准必须用**重打分后的 TP 最小值**:小面积 TP 的 s' 同样被压,
            # 用原始 min_tp 会在"最低分 TP 恰为小面积"时虚计沉底数。
            new_min_tp = min((new_scores[i] for i, (_, _, m) in enumerate(flat)
                              if m), default=float("inf"))
            hi_fp_idx = [i for i, (c, a, m) in enumerate(flat)
                         if not m and a < thr and c > HI_CONF]
            sunk = (sum(1 for i in hi_fp_idx if new_scores[i] < new_min_tp)
                    if n_tp else -1)  # 零 TP 时基准不存在,-1=无效

            if d_glob > 0 and d_case > 0:
                any_positive = True
            print(f"{alpha:5.1f}  {thr:4d}  {d_glob:+.6f}  {d_case:+.6f}  "
                  f"{len(small_tp_idx):6d}  {max_dr:12d}  "
                  f"{len(hi_fp_idx):8d}  {sunk:5d}")

    print(f"\n[判读] 两口径同号 ΔAP>0 的网格存在: {any_positive}"
          "(主会话判读;云端 agent 不判)")


if __name__ == "__main__":
    main()

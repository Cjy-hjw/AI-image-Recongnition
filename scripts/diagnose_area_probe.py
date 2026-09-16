# -*- coding: utf-8 -*-
"""手段4(双阈值滞回生长)方向探针 —— 只读 sidecar,判分割膨胀方向。

背景:手段4 用低阈值把高阈值种子向外"生长"。它只有在分割**欠**分割时
才有收益;若 matched 预测已经 ≥ GT 面积(过分割),生长只会更糟,当场判死。

🔴 本脚本不跑模型、不读影像/mask、不改任何文件、不花跑分名额——
只消费 09-08 全量诊断的 cases.jsonl(与 S0 重打分同数据源)。

口径说明(与 diagnose_detection 落盘字段对齐):
- 静态池:pred.area = 全分辨率绝对像素;gt 只有 area_ratio(占比)。
  侧车行没有 hw,但**静态 GT mask 与预测 mask 同 hw**(build_iou_matrix 的
  shape 断言保证),故 pred_area / (gt_area_ratio × hw) 不可算;
  但对**同一 case 内**的 matched 对,pred_area_px 与 gt_area_ratio 同图,
  比值 = pred_area_px / (area_ratio × hw) 仍缺 hw。
  → 退而求其次:**同池分位数相对比**——每 case 的 Σpred_area(全匹配)
  与 Σgt_area_ratio 的**归一化比**在 case 间可比(除以各自池中位数),
  再给绝对像素 vs 占比的散点分布,主会话判读。
- 动态池:pred.area = 3D 全帧累计体素,gt.area_ratio = 标注帧均值占比,
  两者口径天然不可比(帧数放大),只给原始分布,不合成比值。

输出:分段统计,stdout 即报告(tee 落文件)。不写任何结论判读——
主会话拿到原始数字后自行判死/放行。

用法:
    python -u scripts/diagnose_area_probe.py \
        --sidecar '/2026aicompetition/workspace/logs/detection_diag_*.cases.jsonl'
"""
from __future__ import annotations

import argparse
import glob
import json
import sys


def _pct(v: list, q: float) -> float:
    if not v:
        return float("nan")
    s = sorted(v)
    i = min(int(q * (len(s) - 1)), len(s) - 1)
    return s[i]


def _fmt(x: float) -> str:
    return f"{x:.6g}" if x == x else "nan"


def _safe_print(msg: str) -> None:
    """GBK 控制台/ascii locale 兜底 print(与 official_loader._safe_print 同患)。

    裸中文 print 在 LANG=C 云端容器首行即崩;emoji 在 GBK 必崩(S0 同款已修
    过一次)。encode 失败降级 ASCII 转写,不静默吞整行。
    """
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(msg.encode(enc, "backslashreplace").decode(enc, "replace"))


def load_rows(pattern: str) -> list:
    files = sorted(glob.glob(pattern))
    if not files:
        _safe_print(f"[FATAL] 没找到 sidecar: {pattern}")
        sys.exit(1)
    _safe_print(f"sidecar 文件: {files}")
    rows = []
    n_bad = 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    n_bad += 1
                    if n_bad <= 3:
                        _safe_print(f"[WARN] {fp}:{ln} 解析失败: {e}")
    _safe_print(f"总行数: {len(rows)}(解析失败 {n_bad})")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True,
                    help="cases.jsonl 路径或 glob")
    args = ap.parse_args()

    rows = load_rows(args.sidecar)
    valid = [r for r in rows if not r.get("anomaly")]
    n_anom = len(rows) - len(valid)
    _safe_print(f"有效行: {len(valid)}(anomaly {n_anom})")

    for pool_name, want_dyn in (("静态", False), ("动态", True)):
        pool = [r for r in valid if r.get("is_dynamic") == want_dyn]
        _safe_print(f"\n===== {pool_name}池({len(pool)} case)=====")

        # A. matched 对的 pred 面积与 gt 占比(原始分布,判读在主会话)
        pred_area_matched, gt_ratio_matched = [], []
        # B. matched GT 的 IoU 分布(过分割的间接信号:IoU<1 的构成)
        iou_matched = []
        # C. 漏检 GT(max_iou 0.35-0.55 临界带——生长最可能救回的对象)
        near_miss_ratios, near_miss_ious = [], []
        # D. 每 case Σpred_area vs Σgt_ratio(同 case 内两口径可比性声明:
        #    静态同图,比值跨 case 可比;动态口径不同源,仅列原始值)
        case_sum_pred, case_sum_gt = [], []

        # D. 每 case 成对收集(matched pred 面积和 vs matched GT 占比和)。
        # 🔴 Gate2 P1(2026-09-16 审查):两列必须**同 case 成对**——旧版收集门
        # `if sp or sg` 会把 sp=0(全漏检)的例灌进 pred 列、sg=0 灌进 gt 列,
        # 两列人口不对应,0 值把中位数拉偏、方向判读被系统性带偏(实测:
        # 一例匹配+一例全漏 → Σpred 中位数=0,看着像灾难性欠分割实为污染)。
        case_sum_pred, case_sum_gt = [], []
        n_d_excluded = 0

        def _f(v):
            # NaN 过滤(Gate2 P2:json.loads 接受 NaN 字面量,sum 混入后
            # 分位全 nan 零告警;上游当前产不出 NaN,这是防线不是预期)。
            f = float(v)
            return f if f == f else None

        for r in pool:
            sp_terms, sg_terms = [], []
            for p in r.get("preds", []):
                if p.get("matched"):
                    v = _f(p.get("area"))
                    if v is not None:
                        sp_terms.append(v)
            for g in r.get("gts", []):
                if g.get("matched"):
                    v = _f(g.get("area_ratio"))
                    if v is not None:
                        sg_terms.append(v)
            if sp_terms and sg_terms:
                case_sum_pred.append(sum(sp_terms))
                case_sum_gt.append(sum(sg_terms))
            else:
                n_d_excluded += 1
            for p in r.get("preds", []):
                if p.get("matched"):
                    v = _f(p.get("area"))
                    if v is not None:
                        pred_area_matched.append(v)
                        bi = _f(p.get("best_iou", 0.0))
                        iou_matched.append(bi if bi is not None else 0.0)
            for g in r.get("gts", []):
                if g.get("matched"):
                    v = _f(g.get("area_ratio"))
                    if v is not None:
                        gt_ratio_matched.append(v)
                else:
                    mi = _f(g.get("max_iou", 0.0))
                    if mi is not None and 0.35 <= mi < 0.55:
                        ar = _f(g.get("area_ratio"))
                        if ar is not None:
                            near_miss_ratios.append(ar)
                        near_miss_ious.append(mi)

        # ── A. matched 面积分布 ──
        _safe_print(f"[A] matched pred_area_px: n={len(pred_area_matched)} "
              f"p25={_fmt(_pct(pred_area_matched, .25))} "
              f"p50={_fmt(_pct(pred_area_matched, .5))} "
              f"p75={_fmt(_pct(pred_area_matched, .75))}")
        _safe_print(f"[A] matched gt_area_ratio: n={len(gt_ratio_matched)} "
              f"p25={_fmt(_pct(gt_ratio_matched, .25))} "
              f"p50={_fmt(_pct(gt_ratio_matched, .5))} "
              f"p75={_fmt(_pct(gt_ratio_matched, .75))}")

        # ── B. matched IoU 分布(边缘质量的直接度量:生长赚 Dice 的上限)──
        _safe_print(f"[B] matched best_iou: n={len(iou_matched)} "
              f"p25={_fmt(_pct(iou_matched, .25))} "
              f"p50={_fmt(_pct(iou_matched, .5))} "
              f"p75={_fmt(_pct(iou_matched, .75))} "
              f"min={_fmt(min(iou_matched) if iou_matched else float('nan'))}")

        # ── C. 临界带漏检(生长的目标受众)──
        _safe_print(f"[C] 漏检 GT 临界带[0.35,0.55): n={len(near_miss_ious)}"
              + (f" iou值={sorted(round(v, 3) for v in near_miss_ious)}"
                 if near_miss_ious else "")
              + (f" 对应gt_area_ratio={sorted(round(v, 6) for v in near_miss_ratios)}"
                 if near_miss_ratios else ""))

        # ── D. case 级 Σpred_area(绝对像素) vs Σgt_ratio(占比)──
        # 静态:同图,若能拿到 hw 可换算绝对值;先给中位数比与散点带。
        _safe_print(f"[D] case Σpred_area_px 中位数={_fmt(_pct(case_sum_pred, .5))} | "
              f"case Σgt_area_ratio 中位数={_fmt(_pct(case_sum_gt, .5))} | "
              f"成对 n={len(case_sum_pred)}(排除单侧为空/NaN 的 case {n_d_excluded})| "
              f"(口径声明: {'同图可比,缺hw不能换算绝对比值' if not want_dyn else '动态口径不同源,不合成比值'})")

    # ── E. hw 字段可用性(决定是否需要补落盘后重跑)──
    # 🔴 Gate2 P1:print 不带 emoji(GBK 控制台必崩,S0 同款已修过一次)且
    # 全部输出经 _safe_print——ascii locale(LANG=C)下裸中文 print 首行即崩,
    # 与 official_loader._safe_print 防的是同一种环境。
    n_hw = sum(1 for r in valid if "hw" in r)
    _safe_print(f"\n[E] sidecar 行含 hw 字段: {n_hw}/{len(valid)}"
                + ("(可算绝对面积比)" if n_hw == len(valid) and n_hw
                   else "(警告: 缺 hw,绝对面积比算不出,见脚本 docstring 口径说明)"))

    _safe_print("\nDONE(以上为原始数字,判读在主会话)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

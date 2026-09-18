"""恒定人口分类分与官方模拟总分(plan 任务 B,2026-09-18)。

## 为什么存在:人口漂移陷阱

检测召回提升后,新找回的疑难病灶进入分类池,得分率天然低于原有匹配人口
——"分类指标跌了"可能只是人口变难,不是模型变差;反之候选丢掉难病例后,
"分类指标升了"是虚假改善。对策 = 恒定人口分类分(S_cls_frozen):以 r5
基线实际匹配 GT 的稳定 ID(gt_key)为固定人口,候选权重与 r5 在**同一批 GT**
上比分类指标——检测端变化(漏检/新找回)单独报告,不污染分类读数。

🔴 **不取基线与候选匹配的交集**(plan 裁决):交集会把"候选丢掉的难病例"
静默剔除,制造虚假改善。固定人口 = r5 匹配集全集,候选对这批 GT 的分类
读数从**整图级分类输出**读(当前架构四头整图级,候选漏检该 GT 时仍可读
该 case 的分类输出做固定人口诊断,同时单独报告该 GT 被漏检)。

## 三维度输出

1. S_det_seg = 15*mean_dice + 15*Micro_AP(满分反推: Dice 15/AP 15)
2. S_cls_frozen:固定人口五项分类指标及加权分(良恶性 18/TI-RADS 12/
   ETE·位置·T 各 40/3——plan 裁决:避免 13.33 累计舍入)
3. S_clin_sim:各候选**自己的**实际匹配人口上的分类+检测分割分(对齐
   交付真实得分的模拟口径,与 frozen 对照着读)

**报告必须声明**:60%/40% 内部子权重官方未公布,以上是按当前权重推定的
模拟分,不是预估得分。

## 分类输出的来源(不额外跑分类头)

diagnose_detection.py --mechanism-audit 已把逐 GT 的临床标签快照(clinical)
写进 sidecar v2 行。本脚本从 sidecar 行构造分类 records,分类预测走
`_shared_nodule_context` 同款编码(含 TI-RADS/ETE 期望取整 _qwk_round,
生产交付口径),复用 metrics_official 的五项指标实现。

🔴 交付评分路径 Kappa 显式消费生产输出硬分类值(_qwk_round 期望取整),
**不是** metrics_official 训练口径的 argmax——训练口径与生产脱节是已知
留档(v21_output._qwk_round docstring),交付模拟必须走生产侧。

## 降级路径(plan 裁决)

生产分类接入受阻时,固定 GT ID 筛训练验证 records 复用
collect_official_records/compute_official_metrics,产物命名
S_cls_frozen_train_proxy,不代替生产链分类门。

## 统计口径

- 配对 bootstrap 按 **PatientId** 抽样(患者级标签数据禁随机切分,泄漏铁律)
  ——重复抽中的患者保留抽样倍数(2 次 = 该患者 GT 计 2 份),不被
  lesion ID 去重吞掉(plan 任务 E 裁决)。
- 任一指标 None/NaN = "无法裁决",不按 0 填充。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 满分反推权重(plan 任务 B 裁决:良恶性18/TI-RADS12/ETE·位置·T 各 40/3)
CLS_WEIGHTS = {
    "malignancy_auc_pr": 18.0,
    "tirads_weighted_kappa": 12.0,
    "ete_weighted_kappa": 40.0 / 3.0,
    "location_weighted_auc": 40.0 / 3.0,
    "clinical_t_auc": 40.0 / 3.0,
}
# 检测分割权重(满分反推:Dice 15/AP 15)
DET_SEG_WEIGHTS = {"mean_dice": 15.0, "micro_ap": 15.0}

# 行级跳过统计(P1-D 配套,再审 2026-09-18):build_classification_records
# 对 anomaly 行 / cls_preds 缺失行的 GT 计数——消费前 reset,报告侧可见,
# 防"候选 anomaly 静默缩小 frozen 分母"不可见。非线程安全(单进程工具)。
ROW_SKIP_STATS = {"anomaly_gt": 0, "no_cls_preds_gt": 0}



def load_sidecar_rows(path: str) -> list:
    """读 diagnose_detection sidecar JSONL(须 v2 行:含 gt_key/clinical)。

    v1 行(无 gt_key)在 frozen 人口匹配时会全部失配——入口处计数并 raise,
    不静默跑出"全人口缺失"的空报告(与 diagnose_detection 的键名白名单
    入口拦截同哲学)。
    """
    rows = []
    n_v1 = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "schema_version" not in r:
                n_v1 += 1
            rows.append(r)
    if not rows:
        # 再审 P2(2026-09-18):空文件此前静默返回 [] → frozen_n=0 的空报告
        # (docstring 声称 raise 却只拦了全 v1)。
        raise ValueError(f"{path} 为空文件——没有可消费的行,检查路径/轮次。")
    if n_v1 and n_v1 == len(rows):
        raise ValueError(
            f"{path} 全部为 v1 行(无 --mechanism-audit)。重跑诊断加 "
            "--mechanism-audit 才有 clinical/gt_key 字段。")
    if n_v1:
        # 扫参行混入(v1)——只消费 v2 诊断行,计数提示。
        rows = [r for r in rows if "schema_version" in r]
    return rows


def build_frozen_population(baseline_rows: list) -> set:
    """基线(r5)实际匹配 GT 的稳定 ID 集合 = 固定人口。

    🔴 不取交集(plan 裁决):人口 = 基线匹配集**全集**;候选丢掉的难病例
    留在人口里(其分类读数从整图级输出读),检测端漏检单独报告。
    anomaly 行的 GT 不进人口(官方口径下推理失败病例全 GT 漏检,但基线
    与候选的 anomaly 集可能不同,进人口会把推理失败混进分类比较)。
    """
    pop = set()
    for r in baseline_rows:
        if r.get("anomaly"):
            continue
        for g in r.get("gts", []):
            if g.get("matched") and g.get("gt_key"):
                pop.add(g["gt_key"])
    return pop


def build_classification_records(rows: list, frozen_pop: set | None) -> list:
    """sidecar 行 → metrics_official 兼容的分类 records。

    每条 GT 一条 record(frozen_pop 给定时只留人口内 GT)。
    消费 sidecar v2 的 gts[i] 字段:
      - 标签:clinical.{malignancy,valid_malignancy,lesion_loc,valid_lesion_loc,
        extrathy_ext,valid_extrathy_ext,clin_t_stage,valid_clin_t_stage,
        tirads,valid_tirads} + is_main
      - 预测:case 级分类概率(cls_preds:mal_probs/ete_probs/location_probs/
        t_conf/tirads_probs——由 diagnose_detection 在机制采集 try 内写入;
        含 NaN 时在行级校验抛 ValueError,不把 NaN 带进 sklearn)

    预测硬分类值(交付口径)在 compute_delivery_classification_metrics 里算
    (_qwk_round / argmax+1),record 里只存概率,与 metrics_official 的
    消费契约一致(它吃 probs 自己 argmax——交付口径单独算,见下)。

    行级异常统计(P1-D 配套):anomaly 行 / cls_preds 缺失行各自的 GT 数
    记入本模块级计数器 `ROW_SKIP_STATS`(调用方可读,报告口径可见),
    不再静默消失——docstring 此前声称"单独计数"但零实现(再审 P2 修复)。
    """
    out = []
    for r in rows:
        if r.get("anomaly"):
            # P1-D(再审 2026-09-18):anomaly 行的 GT 人口计数显式入
            # 统计(此前静默 continue,frozen 分母被候选侧缩小不可见)。
            _n = int(r.get("n_gt") or 0)
            ROW_SKIP_STATS["anomaly_gt"] += _n
            continue
        cls_preds = r.get("cls_preds")
        if cls_preds is None:
            ROW_SKIP_STATS["no_cls_preds_gt"] += int(r.get("n_gt") or 0)
            continue        # 无分类输出采集的行:不进 records(计数上行)
        for g in r.get("gts", []):
            key = g.get("gt_key")
            if not key:
                continue
            if frozen_pop is not None and key not in frozen_pop:
                continue
            cl = g.get("clinical") or {}
            out.append({
                "lesion_id": key,
                "patient_id": g.get("patient_id", ""),
                "is_main": int(bool(g.get("is_main"))),
                "gt_key": key,
                "matched": bool(g.get("matched")),
                "mal_prob": float(cls_preds["mal_probs"][1]),
                "gt_malignancy": int(cl.get("malignancy", 0)),
                "valid_malignancy": int(cl.get("valid_malignancy", 0)),
                "location_probs": [float(x) for x in cls_preds["location_probs"]],
                # 🔴 再审 P0(2026-09-18):clinical 快照的 lesion_loc 是
                # xlsx **码值 1..7**,而 location_probs 列是 0-based idx
                # (码-1,location_label_to_index 契约)。指标函数的
                # probs[:, present] 把 present 当列索引用,须喂 0-based
                # ——直接抄码会 off-by-one 取到错误列,AUC 系统性失真
                # (训练侧 metrics_official 之所以对,是它的 records 里
                # gt_lesion_loc 本就是 idx)。
                "gt_lesion_loc": int(cl.get("lesion_loc", 0)) - 1
                if int(cl.get("lesion_loc", 0)) >= 1 else 0,
                "valid_lesion_loc": int(cl.get("valid_lesion_loc", 0)),
                "ete_probs": [float(x) for x in cls_preds["ete_probs"]],
                "gt_extrathy_ext": int(cl.get("extrathy_ext", 0)),
                "valid_extrathy_ext": int(cl.get("valid_extrathy_ext", 0)),
                "t_conf": float(cls_preds["t_conf"]),
                "gt_clin_t_stage": int(cl.get("clin_t_stage", 0)),
                "valid_clin_t_stage": int(cl.get("valid_clin_t_stage", 0)),
                "tirads_probs": [float(x) for x in cls_preds["tirads_probs"]],
                "gt_tirads": int(cl.get("tirads", 0)),
                "valid_tirads": int(cl.get("valid_tirads", 0)),
            })
    return out



def compute_delivery_classification_metrics(records: list) -> dict:
    """五项分类指标:**交付口径**(Kappa 消费 _qwk_round 期望取整硬分类值)。

    与 metrics_official 训练口径的差异(有意,勿"修齐"):训练侧 Kappa 用
    argmax,生产 prediction.json 用 _qwk_round(TI-RADS/ETE)+argmax+1
    (Location)+argmax(T)——交付模拟必须走生产侧,否则本地读数与线上
    脱节(v21_output._qwk_round docstring 留档的已知脱节)。

    AUC 系(良恶性/位置/T)吃概率,与训练口径无决策差异,直接复用
    metrics_official 的实现(AUC-PR/加权宏 AUC/t_conf 排序)。
    Kappa 系(TI-RADS/ETE)用本函数自算(生产决策 + cohen_kappa_score)。
    按主病灶/validity/恶性条件各自筛选(与 metrics_official 同判据),
    每项指标报自己的有效分母(n_valid)。
    """
    from sklearn.metrics import (average_precision_score, cohen_kappa_score,
                                 roc_auc_score)
    from src.models.heads import NUM_ETE_CLASSES, NUM_TIRADS_CLASSES
    from src.inference.v21_output import _qwk_round

    # 再审 P2(2026-09-18):直接复用生产 _qwk_round(单一权威),废弃本地
    # 手抄副本——银行家舍入语义易抄错(v21_output docstring 专警告过),
    # 20k 点对拍虽逐位一致,复制件不随生产演化是漂移面。lo 参数无人用,删。
    _delivery_hard = _qwk_round

    out = {}
    # ── 良恶性 AUC-PR:主病灶 + valid(复用训练侧公式) ────────────────────
    sel_mal = [r for r in records
               if r["is_main"] == 1 and r["valid_malignancy"] == 1]
    if len(sel_mal) >= 2 and len({r["gt_malignancy"] for r in sel_mal}) >= 2:
        out["malignancy_auc_pr"] = float(average_precision_score(
            [r["gt_malignancy"] for r in sel_mal],
            [r["mal_prob"] for r in sel_mal]))
    else:
        out["malignancy_auc_pr"] = None
    out["n_valid_malignancy"] = len(sel_mal)

    # ── 位置 加权宏 AUC:主病灶 + valid(二类分支与训练侧同防御) ──────────
    sel_loc = [r for r in records
               if r["is_main"] == 1 and r["valid_lesion_loc"] == 1]
    y_loc = [r["gt_lesion_loc"] for r in sel_loc]
    present = sorted(set(y_loc))
    if len(present) >= 2:
        probs = np.asarray([r["location_probs"] for r in sel_loc], np.float64)
        probs = probs[:, present]
        rs = probs.sum(axis=1, keepdims=True)
        probs = np.where(rs > 0, probs / np.maximum(rs, 1e-12),
                         1.0 / len(present))
        remap = {c: i for i, c in enumerate(present)}
        ym = [remap[v] for v in y_loc]
        if len(present) == 2:
            out["location_weighted_auc"] = float(roc_auc_score(ym, probs[:, 1]))
        else:
            out["location_weighted_auc"] = float(roc_auc_score(
                ym, probs, multi_class="ovr", average="weighted"))
    else:
        out["location_weighted_auc"] = None
    out["n_valid_location"] = len(sel_loc)

    # ── TI-RADS 加权 Kappa:主病灶 + valid,**不限恶性**;交付决策 _qwk_round ──
    sel_tir = [r for r in records
               if r["is_main"] == 1 and r["valid_tirads"] == 1]
    if len(sel_tir) >= 2:
        y_true = [r["gt_tirads"] for r in sel_tir]
        # 🔴 交付口径:_qwk_round 输出码 1..5(gt 0-based)→ 减 1 回 0-based
        # 对齐 labels;argmax 口径(metrics_official)与它系统性不同,这是
        # 有意的交付对齐(测试钉死 argmax≠_qwk_round 的构造场景)。
        y_pred = [_delivery_hard(r["tirads_probs"]) - 1 for r in sel_tir]
        if not (len(set(y_true)) < 2 and len(set(y_pred)) < 2):
            k = cohen_kappa_score(y_true, y_pred,
                                  labels=list(range(NUM_TIRADS_CLASSES)),
                                  weights="quadratic")
            out["tirads_weighted_kappa"] = None if np.isnan(k) else float(k)
        else:
            out["tirads_weighted_kappa"] = None
    else:
        out["tirads_weighted_kappa"] = None
    out["n_valid_tirads"] = len(sel_tir)

    # ── ETE 加权 Kappa:主病灶 + valid + **恶性**(金标准);交付决策 _qwk_round ──
    sel_ete = [r for r in records
               if r["is_main"] == 1 and r["valid_extrathy_ext"] == 1
               and r["gt_malignancy"] == 1]
    if len(sel_ete) >= 2:
        y_true = [r["gt_extrathy_ext"] for r in sel_ete]
        y_pred = [_delivery_hard(r["ete_probs"]) - 1 for r in sel_ete]
        if not (len(set(y_true)) < 2 and len(set(y_pred)) < 2):
            k = cohen_kappa_score(y_true, y_pred,
                                  labels=list(range(NUM_ETE_CLASSES)),
                                  weights="quadratic")
            out["ete_weighted_kappa"] = None if np.isnan(k) else float(k)
        else:
            out["ete_weighted_kappa"] = None
    else:
        out["ete_weighted_kappa"] = None
    out["n_valid_ete"] = len(sel_ete)

    # ── T 临床显著性 AUC:主病灶 + valid + 恶性;t_conf 排序(无决策差异) ──
    sel_t = [r for r in records
             if r["is_main"] == 1 and r["valid_clin_t_stage"] == 1
             and r["gt_malignancy"] == 1]
    y_t = [r["gt_clin_t_stage"] for r in sel_t]
    if len(y_t) >= 2 and len(set(y_t)) >= 2:
        out["clinical_t_auc"] = float(roc_auc_score(
            y_t, [r["t_conf"] for r in sel_t]))
    else:
        out["clinical_t_auc"] = None
    out["n_valid_t"] = len(sel_t)
    return out



def compute_score_views(baseline_rows: list, candidate_rows: list,
                        det_seg_baseline: dict | None = None,
                        det_seg_candidate: dict | None = None) -> dict:
    """三维度总分视图(frozen / clin_sim / det_seg)。

    Args:
        baseline_rows / candidate_rows: sidecar v2 行(候选可与基线同源——
            r5 自对拍时传同一份两遍)。
        det_seg_baseline / det_seg_candidate: {"mean_dice", "micro_ap"} 的
            池级读数(由诊断报告 side 供给;本函数不重算检测端,专注分类端
            组装——检测口径单一权威在 diagnose_detection)。

    返回:
        {
          "S_det_seg": {run: 分数},
          "S_cls_frozen": {run: {指标: 值, "score": 加权分}},
          "S_clin_sim": {run: 同上(各自实际匹配人口)},
          "frozen_population_n": 人口数,
          "population_delta": {added/removed GT 明细(候选 vs 基线匹配集)},
        }
    任一指标 None → 该 run 的加权分记 None("无法裁决"),不按 0 填充。
    """
    frozen = build_frozen_population(baseline_rows)

    # P1-D(再审 2026-09-18):frozen 侧与候选侧的行级跳过计数分 run 汇总,
    # 防"候选 anomaly 静默缩小分母"不可见——records 数不等时报告里能直接
    # 看到是哪一侧、多少 GT 被 anomaly/无采集剔掉。
    skip_by_run = {}
    for name, rows in (("baseline", baseline_rows),
                       ("candidate", candidate_rows)):
        ROW_SKIP_STATS["anomaly_gt"] = 0
        ROW_SKIP_STATS["no_cls_preds_gt"] = 0
        # 触发计数:全量构造(不筛 frozen,与行级跳过语义一致)
        build_classification_records(rows, None)
        skip_by_run[name] = dict(ROW_SKIP_STATS)

    def _frozen_metrics(rows):
        recs = build_classification_records(rows, frozen)
        return compute_delivery_classification_metrics(recs), len(recs)

    out = {
        "frozen_population_n": len(frozen),
        "row_skip_stats": skip_by_run,
        "S_det_seg": {},
        "S_cls_frozen": {},
        "S_clin_sim": {},
        "population_delta": {},
    }
    for name, rows, ds in (("baseline", baseline_rows, det_seg_baseline),
                           ("candidate", candidate_rows, det_seg_candidate)):
        # S_det_seg(检测分割视图:外部供给的池级读数直算)
        if ds and ds.get("mean_dice") is not None and ds.get("micro_ap") is not None:
            out["S_det_seg"][name] = (
                DET_SEG_WEIGHTS["mean_dice"] * float(ds["mean_dice"])
                + DET_SEG_WEIGHTS["micro_ap"] * float(ds["micro_ap"]))
        else:
            out["S_det_seg"][name] = None

        # S_cls_frozen(固定人口)
        metrics_f, n_f = _frozen_metrics(rows)
        metrics_f["n_records"] = n_f
        out["S_cls_frozen"][name] = metrics_f

        # S_clin_sim(各自实际匹配人口:plan 定义——候选自己的匹配集)
        recs_c = build_classification_records(rows, None)
        matched_recs = [r for r in recs_c if r["matched"]]
        metrics_c = compute_delivery_classification_metrics(matched_recs)
        metrics_c["n_records"] = len(matched_recs)
        out["S_clin_sim"][name] = metrics_c

    # 加权分(None 传染 → 无法裁决)
    for view in ("S_cls_frozen", "S_clin_sim"):
        for name, m in out[view].items():
            vals = [m[k] for k in CLS_WEIGHTS if m.get(k) is not None]
            out[view][name]["score"] = (
                sum(CLS_WEIGHTS[k] * m[k] for k in CLS_WEIGHTS
                    if m.get(k) is not None)
                if len(vals) == len(CLS_WEIGHTS) else None)

    # 人口变化明细(候选 vs 基线匹配集;找回/丢失 GT 各自报,不只净差)
    cand_matched = {g["gt_key"] for r in candidate_rows
                    if not r.get("anomaly")
                    for g in r.get("gts", [])
                    if g.get("matched") and g.get("gt_key")}
    base_matched = frozen
    out["population_delta"] = {
        "recovered": sorted(cand_matched - base_matched),
        "lost": sorted(base_matched - cand_matched),
    }
    return out


def paired_patient_bootstrap(baseline_records: list, candidate_records: list,
                             n_boot: int = 1000, seed: int = 42) -> dict:
    """分类指标差值的配对 bootstrap(按 PatientId 抽样)。

    🔴 抽样单位 = PatientId(患者级标签泄漏铁律);重复抽中的患者**保留
    倍数**(其全部 GT 计入 k 次,不被 lesion 去重吞掉——plan 任务 E 裁决)。
    差值 = candidate − baseline(正 = 候选更好);每项指标独立算
    95% 区间与 P(Δ>0)。输入须同一人口(调用方先筛 frozen),否则差值
    掺人口效应失去配对意义(入口校验 gt_key 集合相等)。
    """
    base_keys = {r["gt_key"] for r in baseline_records}
    cand_keys = {r["gt_key"] for r in candidate_records}
    if base_keys != cand_keys:
        raise ValueError(
            f"配对 bootstrap 须同一人口:基线 {len(base_keys)} vs 候选 "
            f"{len(cand_keys)}(不相等 {len(base_keys ^ cand_keys)} 个)——"
            "先按 frozen 人口筛齐再调用")
    by_pid_base = {}
    by_pid_cand = {}
    for r in baseline_records:
        by_pid_base.setdefault(r["patient_id"], []).append(r)
    for r in candidate_records:
        by_pid_cand.setdefault(r["patient_id"], []).append(r)
    pids = sorted(set(by_pid_base) | set(by_pid_cand))
    rng = np.random.default_rng(seed)

    def _metrics(recs):
        m = compute_delivery_classification_metrics(recs)
        return {k: m[k] for k in CLS_WEIGHTS}

    boot = {k: [] for k in CLS_WEIGHTS}
    for _ in range(n_boot):
        sample_pids = rng.choice(len(pids), size=len(pids), replace=True)
        rb, rc = [], []
        for i in sample_pids:        # 重复患者保留倍数(标签随患者,天然相同)
            rb.extend(by_pid_base.get(pids[i], []))
            rc.extend(by_pid_cand.get(pids[i], []))
        mb = _metrics(rb)
        mc = _metrics(rc)
        for k in CLS_WEIGHTS:
            if mb[k] is not None and mc[k] is not None:
                boot[k].append(mc[k] - mb[k])
    out = {}
    for k, diffs in boot.items():
        if not diffs:
            out[k] = {"mean": None, "ci95": None, "p_positive": None,
                      "n_boot_valid": 0}
            continue
        a = np.asarray(diffs)
        out[k] = {
            "mean": float(a.mean()),
            "ci95": [float(np.percentile(a, 2.5)),
                     float(np.percentile(a, 97.5))],
            "p_positive": float((a > 0).mean()),
            "n_boot_valid": int(a.size),
        }
    return out



def _summarize_delta(a: float | None, b: float | None) -> str:
    """b−a 的简读(null 安全)。"""
    if a is None or b is None:
        return "N/A"
    d = b - a
    return f"{d:+.4f}"


def main():
    ap = argparse.ArgumentParser(
        description="恒定人口分类分与官方模拟总分(plan 任务 B)")
    ap.add_argument("--baseline", required=True,
                    help="基线 sidecar JSONL(r5,须 --mechanism-audit 产物)")
    ap.add_argument("--candidate", default=None,
                    help="候选 sidecar JSONL;缺省=基线自对拍(校验零差)")
    ap.add_argument("--baseline-det-seg", default=None,
                    help="JSON 文件,含 {mean_dice, micro_ap}(诊断报告提取);"
                         "缺省则 S_det_seg 记 None")
    ap.add_argument("--candidate-det-seg", default=None)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help=">0 时跑配对 bootstrap(次数;按 PatientId 抽样)")
    ap.add_argument("--out", default=None, help="报告 JSON 落盘路径")
    args = ap.parse_args()

    baseline_rows = load_sidecar_rows(args.baseline)
    candidate_rows = (load_sidecar_rows(args.candidate)
                      if args.candidate else baseline_rows)

    def _load_ds(path):
        if not path:
            return None
        d = json.load(open(path, encoding="utf-8"))
        return {"mean_dice": d.get("mean_dice"), "micro_ap": d.get("micro_ap")}

    views = compute_score_views(
        baseline_rows, candidate_rows,
        det_seg_baseline=_load_ds(args.baseline_det_seg),
        det_seg_candidate=_load_ds(args.candidate_det_seg))

    report = {
        "meta": {
            "baseline": os.path.abspath(args.baseline),
            "candidate": (os.path.abspath(args.candidate)
                          if args.candidate else "self-pair(baseline)"),
            "caveats": [
                "60%/40% 内部子权重官方未公布,S_* 为按当前权重推定的模拟分,"
                "不是预估得分",
                "S_cls_frozen 人口 = 基线匹配 GT 全集(不取交集);候选漏检的 "
                "GT 分类读数来自整图级输出,检测端漏检另见 population_delta",
                "Kappa 为交付口径(_qwk_round 期望取整),非训练侧 argmax",
                "任一指标 None = 无法裁决(样本不足/单类),不按 0 填充",
            ],
        },
        **views,
    }
    if args.bootstrap > 0:
        recs_b = build_classification_records(baseline_rows, None)
        recs_c = build_classification_records(candidate_rows, None)
        # bootstrap 在全量(非 frozen)人口上不跑——须同人口;用 frozen 筛。
        # P1-D(再审):两侧 frozen 筛后不相等(候选 anomaly 缺采集)时不再
        # 裸崩丢整份报告——先落盘主报告,再以 bootstrap_error 记失败原因,
        # 退出码 1(可见失败优于静默,但报告本体仍可读)。
        pop = build_frozen_population(baseline_rows)
        rb = [r for r in recs_b if r["gt_key"] in pop]
        rc = [r for r in recs_c if r["gt_key"] in pop]
        try:
            report["bootstrap"] = paired_patient_bootstrap(
                rb, rc, n_boot=args.bootstrap)
        except ValueError as e:
            report["bootstrap_error"] = str(e)

    out_path = args.out or (
        os.path.splitext(args.baseline)[0] + "_cmp.json")
    with open(out_path, "w", encoding="utf-8") as f:
        # 再审 P2:严格 JSON 落盘(allow_nan=False)——默认 True 会写字面
        # NaN,严格解析器(Go/Rust/Jackson)拒读整份报告。指标侧 None
        # 语义已保证非 NaN;若真有 NaN 漏进来,此处 ValueError 属于
        # "该炸就炸"(数据坏了该修采集,不是序列化兜底)。
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    # 控制台摘要
    print("===== 恒定人口分类分 =====")
    print(f"frozen 人口: {views['frozen_population_n']}")
    for view in ("S_det_seg", "S_cls_frozen", "S_clin_sim"):
        b = views[view].get("baseline", {})
        c = views[view].get("candidate", {})
        if view == "S_det_seg":
            print(f"[{view}] baseline={b} candidate={c}")
            continue
        print(f"[{view}]")
        for k in list(CLS_WEIGHTS) + ["score", "n_records"]:
            print(f"  {k:24s} base={b.get(k)} cand={c.get(k)}"
                  f"  Δ={_summarize_delta(b.get(k), c.get(k))}")
    if views["population_delta"]:
        d = views["population_delta"]
        print(f"[population] 找回={len(d['recovered'])} "
              f"丢失={len(d['lost'])}")
    if "bootstrap" in report:
        print("[bootstrap]")
        for k, v in report["bootstrap"].items():
            print(f"  {k:24s} Δmean={v['mean']} "
                  f"ci95={v['ci95']} P(Δ>0)={v['p_positive']}")
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()

"""tests for scripts/compare_detection_runs.py(plan 任务 B,2026-09-18)。

覆盖 plan 任务 B 测试清单:
- 固定人口在"新增难主病灶/原有漏检/次病灶标签/非恶性/单类别"下的分母与筛选
- 构造概率让 argmax≠期望取整钉死交付接线(_qwk_round vs argmax)
- frozen 人口 = 基线匹配集全集(不取交集);候选丢难病例不造成虚假改善
- S_det_seg 组装 / population_delta 找回丢失各报 / bootstrap 人口校验
- None 指标不按 0 填充(score=None "无法裁决")
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.compare_detection_runs import (
    CLS_WEIGHTS,
    build_classification_records,
    build_frozen_population,
    compute_delivery_classification_metrics,
    compute_score_views,
    load_sidecar_rows,
    paired_patient_bootstrap,
)


# ── fixture 构造工具 ───────────────────────────────────────────────
def _clinical(**over):
    """GT 临床标签快照(五项 + valid_* 全 1;malignancy=1 恶性)。"""
    base = {
        "malignancy": 1, "valid_malignancy": 1,
        "lesion_loc": 2, "valid_lesion_loc": 1,
        "extrathy_ext": 1, "valid_extrathy_ext": 1,
        "clin_t_stage": 1, "valid_clin_t_stage": 1,
        "tirads": 3, "valid_tirads": 1,
    }
    base.update(over)
    return base


def _cls_preds(mal=0.7, tirads=None, ete=None, loc=None, t_conf=0.6):
    tirads = tirads if tirads is not None else [0.1, 0.2, 0.4, 0.2, 0.1]
    ete = ete if ete is not None else [0.7, 0.2, 0.1, 0.0]
    loc = loc if loc is not None else [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return {
        "mal_probs": [1.0 - mal, mal],
        "ete_probs": list(ete),
        "location_probs": list(loc),
        "t_conf": t_conf,
        "tirads_probs": list(tirads),
    }


def _gt_row(patient, lesion, *, matched=True, is_main=True, clinical=None):
    return {
        "gt_key": f"{patient}::{lesion}",
        "patient_id": patient,
        "lesion_id": lesion,
        "is_main": is_main,
        "matched": matched,
        "clinical": clinical if clinical is not None else _clinical(),
    }


def _case_row(patient, gts, *, is_dynamic=False, anomaly=None, cls_preds="default"):
    """case 行;cls_preds='default' 给标准概率,None 表示该行无采集。"""
    return {
        "case_key": f"/{patient}/S1", "is_dynamic": is_dynamic,
        "anomaly": anomaly, "n_pred": 0, "n_gt": len(gts),
        "schema_version": 2,
        "cls_preds": _cls_probs() if cls_preds == "default" else cls_preds,
        "preds": [], "gts": gts,
    }


def _cls_probs():
    return _cls_preds()


def _write_sidecar(tmp_path, rows, name="sidecar.jsonl"):
    p = tmp_path / name
    p.write_text(chr(10).join(json.dumps(r, ensure_ascii=False) for r in rows),
                 encoding="utf-8")
    return str(p)


# ── 人口构造与筛选 ──────────────────────────────────────────
def test_frozen_population_is_baseline_matched_fullset():
    """frozen = 基线匹配集全集(含次病灶);anomaly 行 GT 不进。"""
    rows = [
        _case_row("P1", [_gt_row("P1", "M", matched=True),
                         _gt_row("P1", "L2", matched=False)]),   # 匹配+漏检
        _case_row("P2", [_gt_row("P2", "M", matched=True)]),
        _case_row("P3", [_gt_row("P3", "M", matched=True)], anomaly="推理失败"),  # anomaly 不进
    ]
    pop = build_frozen_population(rows)
    assert pop == {"P1::M", "P2::M"}
    assert "P1::L2" not in pop and "P3::M" not in pop


def test_records_filter_frozen_and_secondary_lesion():
    """固定人口筛选:人口外 GT 不进 records;次病灶进 records
    (is_main=0 供指标自筛,不在构造时剔除)。"""
    rows = [
        _case_row("P1", [_gt_row("P1", "M", matched=True),
                         _gt_row("P1", "L2", is_main=False, matched=True)]),
        _case_row("P2", [_gt_row("P2", "M", matched=False)]),
        _case_row("PX", [_gt_row("PX", "M", matched=True)]),      # 人口外
    ]
    pop = build_frozen_population(rows)
    # P2::M 基线未匹配 → 不在基线匹配集(=frozen 人口);
    # 漏检 GT 的固定人口读数由 _two_run_rows 场景钉(基线 matched=True)。
    recs = build_classification_records(rows, pop)
    keys = {r["gt_key"] for r in recs}
    assert keys == pop == {"P1::M", "P1::L2", "PX::M"}
    r_sec = next(r for r in recs if r["gt_key"] == "P1::L2")
    assert r_sec["is_main"] == 0


def test_records_missing_cls_preds_excluded():
    """cls_preds=None(未采集)的行不进 records(迭代防口,单独计数)。"""
    rows = [_case_row("P1", [_gt_row("P1", "M")], cls_preds=None)]
    recs = build_classification_records(rows, None)
    assert recs == []


def test_malicious_v1_rows_rejected(tmp_path):
    """v1 行(0.70 无 schema_version)全文 raise;混入则只消费 v2 行。"""
    v1_row = {"case_key": "/a", "gts": [], "preds": []}
    v2_row = _case_row("P1", [_gt_row("P1", "M")])
    # 全 v1
    p = _write_sidecar(tmp_path, [v1_row], name="all_v1.jsonl")
    with pytest.raises(ValueError, match="v1"):
        load_sidecar_rows(p)
    # 混入(v2+v1)：扫参行混入场景
    p2 = _write_sidecar(tmp_path, [v2_row, v1_row], name="mixed.jsonl")
    rows = load_sidecar_rows(p2)
    assert len(rows) == 1 and rows[0].get("schema_version") == 2


# ── 指标筛选与分母 ───────────────────────────────────────────
def _make_records(n, mal=0.9, gt_mal=1, tirads_gt=3, tirads_probs=None,
                  ete_gt=1, ete_probs=None, loc_gt=2, loc_probs=None,
                  t_conf=0.8, t_gt=1, is_main=True, valid=True):
    """每患者一 GT 一 record;概率参数全部生效(mal 是 P(恶))。"""
    out = []
    for i in range(n):
        cl = _clinical(malignancy=gt_mal, lesion_loc=loc_gt,
                       extrathy_ext=ete_gt, clin_t_stage=t_gt, tirads=tirads_gt)
        if not valid:
            cl.update({"valid_malignancy": 0, "valid_lesion_loc": 0,
                       "valid_extrathy_ext": 0, "valid_clin_t_stage": 0,
                       "valid_tirads": 0})
        cp = _cls_preds(mal=mal, tirads=tirads_probs, ete=ete_probs,
                        loc=loc_probs, t_conf=t_conf)
        rows = [_case_row(f"P{i}", [_gt_row(f"P{i}", "M", is_main=is_main,
                                             clinical=cl)], cls_preds=cp)]
        out.extend(build_classification_records(rows, None))
    return out


def test_denominators_by_validity_and_malignancy():
    """各指标自己的有效分母:良性病灶 ETE/T 不入池(n_valid_ete=0)、
    良性仍入良恶性/位置/TI-RADS 池;次病灶全部排除。"""
    recs = _make_records(4, gt_mal=0)          # 全良性
    m = compute_delivery_classification_metrics(recs)
    assert m["n_valid_ete"] == 0 and m["ete_weighted_kappa"] is None
    assert m["n_valid_t"] == 0 and m["clinical_t_auc"] is None
    assert m["n_valid_malignancy"] == 4          # 良性照样入良恶性池
    # 次病灶:主病灶筛选后全空
    recs_sec = _make_records(3, is_main=False)
    m2 = compute_delivery_classification_metrics(recs_sec)
    assert m2["n_valid_malignancy"] == 0


def test_single_class_kappa_none_not_zero():
    """y_true 单类且 y_pred 单类 → Kappa=None(无法裁决,不按 0 填充)。"""
    recs = _make_records(5, tirads_gt=0,
                         tirads_probs=[1.0, 0.0, 0.0, 0.0, 0.0])
    m = compute_delivery_classification_metrics(recs)
    assert m["tirads_weighted_kappa"] is None


def test_argmax_ne_qwk_round_wired():
    """交付接线:argmax≠_qwk_round 的概率下,交付口径跟 _qwk_round 不跟 argmax。
    构造:probs=[0.35,0.33,0.32,0.0](ETE 4类)→ argmax=0,期望=0.33+0.64=0.97→round=1。"""
    from src.inference.v21_output import _qwk_round
    probs = [0.35, 0.33, 0.32, 0.0]
    assert int(np.argmax(probs)) == 0
    assert _qwk_round(probs) == 2                # 0.97 → round 1 → +1
    # 交付口径落到指标:gt 全为代码 2(idx1),交付预测也 idx1 → kappa=1;
    # 若误用 argmax(idx0) → kappa 低。钉死接线。
    # 分裂预测构造(预测单类时 weighted kappa 对小样本
    # 结构性为 0 不敏感,必须让两组概率不同使交付
    # 预测分裂成两类各自命中 gt):
    # P1 = [0.05,0.45,0.30,0.20]:E=1.65 → 码 3 → idx2,argmax=idx1
    # P2 = probs([0.35,0.33,0.32,0.0]):E=0.97 → 码 2 → idx1,argmax=idx0
    # _qwk 口径下两组预测各自命中 gt → kappa=1;
    # argmax 口径两组全错位 1 格 → kappa 降到 ≈0.31。
    P1 = [0.05, 0.45, 0.30, 0.20]
    recs = _make_records(6, ete_gt=2, ete_probs=P1)
    recs += _make_records(3, ete_gt=1, ete_probs=probs)
    m = compute_delivery_classification_metrics(recs)
    assert m["ete_weighted_kappa"] == pytest.approx(1.0)
    # 同构造换 argmax 强峰概率(预测全错位 1 格):
    recs_arg = _make_records(6, ete_gt=2, ete_probs=[0.0, 1.0, 0.0, 0.0])
    recs_arg += _make_records(3, ete_gt=1, ete_probs=[1.0, 0.0, 0.0, 0.0])
    m_arg = compute_delivery_classification_metrics(recs_arg)
    assert m_arg["ete_weighted_kappa"] < 0.5 < m["ete_weighted_kappa"]


def test_tirads_delivery_minus_one_alignment():
    """TI-RADS 交付缁对:_qwk_round 码 1..5 −1 回 0-based。gt=idx 时 kappa=1。"""
    # 分裂预测构造(同 ETE 测试的方法论):
    # PA=[0,0.9,0.1,0,0]:E=1.1 → 码 2 → idx1;PB=[0,0,0.1,0.9,0]:E=2.9 → 码 4 → idx3
    # 交付预测各自命中 gt(idx1/idx3) → kappa=1。
    # 这同时钉死 "-1 回 0-based 对齐"(码 2−1=1==gt1,
    # 码 4−1=3==gt3)——异常时(如忘 -1)两组全位位 2 格。
    PA = [0.0, 0.9, 0.1, 0.0, 0.0]
    PB = [0.0, 0.0, 0.1, 0.9, 0.0]
    recs = _make_records(6, tirads_gt=1, tirads_probs=PA)
    recs += _make_records(3, tirads_gt=3, tirads_probs=PB)
    m = compute_delivery_classification_metrics(recs)
    assert m["tirads_weighted_kappa"] == pytest.approx(1.0)
    # 反例对照:全部概率换成 PA(预测全 idx1)——gt 组
    # idx1 命中、idx3 差 2 格,预测单类 → kappa 降:
    recs_off = _make_records(6, tirads_gt=1, tirads_probs=PA)
    recs_off += _make_records(3, tirads_gt=3, tirads_probs=PA)
    m_off = compute_delivery_classification_metrics(recs_off)
    assert m_off["tirads_weighted_kappa"] < m["tirads_weighted_kappa"]


# ── 三维度视图与人口漂移 ───────────────────────────────────────────
def _two_run_rows():
    """P1 共同主病灶;P2 基线匹配候选丢失(人口保留);
    P3 候选新找回(不在人口)。"""
    base = [
        _case_row("P1", [_gt_row("P1", "M", matched=True)]),
        _case_row("P2", [_gt_row("P2", "M", matched=True)]),
    ]
    cand = [
        _case_row("P1", [_gt_row("P1", "M", matched=True)]),
        _case_row("P2", [_gt_row("P2", "M", matched=False)]),
        _case_row("P3", [_gt_row("P3", "M", matched=True)]),
    ]
    return base, cand


def test_population_delta_reports_both_sides():
    """找回/丢失各自报不只净差:P3 找回、P2 丢失。"""
    base, cand = _two_run_rows()
    views = compute_score_views(base, cand)
    d = views["population_delta"]
    assert d["recovered"] == ["P3::M"]
    assert d["lost"] == ["P2::M"]
    assert views["frozen_population_n"] == 2


def test_frozen_not_polluted_by_candidate_loss():
    """候选丢难病例 P2 不会虚假改善:frozen 分母两 run 相同
    (P1+P2 都在人口,候选从整图级输出读 P2 分类)。"""
    base, cand = _two_run_rows()
    views = compute_score_views(base, cand)
    nb = views["S_cls_frozen"]["baseline"]["n_records"]
    nc = views["S_cls_frozen"]["candidate"]["n_records"]
    assert nb == 2 and nc == 2


def test_clin_sim_uses_own_matched_population():
    """S_clin_sim 各自实际匹配人口:候选丢 P2 但新增 P3,
    匹配 records 数两 run 相同(2 vs 2,不是 2 vs 3)。"""
    base, cand = _two_run_rows()
    views = compute_score_views(base, cand)
    assert views["S_clin_sim"]["baseline"]["n_records"] == 2
    assert views["S_clin_sim"]["candidate"]["n_records"] == 2


def test_det_seg_assembly():
    """S_det_seg = 15*dice + 15*ap;缺 det_seg 时 None。"""
    base, cand = _two_run_rows()
    views = compute_score_views(base, cand,
                                det_seg_baseline={"mean_dice": 0.8, "micro_ap": 0.79},
                                det_seg_candidate=None)
    assert views["S_det_seg"]["baseline"] == pytest.approx(15 * 0.8 + 15 * 0.79)
    assert views["S_det_seg"]["candidate"] is None


def test_none_metric_score_is_none():
    """任一指标 None → score=None(无法裁决)不按 0 填充。"""
    rows = [_case_row("P1", [_gt_row("P1", "M", clinical=_clinical(malignancy=0))])]
    views = compute_score_views(rows, rows)
    assert views["S_cls_frozen"]["baseline"]["ete_weighted_kappa"] is None
    assert views["S_cls_frozen"]["baseline"]["score"] is None


def test_self_pair_zero_diff():
    """r5 自对拍零差:frozen 与 clin_sim 指标逐位相同,population_delta 空。"""
    base, _ = _two_run_rows()
    same = compute_score_views(base, base)
    for view in ("S_cls_frozen", "S_clin_sim"):
        for k in CLS_WEIGHTS:
            assert (same[view]["baseline"][k]
                    == same[view]["candidate"][k])
    assert same["population_delta"]["recovered"] == []
    assert same["population_delta"]["lost"] == []


# ── bootstrap ──────────────────────────────────────────────────────────
def test_bootstrap_population_mismatch_raises():
    """人口不等 → ValueError(入口校验)。"""
    base, cand = _two_run_rows()
    rb = build_classification_records(base, None)
    rc = build_classification_records(cand, None)
    with pytest.raises(ValueError, match="人口"):
        paired_patient_bootstrap(rb, rc, n_boot=10)


def test_bootstrap_patient_sampling_with_multiplicity():
    """PatientId 抽样保留倍数:全部 gt 相同、概率平移
    → AUC 差与抽样无关,P(Δ>0)=1。"""
    rows = []
    # PA 2 GT 一良一恶(gt 二类,AUC-PR 可算;同患者多 GT 验倍数抽样)
    rows.append(_case_row("PA", [_gt_row("PA", "M1"),
                                 _gt_row("PA", "M2",
                                         clinical=_clinical(malignancy=0))]))
    rows.append(_case_row("PB", [_gt_row("PB", "M")]))
    rows.append(_case_row("PC", [_gt_row("PC", "M",
                                         clinical=_clinical(malignancy=0))]))
    recs_b = build_classification_records(rows, None)
    recs_c = build_classification_records(rows, None)
    # 基线反序(恶 0.4/良 0.6)→ AUC 低;候选正序(恶 0.9/良 0.1)
    # → AUC 高。同人口同 gt 下每个重抽样差值恒>0
    # (反序的平移反序、正序的平移正序)。
    for r in recs_b:
        r["mal_prob"] = 0.4 if r["gt_malignancy"] == 1 else 0.6
    for r in recs_c:
        r["mal_prob"] = 0.9 if r["gt_malignancy"] == 1 else 0.1
    out = paired_patient_bootstrap(recs_b, recs_c, n_boot=50, seed=7)
    d = out["malignancy_auc_pr"]
    # 部分 bootstrap 重抽样丢失良性患者(gt 单类)→ 该轮 None,
    # 合法行为(无法裁决不硬算);有效轮次上限 50。
    assert 0 < d["n_boot_valid"] <= 50
    assert d["p_positive"] == 1.0
    assert d["mean"] > 0


# ── CLI 冒烟 ────────────────────────────────────────────────────
def test_cli_self_pair_report(tmp_path, capsys):
    """--baseline 自对拍:report 落盘且控制台摘要打印 frozen 人口。"""
    base, _ = _two_run_rows()
    p = _write_sidecar(tmp_path, base, name="base.jsonl")
    out_json = str(tmp_path / "report.json")
    import sys
    from scripts import compare_detection_runs as CR
    argv_backup = sys.argv
    sys.argv = ["compare_detection_runs.py", "--baseline", p,
                "--out", out_json]
    try:
        CR.main()
    finally:
        sys.argv = argv_backup
    rep = json.load(open(out_json, encoding="utf-8"))
    assert rep["frozen_population_n"] == 2
    assert rep["meta"]["caveats"]
    assert rep["population_delta"]["lost"] == []
    stdout = capsys.readouterr().out
    assert "恒定人口分类分" in stdout


def test_json_dump_strict_no_nan():
    """views 严格序列化(allow_nan=False)不炸:含 None 与正常值。"""
    base, _ = _two_run_rows()
    views = compute_score_views(base, base)
    json.dumps(views, allow_nan=False)


# ── 再审修复钉死(2026-09-18:row_skip_stats/位置AUC/空文件/倍数语义) ────────
def test_row_skip_stats_counts_anomaly_gt():
    """P1-D:anomaly 行 GT 计数上行(取自 n_gt),
    候选侧分母缩小在报告里可见。"""
    from scripts.compare_detection_runs import (ROW_SKIP_STATS,
                                                build_classification_records)
    rows = [
        _case_row("P1", [_gt_row("P1", "M")]),
        {"case_key": "/P2/S", "is_dynamic": False, "anomaly": "X",
         "n_gt": 2, "gts": []},
        {"case_key": "/P3/S", "is_dynamic": False, "anomaly": None,
         "n_gt": 1, "gts": [], "cls_preds": None},
    ]
    ROW_SKIP_STATS["anomaly_gt"] = 0
    ROW_SKIP_STATS["no_cls_preds_gt"] = 0
    build_classification_records(rows, None)
    assert ROW_SKIP_STATS["anomaly_gt"] == 2
    assert ROW_SKIP_STATS["no_cls_preds_gt"] == 1
    # compute_score_views 汇总到 views
    views = compute_score_views(rows[:1], rows)
    assert views["row_skip_stats"]["candidate"]["anomaly_gt"] == 2
    assert views["row_skip_stats"]["candidate"]["no_cls_preds_gt"] == 1


def test_location_auc_directional():
    """P1-E:位置 AUC 专项(反转 probs 列应翻转方向):
    两类 location 人口,正序高 AUC、反序低 AUC。"""
    # 3 患者 loc=2、3 患者 loc=4;probs 峰在各自码列
    # (码=idx+1:loc=2→列 1,loc=4→列 3;两类比较集
    # present=[2,4] 取列 [1,3],峰值必须落在比较集内)
    rows = []
    for i in range(3):
        cl = _clinical(lesion_loc=2)
        rows.append(_case_row(f"PA{i}",
            [_gt_row(f"PA{i}", "M", clinical=cl)],
            cls_preds=_cls_preds(loc=[0.1, 0.9, 0, 0, 0, 0, 0])))
    for i in range(3):
        cl = _clinical(lesion_loc=4)
        rows.append(_case_row(f"PC{i}",
            [_gt_row(f"PC{i}", "M", clinical=cl)],
            cls_preds=_cls_preds(loc=[0, 0, 0, 0.9, 0.1, 0, 0])))
    recs = build_classification_records(rows, None)
    m = compute_delivery_classification_metrics(recs)
    assert m["location_weighted_auc"] == pytest.approx(1.0)
    # 反转:两类的峰列交换(列 1 ↔ 列 3)→ AUC 翻转为 0
    for r in rows:
        lp = r["cls_preds"]["location_probs"]
        lp[1], lp[3] = lp[3], lp[1]
        r["cls_preds"]["location_probs"] = lp
    recs2 = build_classification_records(rows, None)
    m2 = compute_delivery_classification_metrics(recs2)
    assert m2["location_weighted_auc"] == pytest.approx(0.0)


def test_empty_sidecar_raises(tmp_path):
    """空 sidecar 文件 → raise(静默空报告已修)。"""
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="空文件"):
        load_sidecar_rows(str(p))


def test_bootstrap_multiplicity_monkeypatch():
    """P2-有牙化:抽中 k 次的患者记录计 k 份
    (monkeypatch 指标函数记录每轮 records 数)。破坏
    倍数(去重)会改变每轮记录数分布 → 断言可检。"""
    import scripts.compare_detection_runs as CR
    rows = []
    rows.append(_case_row("PA", [_gt_row("PA", "M1"), _gt_row("PA", "M2")]))
    rows.append(_case_row("PB", [_gt_row("PB", "M")]))
    recs_b = build_classification_records(rows, None)
    recs_c = build_classification_records(rows, None)
    for r in recs_b:
        r["mal_prob"] = 0.4 if r["gt_malignancy"] == 1 else 0.6
    for r in recs_c:
        r["mal_prob"] = 0.9 if r["gt_malignancy"] == 1 else 0.1
    seen_sizes = []
    orig = CR.compute_delivery_classification_metrics
    def spy(recs):
        seen_sizes.append(len(recs))
        return orig(recs)
    CR.compute_delivery_classification_metrics = spy
    try:
        CR.paired_patient_bootstrap(recs_b, recs_c, n_boot=60, seed=11)
    finally:
        CR.compute_delivery_classification_metrics = orig
    # n_gt=3(单患者 PA 两条);抽样保留倍数→
    # 应出现 > 3 的轮次和 < 3 的轮次(患者重复/丢失)
    assert max(seen_sizes) > 3, "倍数未生效:抽中患者去重了?"
    assert min(seen_sizes) < 3
    sizes = sorted(set(seen_sizes))
    assert len(sizes) >= 2

"""detection_match.py 单测:IoU / 官方动态口径 / 匈牙利匹配 / AP(n_gt 分母)/
oracle 四件套 / 病因统计聚合。

纯逻辑(合成 numpy 数组),不依赖 pydicom/nibabel/torch —— scipy/sklearn 在
requirements.txt 内,CI 必装。
"""
import numpy as np
import pytest

from scripts.detection_match import (
    aggregate_case_rows, ap_breakdown, average_precision, build_iou_matrix,
    dedup_keep_mask, iou_2d, iou_dynamic, match_instances)


# ── iou_2d ──────────────────────────────────────────────────────────────────

def test_iou_2d_basic():
    a = np.zeros((10, 10), np.uint8); a[2:5, 2:5] = 1     # 9 px
    b = np.zeros((10, 10), np.uint8); b[3:6, 3:6] = 1     # 9 px, 交 4 并 14
    assert iou_2d(a, b) == pytest.approx(4 / 14)
    assert iou_2d(a, a) == 1.0
    assert iou_2d(a, np.zeros_like(a)) == 0.0


def test_iou_2d_both_empty_returns_zero():
    """官方「切片全0 → IoU 记 0」的基础件:双空不得除零。"""
    z = np.zeros((10, 10), np.uint8)
    assert iou_2d(z, z) == 0.0


def test_iou_2d_shape_mismatch_raises():
    with pytest.raises(ValueError):
        iou_2d(np.zeros((10, 10)), np.zeros((10, 11)))


# ── iou_dynamic(官方动态口径两条限定)────────────────────────────────────────

def test_iou_dynamic_uncovered_frame_scores_zero():
    """官方限定①:某标注帧模型无预测(全0)→ 该帧 IoU 记 0 → 病灶级取平均。

    帧1 完美重合(IoU=1)、帧3 无预测 → 病灶级 IoU = (1+0)/2 = 0.5 → 边界恰好
    达 0.5(演示用);若帧1 只有 0.8 → 0.4 → 不匹配。官方明确不设单帧下限。
    """
    gt = np.zeros((5, 10, 10), np.uint8)
    gt[1, 2:5, 2:5] = 1
    gt[3, 2:5, 2:5] = 1
    pred = np.zeros((5, 10, 10), np.uint8)
    pred[1, 2:5, 2:5] = 1                      # 只有帧1 有预测
    assert iou_dynamic(pred, gt, [1, 3]) == pytest.approx(0.5)

    pred2 = np.zeros((5, 10, 10), np.uint8)
    pred2[1, 3:6, 3:6] = 1                     # 帧1 IoU=4/14≈0.2857
    assert iou_dynamic(pred2, gt, [1, 3]) == pytest.approx((4 / 14) / 2)


def test_iou_dynamic_only_annotated_frames_count():
    """官方限定②:只遍历 GT 标注帧——预测在非标注帧上的额外输出不影响病灶级 IoU。"""
    gt = np.zeros((5, 10, 10), np.uint8)
    gt[1, 2:5, 2:5] = 1
    pred = np.zeros((5, 10, 10), np.uint8)
    pred[1, 2:5, 2:5] = 1
    pred[0, 0:9, 0:9] = 1                      # 帧0 全是预测(非标注帧)
    pred[4, 8:10, 8:10] = 1                    # 帧4 亦然
    assert iou_dynamic(pred, gt, [1]) == 1.0


def test_iou_dynamic_empty_annotated_frames():
    gt = np.zeros((5, 10, 10), np.uint8)
    assert iou_dynamic(gt, gt, []) == 0.0


# ── build_iou_matrix + match_instances(匈牙利)──────────────────────────────

def test_hungarian_one_to_one_optimal():
    """匈牙利的一对一最优性:整体次优的逐行贪心会漏配,匈牙利不会。

    构造:pred0 对 gt1=0.9(行内最优)、pred1 对 gt0=0.6/对 gt1=0.0。
    逐行贪心:pred0 抢 gt1,pred1 只剩 gt0(0.6≥0.5)→ 2 TP(此例贪心恰好也对);
    改构造使贪心翻车:pred1 对 gt0 只有 0.4(<0.5)→ 贪心只得 1 TP(pred0-gt1),
    匈牙利全局最优仍是 (pred0,gt1,0.9)+(pred1,gt0,0.4<阈丢弃) → 1 TP。
    本例钉的是"解里 <阈值 的对被正确丢弃且不挤占别人的机会"——max-ΣIoU
    不会为了凑数把 0.4 的对算成命中,也不会让它抢占 0.9 的配对。
    """
    m = np.array([[0.0, 0.9],
                  [0.4, 0.0]])
    r = match_instances(m)
    assert len(r["pairs"]) == 1
    assert r["pred_to_gt"][0] == 1          # 唯一 ≥0.5 的对
    assert r["pred_to_gt"][1] == -1         # 0.4 < 阈值,丢弃


def test_match_drops_below_threshold():
    """解中 IoU<0.5 的对丢弃,双方回未匹配(不是"配上了但分数低")。"""
    m = np.array([[0.4]])
    r = match_instances(m)
    assert r["pairs"] == []
    assert r["pred_to_gt"][0] == -1 and r["gt_to_pred"][0] == -1


def test_match_one_to_one_blocks_double_assignment():
    """一对一约束:两个预测不能同时匹配同一 GT,即使第二个 IoU 也过阈。"""
    m = np.array([[0.9],
                  [0.6]])
    r = match_instances(m)
    assert len(r["pairs"]) == 1
    assert r["pred_to_gt"][0] == 0          # 高 IoU 者胜
    assert r["pred_to_gt"][1] == -1


def test_match_empty_matrices():
    """空矩阵(0 预测或 0 GT)不崩,返回全未匹配。"""
    for shape in [(0, 0), (3, 0), (0, 3)]:
        m = np.zeros(shape)
        r = match_instances(m)
        assert r["pairs"] == []
        assert len(r["pred_to_gt"]) == shape[0]
        assert len(r["gt_to_pred"]) == shape[1]


def test_build_iou_matrix_static_and_dynamic():
    a = np.zeros((10, 10), np.uint8); a[2:5, 2:5] = 1
    b = np.zeros((10, 10), np.uint8); b[2:5, 2:5] = 1
    preds = [{"mask": a, "confidence": 0.8}]
    gts = [{"mask": b}]
    assert build_iou_matrix(preds, gts)[0, 0] == 1.0

    # 动态 GT 带 annotated_frames → 走 iou_dynamic
    p3 = np.zeros((5, 10, 10), np.uint8); p3[1, 2:5, 2:5] = 1
    g3 = np.zeros((5, 10, 10), np.uint8); g3[1, 2:5, 2:5] = 1; g3[3, 2:5, 2:5] = 1
    preds_d = [{"mask": p3, "confidence": 0.5}]
    gts_d = [{"mask": g3, "annotated_frames": [1, 3]}]
    assert build_iou_matrix(preds_d, gts_d)[0, 0] == pytest.approx(0.5)


# ── average_precision(n_gt 分母)────────────────────────────────────────────

def test_ap_denominator_is_total_gt():
    """recall 分母 = n_gt(含漏检),不是 TP 数 —— 本模块与 sklearn 的口径差。"""
    # 1 个 TP(score 0.9)+ 99 个漏检 GT → AP = 0.01。
    assert average_precision([1], [0.9], 100) == pytest.approx(0.01)
    # sklearn 同输入只喂预测实例 → 1.0(虚高)。钉死两种口径的差,防后人误换。
    from sklearn.metrics import average_precision_score
    assert average_precision_score([1], [0.9]) == 1.0


def test_ap_manual_pr_curve():
    """手算 PR 曲线对拍:3 pred(2 TP) + 2 漏检 → n_gt=4。

    降序 [0.9(TP), 0.8(FP), 0.7(TP)]:
      k=1: P=1/1, R=1/4 → ΔR·P = 0.25
      k=3: P=2/3, R=2/4 → ΔR·P = 0.25·(2/3)
      AP = 0.25 + 1/6 = 5/12
    """
    labels = [1, 0, 1]
    scores = [0.9, 0.8, 0.7]
    assert average_precision(labels, scores, 4) == pytest.approx(5 / 12)


def test_ap_ties_merged():
    """同分合并为同一阈值:阈值处同分样本全体进"预测为正"再算 P/R。

    [TP,FP] 同分 0.8 → 单阈值 P=1/2, R=1/1 → AP=0.5。用组内首下标(错误实现)
    会得 1.0——本测试钉死该回归。
    """
    labels = [1, 0]
    scores = [0.8, 0.8]
    assert average_precision(labels, scores, 1) == pytest.approx(0.5)
    # 与 sklearn 同构造(无漏检进池,n_gt=正例数)逐位一致。
    from sklearn.metrics import average_precision_score
    assert average_precision_score(labels, scores) == pytest.approx(0.5)


def test_ap_no_predictions():
    assert average_precision([], [], 5) == 0.0
    assert average_precision([], [], 0) == 0.0


def test_ap_perfect_and_zero():
    assert average_precision([1, 1, 0], [0.9, 0.8, 0.1], 2) == pytest.approx(1.0)
    assert average_precision([0, 0], [0.9, 0.8], 2) == 0.0


# ── dedup_keep_mask + ap_breakdown ───────────────────────────────────────────

def test_dedup_keeps_max_iou_candidate_per_gt():
    """每 GT 只留 max-IoU 候选;纯 FP(与所有 GT 重叠≤0.1)保留。"""
    # pred0/pred1 都盖住 gt0(0.8/0.5),pred2 是纯 FP(0.02)。
    m = np.array([[0.8, 0.0],
                  [0.5, 0.0],
                  [0.02, 0.0]])
    keep = dedup_keep_mask(m)
    assert keep == [True, False, True]


def test_dedup_multi_gt_conflict_keeps_union_of_bests():
    """多 GT 冲突:pred 是 gt0 的最优、同时擦到 gt1(gt1 另有更优)→ 不误删。

    A 路审查 P1 的钉死用例:逐列淘汰法(旧实现)会得 [False,True]——
    gt1 的循环把 gt0 的最优 pred0 删了。"各 GT 最优的并集"语义下应全保留。
    """
    m = np.array([[0.9, 0.2],
                  [0.0, 0.8]])
    assert dedup_keep_mask(m) == [True, True]


def test_ap_breakdown_oracle_equals_match_rate_and_fragments():
    """四件套核心语义:oracle == 匹配率;碎裂(1 GT + 未匹配碎片)时 dedup > current。

    场景确定性构造(避开匈牙利平局):1 GT(16px),预测 3 个——
      frag_a 覆盖 GT 的 2:6×2:5(IoU=12/16=0.75,匹配);
      frag_b 是 GT 右下方的偏移块(交 6 并 26,IoU=0.2308<0.5,未匹配的近漏碎片);
      off 纯 FP。
    conf:碎片 frag_b=0.95(最高但未匹配)、TP=0.60、纯 FP=0.30 → current 被
    高置信碎片压到 0.5;dedup 去掉碎片后 TP 居首 → 1.0。
    """
    gt = np.zeros((10, 10), np.uint8); gt[2:6, 2:6] = 1            # 16 px
    tp_mask = np.zeros((10, 10), np.uint8); tp_mask[2:6, 2:5] = 1  # 12 px, IoU=0.75
    frag = np.zeros((10, 10), np.uint8); frag[3:7, 4:8] = 1        # 交6并26 → 0.2308
    off = np.zeros((10, 10), np.uint8); off[8:10, 8:10] = 1        # 纯 FP
    preds = [{"mask": tp_mask, "confidence": 0.60},
             {"mask": frag, "confidence": 0.95},
             {"mask": off, "confidence": 0.30}]
    gts = [{"mask": gt}]
    m = build_iou_matrix(preds, gts)
    match = match_instances(m)
    assert m[0, 0] == pytest.approx(0.75)          # TP 匹配,唯一确定
    bd = ap_breakdown(preds, m, match, n_gt=1)
    assert bd["n_tp"] == 1
    assert bd["match_rate"] == 1.0
    assert bd["ap_oracle"] == pytest.approx(1.0)   # oracle = 匹配率
    # current:降序 [0.95(碎片FP), 0.60(TP), 0.30(纯FP)] → k=1 处 P=1/2,R=1 → 0.5。
    assert bd["ap_current"] == pytest.approx(0.5)
    # dedup:去掉未匹配且 best_iou>0.1 的碎片 → TP 居首 → 1.0(碎裂损失=0.5)。
    assert bd["ap_dedup"] == pytest.approx(1.0)
    # iou_ranked:TP 0.75 / 碎片 0.2308 / 纯FP 0 → TP 居首 → 1.0(排序空间=0.5)。
    assert bd["ap_iou_ranked"] == pytest.approx(1.0)
    assert bd["ap_sklearn_predonly"] == pytest.approx(0.5)  # 虚高口径对照


# ── aggregate_case_rows ──────────────────────────────────────────────────────

def _case_row(n_pred=1, n_gt=1, preds=None, gts=None, is_dynamic=False,
              anomaly=None):
    return {"case_key": "c", "is_dynamic": is_dynamic, "anomaly": anomaly,
            "n_pred": n_pred, "n_gt": n_gt,
            "preds": preds or [], "gts": gts or []}


def test_aggregate_basic_and_main_vs_secondary():
    rows = [
        _case_row(preds=[{"confidence": 0.9, "area": 10, "best_iou": 0.9,
                          "matched": True}],
                  gts=[{"is_main": True, "area_ratio": 0.01, "max_iou": 0.9,
                        "matched": True, "n_candidates": 1}]),
        _case_row(preds=[{"confidence": 0.4, "area": 8, "best_iou": 0.0,
                          "matched": False}],
                  gts=[{"is_main": False, "area_ratio": 5e-5, "max_iou": 0.0,
                        "matched": False, "n_candidates": 0}]),
    ]
    s = aggregate_case_rows(rows)
    assert s["n_cases"] == 2 and s["n_gt"] == 2 and s["n_tp"] == 1
    assert s["match_rate"] == 0.5
    assert s["match_rate_main"] == 1.0
    assert s["match_rate_secondary"] == 0.0
    assert s["gt_iou_hist"][">=0.5"] == 1
    assert s["gt_iou_hist"]["miss"] == 1
    # 面积 5e-5 落 <1e-4 桶(结构性不可检)。
    assert s["gt_area_hist"]["<1e-4"] == 1
    # 交叉表:漏检 ∩ 小面积 = 1。
    assert s["gt_iou_area_cross"]["miss"]["<1e-4"] == 1
    assert s["preds_per_case"] == {"0": 0, "1": 2, "2": 0, "3-5": 0, "6+": 0}
    assert s["candidates_per_gt"] == {"0": 1, "1": 1, "2": 0, "3+": 0}


def test_aggregate_skips_anomaly_rows():
    rows = [_case_row(anomaly="推理失败", n_pred=3, n_gt=2,
                      preds=[{"confidence": 0.9, "area": 1, "best_iou": 0,
                              "matched": True}] * 3,
                      gts=[{"is_main": True, "area_ratio": 0.01,
                            "max_iou": 0, "matched": False, "n_candidates": 0}] * 2),
            _case_row(n_pred=0, n_gt=1, preds=[], gts=[{"is_main": True,
                                                        "area_ratio": 0.02,
                                                        "max_iou": 0.0,
                                                        "matched": False,
                                                        "n_candidates": 0}])]
    s = aggregate_case_rows(rows)
    assert s["n_anomaly"] == 1
    assert s["n_gt"] == 1 and s["n_pred"] == 0     # anomaly 行不进统计


def test_aggregate_critical_band_fraction():
    gts = [{"is_main": True, "area_ratio": 0.01, "max_iou": 0.4,
            "matched": False, "n_candidates": 1}] * 4
    rows = [_case_row(n_gt=4, preds=[], gts=gts)]
    s = aggregate_case_rows(rows)
    assert s["critical_band_frac"] == 1.0          # 4/4 落 [0.35,0.5)


# ── 2026-09-05 三路审查(A/B/C)修复的回归钉死 ─────────────────────────────

def test_aggregate_nan_and_negative_bucketed_not_crash():
    """A-F1/C-F3:NaN 与负值的 max_iou/area_ratio 进 invalid 桶,不 KeyError 崩报告。

    首版防御只拦了 NaN 返回 "invalid" 桶名,但消费方 dict 初始化无该键 →
    整份报告聚合崩溃(声称"计数暴露"实为地雷)。A-F4:负值同理曾落
    ">=0.5"命中桶(兜底写反)。
    """
    rows = [_case_row(preds=[{"confidence": 0.9, "area": 10, "best_iou": 0.6,
                              "matched": True}],
                      gts=[{"is_main": True, "area_ratio": float("nan"),
                            "max_iou": float("nan"), "matched": True,
                            "n_candidates": 1},
                           {"is_main": False, "area_ratio": -0.01,
                            "max_iou": -0.3, "matched": False,
                            "n_candidates": 0}])]
    s = aggregate_case_rows(rows)
    assert s["gt_iou_hist"]["invalid"] == 2
    assert s["gt_area_hist"]["invalid"] == 2
    assert s["gt_iou_area_cross"]["invalid"]["invalid"] == 2
    # 正常桶不被污染
    assert s["n_gt"] == 2


def test_match_residual_edges_augmented_to_max_cardinality():
    """A-F2:max-Σ+阈值丢弃后,残余 ≥0.5 合法对补上(两形态:free-free 与重排)。

    2 万随机矩阵实测:修复前 531 例 free-free 残余 + 473 例需重排,合计 5%
    的 case 比官方"≥0.5 子图最大匹配"少算 TP。此处钉死一个重排形态
    (free-free 形态已被随机对拍覆盖,不再单列)。
    """
    # max-Σ 会把 gt0 指派给 p1(0.45,sub-thr)以换取 Σ 更大 → 丢弃后 gt0 空置;
    # 但 p2-gt0=0.9 的合法对可救回(需要"允许重排"的增广)。
    m = np.array([[0.467, 0.785, 0.018, 0.109, 0.829],
                  [0.797, 0.233, 0.531, 0.606, 0.868],
                  [0.603, 0.413, 0.374, 0.426, 0.652],
                  [0.867, 0.454, 0.248, 0.237, 0.746]])
    r = match_instances(m)
    # ≥0.5 子图最大匹配 = 4(A 路 2 万矩阵实验的实测形态之一)
    assert len(r["pairs"]) == 4
    # 一对一与阈值合法性
    ps = [p for p, g, _ in r["pairs"]]
    gs = [g for p, g, _ in r["pairs"]]
    assert len(set(ps)) == len(ps) and len(set(gs)) == len(gs)
    assert all(iou >= 0.5 for _, _, iou in r["pairs"])


def test_dedup_keeps_hungarian_tps():
    """A-F3:匈牙利 TP 即使不是任何 GT 的 max-IoU 候选也保留(并入 keep 集)。

    `[[0.75,0.70],[0.60,0.0],[0.0,0.55]]`:匈牙利 2 TP 全中(p0→g0、p2→g1),
    但 p2 是 g1 的次优(g1 最优是 p0)→ 旧 keep 集删掉真 TP、无碎裂却读出
    0.33 的"碎裂损失"。
    """
    m = np.array([[0.75, 0.70],
                  [0.60, 0.0],
                  [0.0, 0.55]])
    match = match_instances(m)
    assert len(match["pairs"]) == 2                    # 全中
    keep = dedup_keep_mask(m, match=match)
    assert keep == [True, False, True]                 # p2(真 TP)保留
    preds = [{"confidence": 0.9}, {"confidence": 0.8}, {"confidence": 0.7}]
    bd = ap_breakdown(preds, m, match, n_gt=2)
    # 无碎裂 → dedup 与 current 的差只能来自删掉的 FP p1,TP 全保留。
    assert bd["ap_dedup"] >= bd["ap_current"]


def test_dedup_without_match_keeps_old_semantics():
    """不传 match 时保持旧接口语义(纯候选并集),供纯逻辑测试/旧调用方。"""
    m = np.array([[0.75, 0.70],
                  [0.60, 0.0],
                  [0.0, 0.55]])
    assert dedup_keep_mask(m) == [True, False, False]  # p0 是两 GT 的最优


def test_match_threshold_boundary_locked():
    """C-F5:IoU 恰 0.5 匹配 / 0.4999 不匹配(MATCH_IOU_THRESHOLD=0.5 官方口径)。"""
    assert len(match_instances(np.array([[0.5]]))["pairs"]) == 1
    assert len(match_instances(np.array([[0.4999]]))["pairs"]) == 0

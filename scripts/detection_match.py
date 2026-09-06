"""检测诊断仪表盘 · 纯逻辑库:IoU / 匈牙利匹配 / AP / 病因统计。

为「检测 AP 3.05(首真分 61.89 全表最弱项)」的诊断而写。无 I/O、无 torch,
只依赖 numpy/scipy/sklearn —— 便于纯逻辑单测,云端脚本 `diagnose_detection.py`
import 本模块完成全部数值计算。

## 官方匹配口径(评审规则 `:201-:227`,逐条实现于本模块)

- **静态**数据:预测 mask 与金标准 mask 的 **2D IoU**。
- **动态**数据:各**标注帧**帧级 IoU 的**算术平均**。两条易漏限定:
  ① 若某标注帧上模型无预测(切片空或全0),该帧 IoU 记 0;
  ② 不设单帧 IoU 下限,仅以病灶级(平均后)IoU 判定。
- **匈牙利算法一对一最优匹配**,IoU≥0.5 判定成功(低于阈值的对丢弃,双方都回到未匹配)。
- 漏检的主病灶「不参与分类/分期指标计算,但在检测 AP 与 Dice 中受双重惩罚」。

## AP 的 recall 分母(本模块最易被"看起来正常"的错法污染)

官方 AP 的召回分母 = **全部金标准病灶数 n_gt(含漏检)**。sklearn 的
`average_precision_score` 分母是**传入 y_true 的正例数**——若只把预测实例喂进去
(漏检 GT 不进 y_true),AP 系统性虚高且数字完全正常(实测:单 TP + 99 漏检时
sklearn=1.0 而真值=0.01)。故主口径用本模块手写的 step-wise `average_precision`,
sklearn 值仅作对照并须在报告里注明其虚高口径。

## oracle AP 分解四件套(同一批实例、同一 n_gt)

| 名称 | score | 读作 |
|---|---|---|
| `ap_current` | `detection_confidence` | 真实表现 |
| `ap_oracle` | TP→1.0,FP→0.0 | **= 匹配率**(召回上限) |
| `ap_iou_ranked` | 每预测对其 case 全部 GT 的 max-IoU | 排序可挖掘空间 |
| `ap_dedup` | confidence(每 GT 留 max-IoU 候选 ∪ 匈牙利 TP) | 碎裂损失 |

静/动**分开各出一套**(另出合并池总览)——动态 `detection_confidence` 经
`scatter_frames_to_full_length` 二值化后语义退化为"帧覆盖率"(详见
`diagnose_detection.py` 模块 docstring),静/动混池会把动态的 conf 饱和误读成排序能力。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# 官方匹配阈值(评审规则:IoU≥0.5 判定成功)。定成常量而非默认参数,
# 防调用方随手改掉后与官方口径漂移。
MATCH_IOU_THRESHOLD = 0.5

# dedup 的重叠判定阈值:预测与 GT 的 IoU 超过此值才算"该 GT 的候选",
# 用于 ap_dedup(碎裂损失)的簇内去重。
DEDUP_OVERLAP_THRESHOLD = 0.1


# ── IoU ────────────────────────────────────────────────────────────────────

def iou_2d(a: np.ndarray, b: np.ndarray) -> float:
    """二值 (H,W) mask 的 IoU。双空(并集 0)返回 0.0 —— 官方「切片全0 → IoU 记 0」的基础件。"""
    m1 = np.asarray(a) != 0
    m2 = np.asarray(b) != 0
    if m1.shape != m2.shape:
        raise ValueError(f"iou_2d: 形状不一致 {m1.shape} vs {m2.shape}")
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(m1, m2).sum()
    return float(inter) / float(union)


def iou_dynamic(pred_3d: np.ndarray, gt_3d: np.ndarray,
                annotated_frames: list) -> float:
    """官方动态口径:各**标注帧**帧级 IoU 的算术平均。

    - 只遍历 GT 标注帧(`annotated_frames` = GT mask 非零帧的 0-based 原始帧号),
      其余帧不参与 —— 官方「评估时仅抽取标注帧计算指标」。
    - 预测帧切片空/全 0 → 该帧 IoU=0(由 `iou_2d` 双空语义 + 单边为空自然覆盖)。
    - 不设单帧下限,直接平均。

    `pred_3d`/`gt_3d` 均为 `(n_total, H, W)`,帧轴语义 = 原始帧号(scatter 后)。
    """
    p = np.asarray(pred_3d)
    g = np.asarray(gt_3d)
    if p.shape != g.shape:
        raise ValueError(f"iou_dynamic: 形状不一致 {p.shape} vs {g.shape}")
    if not annotated_frames:
        return 0.0
    vals = [iou_2d(p[f], g[f]) for f in annotated_frames]
    return float(np.mean(vals))


def build_iou_matrix(preds: list, gts: list) -> np.ndarray:
    """`(n_pred, n_gt)` IoU 矩阵。静态逐对 `iou_2d`;动态逐对 `iou_dynamic`。

    preds/gts 为实例 dict(含 `mask`;动态 GT 另含 `annotated_frames`)。
    """
    m = np.zeros((len(preds), len(gts)), dtype=np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if g.get("annotated_frames") is not None:
                m[i, j] = iou_dynamic(p["mask"], g["mask"], g["annotated_frames"])
            else:
                m[i, j] = iou_2d(p["mask"], g["mask"])
    return m


# ── 匈牙利匹配 ─────────────────────────────────────────────────────────────

def match_instances(iou_matrix: np.ndarray,
                    threshold: float = MATCH_IOU_THRESHOLD) -> dict:
    """一对一最优匹配(IoU≥threshold 判成功)。

    `linear_sum_assignment(-iou)` 求全局最优;解中 IoU < threshold 的对**丢弃**
    (双方都回到未匹配)——官方是一对一 + 阈值判定,低于阈值的配对不算命中。

    ⚠️ 口径假设(A 路 2026-09-05 审查指出 + 2 万随机矩阵实测):评审规则只写
    「匈牙利算法一对一最优匹配,IoU≥0.5 判定成功」,未指明目标函数。本实现为
    **两阶段**:先 max ΣIoU 加权匈牙利求高质量配对,再在 ≥threshold 子图上二次
    分配(保留 max-Σ 配对作 tie-break)——最终匹配**数量**等价于「IoU≥threshold
    二值化后的最大匹配基数」。单独的「max-Σ + 阈值丢弃」实测会少算 TP(2 万随机
    矩阵 531 例 free-free 残余 + 473 例需重排,合计 5%),若官方按最大基数实现,
    本实现与之对齐;若官方真按 max-Σ+丢弃,pair 选择略有不同但数量不会更少。
    pair 级选择仍是实现选择,报告 meta 注明。

    空矩阵((0,N)/(N,0)/(0,0))实测不崩,返回空索引 → 全未匹配。

    Returns: {"pairs": [(pred_i, gt_j, iou)], "pred_to_gt": np.ndarray(-1=未匹配),
              "gt_to_pred": np.ndarray(-1=未匹配)}
    """
    m = np.asarray(iou_matrix, dtype=np.float64)
    n_pred, n_gt = m.shape
    pred_to_gt = np.full(n_pred, -1, dtype=np.int64)
    gt_to_pred = np.full(n_gt, -1, dtype=np.int64)
    if n_pred == 0 or n_gt == 0:
        return {"pairs": [], "pred_to_gt": pred_to_gt, "gt_to_pred": gt_to_pred}

    rows, cols = linear_sum_assignment(-m)
    pairs = []
    for r, c in zip(rows, cols):
        if m[r, c] >= threshold:
            pred_to_gt[r] = c
            gt_to_pred[c] = r
            pairs.append((int(r), int(c), float(m[r, c])))
        # 低于阈值的对:不进 pairs,双方保持 -1(未匹配)。
    # 🔴 ≥threshold 子图二次分配(A 路 F2 修复,2026-09-05):「max-Σ 指派 + 丢弃」
    # 可能比「≥thr 子图最大匹配」少 TP——两种形态:A 路实测的 free-free 残余边
    # (2 万随机矩阵 531 例),以及 max-Σ 配对本身与最大匹配不兼容需**重排**
    # (同实验 473 例,只补 free-free 救不回)。彻底修法:在 ≥thr 子图上重跑一次
    # 分配,代价=非法边 10 / 合法且保留 max-Σ 已配对 0 / 合法新边 0.5−0.1·IoU
    # (高 IoU 优先)——匹配**数量**对齐最大匹配,pair 选择以 max-Σ 结果为
    # tie-break。修复后 2 万随机矩阵与纯二分图最大匹配 0 差异(修复前 531+473)。
    # 代价一次 O(n³),case 级(实例数 ≤ 几十)可忽略。
    n = max(n_pred, n_gt)
    big = m >= threshold
    pad_big = np.zeros((n, n), dtype=bool)
    pad_big[:n_pred, :n_gt] = big
    if pad_big.any():
        kept = {(p, g) for p, g, _ in pairs}
        cost = np.where(pad_big, 0.0, 10.0)
        for i in range(n_pred):
            for j in range(n_gt):
                if big[i, j] and (i, j) not in kept:
                    cost[i, j] = 0.5 - 0.1 * m[i, j]
        # 初始化 pair 列表为空,由二次分配统一产出(保证与代价模型一致)
        pairs = []
        pred_to_gt = np.full(n_pred, -1, dtype=np.int64)
        gt_to_pred = np.full(n_gt, -1, dtype=np.int64)
        rr, cc = linear_sum_assignment(cost)
        for r, c in zip(rr, cc):
            if pad_big[r, c]:
                pred_to_gt[r] = c
                gt_to_pred[c] = r
                pairs.append((int(r), int(c), float(m[r, c])))
    return {"pairs": pairs, "pred_to_gt": pred_to_gt, "gt_to_pred": gt_to_pred}


# ── AP ─────────────────────────────────────────────────────────────────────

def average_precision(labels: list, scores: list, n_gt: int) -> float:
    """手写 step-wise AP,**recall 分母 = n_gt(全部 GT 含漏检)**。

    按分数降序(同分合并为同一阈值),`AP = Σ (R_k − R_{k−1})·P_k`。
    无预测返回 0.0(n_gt>0 时召回为零,AP 恒 0)。

    🔴 别换 sklearn 当主口径:其分母是 y_true 正例数,漏检 GT 不进预测池时
    AP 虚高且数字正常(见模块 docstring)。
    """
    if n_gt <= 0 or not labels:
        return 0.0
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    # 护栏:TP 数不可能超过 GT 总数(A 路审查实测 average_precision([1,1],·,n_gt=1)
    # 会静默返回 2.0)。上游传错 n_gt 时宁可硬拦,不产"看起来正常的超 1 AP"——
    # 本函数的 n_gt 口径正是 docstring 强调的最易错处。
    assert int(np.sum(y)) <= n_gt, \
        f"average_precision: TP 数 {int(np.sum(y))} > n_gt {n_gt},n_gt 口径传错"
    order = np.argsort(-s, kind="stable")
    y = y[order]
    s_sorted = s[order]

    tp_cum = np.cumsum(y)
    fp_cum = np.cumsum(1.0 - y)
    # 同分合并为同一阈值:阈值处的 PR 点必须用**该同分组最后一个下标**的累计
    # tp/fp(即同分样本全体进入"预测为正"集合后再算 P/R),与 sklearn 阈值语义
    # 一致——用组内第一个下标会把同分的 FP 挤出分母、AP 虚高
    # (实测 [1,0]/[0.8,0.8]/n_gt=1:首下标法给 1.0,正确值 0.5)。
    is_group_end = np.concatenate((s_sorted[1:] != s_sorted[:-1], [True]))

    ap = 0.0
    prev_recall = 0.0
    for k in np.flatnonzero(is_group_end):
        tp, fp = tp_cum[k], fp_cum[k]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_gt
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return float(ap)


# ── oracle AP 分解 ─────────────────────────────────────────────────────────

def dedup_keep_mask(iou_matrix: np.ndarray,
                    overlap_thr: float = DEDUP_OVERLAP_THRESHOLD,
                    match: dict | None = None) -> list:
    """碎裂去重:每 GT 在 IoU>overlap_thr 的候选中只留 max-IoU 一个,
    **并集上再并入匈牙利 TP**(A 路 F3 修复,2026-09-05)。

    与所有 GT 重叠 ≤ thr 的纯 FP **保留**(它们是真实假阳性,去重只针对
    "同一 GT 被多个碎片预测覆盖"的情形——那是 AP 的重复 TP 稀释,不是 FP)。

    🔴 实现必须是"**各 GT 最优的并集**"而非逐列淘汰(A 路 2026-09-05 审查
    实测坐实):逐列循环里 `keep[i]=False` 是全局写——pred A 若是 gt0 的
    max-IoU 候选、同时以 IoU>thr 擦到 gt1(而 gt1 另有更优 pred B),gt1 的
    循环会把 A 误删,违反本函数自己的语义。实测 `[[0.9,0.2],[0.0,0.8]]`
    逐列法得 keep=[False,True](正确值 [True,True]),级联 ap_dedup 反转成
    "负碎裂损失"误导归因。

    🔴 并集必须**并入匈牙利 TP**(A 路 F3,同轮实测):只取"各 GT max-IoU 候选"
    会把真实 TP 删掉——`[[0.75,0.70],[0.60,0.0],[0.0,0.55]]` 匈牙利 2 TP 全中
    (p0→g0、p2→g1),但 p2 不是任何 GT 的 max-IoU 候选(它是 g1 的次优,g1 的
    最优是 p0)→ 只留 p0 → 无任何碎裂却读出 ap_current−ap_dedup=0.33 的
    "碎裂损失"。根因:dedup 的"每 GT 留最优"与匈牙利的一对一最优指派是两套
    语义,一个 pred 可以是某 GT 的 max-IoU 候选却实际被指派给另一 GT。修法:
    keep = 各 GT max-IoU 候选 ∪ 全部匈牙利 TP(调用方传 `match=`;不传则
    只做候选并集,保留旧接口供纯逻辑测试)。

    Returns: list[bool],len = n_pred,True=保留。
    """
    m = np.asarray(iou_matrix, dtype=np.float64)
    n_pred = m.shape[0]
    if n_pred == 0 or m.shape[1] == 0:
        return [True] * n_pred
    keep = [False] * n_pred
    for j in range(m.shape[1]):
        col = m[:, j]
        cand = np.flatnonzero(col > overlap_thr)
        if len(cand) > 0:
            best = cand[np.argmax(col[cand])]
            keep[int(best)] = True      # 是任一 GT 的最优即保留(并集)
    # 匈牙利 TP 并入(F3):已成功匹配的 pred 无论是否为某 GT 的 max-IoU 候选都保留。
    if match is not None:
        for i in range(n_pred):
            if match["pred_to_gt"][i] >= 0:
                keep[i] = True
    # 纯 FP(与所有 GT 重叠 ≤ thr):未被任何 GT 标记,按语义保留。
    for i in range(n_pred):
        if m[i].max() <= overlap_thr:
            keep[i] = True
    return keep


def ap_breakdown(preds: list, iou_matrix: np.ndarray, match: dict, n_gt: int) -> dict:
    """四件套(同一批实例、同一 n_gt)。preds 各项含 `confidence`。

    - `ap_current`:`detection_confidence` 原值排序 —— 真实表现。
    - `ap_oracle`:TP→1.0 / FP→0.0 —— 数学上恒等于匹配率(召回上限)。
    - `ap_iou_ranked`:每预测对其 case 全部 GT 的 max-IoU 当分数 —— 若它显著高于
      current,说明"重叠质量信息就在那但 conf 没用它",排序校准有空间。
    - `ap_dedup`:conf 排序但每 GT 只留 max-IoU 候选 ∪ 匈牙利 TP(纯 FP 保留)
      —— 与 current 的差 = 碎裂 FP 的 AP 稀释量(A-F3:TP 并入是防"非列 argmax
      的真 TP"被删、无碎裂却读出碎裂损失)。
    - `ap_sklearn_predonly`:sklearn 对照(只喂预测实例、无 FN 行的虚高口径,
      仅供并排参考,勿当回归锚;见模块 docstring)。
    """
    pred_to_gt = match["pred_to_gt"]
    n_pred = iou_matrix.shape[0]

    labels = [int(pred_to_gt[i] >= 0) for i in range(n_pred)]
    confs = [float(p.get("confidence", 0.0)) for p in preds]
    oracle = [1.0 if l else 0.0 for l in labels]
    max_iou = ([float(iou_matrix[i].max()) for i in range(n_pred)]
               if iou_matrix.shape[1] > 0 else [0.0] * n_pred)
    keep = dedup_keep_mask(iou_matrix, match=match)

    # sklearn 对照:同批实例、同标签,只体现"漏检不在分母"的口径差。
    try:
        from sklearn.metrics import average_precision_score as _ap_sk
        ap_sk = float(_ap_sk(labels, confs)) if n_pred else 0.0
    except Exception:  # noqa: BLE001 — sklearn 缺失时对照列留 None,不影响主口径
        ap_sk = None

    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_tp": int(sum(labels)),
        "match_rate": (float(sum(labels)) / n_gt) if n_gt > 0 else 0.0,
        "ap_current": average_precision(labels, confs, n_gt),
        "ap_oracle": average_precision(labels, oracle, n_gt),
        "ap_iou_ranked": average_precision(labels, max_iou, n_gt),
        "ap_dedup": average_precision(
            [l for l, k in zip(labels, keep) if k],
            [c for c, k in zip(confs, keep) if k],
            n_gt),
        "ap_sklearn_predonly": ap_sk,
    }


# ── 病因统计 ────────────────────────────────────────────────────────────────

# GT max-IoU 桶:0=漏检 / 低重叠 / 中重叠 / 临界带(0.35-0.5,推过阈值即翻盘) / 命中。
IOU_BUCKETS = [(0.0, 0.0, "miss"), (0.0, 0.1, "(0,0.1)"),
               (0.1, 0.35, "(0.1,0.35)"), (0.35, 0.5, "[0.35,0.5)"),
               (0.5, float("inf"), ">=0.5")]
# GT 面积占比桶:<1e-4 低于 split_nodules 默认 min_area_ratio(结构性不可检)。
AREA_BUCKETS = [(0.0, 1e-4, "<1e-4"), (1e-4, 5e-4, "1e-4~5e-4"),
                (5e-4, 2e-3, "5e-4~2e-3"), (2e-3, 1e-2, "2e-3~1e-2"),
                (1e-2, float("inf"), ">1e-2")]


def _bucket(v: float, buckets) -> str:
    # 非有限值单独拦截(A 路审查实测:NaN 落空全部比较后走兜底、静默计入
    # "命中"桶——上游坏数据会虚增匹配表现,与"NaN 令数值门静默失效"的既有
    # 教训同型)。计入 invalid 桶暴露它,不猜。
    # 🔴 同轮 A-F1:负值同理落空比较后走兜底进">=0.5"命中桶(兜底写反),
    # 与 NaN 一起归入 invalid。**消费方(aggregate_case_rows)的 dict 初始化
    # 必须含 "invalid" 键**,否则这里返回的桶名会 KeyError 崩整份报告
    # (A 路实测:含 NaN max_iou 的 case row 令 aggregate_case_rows 抛
    # KeyError('invalid')——声称"计入 invalid 桶暴露"实为崩溃,防御写成地雷)。
    if not np.isfinite(v) or v < 0.0:
        return "invalid"
    for lo, hi, name in buckets:
        # 闭开约定:下界含、上界不含;miss 桶单独处理(精确 =0)。
        # 桶名与约定一致:0.1 落 (0.1,0.35)、0.35 落 [0.35,0.5)、0.5 落 >=0.5。
        if name == "miss":
            if v == 0.0:
                return name
        elif lo <= v < hi:
            return name
    return buckets[-1][2]


def aggregate_case_rows(case_rows: list) -> dict:
    """聚合逐 case 行 → 病因统计。

    case_row 各项字段(`diagnose_detection.py` 产出):
      `case_key/is_dynamic/n_pred/n_gt/anomaly/preds[{confidence,area,best_iou,matched}]/
       gts[{is_main,area_ratio,max_iou,matched,n_candidates}]`

    输出统计键:
      - `gt_iou_area_cross`: max-IoU 桶 × 面积占比桶交叉表(定位"漏检集中在小面积")。
      - `gt_iou_hist` / `pred_conf_hist_*`: 直方图。
      - `match_rate_main` vs `match_rate_secondary`: 主病灶 vs 次病灶匹配率。
      - `preds_per_case`: 每例预测实例数分布(碎裂/爆量的宏观信号)。
    """
    out = {
        "n_cases": len(case_rows),
        "n_anomaly": sum(1 for r in case_rows if r.get("anomaly")),
        "n_gt": 0, "n_pred": 0, "n_tp": 0,
        # 🔴 含 "invalid" 键(A-F1):_bucket 对 NaN/负值返回 "invalid",
        # dict 无此键会 KeyError 崩整份报告——防御必须与消费方对齐。
        "gt_iou_hist": {**{name: 0 for _, _, name in IOU_BUCKETS}, "invalid": 0},
        "gt_area_hist": {**{name: 0 for _, _, name in AREA_BUCKETS}, "invalid": 0},
        "gt_iou_area_cross": {
            **{name: {**{a: 0 for _, _, a in AREA_BUCKETS}, "invalid": 0}
               for _, _, name in IOU_BUCKETS},
            "invalid": {**{a: 0 for _, _, a in AREA_BUCKETS}, "invalid": 0}},
        "n_gt_main": 0, "n_matched_main": 0,
        "n_gt_secondary": 0, "n_matched_secondary": 0,
        "preds_per_case": {"0": 0, "1": 0, "2": 0, "3-5": 0, "6+": 0},
        "candidates_per_gt": {"0": 0, "1": 0, "2": 0, "3+": 0},
        "conf_out_of_range": 0,
    }
    conf_tp, conf_fp = [], []

    for r in case_rows:
        if r.get("anomaly"):
            continue
        n_p = r.get("n_pred", 0)
        k = "0" if n_p == 0 else "1" if n_p == 1 else "2" if n_p == 2 \
            else "3-5" if n_p <= 5 else "6+"
        out["preds_per_case"][k] += 1
        out["n_pred"] += n_p
        for p in r.get("preds", []):
            if p.get("matched"):
                conf_tp.append(float(p.get("confidence", 0.0)))
            else:
                conf_fp.append(float(p.get("confidence", 0.0)))

        for g in r.get("gts", []):
            out["n_gt"] += 1
            max_iou = float(g.get("max_iou", 0.0))
            area_ratio = float(g.get("area_ratio", 0.0))
            b_iou = _bucket(max_iou, IOU_BUCKETS)
            b_area = _bucket(area_ratio, AREA_BUCKETS)
            out["gt_iou_hist"][b_iou] += 1
            out["gt_area_hist"][b_area] += 1
            out["gt_iou_area_cross"][b_iou][b_area] += 1
            matched = bool(g.get("matched"))
            out["n_tp"] += int(matched)
            if g.get("is_main"):
                out["n_gt_main"] += 1
                out["n_matched_main"] += int(matched)
            else:
                out["n_gt_secondary"] += 1
                out["n_matched_secondary"] += int(matched)
            nc = int(g.get("n_candidates", 0))
            ck = "0" if nc == 0 else "1" if nc == 1 else "2" if nc == 2 else "3+"
            out["candidates_per_gt"][ck] += 1

    out["match_rate"] = out["n_tp"] / out["n_gt"] if out["n_gt"] else 0.0
    out["match_rate_main"] = (out["n_matched_main"] / out["n_gt_main"]
                              if out["n_gt_main"] else None)
    out["match_rate_secondary"] = (out["n_matched_secondary"] / out["n_gt_secondary"]
                                   if out["n_gt_secondary"] else None)
    out["critical_band_frac"] = (
        out["gt_iou_hist"]["[0.35,0.5)"] / out["n_gt"] if out["n_gt"] else 0.0)
    # 置信度直方图 10 bins,TP/FP 分列 —— TP/FP 分布重叠度 = 排序能力的直观画像。
    # 越界值(∉[0,1])单独计数:np.histogram 会静默丢弃它们(A 路审查实测 1.5 消失),
    # 而生产 conf 越界正是官方"按 0 处理"的违规形态,必须暴露不能吞。
    edges = [i / 10.0 for i in range(11)]
    for arr in (conf_tp, conf_fp):
        if arr:
            a = np.asarray(arr)
            out["conf_out_of_range"] += int(np.sum((a < 0) | (a > 1)))
    out["conf_hist_tp"] = np.histogram(conf_tp, bins=edges)[0].tolist() if conf_tp \
        else [0] * 10
    out["conf_hist_fp"] = np.histogram(conf_fp, bins=edges)[0].tolist() if conf_fp \
        else [0] * 10
    return out

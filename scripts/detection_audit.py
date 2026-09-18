"""检测诊断 · 机制分账纯逻辑库(2026-09-18,plan 任务 A)。

五种漏检机制判据(DR §6.2 原阈值)——只看 max IoU 分不清漏检成因,分账
是首攻仲裁(任务 D)的证据基础。本模块无 I/O、无 torch,只依赖 numpy,
`diagnose_detection.py` 在 case 级采完证据后调用此处分类。

## 五种机制(判据表,DR §6.2)

| 机制 | 名称 | 判据 |
|---|---|---|
| 1 | 阈值过滤 | GT 内软概率 p95≥0.35 且对应位置无最终保留预测连通域 |
| 2 | 粘连合并 | 至少一预测实例同时与 ≥2 个 GT 的 IoU>0.1 |
| 3 | 碎片化 | 覆盖该 GT 的预测碎块 ≥2 个各自 IoU∈[0.15,0.45] |
| 4 | 定位漂移 | 最大 IoU∈[0.35,0.5) 且预测质心与 GT 质心距离>GT 等效半径 |
| 5 | 零响应 | GT 内软概率 max<0.10 |

**多标签 + 主桶**:保存全部命中标志(一个 GT 可同时命中多机制),
主桶按 `5→2→3→4→1` 优先级生成(信息量从"表征完全没学"到"后处理细节"
递减);另设 unclassified/unobservable 不强塞。路线比例按**唯一 GT 机制
标志并集**算,禁重复计数;报告同时展示主桶分布与交叉表。

## 优先级裁决理由(5→2→3→4→1)

5(零响应)是表征问题的最硬证据——软概率都起不来,任何后处理/损失都救
不了;2(粘连)是实例分离结构性缺口,唯一解是换实例级架构(YOLO);3
(碎片)指向解码器/后处理;4(漂移)是定位质量问题;1(阈值)最"浅"
——调阈值就能救的部分,排最后防"浅药方遮住深病根"。

## 软概率采集契约(由 diagnose_detection.py 保证)

静态:`pred_mask` resize 到 GT 原图坐标后只采样 GT 前景像素。
动态:逐标注帧 resize、只对 GT 该帧前景像素采样、逐帧紧凑保存前景概率
向量(内存随标注前景像素增长,不随动态体积增长)。坏帧/缺失帧标
不可观测,不用零概率伪装零响应。

## 机制1 的"存在性"细分

机制1 同时记录阈值化前后/面积过滤前后的局部存在性(soft_local_response
字段):区分"未过阈值"(模型给了概率但没到 0.35)与"被后处理删掉"
(过了阈值、连通域被 min_area/去重删了)。两者药方完全不同。
"""

from __future__ import annotations

import numpy as np

# DR §6.2 判据阈值(常量而非魔法数,判定用未舍入数值——上游传入前不得 round)。
MECH1_SOFT_P95 = 0.35       # 机制1:GT 内 p95 阈值
MECH2_MERGE_IOU = 0.1       # 机制2:预测与 GT 的粘连重叠门槛
MECH3_MIN_IOU = 0.15        # 机制3:碎片覆盖下界
MECH3_MAX_IOU = 0.45        # 机制3:碎片覆盖上界
MECH4_MIN_IOU = 0.35        # 机制4:漂移候选下界(含)
MECH4_MAX_IOU = 0.5         # 机制4:漂移候选上界(不含)
MECH5_SOFT_MAX = 0.10       # 机制5:零响应线

# 主桶优先级(裁决:5→2→3→4→1;列表顺序即优先级,索引 0 最高)。
MECH_PRIORITY = ["mech5", "mech2", "mech3", "mech4", "mech1"]
MECH_NAMES = {"mech1": "阈值过滤", "mech2": "粘连合并", "mech3": "碎片化",
              "mech4": "定位漂移", "mech5": "零响应"}
MECH_UNCLASSIFIED = "unclassified"
MECH_UNOBSERVABLE = "unobservable"

# sidecar v2 schema 版本号:旧 sidecar 缺字段报"证据缺失",不默认补零误判机制5。
SIDECAR_SCHEMA_VERSION = 2

# 软概率统计的容量上限(像素数):超过则该项统计退出并报告,不偷偷换近似分位数。
# 极端前景量(GT 几乎占满整幅 3D 体积)时保护内存;正常静态 GT 前景 ≤ 8k 像素。
SOFT_STATS_PIXEL_CAP = 2_000_000


def make_gt_key(patient_id: str, lesion_id: str) -> str:
    """稳定 GT ID(跨轮次/跨权重的对账锚点)。

    `patient_id::lesion_id` 双段拼接。lesion_id 本身已含 mask_path 或
    `img_dir::roi` 信息(official_loader 权威定义),跨轮不变;同图多 GT
    天然分流。任务 B 的固定人口(恒定分类分)以此键为锚,不取交集。
    """
    return f"{patient_id}::{lesion_id}"


def collect_gt_probability_stats(soft_probs: list) -> dict:
    """GT 前景软概率统计(机制1/5 的证据基础)。

    Args:
        soft_probs: 该 GT 全部前景像素的概率值列表(静态=全图前景一次采;
            动态=各标注帧前景像素合并——调用方已按帧采完合并传入)。
            空列表 = 不可观测(坏帧/采集超容量),**不是零响应**。

    Returns: {"soft_max", "soft_p95", "soft_mean", "soft_valid"(bool),
              "n_pixels"}。soft_valid=False 时其余数值为 None(不默认 0,
            机制判定跳过该 GT,不误判机制5)。
    """
    if not soft_probs:
        return {"soft_max": None, "soft_p95": None, "soft_mean": None,
                "soft_valid": False, "n_pixels": 0}
    a = np.asarray(soft_probs, dtype=np.float64)
    finite = bool(np.all(np.isfinite(a)))
    if not finite:
        # P2-7:NaN 时数值字段 None(docstring 一直是这么承诺的)——返回 NaN
        # 会被 json.dumps(allow_nan=False) 拒写整行。判定路径靠 soft_valid
        # 短路,数值不消费。
        return {"soft_max": None, "soft_p95": None, "soft_mean": None,
                "soft_valid": False, "n_pixels": int(a.size)}
    return {
        "soft_max": float(a.max()),
        "soft_p95": float(np.percentile(a, 95)),
        "soft_mean": float(a.mean()),
        "soft_valid": True,
        "n_pixels": int(a.size),
    }


def classify_miss_mechanisms(gt_row: dict) -> dict:
    """单 GT(漏检)的机制分类。多标签 flags + 主桶 mechanism_primary。

    Args(gt_row,sidecar v2 字段——判定用未舍入数值):
        soft_p95 / soft_valid:机制1/5(缺失或 invalid → 不可观测,两机制跳过)
        pred_ious: 该 GT 与全部预测的 IoU 列表(len=n_pred,无预测=空)
        merge_pred_count: 与 ≥2 个 GT IoU>0.1 的预测数(机制2,case 级预计算)
        is_main 漏检才进分账(matched=True 的 GT 机制无意义,调用方过滤)
        centroid_distance / equivalent_radius / max_iou:机制4
        动态 GT:centroid_distance/equivalent_radius 为标注帧有效均值
        (逐帧二维,缺预测帧单独标 missing_pred_frames;裁决不混 z 轴)

    Returns: {"mechanism_flags": {mech1..mech5: bool|None},
              "mechanism_primary": str}
    flags 值三态:True=命中 / False=未命中 / None=不可观测(证据缺失)。
    primary 按优先级 5→2→3→4→1;全 False→unclassified;**有任一可判机制
    命中时优先报命中**(证据缺失的机制不让位于 unobservable——只软证据
    缺失但 IoU/质心证据充分时,mech2/3/4 仍可裁决);全 False 且任一
    None → unobservable(不拿"没证据"当"零命中"强归类)。
    """
    flags: dict[str, bool | None] = {"mech1": None, "mech2": None, "mech3": None,
                                     "mech4": None, "mech5": None}

    soft_valid = gt_row.get("soft_valid")
    soft_p95 = gt_row.get("soft_p95")
    soft_max = gt_row.get("soft_max")

    # ── 机制5(零响应):GT 内 max(P)<0.10 ─────────────────────────────
    if soft_valid is False or soft_max is None:
        flags["mech5"] = None          # 不可观测 ≠ 零响应(契约:不伪装)
    else:
        flags["mech5"] = bool(soft_max < MECH5_SOFT_MAX)

    # ── 机制1(阈值过滤):p95≥0.35 且无保留预测覆盖 ─────────────────────
    # 无预测实例时 pred_ious 空 → "无保留连通域" 恒成立(零响应未命中时,
    # p95≥0.35 说明模型有响应但产物被删/未过阈 → 机制1)。
    if soft_valid is False or soft_p95 is None:
        flags["mech1"] = None
    else:
        ious = gt_row.get("pred_ious") or []
        # "对应位置无最终保留预测连通域":该 GT 的保留候选(去重后仍在)里
        # 无任何 IoU>MECH2_MERGE_IOU 的覆盖。matched_pred_index<0 已由调用方
        # 保证(漏检才分账);候选检查用保留候选的 IoU(soft_local_response
        # 由调用方另记,细分"未过阈值"vs"被后处理删")。
        kept = gt_row.get("kept_flags") or [True] * len(ious)
        has_kept_cover = any(iou > MECH2_MERGE_IOU
                             for iou, keep in zip(ious, kept) if keep)
        flags["mech1"] = bool(soft_p95 >= MECH1_SOFT_P95 and not has_kept_cover)

    # ── 机制2(粘连合并):该 GT 被多 GT 共享预测覆盖 ─────────────────────
    # merge_pred_count = 与 ≥2 个 GT IoU>0.1 的预测数(case 级预计算,
    # 覆盖本 GT 的部分)。>0 即本 GT 卷入粘连。
    merge_count = gt_row.get("merge_pred_count")
    if merge_count is None:
        flags["mech2"] = None
    else:
        # 粘连判定的完整形态:存在预测与 ≥2 GT 重叠且本 GT 是其中之一。
        # case 级预计算保证 merge_pred_count>0 ⇒ 本 GT 卷入(见
        # aggregate_mechanism_counts 的调用约定)。
        flags["mech2"] = bool(merge_count > 0)

    # ── 机制3(碎片化):≥2 碎块各自 IoU∈[0.15,0.45] ─────────────────────
    ious = gt_row.get("pred_ious") or []
    if not ious:
        flags["mech3"] = False
    else:
        n_frag = sum(1 for iou in ious
                     if MECH3_MIN_IOU <= iou <= MECH3_MAX_IOU)
        flags["mech3"] = bool(n_frag >= 2)

    # ── 机制4(定位漂移):max IoU∈[0.35,0.5) 且质心距>等效半径 ───────────
    # P1-4(Gate2 2026-09-18):max_iou 已在带外时直接 False——max_iou 是漂移
    # 的必要条件,带外就无所谓质心证据缺失;否则无预测(cd/er=None)的 GT
    # 会被拉进 unobservable,而"max_iou=0 明确非漂移"本可判定。
    max_iou = gt_row.get("max_iou")
    cd = gt_row.get("centroid_distance")
    er = gt_row.get("equivalent_radius")
    if max_iou is not None and not (MECH4_MIN_IOU <= max_iou < MECH4_MAX_IOU):
        flags["mech4"] = False
    elif max_iou is None or cd is None or er is None:
        flags["mech4"] = None
    else:
        flags["mech4"] = bool(cd > er)

    # ── 主桶(优先级 5→2→3→4→1)────────────────────────────────────────
    primary = None
    for mech in MECH_PRIORITY:
        if flags.get(mech) is True:
            primary = mech
            break
    if primary is None:
        if any(v is None for v in flags.values()):
            primary = MECH_UNOBSERVABLE
        else:
            primary = MECH_UNCLASSIFIED
    return {"mechanism_flags": flags, "mechanism_primary": primary}


def compute_merge_pred_count(iou_matrix: np.ndarray, gt_index: int) -> int:
    """机制2 的 case 级预计算:覆盖本 GT 的预测里,与 ≥2 个 GT 重叠>0.1 的个数。

    iou_matrix: (n_pred, n_gt)。对每个与 gt_j 的 IoU>0.1 的预测 i,数它跨
    GT 重叠数 row_overlap = Σ_k [iou[i,k]>0.1];>1 即粘连预测。返回覆盖
    本 GT 的粘连预测数。判定"该 GT 卷入粘连"= 返回值>0。
    """
    m = np.asarray(iou_matrix, dtype=np.float64)
    if m.size == 0:
        return 0
    count = 0
    for i in range(m.shape[0]):
        if m[i, gt_index] <= MECH2_MERGE_IOU:
            continue
        row_overlap = int(np.sum(m[i] > MECH2_MERGE_IOU))
        if row_overlap >= 2:
            count += 1
    return count


def compute_centroid_metrics(gt_mask: np.ndarray,
                             pred_masks: list) -> tuple:
    """机制4 的质心距/等效半径(静态 2D;动态由调用方逐帧二维算后取均值)。

    Args:
        gt_mask: (H,W) 二值 GT。
        pred_masks: 候选预测 mask 列表(通常只传最大 IoU 候选一个——漂移
            候选=最大 IoU 对应预测,同 IoU 按稳定预测索引;调用方选好传入)。

    Returns: (centroid_distance, equivalent_radius, nearest_pred_index)
        无预测 → (None, radius, -1);无前景 GT → (None, None, -1)。
        nearest_pred_index = 质心最近的预测索引(漂移候选≠最近质心候选的
        情形,记录供分析;机制4 判定用最大 IoU 候选的质心距——由调用方
        传 masks 时保证,本函数只算最近索引作参考)。
    """
    g = np.asarray(gt_mask) != 0
    ys, xs = np.nonzero(g)
    if ys.size == 0:
        return None, None, -1
    gy, gx = float(ys.mean()), float(xs.mean())
    area = float(ys.size)
    eq_radius = float(np.sqrt(area / np.pi))
    if not pred_masks:
        return None, eq_radius, -1
    best_i, best_d = -1, None
    for i, pm in enumerate(pred_masks):
        p = np.asarray(pm) != 0
        pys, pxs = np.nonzero(p)
        if pys.size == 0:
            continue
        d = float(np.hypot(pys.mean() - gy, pxs.mean() - gx))
        if best_d is None or d < best_d:
            best_d, best_i = d, i
    if best_i < 0:
        return None, eq_radius, -1
    return best_d, eq_radius, best_i


def aggregate_mechanism_counts(gt_rows: list) -> dict:
    """池级机制分账汇总。

    Args: gt_rows = 全部**漏检** GT 的 sidecar v2 行(matched=False,
        soft stats/pred_ious/merge_pred_count/centroid_distance/...)。
        调用方已按唯一 GT(稳定 gt_key 去重)收集,禁重复计数。

    Returns: {
        "n_miss": 总漏检数(分母),
        "primary_hist": {桶名: 数} —— 主桶分布(含 unclassified/unobservable),
        "union_ratios": {机制名: 占比} —— 按多标签标志并集算(路线比例),
        "unobservable_n": 不可观测 GT 数,
        "cross_table": {主桶: {机制: 数}} —— 主桶 × 多标签交叉表,
    }
    """
    n = len(gt_rows)
    primary_hist = {name: 0 for name in
                    MECH_PRIORITY + [MECH_UNCLASSIFIED, MECH_UNOBSERVABLE]}
    union = {name: 0 for name in MECH_PRIORITY}
    cross = {p: {name: 0 for name in MECH_PRIORITY} for p in primary_hist}
    unobs_n = 0
    for row in gt_rows:
        # 再审 P1-B(2026-09-18):行内已有 mechanism_primary/mechanism_flags
        # (diagnose_one_case 在**未舍入值**上算好随行落盘)时直接信任,
        # 不重算——重算会消费 round(6) 后的落盘副本,round 窗口 ≈5e-7
        # 可翻转 mech1/mech4,报告与行内字段逐例矛盾。缺键(手造行/
        # 异源 sidecar)才走 classify 重算兜底,契约不变。
        if "mechanism_primary" in row and "mechanism_flags" in row:
            cls = {"mechanism_primary": row["mechanism_primary"],
                   "mechanism_flags": row["mechanism_flags"]}
        else:
            cls = classify_miss_mechanisms(row)
        p = cls["mechanism_primary"]
        primary_hist[p] += 1
        if p == MECH_UNOBSERVABLE:
            unobs_n += 1
        for mech, hit in cls["mechanism_flags"].items():
            if hit is True:
                union[mech] += 1
                cross[p][mech] += 1
    return {
        "n_miss": n,
        "primary_hist": primary_hist,
        "union_ratios": {k: (v / n if n else 0.0) for k, v in union.items()},
        "unobservable_n": unobs_n,
        "cross_table": cross,
    }

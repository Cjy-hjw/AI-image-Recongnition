"""官方评审口径的验证指标 —— **训练侧代理实现**,不等于线上分。

口径来源:`docs/8.1赛事公开资料/初赛评审规则简要说明.md`(2026-08-12 官方发布)。
各子任务的真实指标与筛选范围逐条复现如下:

| 维度 | 指标 | 金标准侧筛选 |
|---|---|---|
| 良恶性 | AUC-PR(`average_precision_score`) | 仅匹配成功的主病灶 |
| 病灶位置 | 加权宏平均 AUC(`roc_auc_score` ovr/weighted) | 仅匹配成功的主病灶(**不限恶性**) |
| ETE 侵犯 | 加权 Kappa(`cohen_kappa_score`) | 仅匹配成功且**为恶性**的主病灶 |
| 临床 T | AUC(`roc_auc_score`) | 仅匹配成功且**为恶性**的主病灶 |

🔴 **与线上分的三处已知偏差,选模时必须记住**:
  ① **无 IoU 匈牙利匹配**。官方先按 IoU 把预测结节与金标准配对,未配上的不计入
     分类指标;训练/验证用的是 GT mask,等价于"匹配恒成功"。故这里的数只在
     epoch 之间**相对可比**,不能与官方线上分对齐。
  ② **加权 Kappa 的线性/二次官方未指明**。取 `weights="quadratic"`(ETE 4 级是
     有序等级,二次权重对"差 3 级"的惩罚远重于"差 1 级",更贴合临床代价)。
     若官方实为线性,这里的数会系统性偏低/偏高但方向一致,不影响选模排序。
     已入 `docs/待官方答复清单.md`。
  ③ **位置的"加权宏平均"权重方案官方未定义**。取 `average="weighted"`(按各类
     真实样本占比加权)。若官方实为等权 macro,峡部等稀有类的影响会被低估。
     同样入待答复清单。

**病灶级聚合**:官方对动态数据的分类指标"按病灶汇总"——同一病灶的多帧只算
**一个**样本(帧维度只参与匹配与 Dice)。故先按 `lesion_id` 把多帧的概率取均值、
标签取首个(同病灶标签必同),再算指标。漏了这步会让多帧病灶按帧数加权、
指标被静态/动态样本比例扭曲。
"""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

# 类别数一律 import 权威定义,**不得手写字面量**(CLAUDE.md 标签构造铁律):
# 手抄一份后若官方类别数变动,概率分布与标签会系统性错位——JSON 合法、和为 1、
# 测试仍过,只是指标莫名偏低,本地永远查不出根因。
from src.models.heads import NUM_ETE_CLASSES
from src.models.label_schema import NUM_LOCATIONS


def aggregate_by_lesion(records: List[Dict]) -> List[Dict]:
    """按 `lesion_id` 聚合:概率取均值、标签/validity 取首帧。

    同一病灶的多帧标签必然相同(来自 xlsx 同一行),故取首个即可;概率取算术均值
    是最朴素的帧级融合,与推理侧 `video_predict` 的聚合口径保持一致的前提下
    才有可比性——若推理侧改用取最大/中位数,这里要同步改,否则验证指标与
    交付表现脱节。
    """
    order: List[str] = []
    buckets: Dict[str, List[Dict]] = {}
    for r in records:
        key = r["lesion_id"]
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(r)

    out = []
    for key in order:
        group = buckets[key]
        head = dict(group[0])
        head["n_frames"] = len(group)
        head["mal_prob"] = float(np.mean([g["mal_prob"] for g in group]))
        head["t_conf"] = float(np.mean([g["t_conf"] for g in group]))
        head["location_probs"] = np.mean(
            [g["location_probs"] for g in group], axis=0
        ).tolist()
        head["ete_probs"] = np.mean([g["ete_probs"] for g in group], axis=0).tolist()
        out.append(head)
    return out


def _safe_auc(y_true, y_score) -> Optional[float]:
    """单类/空集返回 None 而非抛异常 —— 早期 epoch 的 val 子集常只剩一类。"""
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _malignancy_auc_pr(rows: List[Dict]) -> Optional[float]:
    sel = [r for r in rows if r["is_main"] == 1 and r["valid_malignancy"] == 1]
    y = [r["gt_malignancy"] for r in sel]
    if len(y) < 2 or len(set(y)) < 2:
        return None
    return float(average_precision_score(y, [r["mal_prob"] for r in sel]))


def _location_weighted_auc(rows: List[Dict]) -> Optional[float]:
    """位置 = 加权宏平均 AUC。**不限恶性**(见模块 docstring ③ 与原文歧义留档)。

    `roc_auc_score` 的 multi_class='ovr' 要求 y_score 每行和为 1 且列数 == 出现的
    类别数。金标准里缺席的类必须先从概率矩阵里剔除并重新归一化,否则 sklearn
    报 "Number of classes in y_true not equal to the number of columns"。
    重归一化不改变 ovr 的排序(同一行内的相对大小不变),故不影响 AUC。

    🔴 **恰好只出现 2 类时必须走二分类分支**:sklearn 对 2 类**不接受** [N,2] 概率
    矩阵 + multi_class='ovr',直接报 `y should be a 1d array, got an array of
    shape (N, 2)`(实测,`tests/test_metrics_official.py` 两条用例钉死)。此时须传
    正类那一列的 1-D 分数。早期 epoch 的小 val 子集经常只覆盖 2 个位置类,
    漏这个分支会让整个验证抛异常、中断训练。
    """
    sel = [r for r in rows if r["is_main"] == 1 and r["valid_lesion_loc"] == 1]
    y = [r["gt_lesion_loc"] for r in sel]
    present = sorted(set(y))
    if len(present) < 2:
        return None
    probs = np.asarray([r["location_probs"] for r in sel], dtype=np.float64)
    probs = probs[:, present]
    row_sum = probs.sum(axis=1, keepdims=True)
    # 全零行(理论不达:softmax 输出恒 > 0)会产生 nan,退化为均匀分布保证可算
    probs = np.where(row_sum > 0, probs / np.maximum(row_sum, 1e-12), 1.0 / len(present))
    remap = {c: i for i, c in enumerate(present)}
    y_mapped = [remap[v] for v in y]
    if len(present) == 2:
        # 2 类时 ovr 的加权宏平均恒等于单一 AUC(两个 one-vs-rest 问题互补、
        # AUC 相同),故直接传正类列分数,与多类分支口径一致。
        return float(roc_auc_score(y_mapped, probs[:, 1]))
    return float(
        roc_auc_score(y_mapped, probs, multi_class="ovr", average="weighted")
    )


def _ete_weighted_kappa(rows: List[Dict]) -> Optional[float]:
    """ETE = 加权 Kappa,**仅恶性主病灶**。Kappa 吃硬预测,故对概率取 argmax。"""
    sel = [
        r
        for r in rows
        if r["is_main"] == 1 and r["valid_extrathy_ext"] == 1 and r["gt_malignancy"] == 1
    ]
    if len(sel) < 2:
        return None
    y_true = [r["gt_extrathy_ext"] for r in sel]
    y_pred = [int(np.argmax(r["ete_probs"])) for r in sel]
    # 全体一致(真值与预测都只有一个类)时 Kappa 未定义,sklearn 返回 nan
    if len(set(y_true)) < 2 and len(set(y_pred)) < 2:
        return None
    k = cohen_kappa_score(
        y_true, y_pred, labels=list(range(NUM_ETE_CLASSES)), weights="quadratic"
    )
    return None if np.isnan(k) else float(k)


def _clinical_t_auc(rows: List[Dict]) -> Optional[float]:
    """T = AUC,**仅恶性主病灶**。`t_conf` 就是 P(T=1),直接当 score。"""
    sel = [
        r
        for r in rows
        if r["is_main"] == 1 and r["valid_clin_t_stage"] == 1 and r["gt_malignancy"] == 1
    ]
    return _safe_auc([r["gt_clin_t_stage"] for r in sel], [r["t_conf"] for r in sel])


def compute_official_metrics(records: List[Dict]) -> Dict[str, Optional[float]]:
    """从 `collect_official_records` 的逐样本记录算全部官方口径指标。

    Args:
        records: `validate_step.collect_official_records` 的输出(可跨批 extend)。
    Returns:
        dict:`malignancy_auc_pr` / `location_weighted_auc` / `ete_weighted_kappa`
        / `clinical_t_auc` / `n_lesions`。任一指标不可算(样本不足/单类)→ None,
        **不抛异常**(早期 epoch 的 val 子集常只剩一类,抛异常会中断整个训练)。

    `composite` 是**我方自定义的选模标量**,不是官方分:取全部可算指标的算术均值。
    官方 60%/40% 的内部子权重未公布,故不做加权;这只是"哪个 checkpoint 更好"的
    单调代理,不得对外声称为预估得分。
    """
    if not records:
        return {
            "malignancy_auc_pr": None,
            "location_weighted_auc": None,
            "ete_weighted_kappa": None,
            "clinical_t_auc": None,
            "composite": None,
            "n_lesions": 0,
        }

    rows = aggregate_by_lesion(records)
    metrics: Dict[str, Optional[float]] = {
        "malignancy_auc_pr": _malignancy_auc_pr(rows),
        "location_weighted_auc": _location_weighted_auc(rows),
        "ete_weighted_kappa": _ete_weighted_kappa(rows),
        "clinical_t_auc": _clinical_t_auc(rows),
    }
    usable = [v for v in metrics.values() if v is not None]
    metrics["composite"] = float(np.mean(usable)) if usable else None
    metrics["n_lesions"] = len(rows)
    return metrics

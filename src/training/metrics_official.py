"""官方评审口径的验证指标 —— **训练侧代理实现**,不等于线上分。

口径来源:`docs/8.1赛事公开资料/初赛评审规则简要说明.md`(2026-08-12 官方发布)。
各子任务的真实指标与筛选范围逐条复现如下:

| 维度 | 指标 | 金标准侧筛选 |
|---|---|---|
| 良恶性 | AUC-PR(`average_precision_score`) | 仅匹配成功的主病灶 |
| 病灶位置 | 加权宏平均 AUC(`roc_auc_score` ovr/weighted) | 仅匹配成功的主病灶(**不限恶性**) |
| ETE 侵犯 | 加权 Kappa(`cohen_kappa_score`) | 仅匹配成功且**为恶性**的主病灶 |
| 临床 T | AUC(`roc_auc_score`) | 仅匹配成功且**为恶性**的主病灶 |
| TI-RADS | 加权 Kappa(`cohen_kappa_score`) | 仅匹配成功的主病灶(**不限恶性**,与位置同、与 ETE/T 不同) |

🔴 **与线上分的三处已知偏差,选模时必须记住**:
  ① **无 IoU 匈牙利匹配**。官方先按 IoU 把预测结节与金标准配对,未配上的不计入
     分类指标;训练/验证用的是 GT mask,等价于"匹配恒成功"。故这里的数只在
     epoch 之间**相对可比**,不能与官方线上分对齐。
  ② **加权 Kappa 的线性/二次官方未指明**。取 `weights="quadratic"`(ETE 4 级/
     TI-RADS 5 级均为有序等级,二次权重对"差 3 级"的惩罚远重于"差 1 级",更贴合
     临床代价)。若官方实为线性,这里的数会系统性偏低/偏高但方向一致,不影响
     选模排序。已入 `docs/待官方答复清单.md`。
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
from src.models.heads import NUM_ETE_CLASSES, NUM_TIRADS_CLASSES
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
        # backlog #128 的 `val_logit_mean_per_class` 监控用。**必须与 probs 一起
        # 聚合**:漏了它,多帧病灶的 logits 会停在首帧值、与已聚合的 probs 不同步,
        # 而两者在监控里是并排看的(probs 看预测分布、logits 看绝对偏移量级)。
        # 🔴 判据是**组内全部帧都有**,不是"首帧有"(2026-08-20 Gate2 抓到):
        # 原写法 `if "ete_logits" in head` 只看首帧(`head = dict(group[0])`),
        # 首帧有而后续帧无时 `[g["ete_logits"] for g in group]` 直接 KeyError。
        # 生产路径不可达(`validate_step.collect_official_records:106` 无条件采集,
        # 同批 records 结构齐一),拦的是**混合来源**——legacy records 与新 records
        # 拼进同一 lesion_id。与 `ete_monitor_metrics` 的 `all("ete_logits" in r ...)`
        # 同一判据(那处本就写对了,此处是漏了)。
        if all("ete_logits" in g for g in group):
            head["ete_logits"] = np.mean(
                [g["ete_logits"] for g in group], axis=0
            ).tolist()
        else:
            # 部分帧缺失 → 整个病灶不给该键,让下游监控走 None 分支(不猜、不填零:
            # 填零会被 `ete_logit_mean_per_class` 当成真实 logits 均值报出去)。
            head.pop("ete_logits", None)
        head["tirads_probs"] = np.mean(
            [g["tirads_probs"] for g in group], axis=0
        ).tolist()
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


def _ete_selection(rows: List[Dict]) -> List[Dict]:
    """ETE 评价子集:主病灶 + 有 ETE 标注 + **金标准为恶性**。

    抽成函数供 `_ete_weighted_kappa` 与 backlog #128 的 4 项监控共用 ——
    监控必须与被监控的指标**同一子集**,否则"监控说预测分布正常、kappa 却在跌"
    这种自相矛盾无从排查(两者看的根本不是同一批样本)。
    """
    return [
        r
        for r in rows
        if r["is_main"] == 1 and r["valid_extrathy_ext"] == 1 and r["gt_malignancy"] == 1
    ]


def _ete_weighted_kappa(rows: List[Dict]) -> Optional[float]:
    """ETE = 加权 Kappa,**仅恶性主病灶**。Kappa 吃硬预测,故对概率取 argmax。"""
    sel = _ete_selection(rows)
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


def ete_monitor_metrics(rows: List[Dict]) -> Dict[str, Optional[object]]:
    """backlog #128 的 4 项 ETE 诊断监控 —— **只告警不 abort**。

    出处:`reference/2026-08月调研/极端不平衡四类有序标签下的加权 Kappa 优化与
    ETE 预测策略研究报告.md` §5(`:118-138`)。DR 设计这四项是为了「在单轮训练中
    同时改善输入质量与损失驱动方向,**并精准区分两者的贡献**」——2-①(只关 CB)
    与 2-②(开 LA+SoftQWK)的 A/B 归因全靠它们,没有这四项就只能看一个 kappa
    标量猜原因。

    🔴 **不做 abort**:DR 全文无 "abort"/"early stop",最强用词是「必须立即判定
    ETE 优化方向失败」(`:128`)。ETE 只是目标四三项之一,为它停掉整轮训练
    (连带分割/良恶性/位置/T 全部作废)不划算 —— 判断留给人看日志。

    Returns:
        dict,四个键各自可为 None(样本不足时):
        - `ete_pred_hist` [P0,P1,P2,P3]:argmax 后各类占比。DR 预警阈值:
          正常 `P0` 应在 85-95%;`P3 > 2%` 或 `P0 < 80%` = **少数类过度激励**,
          kappa 即将转负(本项目 β=0.99 那轮就是这么崩的)。
        - `ete_extreme_error_rate`:混淆矩阵中 `|i−j| ≥ 2` 的样本比例。DR 阈值:
          上升超 1.5% → 无论 loss 是否下降,判定优化方向失败(二次距离惩罚
          `W_ij` 正在迅速积聚)。
        - `ete_kappa_gap`:Soft-Kappa(概率矩阵算)− 硬 Argmax-Kappa。DR 判据:
          Soft 升到 0.4 但 Argmax 恒 0 = softmax 峰值不够陡、跨不过 argmax 门槛;
          Soft 正而 Argmax 负 = 长尾不确定性严重,需提高 `la_tau`。
        - `ete_logit_mean_per_class` [z̄0..z̄3]:各类 **logits** 均值。DR 预警:
          `z̄3 − z̄0 > 0` = 对稀有类过度偏移。**必须用 logits 不能用 probs** ——
          softmax 平移不变,probs 反推不出绝对量级(见 `validate_step` 采集处注释)。

    ⚠️ **Soft-Kappa 的口径**:DR 定义是逐样本概率矩阵算,而这里的 `rows` 已是
    **病灶级聚合**后的(动态数据同一病灶多帧已取均值)。选定按病灶级算 ——
    与 `_ete_weighted_kappa` 同口径,两者的差(`kappa_gap`)才有意义;若一个按帧
    一个按病灶,差值会掺进"聚合方式不同"这个无关变量。
    """
    empty: Dict[str, Optional[object]] = {
        "ete_pred_hist": None,
        "ete_extreme_error_rate": None,
        "ete_kappa_gap": None,
        "ete_logit_mean_per_class": None,
    }
    sel = _ete_selection(rows)
    if len(sel) < 2:
        return empty

    y_true = np.asarray([r["gt_extrathy_ext"] for r in sel], dtype=np.int64)
    probs = np.asarray([r["ete_probs"] for r in sel], dtype=np.float64)  # [N,C]
    # 🔴 长度契约必须硬校验(2026-08-20 Gate2 抓到,与推理侧 `decode_predictions`
    # 同款):`y_pred = probs.argmax(axis=1)` 的上界由**概率向量长度**决定,而
    # `np.bincount(...)[:NUM_ETE_CLASSES]` 会把 idx >= 4 的样本**静默切掉** ——
    # 实测 ete_probs 给 5 列时,直方图占比和变成 **0.750** 而不是 1.0,不报错、
    # 数字只是悄悄不对(比崩溃阴)。而 `ete_pred_hist` 的判读全靠 "P0 应在 85-95%"
    # 这类阈值,分母被静默改小会让预警阈值整体失真、误判"少数类过度激励"。
    # 生产路径不可达(`ExtensionHead` 输出恒 4 类),拦的是未来改类别数时的漏改。
    assert probs.shape[1] == NUM_ETE_CLASSES, (
        f"ete_probs 长度应为 {NUM_ETE_CLASSES}，got {probs.shape[1]}"
        " —— bincount 切片会静默丢弃越界预测，直方图占比和不再为 1"
    )
    y_pred = probs.argmax(axis=1)
    n = len(sel)
    out: Dict[str, Optional[object]] = {}

    # ① 硬预测分布直方图
    hist = np.bincount(y_pred, minlength=NUM_ETE_CLASSES)[:NUM_ETE_CLASSES] / n
    out["ete_pred_hist"] = [round(float(v), 6) for v in hist]

    # ② 大跨度误判率 |i−j| >= 2
    out["ete_extreme_error_rate"] = round(
        float(np.mean(np.abs(y_true - y_pred) >= 2)), 6
    )

    # ③ Soft-Kappa − 硬 Kappa。两者都可能不可算(全体一致 → nan),任一不可算即
    #    整项 None ——「一个有值一个没有」算不出差,填 0 会伪装成"没有背离"。
    hard_k = _ete_weighted_kappa(rows)
    soft_k = _soft_kappa(y_true, probs)
    out["ete_kappa_gap"] = (
        None if (hard_k is None or soft_k is None) else round(soft_k - hard_k, 6)
    )

    # ④ 各类 logits 均值。键缺席(legacy records)→ None,不猜、不拿 probs 顶替。
    if all("ete_logits" in r for r in sel):
        logits = np.asarray([r["ete_logits"] for r in sel], dtype=np.float64)
        out["ete_logit_mean_per_class"] = [
            round(float(v), 6) for v in logits.mean(axis=0)
        ]
    else:
        out["ete_logit_mean_per_class"] = None
    return out


def _soft_kappa(y_true: np.ndarray, probs: np.ndarray) -> Optional[float]:
    """连续 Soft-Kappa:用概率矩阵替代硬预测算二次加权 Kappa。

    与 `soft_qwk_loss`(`losses/components/extension.py`)**同一套 O/E 数学**,
    但那边返回 `1−κ`(损失,越小越好)、这里返回 `κ` 本身(指标,越大越好),
    便于与 `_ete_weighted_kappa` 的硬 Kappa 直接相减。

    `den == 0`(期望一致度为 0,如真值只有一类且预测也集中在同一类)时 Kappa
    未定义 → None,与硬 Kappa 的 nan 处理对齐。
    """
    c = probs.shape[1]
    idx = np.arange(c, dtype=np.float64)
    W = (idx[None, :] - idx[:, None]) ** 2 / max(c - 1, 1) ** 2
    one_hot = np.eye(c, dtype=np.float64)[y_true]
    n = len(y_true)
    O = one_hot.T @ probs / n
    E = np.outer(one_hot.sum(axis=0) / n, probs.sum(axis=0) / n)
    den = float((W * E).sum())
    if den <= 0:
        return None
    return float(1.0 - (W * O).sum() / den)


def _clinical_t_auc(rows: List[Dict]) -> Optional[float]:
    """T = AUC,**仅恶性主病灶**。`t_conf` 就是 P(T=1),直接当 score。"""
    sel = [
        r
        for r in rows
        if r["is_main"] == 1 and r["valid_clin_t_stage"] == 1 and r["gt_malignancy"] == 1
    ]
    return _safe_auc([r["gt_clin_t_stage"] for r in sel], [r["t_conf"] for r in sel])


def _tirads_weighted_kappa(rows: List[Dict]) -> Optional[float]:
    """TI-RADS = 加权 Kappa,**不限恶性**(评审规则原文无"且为恶性"限定,与位置
    同、与 ETE/T 不同——`docs/8.1赛事公开资料/初赛评审规则简要说明.md` 逐条核对
    过,良性主病灶必须参与该项统计)。Kappa 吃硬预测,故对概率取 argmax。"""
    sel = [r for r in rows if r["is_main"] == 1 and r["valid_tirads"] == 1]
    if len(sel) < 2:
        return None
    y_true = [r["gt_tirads"] for r in sel]
    y_pred = [int(np.argmax(r["tirads_probs"])) for r in sel]
    # 全体一致(真值与预测都只有一个类)时 Kappa 未定义,sklearn 返回 nan
    if len(set(y_true)) < 2 and len(set(y_pred)) < 2:
        return None
    k = cohen_kappa_score(
        y_true, y_pred, labels=list(range(NUM_TIRADS_CLASSES)), weights="quadratic"
    )
    return None if np.isnan(k) else float(k)


def _sanitize_seg_dice(seg_dice) -> Optional[float]:
    """把外部传入的 Dice 规整成"可用则 float、不可用则 None"。

    🔴 **只接受 [0,1] 内的有限值**。Dice 的定义域就是 [0,1],越界即"算坏了",
    一律整体丢弃(`seg_dice` 键写 None、也不进 composite)。

    **丢弃而非 clamp**:clamp 会把"算坏了"伪装成一个看似合理的分数,与
    `jsonl_logger._drop_nonfinite` 是同一个取舍(缺字段一眼看得出异常,
    填个数字则永远发现不了)。

    **为什么不能只判 `np.isfinite`**(Gate2 lane1 实测):`isfinite(-0.1)` 为 True,
    负 dice 会被放行并拉低 composite;而 NaN 一旦混进 `np.mean`,composite 也是
    NaN → 选模的 `composite > best_composite` 恒 False → 整轮训练一个 checkpoint
    都存不下来。两种都必须在入口拦掉。

    真实路径下二者均不可达 —— 除零不可达(`validate_step` 的 dice 分母恒带 +1e-6
    平滑),NaN 仅在前向/损失已数值失控时理论可达。**这是防御性入口校验,不是在修
    已发生的 bug;但也不要因为"不可达"就删掉它** —— 护栏的意义正是防不该发生的事,
    且 NaN 混进 `np.mean` 的后果(整轮存不下 checkpoint)远大于这几行的成本。
    """
    if seg_dice is None:
        return None
    v = float(seg_dice)
    return v if (np.isfinite(v) and 0.0 <= v <= 1.0) else None


def compute_official_metrics(
    records: List[Dict], seg_dice=None
) -> Dict[str, Optional[object]]:
    """从 `collect_official_records` 的逐样本记录算全部官方口径指标。

    Args:
        records: `validate_step.collect_official_records` 的输出(可跨批 extend)。
        seg_dice: 本轮 val 的平均分割 Dice(`validate_epoch` 的 `avg_dice`)。给了就
            **并入 composite**;None(默认)则 composite 仍是纯分类均值,老调用点零回归。
            **只接受 [0,1] 内的有限值**,越界/NaN/Inf 一律整体丢弃(`seg_dice` 键写
            None、不进 composite)。
    Returns:
        dict:`malignancy_auc_pr` / `location_weighted_auc` / `ete_weighted_kappa`
        / `clinical_t_auc` / `tirads_weighted_kappa` / `seg_dice` / `n_lesions`。任一
        指标不可算(样本不足/单类)→ None,**不抛异常**(早期 epoch 的 val 子集常只剩
        一类,抛异常会中断整个训练)。

    `composite` 是**我方自定义的选模标量**,不是官方分:取全部可算指标的算术均值。
    官方 60%/40% 的内部子权重未公布,故不做加权;这只是"哪个 checkpoint 更好"的
    单调代理,不得对外声称为预估得分。

    🔴 **seg_dice 必须能进 composite(2026-08-16)**:分割是官方评分的**闸门**——
    预测 mask 全零 → 匈牙利 IoU≥0.5 匹配零命中 → 五项分类/分期全部零样本 → Sclin≈0
    (`docs/项目现状总纲.md` §6③)。此前 composite 只取 5 个分类指标均值、完全不含
    dice,即便分割训起来了,选模也会挑"分类略好但分割崩了"的 checkpoint,重演
    2026-08-15 那轮的覆辙。
    """
    if not records:
        return {
            "malignancy_auc_pr": None,
            "location_weighted_auc": None,
            "ete_weighted_kappa": None,
            "clinical_t_auc": None,
            "tirads_weighted_kappa": None,
            "seg_dice": _sanitize_seg_dice(seg_dice),
            "composite": None,
            "n_lesions": 0,
            # 四项诊断监控在空 records 下同样返回 None(键必须在,否则调用方
            # `metrics.get(k)` 与 `k in metrics` 两种写法行为不一致)。
            "ete_pred_hist": None,
            "ete_extreme_error_rate": None,
            "ete_kappa_gap": None,
            "ete_logit_mean_per_class": None,
        }

    rows = aggregate_by_lesion(records)
    # 值类型是 `object`(不是 `float`):除评分指标外还要并入 backlog #128 的诊断
    # 监控,其中 `ete_pred_hist`/`ete_logit_mean_per_class` 是 list[float]。
    metrics: Dict[str, Optional[object]] = {
        "malignancy_auc_pr": _malignancy_auc_pr(rows),
        "location_weighted_auc": _location_weighted_auc(rows),
        "ete_weighted_kappa": _ete_weighted_kappa(rows),
        "clinical_t_auc": _clinical_t_auc(rows),
        "tirads_weighted_kappa": _tirads_weighted_kappa(rows),
    }
    usable: List[float] = [
        float(v) for v in metrics.values() if isinstance(v, (int, float))
    ]
    # seg_dice 单独加在 usable 之后(而非放进 metrics 再统一筛):它不是"官方口径分类
    # 指标",放进 metrics 字典参与上面的 dict 遍历会让含义混淆;但它必须参与均值。
    # 值域/有限性校验统一在 `_sanitize_seg_dice`(两个 return 分支共用同一份判据)。
    seg_val = _sanitize_seg_dice(seg_dice)
    if seg_val is not None:
        usable.append(seg_val)
    metrics["seg_dice"] = seg_val
    metrics["composite"] = float(np.mean(usable)) if usable else None
    metrics["n_lesions"] = len(rows)
    # 🔴 backlog #128 的 4 项 ETE 诊断监控 **必须加在 composite 算完之后**:
    # 它们是诊断量(直方图/误判率/logits 均值),不是"越大越好"的评分指标 ——
    # 混进 usable 会让 composite 变成一个语义不明的数(选模判据被诊断量污染,
    # 且 `ete_pred_hist` 是 list、`np.mean` 会直接报错或静默产出错误标量)。
    metrics.update(ete_monitor_metrics(rows))
    return metrics

"""`src/training/metrics_official.py` 官方口径指标单测。

重点不是"函数能跑",而是**口径是否真的复现了官方规则**——每条筛选/聚合规则都用
一个"若实现漏掉该规则则必然给出不同数字"的反例钉死:
  - 病灶级聚合:多帧病灶不得按帧数加权(漏聚合 → 数字被帧数扭曲);
  - 主病灶筛选:`is_main==0` 的次要病灶必须被排除;
  - ETE/T 的"仅恶性":良性主病灶必须被排除(而位置/良恶性**不**限恶性);
  - validity=0 的样本必须被排除(不是当成某个默认类);
  - 指标函数选型:AUC-PR vs AUC vs 加权Kappa,用手算值对拍。
"""
import numpy as np
import pytest

from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from src.training.metrics_official import (
    aggregate_by_lesion,
    compute_official_metrics,
)


def _rec(lesion_id, *, mal=1, mal_p=0.5, loc=0, loc_probs=None, ete=0,
         ete_probs=None, t=0, t_conf=0.5, tirads=0, tirads_probs=None, is_main=1,
         v_mal=1, v_loc=1, v_ete=1, v_t=1, v_tirads=1, patient="P"):
    """造一条 `collect_official_records` 形状的记录。默认全 valid、主病灶、恶性。"""
    if loc_probs is None:
        loc_probs = [0.0] * 7
        loc_probs[loc] = 1.0
    if ete_probs is None:
        ete_probs = [0.0] * 4
        ete_probs[ete] = 1.0
    if tirads_probs is None:
        tirads_probs = [0.0] * 5
        tirads_probs[tirads] = 1.0
    return {
        "lesion_id": lesion_id,
        "patient_id": patient,
        "is_main": is_main,
        "mal_prob": mal_p,
        "gt_malignancy": mal,
        "valid_malignancy": v_mal,
        "location_probs": list(loc_probs),
        "gt_lesion_loc": loc,
        "valid_lesion_loc": v_loc,
        "ete_probs": list(ete_probs),
        "gt_extrathy_ext": ete,
        "valid_extrathy_ext": v_ete,
        "t_conf": t_conf,
        "gt_clin_t_stage": t,
        "valid_clin_t_stage": v_t,
        "tirads_probs": list(tirads_probs),
        "gt_tirads": tirads,
        "valid_tirads": v_tirads,
    }


def test_empty_records_returns_all_none():
    m = compute_official_metrics([])
    assert m["n_lesions"] == 0 and m["composite"] is None
    for k in ("malignancy_auc_pr", "location_weighted_auc",
              "ete_weighted_kappa", "clinical_t_auc", "tirads_weighted_kappa"):
        assert m[k] is None


def test_aggregate_averages_probs_and_keeps_first_labels():
    """同 lesion_id 多帧 → 概率取均值、标签取首帧、n_frames 记帧数。"""
    recs = [
        _rec("L1", mal_p=0.2, t_conf=0.1, loc_probs=[1, 0, 0, 0, 0, 0, 0]),
        _rec("L1", mal_p=0.8, t_conf=0.9, loc_probs=[0, 1, 0, 0, 0, 0, 0]),
        _rec("L2", mal_p=0.6, t_conf=0.4),
    ]
    rows = aggregate_by_lesion(recs)
    assert [r["lesion_id"] for r in rows] == ["L1", "L2"]        # 保持首次出现顺序
    l1 = rows[0]
    assert l1["n_frames"] == 2
    assert l1["mal_prob"] == pytest.approx(0.5)
    assert l1["t_conf"] == pytest.approx(0.5)
    assert l1["location_probs"][:2] == pytest.approx([0.5, 0.5])
    assert rows[1]["n_frames"] == 1


def test_aggregate_does_not_mutate_input_records():
    """浅拷贝陷阱:聚合不得改写传入记录的 list 字段(否则跨 epoch 累积污染)。"""
    recs = [_rec("L1", mal_p=0.2, loc_probs=[1, 0, 0, 0, 0, 0, 0]),
            _rec("L1", mal_p=0.8, loc_probs=[0, 1, 0, 0, 0, 0, 0])]
    before = [list(r["location_probs"]) for r in recs]
    aggregate_by_lesion(recs)
    assert [r["location_probs"] for r in recs] == before
    assert recs[0]["mal_prob"] == 0.2 and "n_frames" not in recs[0]


def test_lesion_aggregation_changes_the_number():
    """反证聚合真的生效:一个 5 帧病灶 + 一个 1 帧病灶,漏聚合会被 5 帧压倒。

    构造:L1(5 帧,恶性,mal_prob=1.0)、L2(1 帧,良性,mal_prob=1.0 = 全错)。
    正确(病灶级):2 个样本,预测分数相同 → AP = 0.5(基线率)。
    错误(样本级):6 个样本,5 正 1 负、分数相同 → AP = 5/6 ≈ 0.833。
    两者差 0.33,任何漏聚合的实现都会给出后者。
    """
    recs = [_rec("L1", mal=1, mal_p=1.0) for _ in range(5)]
    recs.append(_rec("L2", mal=0, mal_p=1.0))
    m = compute_official_metrics(recs)
    assert m["n_lesions"] == 2
    assert m["malignancy_auc_pr"] == pytest.approx(0.5, abs=1e-6)
    # 明确排除样本级口径的数字
    assert m["malignancy_auc_pr"] != pytest.approx(5 / 6, abs=1e-3)


def test_secondary_lesions_excluded_from_all_metrics():
    """`is_main==0` 必须被全部指标排除。放行会拉进官方根本不评的次要病灶。"""
    good = [_rec("L1", mal=1, mal_p=0.9), _rec("L2", mal=0, mal_p=0.1)]
    m_clean = compute_official_metrics(good)
    # 掺入 10 个预测完全错的次要病灶
    noisy = good + [
        _rec(f"S{i}", mal=i % 2, mal_p=1.0 - (i % 2), is_main=0) for i in range(10)
    ]
    m_noisy = compute_official_metrics(noisy)
    assert m_noisy["malignancy_auc_pr"] == pytest.approx(m_clean["malignancy_auc_pr"])
    assert m_noisy["location_weighted_auc"] == pytest.approx(
        m_clean["location_weighted_auc"]
    )
    assert m_noisy["n_lesions"] == 12          # 聚合仍统计全部,筛选只在指标内部


def test_ete_and_t_exclude_benign_but_location_does_not():
    """🔴 官方口径的**不对称**:ETE/T 仅恶性主病灶,位置/良恶性不限恶性。

    构造两个良性病灶,其位置标注互不相同、ETE/T 标注也互不相同。
      - 位置指标:良性参与 → 4 个样本、2 类以上 → 可算;
      - ETE/T:良性被排除 → 只剩 2 个恶性且各自单类 → 不可算(None)。
    若实现把"仅恶性"错加到位置上,location 会变 None;若漏加到 ETE/T,它们会有值。
    """
    recs = [
        _rec("M1", mal=1, loc=0, ete=0, t=0),
        _rec("M2", mal=1, loc=1, ete=0, t=0),
        _rec("B1", mal=0, loc=3, ete=1, t=1),
        _rec("B2", mal=0, loc=4, ete=3, t=1),
    ]
    m = compute_official_metrics(recs)
    assert m["location_weighted_auc"] is not None, "位置不该被'仅恶性'过滤"
    assert m["ete_weighted_kappa"] is None, "ETE 必须排除良性 → 剩余单类不可算"
    assert m["clinical_t_auc"] is None, "T 必须排除良性 → 剩余单类不可算"


def test_tirads_includes_benign_unlike_ete_and_t():
    """🔴 TI-RADS 与位置同属"不限恶性",与 ETE/T **不同**(评审规则原文核对过,
    「TI-RADS加权Kappa,仅匹配成功的主病灶」不带"且为恶性"限定)。

    构造 2 恶性(TI-RADS 标注互不相同)+ 2 良性(TI-RADS 标注也互不相同)—— 若实现
    误加了"仅恶性",良性会被过滤、只剩 2 恶性各自单类 → None;正确实现应保留
    全部 4 样本、可算出具体值。
    """
    recs = [
        _rec("M1", mal=1, tirads=0),
        _rec("M2", mal=1, tirads=1),
        _rec("B1", mal=0, tirads=3),
        _rec("B2", mal=0, tirads=4),
    ]
    m = compute_official_metrics(recs)
    assert m["tirads_weighted_kappa"] is not None, "TI-RADS 不该被'仅恶性'过滤"


def test_tirads_uses_quadratic_weighted_kappa_on_argmax():
    """TI-RADS = 加权(quadratic)Kappa,吃 argmax 硬预测。与 sklearn 手算对拍,
    同时反证"不是线性 Kappa"(同 ETE 的钉法)。"""
    def tp(cls):
        v = [0.0] * 5
        v[cls] = 1.0
        return v

    pairs = [(0, 0), (1, 4), (2, 2), (4, 1), (0, 1), (4, 4), (3, 0)]
    recs = [
        _rec(f"L{i}", mal=(i % 2), tirads=gt, tirads_probs=tp(pred))
        for i, (gt, pred) in enumerate(pairs)
    ]
    got = compute_official_metrics(recs)["tirads_weighted_kappa"]

    y_true = [gt for gt, _ in pairs]
    y_pred = [pred for _, pred in pairs]
    quad = cohen_kappa_score(y_true, y_pred, labels=list(range(5)), weights="quadratic")
    lin = cohen_kappa_score(y_true, y_pred, labels=list(range(5)), weights="linear")
    assert got == pytest.approx(float(quad))
    assert not np.isclose(quad, lin), "反例失效:本组数据的线性/二次 Kappa 恰好相同"


def test_invalid_samples_excluded_not_defaulted():
    """validity=0 必须被排除,而不是当成类 0 参与计算。

    构造:2 个有效样本(位置 0 / 1,预测正确)+ 8 个 valid_lesion_loc=0 的样本
    (标签全 0、预测全错)。排除 → AUC=1.0;若当成类 0 参与 → 远低于 1.0。
    """
    recs = [
        _rec("L1", loc=0, loc_probs=[0.9, 0.1, 0, 0, 0, 0, 0]),
        _rec("L2", loc=1, loc_probs=[0.1, 0.9, 0, 0, 0, 0, 0]),
    ]
    recs += [
        _rec(f"X{i}", loc=0, loc_probs=[0.0, 1.0, 0, 0, 0, 0, 0], v_loc=0)
        for i in range(8)
    ]
    m = compute_official_metrics(recs)
    assert m["location_weighted_auc"] == pytest.approx(1.0)


def test_location_handles_absent_classes_and_matches_sklearn():
    """金标准只出现 {0,3,6} 时不得报 sklearn 的类数不匹配,且数值与手算一致。"""
    def probs(p0, p3, p6):
        v = [0.0] * 7
        v[0], v[3], v[6] = p0, p3, p6
        return v

    recs = [
        _rec("A", loc=0, loc_probs=probs(0.7, 0.2, 0.1)),
        _rec("B", loc=3, loc_probs=probs(0.2, 0.6, 0.2)),
        _rec("C", loc=6, loc_probs=probs(0.1, 0.3, 0.6)),
        _rec("D", loc=0, loc_probs=probs(0.5, 0.3, 0.2)),
    ]
    got = compute_official_metrics(recs)["location_weighted_auc"]

    # 手算:剔除缺席类 → 重映射标签 → 重归一化(本例每行已和为 1)
    y = [0, 1, 2, 0]
    p = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6], [0.5, 0.3, 0.2]])
    expect = roc_auc_score(y, p, multi_class="ovr", average="weighted")
    assert got == pytest.approx(float(expect))


def test_location_renormalizes_when_absent_classes_carry_mass():
    """缺席类带走概率质量时必须重归一化(sklearn 要求每行和为 1),且不改排序。"""
    def probs(p0, p1, junk):
        v = [0.0] * 7
        v[0], v[1], v[5] = p0, p1, junk       # 类 5 在金标准里从不出现
        return v

    recs = [
        _rec("A", loc=0, loc_probs=probs(0.5, 0.1, 0.4)),
        _rec("B", loc=1, loc_probs=probs(0.1, 0.5, 0.4)),
    ]
    got = compute_official_metrics(recs)["location_weighted_auc"]
    assert got == pytest.approx(1.0)          # 排序完好 → 完美 AUC


def test_ete_uses_quadratic_weighted_kappa_on_argmax():
    """ETE = 加权(quadratic)Kappa,吃 argmax 硬预测。与 sklearn 手算对拍。

    同时反证"不是线性 Kappa":构造一组线性/二次结果不同的混淆,断言取到二次值。
    """
    def ep(cls):
        v = [0.0] * 4
        v[cls] = 1.0
        return v

    pairs = [(0, 0), (1, 3), (2, 2), (3, 1), (0, 1), (3, 3)]
    recs = [
        _rec(f"L{i}", mal=1, ete=gt, ete_probs=ep(pred))
        for i, (gt, pred) in enumerate(pairs)
    ]
    got = compute_official_metrics(recs)["ete_weighted_kappa"]

    y_true = [gt for gt, _ in pairs]
    y_pred = [pred for _, pred in pairs]
    quad = cohen_kappa_score(y_true, y_pred, labels=list(range(4)), weights="quadratic")
    lin = cohen_kappa_score(y_true, y_pred, labels=list(range(4)), weights="linear")
    assert got == pytest.approx(float(quad))
    assert not np.isclose(quad, lin), "反例失效:本组数据的线性/二次 Kappa 恰好相同"


def test_malignancy_uses_auc_pr_not_auc():
    """良恶性 = AUC-PR(average_precision),不是 ROC-AUC。用二者不等的数据钉死。

    构造要点:正类稀少 + 有一个高分假阳。ROC-AUC 只看排序对数,AP 会因为
    "第一个正例前面挤了个假阳"被显著拉低,两者因此分开。
    """
    recs = [
        _rec("A", mal=0, mal_p=0.95),
        _rec("B", mal=1, mal_p=0.90),
        _rec("C", mal=0, mal_p=0.30),
        _rec("D", mal=0, mal_p=0.20),
        _rec("E", mal=0, mal_p=0.10),
    ]
    got = compute_official_metrics(recs)["malignancy_auc_pr"]
    y = [0, 1, 0, 0, 0]
    s = [0.95, 0.90, 0.30, 0.20, 0.10]
    ap = average_precision_score(y, s)
    auc = roc_auc_score(y, s)
    assert got == pytest.approx(float(ap))
    assert not np.isclose(ap, auc), "反例失效:该组 AP 与 ROC-AUC 恰好相同"


def test_clinical_t_auc_uses_t_conf_as_score():
    recs = [
        _rec("A", mal=1, t=1, t_conf=0.9),
        _rec("B", mal=1, t=0, t_conf=0.2),
        _rec("C", mal=1, t=1, t_conf=0.6),
        _rec("D", mal=1, t=0, t_conf=0.4),
    ]
    got = compute_official_metrics(recs)["clinical_t_auc"]
    assert got == pytest.approx(
        float(roc_auc_score([1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4]))
    )


def test_composite_is_mean_of_available_metrics_only():
    """composite = 可算指标的算术均值;不可算的项不得当 0 拉低(那会误导选模)。"""
    # 只有良恶性可算(位置单类、ETE/T 单类)
    recs = [_rec("A", mal=1, mal_p=0.9, loc=0), _rec("B", mal=0, mal_p=0.1, loc=0)]
    m = compute_official_metrics(recs)
    usable = [m[k] for k in ("malignancy_auc_pr", "location_weighted_auc",
                             "ete_weighted_kappa", "clinical_t_auc")
              if m[k] is not None]
    assert m["composite"] == pytest.approx(float(np.mean(usable)))
    assert len(usable) < 4, "本例应有指标不可算,否则测不到'不当 0 计入'"

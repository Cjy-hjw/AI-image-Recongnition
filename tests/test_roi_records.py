# -*- coding: utf-8 -*-
"""ROI(per-lesion)验证指标链单测 —— P0 尺子先行(新架构 §F 根因 #3)。

被测链路:
  `src/training/collect_roi_records.py`(新,per-lesion 收集)
  + `src/training/loop/validate_step.py`(分流:outputs 含 roi_outputs 键走 ROI 路)
  + `src/training/metrics_official.py`(指标层吃 K 条/图)

核心断言按改造铁律组织:
  ① **零回归**:无 roi_outputs 键时整图路径逐位不变(同输入新旧两路 diff=0);
  ② **K=1 等价**:一图一病灶时 per-lesion 口径与整图口径一致;
  ③ **K=3 形态**:一图 K 病灶 → 记录条数 = K、lesion_id 保全;
  ④ valid 过滤、病灶级聚合与手算一致、metrics 展平正确。
"""
import numpy as np
import pytest

import sklearn.metrics  # noqa: F401  (import 副作用:抢先加载 OpenMP 运行时,
# 防 pytest 收集深度下 sklearn→pandas→pyarrow 链 Windows 栈溢出,test_cliff_loss.py 同款)

import torch

from src.models.heads import NUM_ETE_CLASSES, NUM_TIRADS_CLASSES
from src.models.label_schema import NUM_LOCATIONS
from src.training.collect_roi_records import (
    collect_roi_records,
    roi_metrics,
    roi_records_from_batch,
)
from src.training.loop.validate_step import collect_official_records
from src.training.metrics_official import compute_official_metrics


# ── fixtures ────────────────────────────────────────────────────────────────

def _roi_dict(lesion_id=None, is_main=1, patient=None, *,
              mal_logits=(0.0, 1.0), loc_probs=None, ete_probs=None,
              ete_logits=None, t_conf=0.5, tirads_probs=None):
    """造一条 roi_outputs 元素。概率向量默认均匀(允许非均匀覆盖)。"""
    if loc_probs is None:
        loc_probs = [1.0 / NUM_LOCATIONS] * NUM_LOCATIONS
    if ete_probs is None:
        ete_probs = [1.0 / NUM_ETE_CLASSES] * NUM_ETE_CLASSES
    if ete_logits is None:
        ete_logits = [0.0] * NUM_ETE_CLASSES
    if tirads_probs is None:
        tirads_probs = [1.0 / NUM_TIRADS_CLASSES] * NUM_TIRADS_CLASSES
    d = {
        "lesion_id": lesion_id,
        "is_main": is_main,
        "patient_id": patient,
        "malignancy": list(mal_logits),
        "location_probs": list(loc_probs),
        "ete_probs": list(ete_probs),
        "ete_logits": list(ete_logits),
        "t_conf": t_conf,
        "tirads_probs": list(tirads_probs),
    }
    return d


def _lab_dict(*, mal=1, loc=0, ete=0, t=0, tirads=0,
              v_mal=1, v_loc=1, v_ete=1, v_t=1, v_tirads=1):
    return {
        "gt_malignancy": mal,
        "gt_lesion_loc": loc,
        "gt_extrathy_ext": ete,
        "gt_clin_t_stage": t,
        "gt_tirads": tirads,
        "valid_malignancy": v_mal,
        "valid_lesion_loc": v_loc,
        "valid_extrathy_ext": v_ete,
        "valid_clin_t_stage": v_t,
        "valid_tirads": v_tirads,
    }


def _whole_image_outputs(n_imgs, seed=0, lesion_ids=None):
    """造整图形态 outputs(dict of [B,N] tensors)+ labels/targets,喂整图链路。"""
    rng = np.random.RandomState(seed)
    B = n_imgs
    mal = torch.tensor(rng.randn(B, 2), dtype=torch.float32)
    outputs = {
        "malignancy": mal,
        "location_probs": torch.softmax(
            torch.tensor(rng.randn(B, NUM_LOCATIONS), dtype=torch.float32), dim=1),
        "ete_probs": torch.softmax(
            torch.tensor(rng.randn(B, NUM_ETE_CLASSES), dtype=torch.float32), dim=1),
        "ete_logits": torch.tensor(
            rng.randn(B, NUM_ETE_CLASSES), dtype=torch.float32),
        "t_conf": torch.tensor(rng.rand(B), dtype=torch.float32),
        "tirads_probs": torch.softmax(
            torch.tensor(rng.randn(B, NUM_TIRADS_CLASSES), dtype=torch.float32), dim=1),
    }
    labels = {
        "malignancy": torch.tensor([int(rng.rand() > 0.5) for _ in range(B)]),
        "lesion_loc": torch.tensor([int(rng.randint(0, NUM_LOCATIONS)) for _ in range(B)]),
        "extrathy_ext": torch.tensor([int(rng.randint(0, NUM_ETE_CLASSES)) for _ in range(B)]),
        "clin_t_stage": torch.tensor([int(rng.rand() > 0.5) for _ in range(B)]),
        "tirads": torch.tensor([int(rng.randint(0, NUM_TIRADS_CLASSES)) for _ in range(B)]),
        "valid_malignancy": torch.ones(B),
        "valid_lesion_loc": torch.ones(B),
        "valid_extrathy_ext": torch.ones(B),
        "valid_clin_t_stage": torch.ones(B),
        "valid_tirads": torch.ones(B),
        "is_main": torch.ones(B),
        "lesion_id": lesion_ids or [f"P{i}::M" for i in range(B)],
        "patient_id": [f"P{i}" for i in range(B)],
    }
    targets = {
        "lesion_loc": labels["lesion_loc"],
        "extrathy_ext": labels["extrathy_ext"],
        "clin_t_stage": labels["clin_t_stage"],
        "tirads": labels["tirads"],
        "malignancy": labels["malignancy"],
        "valid_lesion_loc": labels["valid_lesion_loc"],
        "valid_extrathy_ext": labels["valid_extrathy_ext"],
        "valid_clin_t_stage": labels["valid_clin_t_stage"],
        "valid_tirads": labels["valid_tirads"],
    }
    return outputs, labels, targets


def _soft_prob_from_logits(logits):
    """与整图链路 torch.softmax(x)[:,1] 等价的 numpy 实现(供对拍)。"""
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


# ── ① 整图路径逐位一致(零回归铁律)────────────────────────────────────────

def test_whole_image_path_bitwise_identical():
    """无 roi_outputs 键:同输入跑 collect_official_records,结果与改造前
    逐位一致(此处以"重复调用逐位一致 + 不触 ROI 模块"钉分流纯度)。"""
    outputs, labels, targets = _whole_image_outputs(8, seed=3)
    assert "roi_outputs" not in outputs
    r1 = collect_official_records(outputs, labels, targets)
    r2 = collect_official_records(outputs, labels, targets)
    assert r1 == r2, "整图路径必须纯函数式稳定(同输入同输出)"
    assert len(r1) == 8, "整图路径一图一条,B=8 → 8 条"


def test_roi_module_not_imported_on_whole_image_path():
    """整图路径不得 import collect_roi_records(分流零污染)。"""
    import sys
    outputs, labels, targets = _whole_image_outputs(4, seed=1)
    sys.modules.pop("src.training.collect_roi_records", None)
    collect_official_records(outputs, labels, targets)
    assert "src.training.collect_roi_records" not in sys.modules, (
        "整图路径不应加载 ROI 模块 —— 懒加载只在 ROI 分支触发"
    )


# ── ② K=1 时 per-lesion 与整图口径一致 ─────────────────────────────────────

def test_k1_roi_metrics_equal_whole_image_metrics():
    """K=1(一图一病灶):per-lesion 记录喂指标层 = 同数据整图记录喂指标层。

    构造:B=6 张图各 1 病灶;整图链路吃 [B] 张量记录,ROI 链路吃 6 条
    per-lesion 记录(lesion_id/概率/标签逐值相同)。两条路的全部指标
    必须逐位相等 —— 这是"ROI 改造不改变整图口径数字"的直接对拍。
    """
    outputs, labels, targets = _whole_image_outputs(6, seed=7)
    whole = collect_official_records(outputs, labels, targets)

    # 从整图张量逐样本抄出 per-lesion dict(同一数值、不同载体)
    rois, labs = [], []
    for i in range(6):
        rois.append({
            "lesion_id": labels["lesion_id"][i],
            "is_main": 1,
            "patient_id": labels["patient_id"][i],
            "malignancy": outputs["malignancy"][i].tolist(),
            "location_probs": outputs["location_probs"][i].tolist(),
            "ete_probs": outputs["ete_probs"][i].tolist(),
            "ete_logits": outputs["ete_logits"][i].tolist(),
            "t_conf": outputs["t_conf"][i].item(),
            "tirads_probs": outputs["tirads_probs"][i].tolist(),
        })
        labs.append({
            "gt_malignancy": int(labels["malignancy"][i]),
            "gt_lesion_loc": int(labels["lesion_loc"][i]),
            "gt_extrathy_ext": int(labels["extrathy_ext"][i]),
            "gt_clin_t_stage": int(labels["clin_t_stage"][i]),
            "gt_tirads": int(labels["tirads"][i]),
            "valid_malignancy": 1, "valid_lesion_loc": 1,
            "valid_extrathy_ext": 1, "valid_clin_t_stage": 1,
            "valid_tirads": 1,
        })

    roi = collect_roi_records(rois, labs)
    m_whole = compute_official_metrics(whole)
    m_roi = roi_metrics(roi)

    for k in ("malignancy_auc_pr", "location_weighted_auc", "ete_weighted_kappa",
              "clinical_t_auc", "tirads_weighted_kappa", "composite", "n_lesions"):
        assert m_roi[k] == m_whole[k], (
            f"K=1 时指标 {k} 不一致:roi={m_roi[k]} vs whole={m_whole[k]}"
        )


def test_k1_record_fields_match_whole_image_record():
    """K=1 时单条 ROI 记录与整图记录逐键逐值对齐(字段集契约)。

    E1-F1 硬前置②(2026-09-25)后 ROI 记录多一个加性键 `instance_idx`
    (实例级聚合键,缺省 0 = 单实例;整图记录无此键)—— 键集断言把
    它单列,数值断言逐位不变。"""
    outputs, labels, targets = _whole_image_outputs(1, seed=11)
    whole = collect_official_records(outputs, labels, targets)[0]
    roi = collect_roi_records(
        [_roi_dict(lesion_id="P0::M", patient="P0",
                   mal_logits=outputs["malignancy"][0].tolist(),
                   loc_probs=outputs["location_probs"][0].tolist(),
                   ete_probs=outputs["ete_probs"][0].tolist(),
                   ete_logits=outputs["ete_logits"][0].tolist(),
                   t_conf=outputs["t_conf"][0].item(),
                   tirads_probs=outputs["tirads_probs"][0].tolist())],
        [_lab_dict(mal=int(labels["malignancy"][0]),
                   loc=int(labels["lesion_loc"][0]),
                   ete=int(labels["extrathy_ext"][0]),
                   t=int(labels["clin_t_stage"][0]),
                   tirads=int(labels["tirads"][0]))],
    )[0]
    assert set(roi.keys()) - {"instance_idx"} == set(whole.keys()), (
        f"字段集不一致:仅 roi 有 {set(roi) - set(whole) - {'instance_idx'}},"
        f"仅 whole 有 {set(whole) - set(roi)}"
    )
    assert roi["instance_idx"] == 0, "缺省实例键 = 0(单实例语义)"
    for k, v in whole.items():
        if isinstance(v, float):
            assert roi[k] == pytest.approx(v), f"键 {k} 数值不一致"
        else:
            assert roi[k] == v, f"键 {k} 值不一致:roi={roi[k]} vs whole={v}"


# ── ③ K=3 形态:条数 / lesion_id / valid 过滤 ───────────────────────────────

def test_k3_record_count_is_k():
    """一图 3 病灶 → 3 条记录,lesion_id 各自保全(不并、不丢)。"""
    rois = [
        _roi_dict(lesion_id=f"img1::roi{j}", is_main=1 if j == 0 else 0)
        for j in range(3)
    ]
    labs = [_lab_dict() for _ in range(3)]
    recs = collect_roi_records(rois, labs)
    assert len(recs) == 3
    assert [r["lesion_id"] for r in recs] == [
        "img1::roi0", "img1::roi1", "img1::roi2"]
    # 逐条 is_main 保留(官方筛选键,不许图级平均)
    assert [r["is_main"] for r in recs] == [1, 0, 0]


def test_k3_metrics_do_not_collapse_to_one_record():
    """🔴 核心反例:K=3 的三条记录喂指标层,不得被'平均成一条'。

    构造:3 条记录 mal_prob 分别 0.9/0.5/0.1、gt 恶性/良性/恶性。
      - 正确(per-lesion):AUC-PR 吃 3 个分数 → AP = (1 + 2/3)/2 = 5/6
        (负样本 0.5 排中间,压低精度曲线,非退化值);
      - 错误(平均成一条):只剩 1 条 → len(y)<2 → 指标 None,给不出 5/6。
    任何把 K 条平均的实现都给不出这个数。
    """
    rois = [
        _roi_dict(lesion_id=f"L{j}", mal_logits=(0.0, _logit(p)))
        for j, p in enumerate([0.9, 0.5, 0.1])
    ]
    labs = [_lab_dict(mal=gt) for gt in (1, 0, 1)]
    recs = collect_roi_records(rois, labs)
    m = roi_metrics(recs)
    # n_lesions=3(不是 1)+ 非退化 AP 精确值(平均实现必给不出)
    assert m["n_lesions"] == 3
    assert m["malignancy_auc_pr"] == pytest.approx(5.0 / 6.0)


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def test_valid_flags_filter_in_roi_records():
    """valid=0 的 per-lesion 记录必须被指标层排除(不是当默认类)。"""
    # 2 有效(位置 0/1,预测完美)+ 1 invalid(预测全错)
    rois = [
        _roi_dict(lesion_id="A", loc_probs=_onehot(NUM_LOCATIONS, 0)),
        _roi_dict(lesion_id="B", loc_probs=_onehot(NUM_LOCATIONS, 1)),
        _roi_dict(lesion_id="C", loc_probs=_onehot(NUM_LOCATIONS, 6)),
    ]
    labs = [
        _lab_dict(loc=0),
        _lab_dict(loc=1),
        _lab_dict(loc=0, v_loc=0),
    ]
    m = roi_metrics(collect_roi_records(rois, labs))
    assert m["location_weighted_auc"] == 1.0, (
        "invalid 样本必须被排除;若被当默认类,AUC 会低于 1.0"
    )


def _onehot(n, k):
    v = [0.0] * n
    v[k] = 1.0
    return v


def test_lesion_id_kept_as_str():
    """lesion_id 必须保持 list[str] 语义(病灶级聚合键,类型漂移=聚合失效)。"""
    rois = [_roi_dict(lesion_id=f"pid32hex::{j}") for j in range(3)]
    recs = collect_roi_records(rois, [_lab_dict() for _ in range(3)])
    assert all(isinstance(r["lesion_id"], str) for r in recs)
    assert [r["lesion_id"] for r in recs] == [
        "pid32hex::0", "pid32hex::1", "pid32hex::2"]


def test_missing_lesion_id_gets_unique_placeholder():
    """lesion_id 缺失 → 进程内单调占位,绝不互撞(宁可不聚合,不可错聚合)。"""
    rois = [_roi_dict(lesion_id=None) for _ in range(3)]
    recs = collect_roi_records(rois, [_lab_dict() for _ in range(3)])
    lids = [r["lesion_id"] for r in recs]
    assert len(set(lids)) == 3, "占位键必须互不相同"
    assert all(str(l).startswith("__unaggregatable__") for l in lids)
    # 聚合后 3 条独立(不因占位撞名而并成 1 条)
    m = roi_metrics(recs)
    assert m["n_lesions"] == 3


# ── ④ 病灶级聚合与手算一致 / metrics 展平 ──────────────────────────────────

def test_lesion_level_aggregation_matches_hand_computation():
    """同病灶多帧(动态)在 ROI 记录下仍按 lesion_id 聚合,概率均值与手算一致。"""
    lp_a = [0.6, 0.4] + [0.0] * (NUM_LOCATIONS - 2)
    lp_b = [0.2, 0.8] + [0.0] * (NUM_LOCATIONS - 2)
    rois = [
        _roi_dict(lesion_id="SAME", loc_probs=lp_a, t_conf=0.2,
                  mal_logits=(0.0, _logit(0.3))),
        _roi_dict(lesion_id="SAME", loc_probs=lp_b, t_conf=0.8,
                  mal_logits=(0.0, _logit(0.7))),
        _roi_dict(lesion_id="OTHER", t_conf=0.5),
    ]
    labs = [_lab_dict(loc=0), _lab_dict(loc=0), _lab_dict(loc=1)]
    recs = collect_roi_records(rois, labs)
    m = roi_metrics(recs)
    # SAME 两帧聚合:t_conf = (0.2+0.8)/2 = 0.5;OTHER 0.5 → n_lesions=2
    assert m["n_lesions"] == 2
    # 聚合概率均值手算:location_probs[0] = (0.6+0.2)/2 = 0.4
    from src.training.metrics_official import aggregate_by_lesion
    rows = aggregate_by_lesion(recs)
    same = rows[0]
    assert same["n_frames"] == 2
    assert abs(same["location_probs"][0] - 0.4) < 1e-9
    assert abs(same["location_probs"][1] - 0.6) < 1e-9
    assert abs(same["t_conf"] - 0.5) < 1e-9
    # mal_prob 均值 = (0.3+0.7)/2 = 0.5(由 logit 反推,softmax 精度内)
    assert abs(same["mal_prob"] - 0.5) < 1e-6


def test_metrics_flatten_k_per_image_correct():
    """🔴 指标层吃 K 条/图:records 是展平的 list,不按图分组。

    构造 2 图 × 3 病灶(lesion_id 互异)= 6 条记录。指标函数吃的是
    list[dict](展平),只要 lesion_id 互异,聚合 = 恒等(每条一病灶)。
    位置 AUC 手算:6 样本、2 类,预测完美 → 1.0。
    """
    rois, labs = [], []
    for img in range(2):
        for j in range(3):
            loc = (img + j) % 2
            rois.append(_roi_dict(
                lesion_id=f"img{img}::roi{j}",
                loc_probs=_onehot(NUM_LOCATIONS, loc)))
            labs.append(_lab_dict(loc=loc))
    recs = collect_roi_records(rois, labs)
    assert len(recs) == 6
    m = roi_metrics(recs)
    assert m["n_lesions"] == 6, "6 个互异 lesion_id → 6 病灶"
    assert m["location_weighted_auc"] == 1.0


def test_labels_length_mismatch_fails_loud():
    """roi_outputs 与 labels 长度不齐 → ValueError,绝不静默截断。"""
    rois = [_roi_dict(lesion_id="A"), _roi_dict(lesion_id="B")]
    with pytest.raises(ValueError, match="长度不齐"):
        collect_roi_records(rois, [_lab_dict()])


# ── ⑤ validate_batch 分流入口 ───────────────────────────────────────────────

def test_roi_records_from_batch_dispatch():
    """roi_records_from_batch = collect_roi_records(outputs['roi_outputs'], labels)。"""
    rois = [_roi_dict(lesion_id="A"), _roi_dict(lesion_id="B")]
    labs = [_lab_dict(), _lab_dict()]
    outputs = {"roi_outputs": rois}
    recs = roi_records_from_batch(outputs, labs)
    assert len(recs) == 2
    assert [r["lesion_id"] for r in recs] == ["A", "B"]


def test_collect_official_records_dispatches_to_roi_path():
    """validate_step.collect_official_records 对含 roi_outputs 的 outputs 走 ROI 路。"""
    rois = [_roi_dict(lesion_id="X"), _roi_dict(lesion_id="Y"),
            _roi_dict(lesion_id="Z")]
    labs = [_lab_dict(), _lab_dict(), _lab_dict()]
    outputs = {"roi_outputs": rois}
    recs = collect_official_records(outputs, labs, targets={})
    assert len(recs) == 3, "ROI 分流生效:3 病灶 → 3 条记录"
    assert [r["lesion_id"] for r in recs] == ["X", "Y", "Z"]


def test_tensor_inputs_accepted():
    """roi_outputs 元素的四头值允许 torch 张量(自动 cpu/float)。"""
    rois = [{
        "lesion_id": "T1",
        "is_main": torch.tensor(1),
        "patient_id": None,
        "malignancy": torch.tensor([0.3, 0.9]),
        "location_probs": torch.softmax(
            torch.randn(NUM_LOCATIONS), dim=0),
        "ete_probs": torch.softmax(torch.randn(NUM_ETE_CLASSES), dim=0),
        "ete_logits": torch.randn(NUM_ETE_CLASSES),
        "t_conf": torch.tensor(0.7),
        "tirads_probs": torch.softmax(
            torch.randn(NUM_TIRADS_CLASSES), dim=0),
    }]
    recs = collect_roi_records(rois, [_lab_dict()])
    r = recs[0]
    assert isinstance(r["mal_prob"], float)
    assert len(r["location_probs"]) == NUM_LOCATIONS
    assert 0.0 <= r["mal_prob"] <= 1.0
    # mal_logits (0.3, 0.9) → softmax 正类 ≈ 0.659
    assert abs(r["mal_prob"] - 1.0 / (1.0 + np.exp(0.3 - 0.9))) < 1e-6


def test_wrong_probs_length_raises():
    """概率向量长度错 → 收集层即抛(不等到 ete_monitor 的 assert)。"""
    bad = _roi_dict(lesion_id="BAD", ete_probs=[0.25] * (NUM_ETE_CLASSES + 1))
    with pytest.raises(ValueError, match="ete_probs"):
        collect_roi_records([bad], [_lab_dict()])

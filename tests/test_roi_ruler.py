# -*- coding: utf-8 -*-
"""E1-F1 硬前置:ROI 训练尺子升**病灶(实例)级**两层聚合 —— 与推理侧同构。

## 背景(E1 review F1 实锤,docs/新架构文件级改动清单_20260925.md §J.7)

推理 ROI 路是**实例级**:同帧两实例各自独立 ROI 前向,得 0.9975/0.0025 两个
独立置信度,各自单独进 prediction.json(`v21_output._run_roi_static`)。
训练尺子 `aggregate_by_lesion` 是**帧级**:同 lesion_id 的多条记录被均值摊平
—— 同帧两实例 0.9975/0.0025 摊成同一个 0.5。后果:ROI 训练的本地 val
系统性乐观,判线对拍对不上。

## 两层聚合语义(与推理侧对齐,本文件的每条断言都锚它)

- **层 1(实例内跨帧,动态)**:同一实例跨多帧的记录按帧取均值 —— 与推理侧
  `_run_roi_dynamic → _aggregate_roi_frame_rows` 的病灶级帧均值同构:
  malignancy 逐帧 softmax 后对概率取均值(记录里的 `mal_prob` 已是逐条
  softmax 值,均值即同式),其余四头概率/t_conf 逐帧算术平均。
- **层 2(实例间独立)**:同帧/同 lesion 的多个实例各自独立成记录,互相
  之间**绝不平均**(0.9975 与 0.0025 必须分开进指标)。
- 聚合键 = `(lesion_id, instance_idx)`;`instance_idx` 由记录携带
  (`collect_roi_records` 从 roi/labels dict 读取,缺省 0 = 单实例语义,
  GT-mask ROI 流的既有行为天然保持)。

## 整图链零回归红线

`aggregate_by_lesion` 原函数一字不动;`compute_official_metrics` 新增
keyword-only `aggregation` 参数(缺省 = `aggregate_by_lesion`,整图调用点
逐位不变)。未携带 `instance_idx` 的记录流(整图链/旧 GT 流)在新聚合下
逐位等价于旧口径 —— 有专门对拍测试钉死。
"""
import numpy as np
import pytest

import sklearn.metrics  # noqa: F401  (import 副作用:抢先加载 OpenMP 运行时,
# 防 pytest 收集深度下 sklearn→pandas→pyarrow 链 Windows 栈溢出,
# test_cliff_loss.py / test_roi_records.py 同款)

import torch

from src.models.heads import NUM_ETE_CLASSES, NUM_TIRADS_CLASSES
from src.models.label_schema import NUM_LOCATIONS
from src.training.collect_roi_records import collect_roi_records, roi_metrics
from src.training.metrics_official import aggregate_by_lesion, compute_official_metrics


def aggregate_by_roi_instance(records):
    """被测入口(函数体 import 让 RED 落在断言层,见文件头变异纪律)。"""
    from src.training.metrics_official import aggregate_by_roi_instance as f

    return f(records)


# ── fixtures ────────────────────────────────────────────────────────────────

def _roi_dict(lesion_id=None, is_main=1, instance_idx=None, *,
              mal_logits=(0.0, 1.0), loc_probs=None, ete_probs=None,
              ete_logits=None, t_conf=0.5, tirads_probs=None, patient=None):
    """造一条 roi_outputs 元素。概率向量默认均匀;instance_idx 显式 None
    = 不携带该键(测缺省退化语义时用),int = 生产方实例标注。"""
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
    if instance_idx is not None:
        d["instance_idx"] = int(instance_idx)
    return d


def _lab_dict(*, mal=1, loc=0, ete=0, t=0, tirads=0,
              v_mal=1, v_loc=1, v_ete=1, v_t=1, v_tirads=1, instance_idx=None):
    d = {
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
    if instance_idx is not None:
        d["instance_idx"] = int(instance_idx)
    return d


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


# softmax(−3,3)[1] ≈ 0.9975(推理侧 test_roi_inference.py 同数值语义)
_P_HOT = float(np.exp(3.0) / (np.exp(-3.0) + np.exp(3.0)))
_P_COLD = float(np.exp(-3.0) / (np.exp(-3.0) + np.exp(3.0)))


# ── 层 2:同帧双实例绝不平均(E1-F1 区分力核心,旧口径必须死)────────────────

class TestSameFrameTwoInstances:
    def test_two_instances_not_averaged_kills_old_ruler(self):
        """🔴 E1-F1 原始反例:同帧同 lesion 双实例 0.9975/0.0025。

        正确(实例级):两条独立记录 → n_lesions=2,AUC-PR 吃两个分数;
        旧口径(lesion_id 帧级均值):摊平成一条 0.5 → n_lesions=1、指标
        退化为 None —— 本测试在旧代码上必须 RED(均值 0.5 杀死)。
        """
        rois = [
            _roi_dict(lesion_id="L", instance_idx=0, mal_logits=(-3.0, 3.0)),
            _roi_dict(lesion_id="L", instance_idx=1, mal_logits=(3.0, -3.0)),
        ]
        # 同帧(动态帧 5)双实例:gt 相同但分数天差地别
        labs = [_lab_dict(mal=1), _lab_dict(mal=0)]
        recs = collect_roi_records(rois, labs)
        m = roi_metrics(recs)
        assert m["n_lesions"] == 2, (
            f"同帧双实例必须各成一条记录,得到 n_lesions={m['n_lesions']}"
            " —— 旧帧级口径会把两实例平均成一条(系统性乐观)"
        )
        # 两条记录的分数必须原样保全(0.9975/0.0025),不是均值 0.5
        rows = aggregate_by_roi_instance(recs)
        probs = sorted(r["mal_prob"] for r in rows)
        assert probs[0] == pytest.approx(_P_COLD, abs=2e-3)
        assert probs[1] == pytest.approx(_P_HOT, abs=2e-3)
        assert m["malignancy_auc_pr"] == pytest.approx(1.0), (
            "双实例分开后 0.9975(gt=恶)/0.0025(gt=良)是完美排序,AP=1.0;"
            "任何平均实现给不出这个数"
        )

    def test_rows_level_two_instances_two_rows(self):
        """行级直查:aggregate_by_roi_instance 输出两条独立行,概率逐值保全。"""
        rois = [
            _roi_dict(lesion_id="L", instance_idx=0,
                      mal_logits=(-3.0, 3.0), t_conf=0.9),
            _roi_dict(lesion_id="L", instance_idx=1,
                      mal_logits=(3.0, -3.0), t_conf=0.1),
        ]
        labs = [_lab_dict(mal=1), _lab_dict(mal=0)]
        rows = aggregate_by_roi_instance(collect_roi_records(rois, labs))
        assert len(rows) == 2
        assert all(r["n_frames"] == 1 for r in rows)
        by_idx = {r["instance_idx"]: r for r in rows}
        assert by_idx[0]["mal_prob"] == pytest.approx(_P_HOT, abs=2e-3)
        assert by_idx[1]["mal_prob"] == pytest.approx(_P_COLD, abs=2e-3)
        assert by_idx[0]["t_conf"] == 0.9 and by_idx[1]["t_conf"] == 0.1

    def test_instance_idx_from_labels_dict_fallback(self):
        """instance_idx 允许由 labels 侧(meta 桥透传)提供:roi 缺席 → labels。"""
        rois = [
            _roi_dict(lesion_id="L"),
            _roi_dict(lesion_id="L"),
        ]
        labs = [_lab_dict(mal=1, instance_idx=0), _lab_dict(mal=0, instance_idx=1)]
        recs = collect_roi_records(rois, labs)
        assert [r["instance_idx"] for r in recs] == [0, 1]
        rows = aggregate_by_roi_instance(recs)
        assert len(rows) == 2

    def test_roi_dict_instance_idx_wins_over_labels(self):
        """两侧都带 instance_idx 时 roi 侧优先(生产方 = 模型输出侧)。"""
        rois = [
            _roi_dict(lesion_id="L", instance_idx=2),
            _roi_dict(lesion_id="L", instance_idx=5),
        ]
        labs = [_lab_dict(instance_idx=0), _lab_dict(instance_idx=1)]
        recs = collect_roi_records(rois, labs)
        assert [r["instance_idx"] for r in recs] == [2, 5]

    def test_missing_instance_idx_defaults_to_zero(self):
        """不携带 instance_idx 的记录 → 缺省 0(单实例语义,GT 流既有行为)。"""
        rois = [_roi_dict(lesion_id="L"), _roi_dict(lesion_id="L")]
        labs = [_lab_dict(), _lab_dict()]
        recs = collect_roi_records(rois, labs)
        assert all(r["instance_idx"] == 0 for r in recs)

    def test_batch_tensor_instance_idx_rejected(self):
        """instance_idx 误接 batch 张量([B] 形)→ fail-loud(同 _label_val 纪律:
        整批标签被静默当单实例标签 = 错得无声)。"""
        rois = [_roi_dict(lesion_id="L"), _roi_dict(lesion_id="L")]
        labs = [{**_lab_dict(), "instance_idx": torch.tensor([1, 2])},
                _lab_dict()]
        with pytest.raises(ValueError, match="instance_idx"):
            collect_roi_records(rois, labs)


# ── 层 1:同实例跨帧均值(与推理侧 _aggregate_roi_frame_rows 同构)────────────

class TestDynamicFrameMean:
    def test_frame_mean_matches_inference_semantics(self):
        """同实例 3 帧均值 = 推理侧 `_run_roi_dynamic` 的聚合式,数值语义
        从 tests/test_roi_inference.py:500-501 抄来(同一对 logit/值):
        mal = (0.9975 + 0.0025 + 0.0025)/3;t_conf = (0.9+0.1+0.1)/3;
        其余四头概率逐帧算术平均(推理侧逐条 validate 后均值,记录链同式)。
        """
        tir_f0 = [0.02, 0.02, 0.02, 0.02, 0.92]   # 帧 0:峰 idx4(hot)
        tir_f1 = [0.02, 0.6, 0.2, 0.1, 0.05]      # 帧 1(推理侧同款值)
        tir_f2 = [0.02, 0.6, 0.2, 0.1, 0.05]
        rois = [
            _roi_dict(lesion_id="L", instance_idx=0,
                      mal_logits=(-3.0, 3.0), t_conf=0.9, tirads_probs=tir_f0),
            _roi_dict(lesion_id="L", instance_idx=0,
                      mal_logits=(3.0, -3.0), t_conf=0.1, tirads_probs=tir_f1),
            _roi_dict(lesion_id="L", instance_idx=0,
                      mal_logits=(3.0, -3.0), t_conf=0.1, tirads_probs=tir_f2),
        ]
        labs = [_lab_dict() for _ in range(3)]
        rows = aggregate_by_roi_instance(collect_roi_records(rois, labs))
        assert len(rows) == 1, "同实例跨帧 → 一条病灶级记录"
        row = rows[0]
        assert row["n_frames"] == 3
        # 推理侧同款断言(malignancy 逐帧 softmax 后均值)
        mal_mean = (_P_HOT + _P_COLD + _P_COLD) / 3.0
        assert abs(row["mal_prob"] - mal_mean) < 2e-3
        assert abs(row["t_conf"] - (0.9 + 0.1 + 0.1) / 3.0) < 1e-8
        # tirads 概率均值 = 推理侧 test_dynamic_frame_mean_not_keyframe 同式
        for j in range(NUM_TIRADS_CLASSES):
            expected = (tir_f0[j] + tir_f1[j] + tir_f2[j]) / 3.0
            assert abs(row["tirads_probs"][j] - expected) < 1e-9

    def test_frame_mean_is_prob_mean_not_logit_resoftmax(self):
        """层 1 的 malignancy 聚合是**逐帧 softmax 后取均值**,不是 logits
        均值再 softmax(两条路径数学不等价,后者系统性压扁置信度 —— 推理侧
        `_softmax_mal_logits` docstring 同款红线)。构造:两帧 logits
        (−3,3)/(3,−3):概率均值 = 0.5;logits 均值 = (0,0) → softmax 0.5
        —— 该构造两式同值无区分力,故用三帧 (−3,3)/(3,−3)/(3,−3):
        概率均值 = (0.9975+0.0025+0.0025)/3 ≈ 0.3342;logits 均值 (1,−1)
        → softmax[1] ≈ 0.1192 —— 两值差 3 倍,断言钉死正确侧。
        """
        rois = [
            _roi_dict(lesion_id="L", mal_logits=(-3.0, 3.0)),
            _roi_dict(lesion_id="L", mal_logits=(3.0, -3.0)),
            _roi_dict(lesion_id="L", mal_logits=(3.0, -3.0)),
        ]
        recs = collect_roi_records(rois, [_lab_dict() for _ in range(3)])
        row = aggregate_by_roi_instance(recs)[0]
        assert abs(row["mal_prob"] - (_P_HOT + 2 * _P_COLD) / 3.0) < 2e-3


# ── 两层合体:实例内跨帧均值 + 实例间独立 ────────────────────────────────────

class TestTwoLayerCombined:
    def test_two_instances_one_multiframe(self):
        """完整两层:实例 0 跨 2 帧(均值),实例 1 单帧(独立)→ 2 行。"""
        rois = [
            # 实例 0:帧 a/b,概率 0.9/0.7 → 均值 0.8
            _roi_dict(lesion_id="L", instance_idx=0,
                      mal_logits=(0.0, _logit(0.9)), t_conf=0.8),
            _roi_dict(lesion_id="L", instance_idx=0,
                      mal_logits=(0.0, _logit(0.7)), t_conf=0.6),
            # 实例 1:单帧 0.1(绝不与实例 0 平均)
            _roi_dict(lesion_id="L", instance_idx=1,
                      mal_logits=(0.0, _logit(0.1)), t_conf=0.2),
        ]
        labs = [_lab_dict(mal=1), _lab_dict(mal=1), _lab_dict(mal=0)]
        rows = aggregate_by_roi_instance(collect_roi_records(rois, labs))
        assert len(rows) == 2
        by_idx = {r["instance_idx"]: r for r in rows}
        assert by_idx[0]["n_frames"] == 2
        assert by_idx[0]["mal_prob"] == pytest.approx(0.8, abs=1e-6)
        assert by_idx[0]["t_conf"] == pytest.approx(0.7, abs=1e-9)
        assert by_idx[1]["n_frames"] == 1
        assert by_idx[1]["mal_prob"] == pytest.approx(0.1, abs=1e-6)
        assert by_idx[1]["t_conf"] == pytest.approx(0.2, abs=1e-9)

    def test_instance_rows_keep_is_main_per_instance(self):
        """is_main 按实例逐条保留(官方筛选键,不许图级/lesion 级平均)。"""
        rois = [
            _roi_dict(lesion_id="L", instance_idx=0, is_main=1),
            _roi_dict(lesion_id="L", instance_idx=1, is_main=0),
        ]
        labs = [_lab_dict(), _lab_dict()]
        rows = aggregate_by_roi_instance(collect_roi_records(rois, labs))
        by_idx = {r["instance_idx"]: r for r in rows}
        assert by_idx[0]["is_main"] == 1 and by_idx[1]["is_main"] == 0


# ── 整图链零回归 + 未标注流的逐位退化等价 ────────────────────────────────────

class TestWholeImageZeroRegression:
    def test_default_aggregation_is_aggregate_by_lesion(self):
        """compute_official_metrics 缺省 aggregation = aggregate_by_lesion
        (整图调用点零改动的直接证据:显式传旧函数 == 缺省,逐位相等)。"""
        recs = []
        for i in range(4):
            recs.append({
                "lesion_id": f"L{i % 2}",   # L0/L1 各 2 帧(多帧聚合被消费)
                "patient_id": f"P{i}",
                "is_main": 1,
                "mal_prob": 0.1 + 0.2 * i,
                "gt_malignancy": i % 2,
                "valid_malignancy": 1,
                "location_probs": [0.5, 0.5] + [0.0] * (NUM_LOCATIONS - 2),
                "gt_lesion_loc": i % 2,
                "valid_lesion_loc": 1,
                "ete_probs": [0.25] * NUM_ETE_CLASSES,
                "ete_logits": [0.1, 0.2, 0.3, 0.4],
                "gt_extrathy_ext": i % NUM_ETE_CLASSES,
                "valid_extrathy_ext": 1,
                "t_conf": 0.3 + 0.1 * i,
                "gt_clin_t_stage": i % 2,
                "valid_clin_t_stage": 1,
                "tirads_probs": [0.2] * NUM_TIRADS_CLASSES,
                "gt_tirads": i % NUM_TIRADS_CLASSES,
                "valid_tirads": 1,
            })
        m_default = compute_official_metrics(recs)
        m_explicit = compute_official_metrics(
            recs, aggregation=aggregate_by_lesion)
        assert m_default == m_explicit, (
            "缺省聚合必须仍是 aggregate_by_lesion(整图链零回归)"
        )

    def test_untagged_records_degrade_to_lesion_level_bitwise(self):
        """未携带 instance_idx 的记录流(整图链/旧 GT 流)在新聚合下
        逐位等价于旧 aggregate_by_lesion —— 行序/逐键全等(退化保证)。"""
        rng = np.random.RandomState(0)
        recs = []
        for i in range(6):
            recs.append({
                "lesion_id": f"L{i % 3}",   # 3 病灶 × 各 2 帧
                "patient_id": f"P{i}",
                "is_main": 1,
                "mal_prob": float(rng.rand()),
                "gt_malignancy": i % 2,
                "valid_malignancy": 1,
                "location_probs": list(rng.rand(NUM_LOCATIONS)),
                "gt_lesion_loc": i % NUM_LOCATIONS,
                "valid_lesion_loc": 1,
                "ete_probs": list(rng.rand(NUM_ETE_CLASSES)),
                "ete_logits": list(rng.randn(NUM_ETE_CLASSES)),
                "gt_extrathy_ext": i % NUM_ETE_CLASSES,
                "valid_extrathy_ext": 1,
                "t_conf": float(rng.rand()),
                "gt_clin_t_stage": i % 2,
                "valid_clin_t_stage": 1,
                "tirads_probs": list(rng.rand(NUM_TIRADS_CLASSES)),
                "gt_tirads": i % NUM_TIRADS_CLASSES,
                "valid_tirads": 1,
            })
        old_rows = aggregate_by_lesion(recs)
        new_rows = aggregate_by_roi_instance(recs)
        assert len(old_rows) == len(new_rows) == 3
        for a, b in zip(old_rows, new_rows):
            # instance_idx 是新输出的**加性诊断键**(值恒 0),旧口径没有;
            # 除它之外键集必须全等、逐键逐值等价。
            assert set(a.keys()) == set(b.keys()) - {"instance_idx"}
            assert b["instance_idx"] == 0
            for k, v in a.items():
                assert b[k] == v, f"未标注流退化必须逐位等价,键 {k} 漂移"

    def test_metrics_unchanged_for_untagged_stream(self):
        """指标层:未标注流经新聚合 == 经旧聚合,全部指标逐位相等。"""
        rng = np.random.RandomState(1)
        recs = []
        for i in range(8):
            lp = list(rng.rand(NUM_LOCATIONS))
            recs.append({
                "lesion_id": f"L{i % 4}",
                "patient_id": f"P{i}",
                "is_main": 1,
                "mal_prob": float(rng.rand()),
                "gt_malignancy": i % 2,
                "valid_malignancy": 1,
                "location_probs": lp,
                "gt_lesion_loc": i % 2,
                "valid_lesion_loc": 1,
                "ete_probs": list(rng.rand(NUM_ETE_CLASSES)),
                "ete_logits": list(rng.randn(NUM_ETE_CLASSES)),
                "gt_extrathy_ext": i % 2,
                "valid_extrathy_ext": 1,
                "t_conf": float(rng.rand()),
                "gt_clin_t_stage": i % 2,
                "valid_clin_t_stage": 1,
                "tirads_probs": list(rng.rand(NUM_TIRADS_CLASSES)),
                "gt_tirads": i % 2,
                "valid_tirads": 1,
            })
        m_old = compute_official_metrics(recs, aggregation=aggregate_by_lesion)
        m_new = compute_official_metrics(recs, aggregation=aggregate_by_roi_instance)
        assert m_old == m_new

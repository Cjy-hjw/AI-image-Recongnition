# -*- coding: utf-8 -*-
"""evaluate_seg_only_gate 纯函数契约测试(Gate2 修复轮专审 F2,2026-09-14)。

抽出该函数的理由(F7-1)就是"内联在 main 里=零测试覆盖",抽出后必须补测,
否则抽出的动作只换了位置没换覆盖。同型先例:tests/test_seg_guards.py 之于
evaluate_seg_guards。

关键分支(修复轮专审 F6 一并钉死:顶层 None 不崩、报"无法判定"):
1. val_dice 达标 + 五项全不跌 → (True, 已过)
2. val_dice 不达标 → (False, 未过)
3. 某项跌超 0.01 → (False, 未过+列明细)
4. 指标缺值(基线/当轮 None)→ (None, 无法判定)——🔴 pass 门跳过 None 是
   把"测不出"当"通过",与 abort 门的保守方向相反
5. 顶层入参 None(val_dice/seg_baseline/metrics)→ (None, 无法判定)不崩
   (2026-09-14 修复轮专审 F6:此前 math.isfinite(None) 直接 TypeError)
6. 恰好跌 0.01 不算越界(> 0.01 判据)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_official import evaluate_seg_only_gate  # noqa: E402

_BASE = {
    "malignancy_auc_pr": 0.9359, "tirads_weighted_kappa": 0.8733,
    "location_weighted_auc": 0.9406, "clinical_t_auc": 0.9002,
    "ete_weighted_kappa": 0.5264,
}


def _cur(**overrides):
    d = dict(_BASE)
    d.update(overrides)
    return d


def test_all_pass():
    ok, lines = evaluate_seg_only_gate(0.75, dict(_BASE), _cur())
    assert ok is True
    assert any("已过" in ln for ln in lines)


def test_dice_below_threshold_fails():
    ok, lines = evaluate_seg_only_gate(0.71, dict(_BASE), _cur())
    assert ok is False
    assert any("未过" in ln for ln in lines)


def test_classification_drop_fails_with_detail():
    ok, lines = evaluate_seg_only_gate(0.75, dict(_BASE),
                                       _cur(ete_weighted_kappa=0.50))
    assert ok is False
    assert any("ete:0.5264→0.5000" in ln.replace(" ", "") for ln in lines)


def test_missing_metric_values_reports_undecidable():
    # 当轮某指标 None:不得报"已过"(pass 门跳过 None=把测不出当通过)
    ok, lines = evaluate_seg_only_gate(
        0.75, dict(_BASE), _cur(tirads_weighted_kappa=None))
    assert ok is None
    assert any("无法判定" in ln for ln in lines)
    # 基线 None 同理
    base = dict(_BASE); base["clinical_t_auc"] = None
    ok2, _ = evaluate_seg_only_gate(0.75, base, _cur())
    assert ok2 is None


def test_toplevel_none_inputs_report_undecidable_not_crash():
    # 修复轮专审 F6:math.isfinite(None) 曾 TypeError,现报无法判定
    ok, lines = evaluate_seg_only_gate(None, dict(_BASE), _cur())
    assert ok is None and any("无法判定" in ln for ln in lines)
    ok2, _ = evaluate_seg_only_gate(0.75, None, _cur())
    assert ok2 is None
    ok3, _ = evaluate_seg_only_gate(0.75, dict(_BASE), None)
    assert ok3 is None


def test_nan_metric_reports_undecidable_not_pass():
    """NaN/+inf 与 None 同等报"无法判定"(3B 专审 P2-1:nan>0.01 恒 False、
    baseline-inf 也不触发 bad——两种"测不出"曾被判"已过",与 docstring
    保守纪律矛盾)。当前生产链路大概率不可达,防御性自持。"""
    nan = float("nan")
    # 当轮 NaN
    ok, lines = evaluate_seg_only_gate(0.75, dict(_BASE),
                                       _cur(ete_weighted_kappa=nan))
    assert ok is None and any("无法判定" in ln for ln in lines)
    # 基线 NaN
    base = dict(_BASE); base["clinical_t_auc"] = nan
    ok2, _ = evaluate_seg_only_gate(0.75, base, _cur())
    assert ok2 is None
    # 当轮 +inf
    ok3, _ = evaluate_seg_only_gate(0.75, dict(_BASE),
                                    _cur(malignancy_auc_pr=float("inf")))
    assert ok3 is None


def test_exactly_01_drop_is_not_violation():
    """门限边界必须可区分(3B 专审 P2-5:旧版只覆盖 0.0099/0.02 两档,
    > 与 >= 的变异对全文件零区分力)。用 nextafter 构造严格边界:
    drop 恰好 == 0.01(不越界,> 判据)与刚好超过 0.01(越界)两档。"""
    import math as _m
    base = dict(_BASE); base["ete_weighted_kappa"] = 0.6
    gate = 0.01
    # 恰好跌 0.01:cur = base - 0.01 的"下邻"使差值严格小于 0.01+ulp
    cur_exact = _m.nextafter(0.6 - gate, 0.6)  # 差值 < 0.01 → 不越界
    ok, _ = evaluate_seg_only_gate(0.75, base,
                                   _cur(ete_weighted_kappa=cur_exact))
    assert ok is True, f"差值 {0.6 - cur_exact!r} 不应越界"
    # 刚好超过:cur 再往下压一档使差值 > 0.01 → 越界
    cur_over = _m.nextafter(0.6 - gate, -1.0)
    ok2, _ = evaluate_seg_only_gate(0.75, base,
                                    _cur(ete_weighted_kappa=cur_over))
    assert ok2 is False, f"差值 {0.6 - cur_over!r} 应越界"

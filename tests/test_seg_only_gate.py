# -*- coding: utf-8 -*-
"""evaluate_seg_only_gate 纯函数契约测试(Gate2 修复轮专审 F2,2026-09-14)。

抽出该函数的理由(F7-1)就是"内联在 main 里=零测试覆盖",抽出后必须补测,
否则抽出的动作只换了位置没换覆盖。同型先例:tests/test_seg_guards.py 之于
evaluate_seg_guards。

🔴 2026-09-18 seg_only 调教:**val_dice≥0.72 硬门已删**(当年判死用的错尺子,
P0 验尸实锤官方口径下 seg_only 双超 r5;选模改由训后快照官方口径筛)。本文件
同步改写:dice 不再参与判定(低 dice 不再 False),并补快照路径纯函数测试。

关键分支:
1. 五项全不跌 → (True, 已过)——**即使 val_dice 很低**(0.60 也过,门只管冻结泄漏)
2. 某项跌超 0.01 → (False, 未过+列明细)
3. 指标缺值(基线/当轮 None)→ (None, 无法判定)——🔴 pass 门跳过 None 是
   把"测不出"当"通过",与 abort 门的保守方向相反
4. 顶层入参 None(seg_baseline/metrics)→ (None, 无法判定)不崩
5. 恰好跌 0.01 不算越界(> 0.01 判据)
6. 快照路径:<out 去后缀>_ep<N>.pth;快照点上限锚 SEG_ABORT_EPOCH
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_official import (  # noqa: E402
    evaluate_seg_only_gate, _snapshot_path, _SEG_SNAPSHOT_EPOCHS,
    SEG_ABORT_EPOCH,
)

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


def test_low_dice_no_longer_fails():
    """🔴 2026-09-18 删 dice 硬门:低 dice 不再 False。当年 0.6861<0.72 被判
    死,而官方口径下它是最强的——门只管"分类是否冻住",dice 留档不判。"""
    ok, lines = evaluate_seg_only_gate(0.60, dict(_BASE), _cur())
    assert ok is True
    assert any("已过" in ln for ln in lines)
    # dice 仍出现在留档文案里(操作者能对照)
    assert any("0.6000" in ln for ln in lines)


def test_zero_dice_no_longer_fails():
    """🔴 dice=0.0 也过:这条断言专门杀"dice 门被加回"变异——审查实锤
    val_dice=0.60/None 对任何合理阈值(0.0,0.6)恒 GREEN 零区分力,加回
    0.3/0.5 门只有 0.0 这种极值才 RED。0 是真实可达值(分割未起步)。"""
    ok, _ = evaluate_seg_only_gate(0.0, dict(_BASE), _cur())
    assert ok is True


def test_dice_none_still_decidable():
    """val_dice=None(旧版顶层 None 报无法判定):删门后不再参与判定,
    缺失也能判分类门——但这不是"把测不出当通过"(测的是分类,dice 只是留档)。"""
    ok, _ = evaluate_seg_only_gate(None, dict(_BASE), _cur())
    assert ok is True


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
    base: dict = dict(_BASE); base["clinical_t_auc"] = None
    ok2, _ = evaluate_seg_only_gate(0.75, base, _cur())
    assert ok2 is None


def test_toplevel_none_inputs_report_undecidable_not_crash():
    # seg_baseline/metrics None → 无法判定;val_dice None 不再拦(删门后仅留档)
    ok, _ = evaluate_seg_only_gate(0.75, None, _cur())
    assert ok is None
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


# ── 快照路径与快照点(2026-09-18 调教)────────────────────────────────────


def test_snapshot_path_versioned():
    assert _snapshot_path("checkpoints/seg_v2.pth", 8) == \
        "checkpoints/seg_v2_ep8.pth"
    # 无后缀/多后缀:splitext 只去最后一段
    assert _snapshot_path("checkpoints/seg_v2", 4) == "checkpoints/seg_v2_ep4.pth"
    assert _snapshot_path("checkpoints/a.b.pth", 12) == "checkpoints/a.b_ep12.pth"


def test_snapshot_epochs_within_gamma_window():
    """快照点上限必须 ≤ SEG_ABORT_EPOCH(γ_prior ep≤20 恒 0 单变量窗口):
    越界 = 快照落在 γ>0 区间,单变量前提被污染。常量同源防漂移。"""
    assert _SEG_SNAPSHOT_EPOCHS[-1] <= SEG_ABORT_EPOCH
    assert _SEG_SNAPSHOT_EPOCHS == tuple(sorted(set(_SEG_SNAPSHOT_EPOCHS)))

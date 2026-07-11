"""report_generator.py 回归测试（2026-07-10 Gate2审查发现该文件此前零测试覆盖）。

覆盖两类此前静默的真实bug：
  1. calculate_tirads_points 对越界 idx 静默按0分处理，会悄悄跨越临床处置边界
     （如 margin_idx=5 本应算2分被吞成0分，TR3→TR2，穿刺建议消失）。
  2. generate_report 对 malignancy_prob=nan/None 无校验，nan>=0.5 恒为False导致
     "确定性良性倾向"结论被静默给出——癌症筛查场景的假阴性风险。
两者修复后均改为硬 raise ValueError，交给 predict.py/video_predict.py 的外层
except Exception 转 ERROR result.json，不允许带着无效/越界数据继续生成报告。
"""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.utils.report_generator import calculate_tirads_points, generate_report


class TestCalculateTiradsPoints:
    def test_all_zero_indices_gives_tr1(self):
        b = calculate_tirads_points(0, 0, 0, 0, [0.0, 0.0, 0.0, 0.0])
        assert b["total_points"] == 0
        assert b["tr_level"].startswith("TR1")

    def test_known_combo_matches_acr_whitepaper(self):
        # solid(2) + very hypoechoic(3) + taller-than-wide(3) + smooth(0) + no foci(0) = 8 → TR5
        b = calculate_tirads_points(3, 3, 1, 0, [0.0, 0.0, 0.0, 0.0])
        assert b["total_points"] == 8
        assert b["tr_level"].startswith("TR5")

    def test_margin_lobulated_contributes_2_points(self):
        # margin_idx=2 (Lobulated/Irregular) 应计2分——回归本次修复前"越界idx静默
        # 记0分"的对照:合法idx=2必须正确计分,不能被误伤。
        b = calculate_tirads_points(0, 1, 0, 2, [0.0, 0.0, 0.0, 0.0])
        assert b["margin"] == ("边缘不规则或分叶状 (Lobulated / Irregular)", 2)
        assert b["total_points"] == 1 + 2  # echo(1分) + margin(2分)

    def test_out_of_range_margin_idx_raises_instead_of_silent_zero(self):
        """回归 Gate2 发现的具体案例:margin_idx=5(非法)此前被 `.get(idx,("未知",0))`
        静默记0分,总分从真值3(TR3)算成1(TR2)——静默跨越穿刺建议边界。现在必须
        硬 raise,不允许静默算错。"""
        with pytest.raises(ValueError, match="margin_idx"):
            calculate_tirads_points(1, 1, 0, 5, [0.0, 0.0, 0.0, 0.0])

    def test_out_of_range_comp_echo_shape_idx_all_raise(self):
        for kwargs, bad_key in [
            (dict(comp_idx=99, echo_idx=0, shape_idx=0, margin_idx=0), "comp_idx"),
            (dict(comp_idx=0, echo_idx=99, shape_idx=0, margin_idx=0), "echo_idx"),
            (dict(comp_idx=0, echo_idx=0, shape_idx=99, margin_idx=0), "shape_idx"),
        ]:
            with pytest.raises(ValueError, match=bad_key):
                calculate_tirads_points(foci_probs=[0.0, 0.0, 0.0, 0.0], **kwargs)

    def test_foci_index_out_of_range_raises_instead_of_crashing(self):
        """回归 Gate2 发现的崩溃bug:FOCI_MAP.get(i) 此前无默认值,foci_probs 若
        >4维会在越界index处 TypeError 崩溃(解包None)。现在应给出清晰的
        ValueError,而不是裸 TypeError。"""
        with pytest.raises(ValueError, match="foci_idx"):
            calculate_tirads_points(0, 0, 0, 0, [0.1, 0.2, 0.3, 0.4, 0.6])  # 5维,index4越界

    def test_foci_within_range_multi_label_sums_correctly(self):
        # foci多标签:index0(0分)与index3(3分)同时命中
        b = calculate_tirads_points(0, 0, 0, 0, [0.9, 0.0, 0.0, 0.9])
        assert b["total_points"] == 0 + 3
        assert len(b["foci"]) == 2

    def test_tr_boundary_edges(self):
        assert calculate_tirads_points(0, 0, 0, 0, [0, 0, 0, 0])["tr_level"].startswith("TR1")
        assert calculate_tirads_points(2, 0, 0, 0, [0, 0, 0, 0])["tr_level"].startswith("TR2")  # 1分
        assert calculate_tirads_points(3, 1, 0, 0, [0, 0, 0, 0])["tr_level"].startswith("TR3")  # solid(2)+hyper(1)=3分
        assert calculate_tirads_points(3, 3, 0, 0, [0, 0, 0, 0])["tr_level"].startswith("TR4")  # 5分
        assert calculate_tirads_points(3, 3, 1, 0, [0, 0, 0, 0])["tr_level"].startswith("TR5")  # 8分


class TestGenerateReport:
    def _valid_breakdown(self):
        return calculate_tirads_points(0, 1, 0, 0, [0.0, 0.0, 0.0, 0.0])

    def test_valid_prob_produces_report(self):
        txt = generate_report("img1", "p1", self._valid_breakdown(), 0.73)
        assert "73.00%" in txt
        assert "恶性可疑" in txt

    def test_benign_boundary_below_half(self):
        txt = generate_report("img1", "p1", self._valid_breakdown(), 0.49)
        assert "良性倾向" in txt

    def test_none_malignancy_prob_raises(self):
        """回归崩溃bug:此前 malignancy_prob*100 对 None 直接 TypeError。"""
        with pytest.raises(ValueError, match="malignancy_prob"):
            generate_report("img1", "p1", self._valid_breakdown(), None)

    def test_nan_malignancy_prob_raises_instead_of_silently_reporting_benign(self):
        """回归静默临床安全bug:此前 nan>=0.5 恒为False,会**无任何报错**地打印
        "良性倾向"——本该判定为"无效/不确定"的情况被静默判成良性,是癌症筛查
        场景的假阴性风险。现在必须硬拦截。"""
        with pytest.raises(ValueError, match="malignancy_prob"):
            generate_report("img1", "p1", self._valid_breakdown(), float("nan"))

    def test_inf_malignancy_prob_raises(self):
        with pytest.raises(ValueError, match="malignancy_prob"):
            generate_report("img1", "p1", self._valid_breakdown(), float("inf"))

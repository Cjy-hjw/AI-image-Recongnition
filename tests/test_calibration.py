"""温度缩放校准骨架测试(Gate 1 可量化验证)。

核心保证:
1. 降级正确性 —— 缺参数 / 全 T=1.0 时,decode_predictions 输出与无校准逐位相等。
2. T>1 确实压低 max 概率,但 **不改 argmax**(温度缩放单调性)。
3. ECE 在已知合成数据上数值正确。
4. 参数文件缺失 / 损坏不崩溃。
"""

import json

import numpy as np
import torch

from src.inference.calibration import (
    TemperatureScaler,
    compute_ece,
    fit_temperature,
)
from src.inference.postprocess import decode_predictions


def _fake_outputs(seed: int = 0) -> dict:
    """构造一份各头 logits(模拟模型输出),形状与 MultiTaskThyroidModel 对齐。"""
    torch.manual_seed(seed)
    return {
        "mask": torch.randn(1, 1, 8, 8),
        "composition": torch.randn(1, 4),
        "echogenicity": torch.randn(1, 4),
        "shape": torch.randn(1, 2),
        "margin": torch.randn(1, 4),
        "foci": torch.randn(1, 4),
        "malignancy": torch.randn(1, 2),
    }


class TestDegradation:
    """缺参数 / 全 T=1.0 → 必须无任何行为变化。"""

    def test_missing_file_is_identity(self):
        cal = TemperatureScaler.from_file("checkpoints/__definitely_absent__.json")
        assert cal.is_identity()

    def test_decode_identical_when_uncalibrated(self):
        outputs = _fake_outputs()
        base = decode_predictions(outputs, calibrator=None)
        cal = TemperatureScaler.from_file(None)  # 全 T=1.0
        calibrated = decode_predictions(_fake_outputs(), calibrator=cal)
        # 逐位相等(降级 = 不校准)
        assert base["comp_idx"] == calibrated["comp_idx"]
        assert base["echo_idx"] == calibrated["echo_idx"]
        assert base["shape_idx"] == calibrated["shape_idx"]
        assert base["margin_idx"] == calibrated["margin_idx"]
        assert base["mal_prob"] == calibrated["mal_prob"]
        np.testing.assert_array_equal(base["foci_prob"], calibrated["foci_prob"])

    def test_corrupt_file_degrades(self, tmp_path):
        bad = tmp_path / "calibration.json"
        bad.write_text("{not valid json", encoding="utf-8")
        cal = TemperatureScaler.from_file(str(bad))
        assert cal.is_identity()


class TestTemperatureEffect:
    """T>1 压低置信度但不改 argmax。"""

    def test_softens_but_preserves_argmax(self):
        outputs = _fake_outputs(seed=3)
        cal = TemperatureScaler({"malignancy": 2.5})
        base = decode_predictions(_fake_outputs(seed=3), calibrator=None)
        warm = decode_predictions(outputs, calibrator=cal)
        # malignancy 概率向 0.5 收缩(过度自信被压低)
        assert abs(warm["mal_prob"] - 0.5) < abs(base["mal_prob"] - 0.5)

    def test_argmax_invariant_under_temperature(self):
        logits = torch.tensor([[1.0, 3.0, 0.5, 2.0]])
        cal = TemperatureScaler({"composition": 5.0})
        scaled = cal.apply("composition", logits)
        assert torch.argmax(scaled, dim=1).item() == torch.argmax(logits, dim=1).item()

    def test_invalid_temperature_rejected(self):
        cal = TemperatureScaler({"malignancy": -1.0, "shape": 0.0})
        assert cal.temperatures["malignancy"] == 1.0
        assert cal.temperatures["shape"] == 1.0


class TestECE:
    """ECE 数值正确性。"""

    def test_perfectly_calibrated_low_ece(self):
        # 全预测置信度=1.0 且全部正确 → ECE=0
        probs = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        labels = np.array([1, 1, 0])
        assert compute_ece(probs, labels, n_bins=10) < 1e-9

    def test_overconfident_wrong_high_ece(self):
        # 置信度=1.0 但全错 → ECE=1.0
        probs = np.array([[0.0, 1.0], [0.0, 1.0]])
        labels = np.array([0, 0])
        assert abs(compute_ece(probs, labels, n_bins=10) - 1.0) < 1e-9

    def test_binary_prob_vector_accepted(self):
        # 一维正类概率应自动补成两列
        probs = np.array([0.9, 0.8, 0.1])
        labels = np.array([1, 1, 0])
        ece = compute_ece(probs, labels, n_bins=10)
        assert 0.0 <= ece <= 1.0


class TestFit:
    """拟合函数:可跑通且返回正温度。"""

    def test_fit_returns_positive_temperature(self):
        torch.manual_seed(0)
        logits = torch.randn(200, 2) * 4.0  # 故意放大 → 过度自信
        labels = (logits[:, 1] > logits[:, 0]).long()
        t = fit_temperature(logits, labels)
        assert t > 0

    def test_fit_from_file_roundtrip(self, tmp_path):
        p = tmp_path / "calibration.json"
        p.write_text(json.dumps({"temperatures": {"malignancy": 1.7}}), encoding="utf-8")
        cal = TemperatureScaler.from_file(str(p))
        assert cal.temperatures["malignancy"] == 1.7
        assert not cal.is_identity()

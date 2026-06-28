"""多源 DataLoader 骨架测试(Phase 4 预留,不接真数据)。

用临时目录造合成样本,验证:源级标签有效性默认、多源拼接、加权采样权重、
与单源路径向后兼容。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from src.dataset.loader import ThyroidDataset, MultiSourceThyroidDataset


def _make_sample(root, name, with_mask=True, meta=None):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    cv2.imwrite(os.path.join(d, "image.png"), img)
    if with_mask:
        cv2.imwrite(os.path.join(d, "mask.png"), (img > 128).astype(np.uint8) * 255)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta or {"patient_id": name}, f)


def _make_source(root, n, meta=None):
    os.makedirs(root, exist_ok=True)
    for i in range(n):
        _make_sample(root, f"s{i:03d}", meta=meta)
    return root


# ═════════════════════════════════════════════════════════════════════════════
# 源级标签有效性默认
# ═════════════════════════════════════════════════════════════════════════════

class TestSourceLevelMask:
    def test_default_all_valid_backward_compat(self, tmp_path):
        """无 source_label_mask、无 meta label_mask → 全部 valid=1(旧行为不变)。"""
        path = _make_source(str(tmp_path / "src"), 3)
        ds = ThyroidDataset(path, img_size=64)
        _, _, labels = ds[0]
        for t in ("composition", "malignancy", "foci"):
            assert labels[f"valid_{t}"] == 1
        assert labels["valid_mask"] == 1

    def test_source_mask_marks_missing(self, tmp_path):
        """源级声明缺 malignancy → valid_malignancy=0,其余仍 1。"""
        path = _make_source(str(tmp_path / "src"), 3)
        ds = ThyroidDataset(path, img_size=64,
                            source_label_mask={"malignancy": 0, "echogenicity": 0})
        _, _, labels = ds[0]
        assert labels["valid_malignancy"] == 0
        assert labels["valid_echogenicity"] == 0
        assert labels["valid_composition"] == 1  # 未声明 → 默认 1

    def test_meta_overrides_source_mask(self, tmp_path):
        """样本 meta.json 的 label_mask 优先于源级默认。"""
        path = str(tmp_path / "src")
        os.makedirs(path, exist_ok=True)
        # 该样本 meta 显式标 malignancy 有效,覆盖源级的 0
        _make_sample(path, "s000", meta={"patient_id": "p", "label_mask": {"malignancy": 1}})
        ds = ThyroidDataset(path, img_size=64, source_label_mask={"malignancy": 0})
        _, _, labels = ds[0]
        assert labels["valid_malignancy"] == 1, "meta label_mask 应优先于源级"


# ═════════════════════════════════════════════════════════════════════════════
# 多源拼接 + 加权采样
# ═════════════════════════════════════════════════════════════════════════════

class TestMultiSource:
    def test_concatenation_length(self, tmp_path):
        a = _make_source(str(tmp_path / "a"), 4)
        b = _make_source(str(tmp_path / "b"), 6)
        mds = MultiSourceThyroidDataset(
            [{"name": "a", "path": a}, {"name": "b", "path": b}], img_size=64)
        assert len(mds) == 10

    def test_per_source_mask_applied(self, tmp_path):
        """源 a 缺 malignancy、源 b 全标 → 取样验证各自 valid 标记正确。"""
        a = _make_source(str(tmp_path / "a"), 2)
        b = _make_source(str(tmp_path / "b"), 2)
        mds = MultiSourceThyroidDataset([
            {"name": "a", "path": a, "label_mask": {"malignancy": 0}},
            {"name": "b", "path": b},
        ], img_size=64)
        # 前 2 个来自源 a,后 2 个来自源 b
        _, _, la = mds[0]
        _, _, lb = mds[3]
        assert la["valid_malignancy"] == 0
        assert lb["valid_malignancy"] == 1

    def test_sample_weights_reflect_ratio(self, tmp_path):
        """ratio 均摊到样本:源权重和 ≈ ratio,逐样本 = ratio/n。"""
        a = _make_source(str(tmp_path / "a"), 4)
        b = _make_source(str(tmp_path / "b"), 2)
        mds = MultiSourceThyroidDataset([
            {"name": "a", "path": a, "ratio": 0.6},
            {"name": "b", "path": b, "ratio": 0.4},
        ], img_size=64)
        w = mds.get_sample_weights()
        assert len(w) == 6
        assert abs(sum(w[:4]) - 0.6) < 1e-6  # 源 a 总权重
        assert abs(sum(w[4:]) - 0.4) < 1e-6  # 源 b 总权重
        assert abs(w[0] - 0.6 / 4) < 1e-6    # 逐样本

    def test_empty_source_no_crash(self, tmp_path):
        a = _make_source(str(tmp_path / "a"), 3)
        empty = str(tmp_path / "empty")
        os.makedirs(empty, exist_ok=True)
        mds = MultiSourceThyroidDataset([
            {"name": "a", "path": a, "ratio": 1.0},
            {"name": "empty", "path": empty, "ratio": 1.0},
        ], img_size=64)
        assert len(mds) == 3  # 空源贡献 0 样本,不崩

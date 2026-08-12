"""PathologyDataset 单元测试。

覆盖:
1. build_pathology_samples: 标签广播、文件缺失跳过、batch列顺序容错
2. patient_grouped_split: 无泄漏、分层、可复现、两类均有
3. PathologyDataset.__getitem__: 输出形状/dtype/valid_* 键正确
4. class_counts: 计数准确
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest
import torch

from src.dataset.pathology_dataset import (
    PathologyDataset,
    build_pathology_samples,
    patient_grouped_split,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _make_batch(root, batch_name, patients, col_order="patient_first"):
    """
    patients: list of (patient_id_str, histo_label, n_images)
    col_order: 'patient_first' → header patient_name,path
               'path_first'    → header path,patient_name  (batch2风格)
    """
    base = os.path.join(root, batch_name)
    ds_dir = os.path.join(base, "dataset")
    os.makedirs(ds_dir, exist_ok=True)

    label_rows = []
    img_rows = []
    for pid, label, n in patients:
        label_rows.append([pid, str(label)])
        for i in range(1, n + 1):
            fname = f"{pid}_{i:03d}.Jpg"
            # 写一张小 JPG
            img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
            cv2.imwrite(os.path.join(ds_dir, fname), img)
            if col_order == "patient_first":
                img_rows.append([pid, fname])
            else:
                img_rows.append([fname, pid])

    _write_csv(
        os.path.join(base, f"{batch_name}_label.csv"),
        ["patient_name", "histo_label"],
        label_rows,
    )
    if col_order == "patient_first":
        _write_csv(
            os.path.join(base, f"{batch_name}.csv"),
            ["patient_name", "path"],
            img_rows,
        )
    else:
        _write_csv(
            os.path.join(base, f"{batch_name}.csv"),
            ["path", "patient_name"],
            img_rows,
        )
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 1. build_pathology_samples
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPathologySamples:

    def test_label_broadcast_to_all_images(self, tmp_path):
        """患者 label 正确广播到该患者所有图像。"""
        _make_batch(str(tmp_path), "batch1_image",
                    [("1", 0, 3), ("2", 1, 2)])
        samples = build_pathology_samples(str(tmp_path))
        by_pat = {}
        for s in samples:
            by_pat.setdefault(s["patient_key"], set()).add(s["label"])
        # 同一患者所有图像 label 一致
        assert by_pat["b1_1"] == {0}
        assert by_pat["b1_2"] == {1}
        assert len(samples) == 5  # 3 + 2

    def test_patient_prefix_isolation(self, tmp_path):
        """batch1/batch2 同名患者 ID 经前缀隔离为不同 patient_key。"""
        _make_batch(str(tmp_path), "batch1_image", [("0", 0, 2)], col_order="patient_first")
        _make_batch(str(tmp_path), "batch2_image", [("0", 1, 2)], col_order="path_first")
        samples = build_pathology_samples(str(tmp_path))
        keys = {s["patient_key"] for s in samples}
        assert "b1_0" in keys
        assert "b2_0" in keys
        # 两者 label 不同,验证未混淆
        b1_labels = {s["label"] for s in samples if s["patient_key"] == "b1_0"}
        b2_labels = {s["label"] for s in samples if s["patient_key"] == "b2_0"}
        assert b1_labels == {0}
        assert b2_labels == {1}

    def test_batch2_path_first_column_order(self, tmp_path):
        """batch2 CSV 列顺序 path,patient_name —— 标签广播仍正确。"""
        _make_batch(str(tmp_path), "batch2_image",
                    [("10", 1, 3)], col_order="path_first")
        samples = build_pathology_samples(str(tmp_path))
        assert len(samples) == 3
        assert all(s["label"] == 1 for s in samples)

    def test_skip_patient_without_label(self, tmp_path):
        """image.csv 有文件但 label.csv 无该患者 → 图像被跳过,不崩。"""
        base = str(tmp_path)
        batch_dir = "batch1_image"
        ds_dir = os.path.join(base, batch_dir, "dataset")
        os.makedirs(ds_dir, exist_ok=True)
        # label.csv: 只有患者1
        _write_csv(os.path.join(base, batch_dir, f"{batch_dir}_label.csv"),
                   ["patient_name", "histo_label"], [["1", "0"]])
        # 磁盘: 患者1和患者99
        for pid in ["1", "99"]:
            img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
            cv2.imwrite(os.path.join(ds_dir, f"{pid}_001.Jpg"), img)
        samples = build_pathology_samples(str(tmp_path))
        pkeys = {s["patient_key"] for s in samples}
        assert "b1_1" in pkeys
        assert "b1_99" not in pkeys  # 无标签跳过

    def test_no_samples_raises(self, tmp_path):
        """空目录 → RuntimeError,不静默。"""
        with pytest.raises(RuntimeError):
            build_pathology_samples(str(tmp_path))


# ─────────────────────────────────────────────────────────────────────────────
# 2. patient_grouped_split
# ─────────────────────────────────────────────────────────────────────────────

class TestPatientGroupedSplit:

    def _make_samples(self, n_patients_per_class=10, imgs_per_patient=5):
        samples = []
        for cls in [0, 1]:
            for i in range(n_patients_per_class):
                pid = f"b1_{cls}_{i}"
                for k in range(imgs_per_patient):
                    samples.append({"img_path": f"/x/{pid}_{k}.jpg",
                                    "label": cls, "patient_key": pid})
        return samples

    def test_no_patient_leaks_across_sets(self):
        """同一患者的所有图像必须在同一侧。"""
        samples = self._make_samples(20, 5)
        train, val = patient_grouped_split(samples, val_frac=0.2, seed=42)
        train_pats = {s["patient_key"] for s in train}
        val_pats = {s["patient_key"] for s in val}
        assert train_pats.isdisjoint(val_pats), "同患者跨 train/val 泄漏!"

    def test_all_samples_accounted(self):
        """train + val 图像数 == 总数。"""
        samples = self._make_samples(10, 5)
        train, val = patient_grouped_split(samples, val_frac=0.2, seed=0)
        assert len(train) + len(val) == len(samples)

    def test_stratified_both_classes_in_val(self):
        """val 集中两类都有(分层采样保障)。"""
        samples = self._make_samples(10, 5)
        _, val = patient_grouped_split(samples, val_frac=0.2, seed=42)
        val_labels = {s["label"] for s in val}
        assert val_labels == {0, 1}, "val 集缺少一个类,分层采样失败"

    def test_reproducible_with_same_seed(self):
        """相同 seed → 划分完全一致。"""
        samples = self._make_samples(10, 5)
        _, v1 = patient_grouped_split(samples, val_frac=0.2, seed=7)
        _, v2 = patient_grouped_split(samples, val_frac=0.2, seed=7)
        assert [s["patient_key"] for s in v1] == [s["patient_key"] for s in v2]

    def test_different_seeds_give_different_splits(self):
        """不同 seed → 划分不同(大概率,patients数够多时)。"""
        samples = self._make_samples(20, 3)
        _, v1 = patient_grouped_split(samples, val_frac=0.2, seed=0)
        _, v2 = patient_grouped_split(samples, val_frac=0.2, seed=99)
        p1 = {s["patient_key"] for s in v1}
        p2 = {s["patient_key"] for s in v2}
        assert p1 != p2

    def test_val_frac_boundary_raises(self):
        samples = self._make_samples(2, 2)
        with pytest.raises(AssertionError):
            patient_grouped_split(samples, val_frac=0.0)
        with pytest.raises(AssertionError):
            patient_grouped_split(samples, val_frac=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PathologyDataset.__getitem__
# ─────────────────────────────────────────────────────────────────────────────

class TestPathologyDatasetGetitem:

    @pytest.fixture()
    def tiny_ds(self, tmp_path):
        samples = []
        for i in range(4):
            fname = str(tmp_path / f"img_{i}.jpg")
            cv2.imwrite(fname, np.random.randint(0, 255, (64, 64), dtype=np.uint8))
            samples.append({"img_path": fname, "label": i % 2,
                             "patient_key": f"b1_{i}"})
        return PathologyDataset(samples, img_size=64)

    def test_output_shapes(self, tiny_ds):
        img, mask, labels = tiny_ds[0]
        assert img.shape == (1, 64, 64)
        assert mask.shape == (1, 64, 64)

    def test_img_dtype_range(self, tiny_ds):
        img, _, _ = tiny_ds[0]
        assert img.dtype == torch.float32
        assert img.min() >= 0.0 and img.max() <= 1.0

    def test_mask_all_zero(self, tiny_ds):
        """无分割标注 → mask 全零。"""
        _, mask, _ = tiny_ds[0]
        assert mask.sum() == 0.0

    def test_valid_malignancy_is_one(self, tiny_ds):
        _, _, labels = tiny_ds[0]
        assert labels["valid_malignancy"] == 1

    def test_valid_mask_is_zero(self, tiny_ds):
        _, _, labels = tiny_ds[0]
        assert labels["valid_mask"] == 0

    def test_other_valid_tasks_zero(self, tiny_ds):
        _, _, labels = tiny_ds[0]
        for t in ("composition", "echogenicity", "shape", "margin", "foci"):
            assert labels[f"valid_{t}"] == 0, f"valid_{t} 应为 0"

    def test_foci_is_zero_tensor(self, tiny_ds):
        _, _, labels = tiny_ds[0]
        assert isinstance(labels["foci"], torch.Tensor)
        assert labels["foci"].shape == (4,)
        assert labels["foci"].sum() == 0.0

    def test_malignancy_label_correct(self, tmp_path):
        """确认标签值与构造时一致。"""
        fname = str(tmp_path / "img.jpg")
        cv2.imwrite(fname, np.zeros((32, 32), dtype=np.uint8))
        ds = PathologyDataset(
            [{"img_path": fname, "label": 1, "patient_key": "b1_0"}],
            img_size=32,
        )
        _, _, labels = ds[0]
        assert labels["malignancy"] == 1

    def test_class_counts(self, tmp_path):
        samples = []
        for i in range(6):
            fname = str(tmp_path / f"img_{i}.jpg")
            cv2.imwrite(fname, np.zeros((32, 32), dtype=np.uint8))
            samples.append({"img_path": fname, "label": 0 if i < 4 else 1,
                             "patient_key": f"b1_{i}"})
        ds = PathologyDataset(samples, img_size=32)
        counts = ds.class_counts()
        assert counts[0] == 4
        assert counts[1] == 2

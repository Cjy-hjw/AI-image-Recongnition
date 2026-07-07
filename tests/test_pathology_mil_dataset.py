"""PathologyMILDataset + MIL 模型单元测试。

覆盖:
1. build_patient_bags: 帧按患者聚合、bag label=患者 label、可复现顺序
2. PathologyMILDataset: bag 形状 [N,1,H,W]、max_frames 截断、患者级计数
3. 无泄漏: train/val bag 患者集合不相交(继承 patient_grouped_split)
4. GatedAttentionPooling: 输出形状、权重归一化(sum≈1)
5. ThyroidMILNet: 冻结后 backbone 无梯度、前向 [N,1,H,W]→[1,2]

不碰 test_pathology_dataset.py(现有 20 测试独立)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from src.dataset.pathology_mil_dataset import build_patient_bags, PathologyMILDataset


# ─────────────────────────────────────────────────────────────────────────────
# helpers:构造 flat 单帧 samples(与 build_pathology_samples 输出同构)
# ─────────────────────────────────────────────────────────────────────────────

def _make_samples(tmp_path, spec):
    """spec: list of (patient_key, label, n_frames)。写小 JPG,返回 samples 列表。"""
    import cv2
    samples = []
    for pkey, label, n in spec:
        for i in range(n):
            fp = os.path.join(tmp_path, f"{pkey}_{i:03d}.jpg")
            cv2.imwrite(fp, np.random.randint(0, 255, (32, 32), dtype=np.uint8))
            samples.append({"img_path": fp, "label": label, "patient_key": pkey})
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# build_patient_bags
# ─────────────────────────────────────────────────────────────────────────────

def test_bags_group_frames_by_patient(tmp_path):
    samples = _make_samples(tmp_path, [("b1_0", 1, 3), ("b1_1", 0, 5)])
    bags = build_patient_bags(samples)
    assert len(bags) == 2
    by = {b["patient_key"]: b for b in bags}
    assert len(by["b1_0"]["img_paths"]) == 3
    assert len(by["b1_1"]["img_paths"]) == 5


def test_bag_label_matches_patient(tmp_path):
    samples = _make_samples(tmp_path, [("b1_0", 1, 2), ("b1_1", 0, 2)])
    bags = build_patient_bags(samples)
    by = {b["patient_key"]: b for b in bags}
    assert by["b1_0"]["label"] == 1
    assert by["b1_1"]["label"] == 0


def test_bags_reproducible_order(tmp_path):
    samples = _make_samples(tmp_path, [("b1_2", 1, 1), ("b1_0", 0, 1), ("b1_1", 1, 1)])
    bags = build_patient_bags(samples)
    keys = [b["patient_key"] for b in bags]
    assert keys == sorted(keys)  # 按 patient_key 排序,可复现


# ─────────────────────────────────────────────────────────────────────────────
# PathologyMILDataset
# ─────────────────────────────────────────────────────────────────────────────

def test_getitem_bag_shape(tmp_path):
    samples = _make_samples(tmp_path, [("b1_0", 1, 4)])
    bags = build_patient_bags(samples)
    ds = PathologyMILDataset(bags, img_size=64, max_frames_per_bag=32)
    bag, label, pkey = ds[0]
    assert bag.shape == (4, 1, 64, 64)
    assert bag.dtype == torch.float32
    assert 0.0 <= bag.min() and bag.max() <= 1.0
    assert label == 1
    assert pkey == "b1_0"


def test_max_frames_truncation(tmp_path):
    samples = _make_samples(tmp_path, [("b1_0", 1, 50)])
    bags = build_patient_bags(samples)
    ds = PathologyMILDataset(bags, img_size=32, max_frames_per_bag=10)
    bag, _, _ = ds[0]
    assert bag.shape[0] == 10  # 截断到 10 帧


def test_under_max_frames_keeps_all(tmp_path):
    samples = _make_samples(tmp_path, [("b1_0", 0, 6)])
    bags = build_patient_bags(samples)
    ds = PathologyMILDataset(bags, img_size=32, max_frames_per_bag=32)
    bag, _, _ = ds[0]
    assert bag.shape[0] == 6  # 不足上界全取


def test_patient_class_counts(tmp_path):
    samples = _make_samples(tmp_path, [("b1_0", 1, 3), ("b1_1", 1, 2), ("b1_2", 0, 4)])
    bags = build_patient_bags(samples)
    ds = PathologyMILDataset(bags, img_size=32)
    # 患者级计数:2 恶(b1_0,b1_1) + 1 良(b1_2),不受帧数影响
    assert ds.patient_class_counts() == {1: 2, 0: 1}


# ─────────────────────────────────────────────────────────────────────────────
# 无泄漏(结合 patient_grouped_split)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_patient_leak_after_bagging(tmp_path):
    from src.dataset.pathology_dataset import patient_grouped_split
    spec = [(f"b1_{i}", i % 2, 3) for i in range(20)]
    samples = _make_samples(tmp_path, spec)
    train_s, val_s = patient_grouped_split(samples, val_frac=0.3, seed=42)
    train_bags = build_patient_bags(train_s)
    val_bags = build_patient_bags(val_s)
    tr = {b["patient_key"] for b in train_bags}
    va = {b["patient_key"] for b in val_bags}
    assert tr & va == set()  # 无患者跨集


# ─────────────────────────────────────────────────────────────────────────────
# MIL 模型(纯前向,不加载权重,--no-pretrained 等效)
# ─────────────────────────────────────────────────────────────────────────────

def test_gated_attention_pooling_shape_and_norm():
    from src.models.mil_pooling import GatedAttentionPooling
    pool = GatedAttentionPooling(feature_dim=1408, attn_dim=128)
    x = torch.randn(7, 1408)  # 7 帧
    bag_repr, weights = pool(x)
    assert bag_repr.shape == (1, 1408)
    assert weights.shape == (7, 1)
    assert abs(weights.sum().item() - 1.0) < 1e-5  # softmax 归一


def test_mil_net_forward_and_freeze():
    from src.models.mil_pooling import ThyroidMILNet
    # pretrained=False 避免测试依赖权重文件
    model = ThyroidMILNet(pretrained=False, freeze_backbone=True)
    # backbone 全冻
    assert all(not p.requires_grad for p in model.backbone.parameters())
    # 头可训
    assert any(p.requires_grad for p in model.mil_pooling.parameters())
    assert any(p.requires_grad for p in model.classifier.parameters())
    # 前向:一个 5 帧 bag → [1,2]
    bag = torch.randn(5, 1, 64, 64)
    model.eval()
    logits, weights = model(bag)
    assert logits.shape == (1, 2)
    assert weights.shape == (5, 1)


def test_mil_net_trainable_params_exclude_backbone():
    from src.models.mil_pooling import ThyroidMILNet
    model = ThyroidMILNet(pretrained=False, freeze_backbone=True)
    trainable = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable)
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    # 可训参数 = 总 - backbone(冻结)
    assert n_trainable == n_total - n_backbone
    assert n_trainable > 0

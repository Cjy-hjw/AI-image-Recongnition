"""真实性分类器训练器 smoke test。

本地无 TN3K 真实超声底图,无法产出可上场的 baseline 权重(真训留服务器)。
此 smoke test 用合成"伪超声"张量喂训练器,验证端到端:
  特征集构建 → 训练 → 保存 dict(state_dict+feat_mean+feat_std) → 推理端加载并应用。
重点守卫此前修过的"训练/推理归一化参数一致性"契约(feat_mean/feat_std 必须
随权重保存且推理端 predict 套用同一组 (x-mean)/(std+eps))。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest
import torch

from src.training.authenticity_trainer import train_authenticity_classifier
from src.gatekeeper.authenticity import AuthenticityModule


def _fake_ultrasound_tensor(seed):
    """合成 [1,H,W] float(0~1) 伪超声张量(渐变+椭圆+散斑),作训练器输入底图。"""
    rng = np.random.RandomState(seed)
    img = np.zeros((128, 128), np.float32)
    yy, xx = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
    img += 80 + 40 * np.sin(xx / 30.0) + 30 * (yy / 128.0)
    cv2.ellipse(img, (64, 60), (25, 18), int(rng.randint(0, 180)), 0, 360, 180, -1)
    img += rng.normal(0, 8, (128, 128))
    img = np.clip(img, 0, 255) / 255.0
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


class _FakeDataset:
    """最小数据集契约:__len__ + __getitem__ → (img_tensor, ...)。"""
    def __init__(self, n, seed_base=0):
        self.items = [_fake_ultrasound_tensor(seed_base + i) for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return (self.items[i], None, {})


def test_authenticity_trainer_end_to_end(tmp_path):
    """训练器跑通并产出含归一化参数的 checkpoint,推理端能加载并应用。"""
    train_ds = _FakeDataset(12, seed_base=0)
    val_ds = _FakeDataset(6, seed_base=10000)
    save_path = str(tmp_path / "authenticity_model.pth")

    train_authenticity_classifier(train_ds, val_ds, torch.device("cpu"), save_path)

    assert os.path.exists(save_path), "训练器未产出 checkpoint"
    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    # 契约:必须含 state_dict + feat_mean + feat_std(归一化参数随权重保存)
    assert "state_dict" in ckpt
    assert "feat_mean" in ckpt and "feat_std" in ckpt
    assert np.asarray(ckpt["feat_mean"]).shape == (13,)
    assert np.asarray(ckpt["feat_std"]).shape == (13,)


def test_authenticity_module_loads_and_normalizes(tmp_path, monkeypatch):
    """推理端 AuthenticityModule 加载训练产物后,feat_mean/feat_std 被填充且 predict 可跑。"""
    # 2026-07-30 交付封印:EXPECTED_CKPT_SHA256 已设为当前生产权重的实际哈希。
    # 本测试训练一个**全新的**随机权重,哈希必然不匹配——须临时把校验关掉
    # (monkeypatch 置 None),否则测的不是"加载+归一化"而是"哈希校验拦截"。
    # 哈希校验本身的正确性由 test_sha256_mismatch_raises / test_sha256_match_passes 覆盖。
    monkeypatch.setattr(AuthenticityModule, "EXPECTED_CKPT_SHA256", None)
    train_ds = _FakeDataset(12, seed_base=0)
    val_ds = _FakeDataset(6, seed_base=10000)
    save_path = str(tmp_path / "authenticity_model.pth")
    train_authenticity_classifier(train_ds, val_ds, torch.device("cpu"), save_path)

    module = AuthenticityModule()
    module.load_weights(save_path)
    # 归一化参数确实被加载(此前 bug:训练存了但推理没用 → 训练/推理分布不一致)
    assert module.feat_mean is not None and module.feat_std is not None
    assert np.asarray(module.feat_mean).shape == (13,)

    # predict 端到端可跑,返回 (类别, 概率向量[3])
    img = (np.clip(
        np.random.RandomState(1).normal(120, 30, (128, 128)), 0, 255)).astype(np.uint8)
    pred, prob = module.predict(img)
    assert pred in (0, 1, 2)
    assert np.asarray(prob).shape == (3,)
    assert abs(float(np.asarray(prob).sum()) - 1.0) < 1e-4, "softmax 概率应和为 1"


def test_load_weights_raises_when_file_missing(tmp_path):
    """回归 2026-07-10 Gate2审查发现的静默bug:load_weights 此前对不存在的路径
    直接返回(无 else 分支),模型停留在随机初始化态,predict() 仍返回合法 0/1/2、
    零日志零标志位——真实性检测器悄悄变成随机分类器,gatekeeper 一票否决门
    失效或随机误杀且无人察觉。现在必须硬报错,不允许静默退化。"""
    module = AuthenticityModule()
    missing_path = str(tmp_path / "does_not_exist.pth")
    with pytest.raises(RuntimeError, match="权重文件不存在"):
        module.load_weights(missing_path)
    # 硬报错前不能悄悄改动任何状态(仍是 __init__ 的干净初始状态)
    assert module.feat_mean is None and module.feat_std is None


class TestExpectedCkptSha256:
    """EXPECTED_CKPT_SHA256 完整性校验(2026-07-30 交付封印启用)的双向回归。

    配套 `weights_only=False`(ckpt 含 numpy 标准化统计)防 pickle 任意代码执行。
    校验启用后:哈希匹配 → 正常加载;不匹配(文件被篡改/损坏/重训后忘更新常量)
    → 硬 RuntimeError,不允许带不可信权重继续跑。
    """

    def _make_ckpt(self, tmp_path, seed=0):
        """训练并保存一个临时 checkpoint,返回路径。"""
        train_ds = _FakeDataset(8, seed_base=seed)
        val_ds = _FakeDataset(4, seed_base=seed + 5000)
        save_path = str(tmp_path / "authenticity_model.pth")
        train_authenticity_classifier(train_ds, val_ds, torch.device("cpu"), save_path)
        return save_path

    def test_sha256_match_passes(self, tmp_path, monkeypatch):
        import hashlib
        save_path = self._make_ckpt(tmp_path)
        with open(save_path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        monkeypatch.setattr(AuthenticityModule, "EXPECTED_CKPT_SHA256", actual)
        module = AuthenticityModule()
        module.load_weights(save_path)  # 不抛即达标
        assert module.feat_mean is not None

    def test_sha256_mismatch_raises(self, tmp_path, monkeypatch):
        save_path = self._make_ckpt(tmp_path)
        monkeypatch.setattr(
            AuthenticityModule, "EXPECTED_CKPT_SHA256", "0" * 64)  # 必不匹配
        module = AuthenticityModule()
        with pytest.raises(RuntimeError, match="integrity check failed"):
            module.load_weights(save_path)

    def test_sha256_none_skips_check(self, tmp_path, monkeypatch):
        """开发态(EXPECTED_CKPT_SHA256=None)跳过校验,向后兼容。"""
        save_path = self._make_ckpt(tmp_path)
        monkeypatch.setattr(AuthenticityModule, "EXPECTED_CKPT_SHA256", None)
        module = AuthenticityModule()
        module.load_weights(save_path)  # 不抛即达标


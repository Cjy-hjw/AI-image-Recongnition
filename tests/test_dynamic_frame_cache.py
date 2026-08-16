"""动态帧预抽取缓存路径 —— 钉住 2026-08-15 的性能修复。

## 背景（勿删）

`OfficialThyroidDataset.__getitem__` 的动态分支原本对**每个样本**全量解压两个体积
（影像 + mask,各 258MB 量级、~4s）。1400 动态样本 × 35 epoch ≈ 24 小时纯解压,
且 `num_workers=12` 时瞬时驻留 ~8.4GB —— 首轮真训的 `invalid block type` 与步耗时
22 倍抖动即由此而来（根因 B:我方读法炸了,非文件损坏）。

修复:`scripts/data/preextract_dynamic.py` 预抽取成 512×512 PNG,`__getitem__` 命中
`cache_img` 就读 PNG、**完全不碰 .nii.gz**。

本文件钉三件事:
1. 缓存命中时确实不碰 `.nii.gz`（用不存在的路径作反证:碰了必崩）;
2. 读出来的像素/掩膜与写进去的**逐位一致**（缓存是等价替换,不是有损近似）;
3. 🔴 缓存缺失必须**抛异常**,不得静默返回零图 —— 静默补零 = 拿全黑图 + 全零 mask
   当真样本训,loss 照常下降、本地永远查不出根因。
"""
import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.official_loader import (  # noqa: E402
    OfficialThyroidDataset,
    _read_cached_pair,
)

SIZE = 512


def _labels():
    return {
        "composition": 0, "echogenicity": 0, "shape": 0, "margin": 0,
        "foci": torch.zeros(4), "malignancy": 1, "patient_id": "p1",
        "lesion_loc": 2, "valid_lesion_loc": 1,
        "extrathy_ext": 0, "valid_extrathy_ext": 1,
        "clin_t_stage": 0, "valid_clin_t_stage": 1,
        "tirads": 2, "valid_tirads": 1,
        "valid_malignancy": 1, "valid_mask": 1, "is_main": 1,
    }


def _make_cache(tmp_path, box=(100, 200, 100, 200)):
    """写一对缓存 PNG,返回 (sample, 原始 img 数组, mask 前景像素数)。"""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (SIZE, SIZE), dtype=np.uint8)
    msk = np.zeros((SIZE, SIZE), np.uint8)
    y0, y1, x0, x1 = box
    msk[y0:y1, x0:x1] = 255           # 预抽取脚本存 0/255,loader 侧统一 `>0`
    pi = str(tmp_path / "a_img.png")
    pm = str(tmp_path / "a_msk.png")
    cv2.imwrite(pi, img)
    cv2.imwrite(pm, msk)
    sample = {
        # 🔴 故意用不存在的路径:一旦代码退回 .nii.gz 分支,必然抛错而不是静默变慢
        "img_path": "/nonexistent/never_read.nii.gz",
        "mask_path": "/nonexistent/never_read_mask.nii.gz",
        "is_dynamic": True, "frame_index": 7, "patient_key": "p1", "label": 1,
        "labels": _labels(), "cache_img": pi, "cache_msk": pm,
    }
    return sample, img, (y1 - y0) * (x1 - x0)


def test_cache_hit_never_touches_niigz(tmp_path):
    """缓存命中 → 不读 .nii.gz。img_path 指向不存在的文件,能跑通即是证明。"""
    sample, _, _ = _make_cache(tmp_path)
    ds = OfficialThyroidDataset([sample], img_size=SIZE)
    img_t, mask_t, _ = ds[0]
    assert img_t.shape == (1, SIZE, SIZE)
    assert mask_t.shape == (1, SIZE, SIZE)


def test_cached_pixels_roundtrip_exactly(tmp_path):
    """PNG 无损:读回的灰度与写入逐位一致(归一化后 ±1/255 容差)。"""
    sample, img, _ = _make_cache(tmp_path)
    ds = OfficialThyroidDataset([sample], img_size=SIZE)
    img_t, _, _ = ds[0]
    assert np.allclose(img_t.numpy()[0] * 255.0, img, atol=1.0)


def test_cached_mask_binarized_to_foreground_count(tmp_path):
    """mask 存 0/255,loader `>0` 二值化后前景像素数必须精确等于写入的框面积。

    这条防的是"存 0/1 → PNG 存活但被某处当成近乎全黑"或"resize 插值糊边"。
    """
    sample, _, n_fg = _make_cache(tmp_path)
    ds = OfficialThyroidDataset([sample], img_size=SIZE)
    _, mask_t, _ = ds[0]
    assert float(mask_t.sum()) == pytest.approx(n_fg, abs=1.0)
    assert set(np.unique(mask_t.numpy()).tolist()) <= {0.0, 1.0}


def test_missing_cache_raises_not_silent_zeros(tmp_path):
    """🔴 缓存缺失必须炸,不得静默返回零图。

    静默补零会让训练拿全黑图 + 全零 mask 当真样本(标签还在),分割监督被污染而
    loss 照常下降 —— 正是本项目反复吃亏的那类"合法但全错"。
    """
    sample, _, _ = _make_cache(tmp_path)
    with pytest.raises(RuntimeError, match="预抽取缓存缺失"):
        _read_cached_pair(str(tmp_path / "missing.png"), sample["cache_msk"])


def test_no_cache_key_falls_back_to_niigz_path(tmp_path):
    """无 `cache_img` 键 → 走原 .nii.gz 路径(向后兼容,老 samples 零回归)。

    用不存在的路径反证:确实进了解压分支才会抛这个错。
    """
    sample, _, _ = _make_cache(tmp_path)
    sample.pop("cache_img")
    ds = OfficialThyroidDataset([sample], img_size=SIZE)
    with pytest.raises(Exception):
        _ = ds[0]


def test_frame_key_distinguishes_mask_when_image_shared(tmp_path):
    """同一影像下不同病灶必须得到不同缓存键。

    动态患者一个 .nii.gz 可能对应多个病灶 mask;只哈希影像路径会撞车,让两个病灶
    读到同一张 mask —— 不报错,标签与像素静默错配。
    """
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "data"))
    from preextract_dynamic import frame_key

    a = frame_key("/d/p.nii.gz", "/d/M_1_mask.nii.gz", 7)
    b = frame_key("/d/p.nii.gz", "/d/病灶1_1_mask.nii.gz", 7)
    c = frame_key("/d/p.nii.gz", "/d/M_1_mask.nii.gz", 8)
    assert a != b, "同影像不同 mask 必须不同键(否则两个病灶读到同一张 mask)"
    assert a != c, "同 mask 不同帧必须不同键"
    assert a == frame_key("/d/p.nii.gz", "/d/M_1_mask.nii.gz", 7), "同输入必须稳定"


def test_partial_pkl_is_rejected(tmp_path):
    """🔴 `--limit` 产出的残件 pkl 必须被硬拒,不能只告警。

    残件里未处理的动态样本没有缓存 → 静默回落到全量解压原路径:既没提速、又只验了
    一小部分,而训练照跑、日志照出。属"合法但全错",堵在入口才安全。
    """
    import pickle

    from src.dataset.official_loader import build_official_datasets

    p = tmp_path / "samples.pkl"
    with open(p, "wb") as f:
        pickle.dump({"_meta": {"img_size": SIZE, "partial": True}, "samples": [1]}, f)
    with pytest.raises(RuntimeError, match="partial"):
        build_official_datasets("/r", "/x.xlsx", samples_pkl=str(p))


@pytest.mark.parametrize("meta_extra", [
    {"partial": False},   # 显式标完整
    {},                   # 键缺失(预抽取脚本加 partial 之前产出的旧 pkl)
])
def test_complete_pkl_is_accepted_by_the_real_gate(tmp_path, meta_extra):
    """反向边界:完整 pkl 不得被误拒 —— 且必须**直接打到 build_official_datasets 那道闸门**。

    🔴 本测试的前一版只断言 `_load_samples_pkl` 返回 `meta["partial"] is False`,
    那对闸门的实现写法**零区分力**:`meta.get("partial")` 与 `"partial" in meta`
    两种写法都能让它通过,而后者会把所有带 partial 键的完整 pkl 全部误拒。
    所以这里必须真的调 `build_official_datasets` 并喂**非空** samples
    (空 samples 会先撞"未构造出任何样本"那道防护,闸门根本轮不到执行)。
    """
    import pickle

    from src.dataset.official_loader import build_official_datasets

    s = {
        "img_path": "/x.dcm", "mask_path": None, "is_dynamic": False,
        "frame_index": None, "patient_key": "p1", "label": 0, "labels": _labels(),
    }
    p = tmp_path / "samples.pkl"
    with open(p, "wb") as f:
        pickle.dump({"_meta": {"img_size": SIZE, **meta_extra}, "samples": [s, dict(s)]}, f)

    train_ds, val_ds = build_official_datasets(
        "/r", "/x.xlsx", img_size=SIZE, samples_pkl=str(p))
    assert len(train_ds) + len(val_ds) == 2, "完整 pkl 应正常放行并建出数据集"

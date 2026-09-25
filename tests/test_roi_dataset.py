"""roi_dataset.py + build_official_samples(roi_meta=True) 契约测试(2026-09-25)。

两阶段 ROI 范式数据面(方案一 P1)的单测,fixture 照 test_official_loader.py
的双行表头 mock 形态(官方 xlsx 表头在第 2 行,header=1)。

覆盖:
1. roi_meta 默认关 → build 输出与现状逐位一致(零回归);
2. 开时 roi_lesions 数 = 同图 GT mask 数(病灶行数,非样本数);
3. bbox 与手造 mask 非零区精确一致(静态 + 动态多帧并集口径);
4. collate 后 lesion_id 仍是 list[str](不变 [B,32] 字符矩阵);
5. spacing None 传递(无来源不编造);
6. 动态样本每病灶逐帧展开;
7. ROI 裁剪坐标可逆(crop → 还原逐像素相等,P1 验收门);
8. 旧 pkl 式样本(无 roi_lesions 键)跳过且计数。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pydicom = pytest.importorskip("pydicom")
nib = pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

import sklearn.metrics  # noqa: F401  (import 副作用:抢先加载 OpenMP 运行时,
# 防 pytest 收集深度下 Windows 栈溢出,test_cliff_loss.py 同款)

import torch

from src.dataset import official_loader as O
from src.dataset.roi_dataset import RoiDataset, roi_collate, expand_roi_samples
from src.inference.nifti_io import save_binary_mask
from test_official_loader import _make_dcm, _save_image_nifti_3d


# ── fixture:双病灶静态 + 双病灶动态(与 L2 fixture 同型,独立构造)──────────
@pytest.fixture
def roi_root(tmp_path):
    """QA 静态双病灶(M@20:40x20:40 + 病灶1@30:50x30:50)+ QB 动态双病灶。"""
    root = tmp_path / "annotation"
    H = W = 64

    qa = root / "QA" / "SA"
    _make_dcm(str(qa / "img.dcm"), np.full((H, W), 120, np.uint8))
    m1 = np.zeros((H, W), np.uint8); m1[20:40, 20:40] = 1
    m2 = np.zeros((H, W), np.uint8); m2[30:50, 30:50] = 1
    save_binary_mask(m1, str(qa / "M_1_mask.nii.gz"))
    save_binary_mask(m2, str(qa / "病灶1_1_mask.nii.gz"))

    qb = root / "QB" / "QB"
    vol = np.stack([np.full((H, W), 40 + 10 * i, np.uint8) for i in range(5)])
    _save_image_nifti_3d(str(qb / "QB.nii.gz"), vol)
    # M:帧1 @10:30x10:30 + 帧3 @25:45x25:45(跨帧幅面并集 → 10:45)
    dm = np.zeros((5, H, W), np.uint8)
    dm[1, 10:30, 10:30] = 1
    dm[3, 25:45, 25:45] = 1
    save_binary_mask(dm, str(qb / "M_1_mask.nii.gz"))
    # 病灶1:仅帧2 @40:55x5:20
    ds = np.zeros((5, H, W), np.uint8); ds[2, 40:55, 5:20] = 1
    save_binary_mask(ds, str(qb / "病灶1_1_mask.nii.gz"))

    rows = [
        ["QA", "SA", "1", "M", "2左中叶", "2甲状腺包膜", "1有", "恶", "", "V类"],
        ["QA", "SA", "1", "病灶1", "5右中叶", "1无", "0无", "良", "", "IV类"],
        ["QB", "", "2", "M", "7峡部", "1无", "1有", "恶", "", ""],
        ["QB", "", "2", "病灶1", "7峡部", "1无", "1有", "良", "", ""],
    ]
    df = pd.DataFrame(rows, columns=[
        "PatientId", "SeriesUid", "图像性质", "RoiName", "主病灶位置",
        "甲状腺外侵犯", "临床T分期", "ROI良恶性", "ANNOTATION->SliceIndex", "ROI-TIRADS",
    ])
    xlsx = tmp_path / "labels_roi.xlsx"
    df.to_excel(xlsx, sheet_name="ROI级别", index=False, startrow=1)
    return str(root), str(xlsx)


# ── 1. 零回归:默认关逐位一致 ───────────────────────────────────────────────
def test_roi_meta_off_bitwise_identical(roi_root):
    """roi_meta=False(默认)输出与旧签名逐位一致:全部既有键值相等、无新键。"""
    root, xlsx = roi_root
    plain = O.build_official_samples(root, xlsx)
    again = O.build_official_samples(root, xlsx, roi_meta=False)
    assert len(plain) == len(again) > 0
    for a, b in zip(plain, again):
        assert set(a.keys()) == set(b.keys())
        assert set(a.keys()) == {"img_path", "mask_path", "is_dynamic",
                                 "patient_key", "label", "labels", "frame_index"}
        for k in a:
            va, vb = a[k], b[k]
            if isinstance(va, torch.Tensor):
                assert torch.equal(va, vb), k
            elif isinstance(va, dict):
                for kk in va:
                    xa, xb = va[kk], vb[kk]
                    if isinstance(xa, torch.Tensor):
                        assert torch.equal(xa, xb), kk
                    else:
                        assert xa == xb, kk
            else:
                assert va == vb, k


def test_roi_meta_off_with_union_unchanged(roi_root):
    """roi_meta 与 union 组合:仅 union 开时,键集与 union-only 逐位一致。"""
    root, xlsx = roi_root
    uni = O.build_official_samples(root, xlsx, union_mask=True)
    uni_roi = O.build_official_samples(root, xlsx, union_mask=True, roi_meta=False)
    assert len(uni) == len(uni_roi)
    for a, b in zip(uni, uni_roi):
        assert a.keys() == b.keys()
        assert a["union_mask_paths"] == b["union_mask_paths"]


# ── 2. roi_lesions 数 = GT mask 数 ──────────────────────────────────────────
def test_roi_lesions_count_equals_gt_masks(roi_root):
    """QA 双病灶 → 每样本 roi_lesions 恰 2 条;各条 lesion_id 互异。"""
    root, xlsx = roi_root
    s = O.build_official_samples(root, xlsx, roi_meta=True)
    qa = [x for x in s if x["patient_key"] == "QA"]
    assert len(qa) == 2                      # M + 病灶1 各一样本(静态)
    for x in qa:
        assert len(x["roi_lesions"]) == 2    # 每样本同图 2 病灶元数据
    ids = {d["lesion_id"] for d in qa[0]["roi_lesions"]}
    assert len(ids) == 2


def test_roi_lesions_shared_across_frames(roi_root):
    """动态多帧样本共享同一 roi_lesions(病灶级语义不随帧变)。"""
    root, xlsx = roi_root
    s = O.build_official_samples(root, xlsx, roi_meta=True)
    qb_m = [x for x in s if x["patient_key"] == "QB"
            and os.path.basename(x["mask_path"]) == "M_1_mask.nii.gz"]
    assert len(qb_m) == 2                    # M 的非零帧 1、3 → 2 样本
    assert qb_m[0]["roi_lesions"] is qb_m[1]["roi_lesions"]


# ── 3. bbox 与手造 mask 非零区一致 ──────────────────────────────────────────
def test_static_bbox_matches_mask(roi_root):
    """QA 静态 bbox = mask 非零区外接框(半开区间,切片精确复现)。"""
    root, xlsx = roi_root
    s = O.build_official_samples(root, xlsx, roi_meta=True)
    les = {d["lesion_id"].split("::")[-1] or "M": d
           for d in s[0]["roi_lesions"]}
    by_mask = {os.path.basename(d["lesion_id"]): d
               for d in s[0]["roi_lesions"]}
    m = by_mask["M_1_mask.nii.gz"]
    x0, y0, x1, y1 = m["bbox_xyxy"]
    assert (x0, y0, x1, y1) == (20, 20, 40, 40)
    sec = by_mask["病灶1_1_mask.nii.gz"]
    assert sec["bbox_xyxy"] == (30, 30, 50, 50)
    assert m["orig_w"] == 64 and m["orig_h"] == 64
    assert m["frame_indices"] is None        # 静态无帧维


def test_dynamic_bbox_union_across_frames(roi_root):
    """动态 bbox = 全部非零帧外接框并集(M 帧1@10:30 + 帧3@25:45 → 10:45)。"""
    root, xlsx = roi_root
    s = O.build_official_samples(root, xlsx, roi_meta=True)
    qb = next(x for x in s if x["patient_key"] == "QB")
    by_mask = {os.path.basename(d["lesion_id"]): d
               for d in qb["roi_lesions"]}
    m = by_mask["M_1_mask.nii.gz"]
    assert m["bbox_xyxy"] == (10, 10, 45, 45)          # 并集幅面
    assert sorted(m["frame_indices"]) == [1, 3]        # 与 _nonzero_mask_frames 同源
    sec = by_mask["病灶1_1_mask.nii.gz"]
    assert sec["bbox_xyxy"] == (5, 40, 20, 55)
    assert sec["frame_indices"] == [2]


# ── 4/5. collate 契约 + spacing ─────────────────────────────────────────────
def test_collate_keeps_lesion_id_list_of_str(roi_root):
    """collate 后 lesion_id 是 list[str];default_collate 的 [B,32] 字符矩阵被
    显式拒绝(长度不等时其报错、等长时静默变矩阵)。"""
    root, xlsx = roi_root
    s = O.build_official_samples(root, xlsx, roi_meta=True)
    rois = expand_roi_samples(s)
    assert len(rois) > 1
    ds = RoiDataset(rois, roi_size=32)
    batch = [ds[i] for i in range(min(3, len(ds)))]
    imgs, metas = roi_collate(batch)
    lid = metas["lesion_id"]
    assert isinstance(lid, list)
    assert all(isinstance(x, str) for x in lid)
    assert imgs.shape[0] == len(batch)
    # 变异杀测:default_collate 的字符矩阵行为对同长 hex 串"静默变矩阵",
    # 这里断言 list 语义(矩阵会是 tensor 且 ndim==2)。
    assert not torch.is_tensor(lid)


def test_collate_scalar_keys_stack(roi_root):
    """label/valid 标量键堆成 [B] int64;frame_index 全 int 同堆。"""
    root, xlsx = roi_root
    s = O.build_official_samples(root, xlsx, roi_meta=True)
    rois = [r for r in expand_roi_samples(s) if r["is_dynamic"]]
    ds = RoiDataset(rois, roi_size=32)
    imgs, metas = roi_collate([ds[i] for i in range(len(ds))])
    assert metas["malignancy"].shape == (len(ds),)
    assert metas["valid_tirads"].dtype == torch.int64
    assert torch.is_tensor(metas["frame_index"])
    assert metas["lesion_id"] == [r["lesion_id"] for r in rois]


def test_spacing_none_passthrough(roi_root):
    """spacing 无官方来源:build 与 collate 全链恒 None(不编造数值)。"""
    root, xlsx = roi_root
    s = O.build_official_samples(root, xlsx, roi_meta=True)
    for x in s:
        for d in x["roi_lesions"]:
            assert d["spacing_mm"] is None
    rois = expand_roi_samples(s)
    ds = RoiDataset(rois, roi_size=32)
    for i in range(len(ds)):
        assert ds[i][1]["spacing_mm"] is None
    _, metas = roi_collate([ds[i] for i in range(len(ds))])
    assert metas["spacing_mm"] == [None] * len(ds)


# ── 6. 动态每病灶逐帧展开 ───────────────────────────────────────────────────
def test_dynamic_expand_per_frame(roi_root):
    """QB 动态:整图帧样本(M{1,3} + 病灶1{2})逐帧取**交集**展开 → 3 条 ROI 样本。

    整图样本 3 条(M@1、M@3、病灶1@2);ROI 展开按"该样本帧落在此病灶
    frame_indices 内"取交集并按 (lesion_id, frame) 去重——M@1 样本只产 M@1
    一条(病灶1 在帧1 无 GT),故总数 = 3,与整图帧样本一一对应。
    """
    root, xlsx = roi_root
    s = O.build_official_samples(root, xlsx, roi_meta=True)
    rois = expand_roi_samples(s)
    qb = [r for r in rois if r["patient_key"] == "QB"]
    assert len(qb) == 3
    by_lid = {}
    for r in qb:
        by_lid.setdefault(r["lesion_id"], []).append(r["frame_index"])
    assert len(by_lid) == 2
    for frames in by_lid.values():
        assert all(isinstance(f, int) for f in frames)
    assert sum(len(v) for v in by_lid.values()) == 3


# ── 7. 裁剪坐标可逆(P1 验收门)──────────────────────────────────────────────
def test_crop_roundtrip_pixelwise(roi_root):
    """裁 → pad → resize 前两步(原分辨率域)可逆:还原 ROI 与原帧切片逐像素相等。"""
    root, xlsx = roi_root
    s = O.build_official_samples(root, xlsx, roi_meta=True)
    rois = expand_roi_samples(s)
    ds = RoiDataset(rois, roi_size=32)
    frame = np.full((64, 64), 77, np.uint8)
    frame[20:40, 20:40] = 200
    r = next(x for x in rois if x["patient_key"] == "QA"
             and os.path.basename(x["lesion_id"]) == "M_1_mask.nii.gz")
    x0, y0, x1, y1 = r["bbox_xyxy"]
    expect_crop = frame[y0:y1, x0:x1]
    roi, (pad_h, pad_w) = ds.crop_roi(frame, r["bbox_xyxy"])
    assert pad_h == pad_w == 0                     # 该 bbox 已是正方形
    restored = roi[:roi.shape[0] - pad_h, :roi.shape[1] - pad_w]
    assert np.array_equal(restored, expect_crop)
    # 非方形 bbox(QB 病灶1:15 宽 x20 高 → x1-x0=15, y1-y0=15? 复核:5:20x40:55
    # → w=15,h=15 已方形;改用 QA M 的 20x20 已验,此处用自定义非方形手验)
    r2 = {"bbox_xyxy": (5, 40, 20, 60)}       # w=15, h=20 → pad w 侧 5
    roi2, (ph, pw) = ds.crop_roi(frame, r2["bbox_xyxy"])
    x0, y0, x1, y1 = r2["bbox_xyxy"]
    assert (ph, pw) == (0, 5) and roi2.shape == (20, 20)
    assert np.array_equal(roi2[:20 - ph, :20 - pw], frame[y0:y1, x0:x1])


def test_roi_getitem_shape_and_labels(roi_root):
    """__getitem__ 输出契约:img [1,S,S]∈[0,1]、global_loc 归一化、label 子集在。"""
    root, xlsx = roi_root
    s = O.build_official_samples(root, xlsx, roi_meta=True)
    rois = expand_roi_samples(s)
    ds = RoiDataset(rois, roi_size=32)
    img, meta = ds[0]
    assert tuple(img.shape) == (1, 32, 32)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert 0.0 <= meta["global_loc"][0] <= 1.0
    assert 0.0 <= meta["global_loc"][1] <= 1.0
    for k in ("lesion_loc", "valid_lesion_loc", "malignancy", "is_main"):
        assert k in meta
    assert isinstance(meta["pad_hw"], tuple)


# ── 8. 旧 pkl 式样本(无键)跳过 ────────────────────────────────────────────
def test_expand_skips_samples_without_key(roi_root):
    """无 roi_lesions 键的整图样本(旧 pkl)被跳过,不编造 bbox。"""
    root, xlsx = roi_root
    plain = O.build_official_samples(root, xlsx)   # 默认 build:无键
    rois = expand_roi_samples(plain)
    assert rois == []
    ds = RoiDataset([], roi_size=32)
    assert len(ds) == 0 and ds.n_skipped_no_meta == 0


def test_dataset_counts_skipped_no_bbox():
    """手构样本缺 bbox → 构造期计数并在 len 前打点(零反馈防御)。"""
    bad = [{"bbox_xyxy": None, "lesion_id": "x", "img_path": "",
            "is_dynamic": False, "frame_index": None, "patient_key": "p",
            "orig_w": 64, "orig_h": 64, "global_loc": (0.5, 0.5)}]
    ds = RoiDataset(bad, roi_size=32)
    assert len(ds) == 0
    assert ds.n_skipped_no_meta == 1

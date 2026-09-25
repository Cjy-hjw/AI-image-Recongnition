# -*- coding: utf-8 -*-
"""preextract_dynamic.py 并行化(2026-09-11)的针对性测试。

覆盖三层:
1. **回填正确性**:_process_one_file 按 s_idx 回填,失败/越界样本 cache 置 None
   (落盘后被剔除),成功样本拿到真实 PNG 路径;
2. **进程池真跑**:workers=2 的 extract() 端到端产出与串行 workers=1 **逐字节
   等价**的 PNG 与等价 samples.pkl;
3. **分辨率抽检护栏在并行路径仍然硬拦**:旧缓存 + 新 --img-size → SystemExit
   (probe_hw 从 worker 报回主进程的链路)。

🔴 测试策略(与 mock 的不兼容教训):`ProcessPoolExecutor` 的子进程重新 import
模块,**monkeypatch 只在主进程生效** —— 第一版用 monkeypatch mock
`load_nifti_frames`/`_read_nifti_binmasks`,子进程里 patch 不存在、真函数去读
不存在的 .nii.gz → 并行臂全失败。故这里**不 mock 任何 IO**:用 nibabel 造
真实的 mini .nii.gz(6 帧 32×32,~几 KB),主/子进程走完全相同的真代码路径。
build_official_samples 仍 mock(它只消费 xlsx,与并行边界无关,主进程单点调用)。
"""
import os
import pickle
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "data"))

import preextract_dynamic as pex  # noqa: E402


SIZE = 64          # 目标 PNG 尺寸(小,快)
FRAMES = 6
RAW = 32           # 原始 nii.gz 帧尺寸


def _mk_vol(fg_frames=(1, 3), size=RAW, dtype=np.uint8):
    """造一个 (size,size,FRAMES) 体积:指定帧放 8x8 前景块(影像随机、mask 前景)。"""
    rng = np.random.default_rng(0)
    if fg_frames is None:                      # 影像:纯随机灰度
        return rng.integers(0, 255, (size, size, FRAMES), dtype=dtype)
    vol = np.zeros((size, size, FRAMES), dtype=np.uint8)
    for f in fg_frames:
        vol[8:16, 8:16, f] = 1                 # mask 前景
    return vol


@pytest.fixture()
def real_data(tmp_path, monkeypatch):
    """造 3 个动态患者(pa/pb 正常 + bad 用真损坏文件)+ 静态样本行。

    产物是**真实 .nii.gz**,worker 子进程用真 load_nifti_frames/_read_nifti_binmasks
    读它们 —— 并行/串行走完全一致的真代码路径。
    """
    root = tmp_path / "data"
    root.mkdir()

    for pid in ("pa", "pb"):
        img = nib.Nifti1Image(_mk_vol(fg_frames=None), np.eye(4))
        msk = nib.Nifti1Image(_mk_vol(fg_frames=(1, 3)), np.eye(4))
        nib.save(img, str(root / f"{pid}.nii.gz"))
        nib.save(msk, str(root / f"{pid}_M.nii.gz"))
    # bad:写入后截断一半 → 真解码失败(IMG 失败路径)
    p = root / "bad.nii.gz"
    with open(p, "wb") as f:
        f.write(open(root / "pa.nii.gz", "rb").read()[:64])

    def fake_build(data_root, xlsx, sheet_name="ROI级别", union_mask=False,
                   roi_meta=False):
        samples = []
        for pi, pid in enumerate(["pa", "pb"]):
            for fidx in (1, 3):                  # 每患者 2 帧 = 4 个动态样本
                samples.append({
                    "img_path": str(root / f"{pid}.nii.gz"),
                    "mask_path": str(root / f"{pid}_M.nii.gz"),
                    "is_dynamic": True, "frame_index": fidx,
                    "patient_key": pid, "label": pi,
                })
            samples.append({                     # 静态样本:预抽取不碰
                "img_path": str(root / f"{pid}.dcm"), "mask_path": None,
                "is_dynamic": False, "frame_index": None,
                "patient_key": pid, "label": pi,
            })
        samples.append({                         # bad:解压失败 → 剔除
            "img_path": str(p), "mask_path": str(root / "bad_M.nii.gz"),
            "is_dynamic": True, "frame_index": 0,
            "patient_key": "bad", "label": 1,
        })
        return samples

    monkeypatch.setattr(pex, "build_official_samples", fake_build)
    return root


def _load_pkl(cache_dir):
    with open(os.path.join(cache_dir, "samples.pkl"), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- tests ----
def test_worker_unit_fills_and_failures(real_data, tmp_path):
    """_process_one_file 单元(串行直调):成功回填路径、越界帧回填 None 且记 failure。

    🔴 用真 .nii.gz 而非 mock:worker 也要在无 monkeypatch 的子进程里跑同一函数,
    单元层先用真文件直调,与并行行为天然一致。
    """
    fi = str(tmp_path / "a_img.png")
    fm = str(tmp_path / "a_msk.png")
    job = {
        "img_path": str(real_data / "pa.nii.gz"), "img_size": SIZE,
        "entries": [
            {"k": "k0", "fi": fi, "fm": fm,
             "mask_path": str(real_data / "pa_M.nii.gz"),
             "frame_index": 1, "s_idx": 7},
            # 越界帧:回填 None 且记 failure
            {"k": "k1", "fi": str(tmp_path / "b_img.png"), "fm": str(tmp_path / "b_msk.png"),
             "mask_path": str(real_data / "pa_M.nii.gz"),
             "frame_index": 99, "s_idx": 8},
        ],
    }
    res = pex._process_one_file(job)
    assert res["fills"][7]["cache_img"] == fi
    assert os.path.getsize(fi) > 0
    assert res["fills"][8]["cache_img"] is None            # 越界 → 剔除语义
    assert any("FRM" in f for f in res["failures"])
    assert res["done_frames"] == 1


def test_parallel_equals_serial_end_to_end(real_data, tmp_path):
    """workers=2 与 workers=1:PNG 逐字节一致、pkl 样本逐字段等价。"""
    cache_s = str(tmp_path / "cache_serial")
    cache_p = str(tmp_path / "cache_parallel")
    common = dict(data_root=str(real_data), xlsx="fake.xlsx",
                  sheet="ROI级别", img_size=SIZE, limit=0)
    assert pex.extract(cache_dir=cache_s, workers=1, **common) == 0
    assert pex.extract(cache_dir=cache_p, workers=2, **common) == 0

    ps, pp = _load_pkl(cache_s), _load_pkl(cache_p)
    assert len(ps["samples"]) == len(pp["samples"]) == 6  # 4 动态成功 + 2 静态;bad 剔除
    for a, b in zip(ps["samples"], pp["samples"]):
        if not a["is_dynamic"]:
            continue                                    # 静态样本无缓存键,不比
        # 路径分属两个 cache-dir,只比相对部分(frames/<key>_*.png)与回填语义
        assert a["cache_img"].replace(cache_s, "") == b["cache_img"].replace(cache_p, "")
        assert a["cache_msk"].replace(cache_s, "") == b["cache_msk"].replace(cache_p, "")
        with open(a["cache_img"], "rb") as f1, open(b["cache_img"], "rb") as f2:
            assert f1.read() == f2.read(), "并行与串行产出的 PNG 必须逐字节一致"


def test_idempotent_rerun_parallel(real_data, tmp_path):
    """幂等:同 cache-dir 二次跑(workers=2)不重写已有 PNG、样本数不变。"""
    cache = str(tmp_path / "cache")
    common = dict(data_root=str(real_data), xlsx="fake.xlsx",
                  sheet="ROI级别", img_size=SIZE, limit=0)
    pex.extract(cache_dir=cache, workers=2, **common)
    p1 = _load_pkl(cache)
    mtime1 = {a["cache_img"]: os.path.getmtime(a["cache_img"])
              for a in p1["samples"] if a["is_dynamic"]}
    pex.extract(cache_dir=cache, workers=2, **common)
    p2 = _load_pkl(cache)
    for path, m in mtime1.items():
        assert os.path.getmtime(path) == m, "幂等重跑不得重写已存在的 PNG"
    assert len(p1["samples"]) == len(p2["samples"])


@pytest.mark.parametrize("workers", [1, 2])
def test_probe_guard_fires_both_paths(real_data, tmp_path, workers):
    """分辨率抽检:同 cache-dir 换 --img-size → SystemExit(串行/并行两路都拦)。

    这是 A 路审查闭环的关键护栏,并行化不得让它失效:probe_hw 由 worker 报回
    主进程统一硬拦(串行路径 = 主进程自己算,同一断言)。
    """
    cache = str(tmp_path / "cache")
    kw = dict(data_root=str(real_data), xlsx="f.xlsx", cache_dir=cache,
              sheet="ROI级别", img_size=SIZE, limit=0, workers=workers)
    pex.extract(**kw)                                # 先以 SIZE 造全量缓存
    with pytest.raises(SystemExit, match="缓存 PNG"):
        pex.extract(**{**kw, "img_size": SIZE + 8})  # 复用 + 换分辨率 → 硬拦


@pytest.mark.parametrize("workers", [1, 2])
def test_union_export_matches_loader_union(real_data, tmp_path, workers):
    """L2(路B P2-3,2026-09-22):union 导出端 e2e——缓存 PNG 与 loader 原路径
    union 读取逐位一致(预注册不变式"cache miss 与无 cache 两条路径逐位一致")。

    fake_build 的 union 版:同患者加第二个病灶 mask(帧 1 与 M 重叠、帧 3 独立),
    每动态样本带 union_mask_paths → extract(union_mask=True) 走 worker OR 合成;
    对照 = OfficialThyroidDataset._dynamic_union_mask_frame(无缓存原路径)。
    """
    import cv2
    import numpy as _np

    root = real_data
    # 第二病灶 mask:帧 1 中央方块(与 M 的 fg_frames=(1,3) 部分同帧),帧 3 同位置
    sec = _mk_vol(fg_frames=(1, 3))
    sec_path = str(root / "pa_S.nii.gz")
    nib.save(nib.Nifti1Image(sec, np.eye(4)), sec_path)
    # M 的体素:换一个互补位置(左上角),保证 并集 != 任一单 mask(fixture 区分力,
    # 路A P2-1 教训:并集恰等于某单 mask 时测试对"union 分支整体失效"零区分力)
    m_alt = _mk_vol(fg_frames=(1, 3))
    # _mk_vol 的前景位置不知,直接手造:取全 0 体 + 两个不同角块
    vol_m = np.zeros(_mk_vol(fg_frames=None).shape, np.uint8)
    vol_s = np.zeros_like(vol_m)
    # NIfTI 存前需 transpose(2,1,0) —— 与 _save_image_nifti_3d 同约定
    vm = np.zeros((4, SIZE, SIZE), np.uint8); vm[1, :16, :16] = 1; vm[3, :16, :16] = 1
    vs = np.zeros((4, SIZE, SIZE), np.uint8); vs[1, 32:64, 32:64] = 1; vs[3, 32:64, 32:64] = 1
    nib.save(nib.Nifti1Image(np.ascontiguousarray(
        np.transpose(vm, (2, 1, 0))), np.eye(4)), str(root / "pa_M.nii.gz"))
    nib.save(nib.Nifti1Image(np.ascontiguousarray(
        np.transpose(vs, (2, 1, 0))), np.eye(4)), sec_path)

    def fake_build_u(data_root, xlsx, sheet_name="ROI级别", union_mask=False,
                     roi_meta=False):
        samples = []
        for fidx in (1, 3):
            samples.append({
                "img_path": str(root / "pa.nii.gz"),
                "mask_path": str(root / "pa_M.nii.gz"),
                "is_dynamic": True, "frame_index": fidx,
                "patient_key": "pa", "label": 0,
                "union_mask_paths": sorted([str(root / "pa_M.nii.gz"), sec_path]),
            })
        return samples

    orig_build = pex.build_official_samples
    pex.build_official_samples = fake_build_u
    try:
        cache = str(tmp_path / "ucache")
        assert pex.extract(data_root=str(root), xlsx="f.xlsx", cache_dir=cache,
                           sheet="ROI级别", img_size=SIZE, limit=0,
                           workers=workers, union_mask=True) == 0
    finally:
        pex.build_official_samples = orig_build

    payload = _load_pkl(cache)
    assert payload["_meta"]["union_mask"] is True
    dyn = [s for s in payload["samples"] if s["is_dynamic"]]
    assert len(dyn) == 2
    for s in dyn:
        assert s["cache_msk"] and os.path.exists(s["cache_msk"])

    # 对照:loader 原路径(无缓存)对同 union_paths 同帧的读取
    from src.dataset import official_loader as OL
    ds = OL.OfficialThyroidDataset(dyn, img_size=SIZE)
    for i, s in enumerate(dyn):
        png = _np.fromfile(s["cache_msk"], dtype=_np.uint8)
        png = cv2.imdecode(png, cv2.IMREAD_GRAYSCALE)
        expect = ds._dynamic_union_mask_frame(
            s["union_mask_paths"], s["frame_index"])
        assert _np.array_equal(png > 0, expect > 0), \
            f"union 导出 PNG 与 loader 原路径 frame={s['frame_index']} 不一致"
        # 区分力:并集必须严格大于任一单 mask(防"union 分支失效仍绿")
        m_only = ds._dynamic_mask_frame(s["mask_path"], s["frame_index"])
        assert (png > 0).sum() > (m_only > 0).sum(), "并集应大于单病灶 mask"

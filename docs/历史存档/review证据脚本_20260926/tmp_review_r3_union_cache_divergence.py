# -*- coding: utf-8 -*-
"""R3 证据 1:union 直建 vs pkl 缓存命中 —— 两者**不逐位等价**。

复刻生产两条路径的 mask 计算链(不 import 训练脚本,避免副作用):
  A. "pkl 路径"(装载分支):直接 _read_cached_pair(cache_img, cache_msk)
  B. "直建路径"(原分支):逐 mask OR 合成(union_mask_frame_from_masks 同式的数学)

帧号选在**动态抽样点**(fidx=0 或 n-1,见 uniform_sample_indices 必含首尾),
这正是生产会走到的情形。
"""
import os
import sys
import tempfile

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.official_loader import _read_cached_pair  # noqa: E402

N_FRAMES = 8
SIZE = 16
H, W = 16, 16
FX, FY = 0, 0          # 次病灶只在 (0,0) 置 1
PX, PY = 8, 8          # 主病灶只在 (8,8) 置 1


def _mk_mask(fx, fy, tmp, name):
    """单个病灶 mask:全部帧仅在 (fy,fx) 处为 1。"""
    arr = np.zeros((W, H, N_FRAMES), dtype=np.uint8)
    arr[fx, fy, :] = 1
    p = os.path.join(tmp, name)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), p)
    return p


def main():
    tmp = tempfile.mkdtemp(prefix="tmp_review_r3_")
    m_main = _mk_mask(PX, PY, tmp, "M_1_mask.nii.gz")
    m_second = _mk_mask(FX, FY, tmp, "病灶1_1_mask.nii.gz")

    # union 路(直建):binf[0] | binf[1] —— mask 帧恒 = 帧 0(binf 只含 1 帧)
    from src.dataset.official_loader import _read_nifti_binmasks
    b_main, _ = _read_nifti_binmasks(m_main)
    b_second, _ = _read_nifti_binmasks(m_second)
    union_true = ((b_main[0] > 0) | (b_second[0] > 0)).astype(np.uint8)

    # pkl 缓存路:预抽取导出端对动态样本写 cache_msk 时取的是 binf[mp](mp=fidx)
    # 只勾了"本 mask"(不含 union OR);这里用帧 fidx=0 与 fidx=N_FRAMES-1 各造一份
    cache_dir = os.path.join(tmp, "frames")
    os.makedirs(cache_dir, exist_ok=True)
    from scripts.data.preextract_dynamic import _write_png, _resize_mask
    import cv2

    results = {}
    for fidx in (0, N_FRAMES - 1):
        # 导出端(union 分支)应产出并集
        exp = os.path.join(cache_dir, f"k{fidx}_msk.png")
        _write_png(exp, _resize_mask(union_true, H) * 255)
        img_p = os.path.join(cache_dir, f"k{fidx}_img.png")
        _write_png(img_p, np.zeros((H, W), np.uint8))
        _, m_cached = _read_cached_pair(img_p, exp)
        ok = np.array_equal((m_cached > 0).astype(np.uint8), union_true)
        results[fidx] = {"cached_is_union": bool(ok),
                         "cached_foreground": int((m_cached > 0).sum()),
                         "union_foreground": int(union_true.sum())}

    # fmt: off
    print("[A] union 导出端正确时,缓存路 == 直建路:",
          all(v["cached_is_union"] for v in results.values()))
    # fmt: on
    for fidx, v in results.items():
        print(f"    fidx={fidx}: {v}")

    from src.dataset import official_loader as OL

    # ---- [B] 跨帧错配:union 的"帧定位回退 frame_index"把别的帧的标注 OR 进本帧 ----
    # 主病灶 M 只在帧 1 有标注 → 整图样本 frame_index=1;
    # 次病灶 病灶1 在帧 1 与帧 3 有标注 → 它另产 frame_index=3 的整图样本。
    # 但 union_mask_frame_paths 对**每个样本**都含 病灶1,于是 M 的样本在
    # 帧 3、病灶1 的样本在帧 1 时,都取"本 mask 不含该帧 → pos=frame_index"
    # 的越界帧 = 别的帧的标注。
    print("\n[B-1] 生产消费端 _dynamic_union_mask_frame 的帧定位:")
    import nibabel as _nib
    from src.dataset.official_loader import (
        _read_nifti_binmasks, OfficialThyroidDataset)
    ds_b = OfficialThyroidDataset.__new__(OfficialThyroidDataset)
    ds_b.img_size = H
    paths = sorted([m_main, m_second])
    for fidx in (0, 1, 3, 7):
        u = ds_b._dynamic_union_mask_frame(paths, fidx)
        s = ds_b._dynamic_mask_frame(m_main, fidx)
        print(f"    帧 {fidx}: union 前景={int((u > 0).sum())} "
              f"(应 = 该帧真实并集) | 单主灶(对照)={int((s > 0).sum())}")
    print("    → 帧 0/7 主灶无标注、本应全零,union 却给出 2 个前景像素"
          "(值取自帧 1/3 之外的越界下标,随帧长而变)")

    # ---- [B-2] 同缺陷在真实 build 样本上的取值(直建链) ----
    print("\n[B-2] 真实 build 样本逐帧 union 前景(直建链):")
    import pandas as _pd, tempfile as _tf
    from pathlib import Path as _P
    _tmp2 = _P(_tf.mkdtemp(prefix="tmp_review_r3b2_"))
    _pd_dir = _tmp2 / "PA" / "PA"
    os.makedirs(_pd_dir, exist_ok=True)
    _vol = np.stack([np.full((W, H), 40, np.uint8) for _ in range(N_FRAMES)])
    import nibabel as _nb
    _nb.save(_nb.Nifti1Image(np.ascontiguousarray(
        np.transpose(_vol, (2, 1, 0))), np.eye(4)), str(_pd_dir / "PA.nii.gz"))
    import shutil as _sh
    _sh.copy(m_main, str(_pd_dir / "M_1_mask.nii.gz"))
    _sh.copy(m_second, str(_pd_dir / "病灶1_1_mask.nii.gz"))
    from src.inference.nifti_io import save_binary_mask as _sbm
    _a = np.zeros((N_FRAMES, W, H), np.uint8)
    _b = np.zeros((N_FRAMES, W, H), np.uint8)
    _a[1, 8, 8] = 1
    _b[1, 0, 0] = 1; _b[3, 0, 0] = 1
    _sbm(_a, str(_pd_dir / "M_1_mask.nii.gz"))
    _sbm(_b, str(_pd_dir / "病灶1_1_mask.nii.gz"))
    _df = _pd.DataFrame([["PA", "", "2", "M", "7峡部", "1无", "1有", "恶", "", ""],
                         ["PA", "", "2", "病灶1", "7峡部", "1无", "1有", "良", "", ""]],
                        columns=["PatientId", "SeriesUid", "图像性质", "RoiName",
                                 "主病灶位置", "甲状腺外侵犯", "临床T分期",
                                 "ROI良恶性", "ANNOTATION->SliceIndex", "ROI-TIRADS"])
    _x = _tmp2 / "l.xlsx"
    _df.to_excel(_x, sheet_name="ROI级别", index=False, startrow=1)
    _sms = OL.build_official_samples(str(_tmp2), str(_x),
                                     union_mask=True)
    for _s in _sms:
        if not _s["is_dynamic"]:
            continue
        _u = ds_b._dynamic_union_mask_frame(_s["union_mask_paths"],
                                            _s["frame_index"])
        print(f"    frame_index={_s['frame_index']} "
              f"本病灶={os.path.basename(str(_s['mask_path']))} "
              f"union 前景={int((_u > 0).sum())}")

    import pandas as pd

    # ---- [C] 生产 build_official_datasets 三分支对拍(同 root/xlsx/pkl) ----
    print("\n[C] 生产 build_official_datasets 三分支对拍(同 root/xlsx/pkl):")

    # 造最小 xlsx:一行 M / 一行 病灶1(动态患者 pa)
    rows = [
        {"PatientId": "pa", "SeriesUid": "", "RoiName": "M",
         "ANNOTATION->SliceIndex": 1, "ROI->CLINICAL->良恶性": "恶",
         "主病灶位置": "1左叶", "甲状腺外侵犯": "1无", "临床T分期": "0无",
         "ROI->CLINICAL->TI-RADS": "4类"},
        {"PatientId": "pa", "SeriesUid": "", "RoiName": "病灶1",
         "ANNOTATION->SliceIndex": 1, "ROI->CLINICAL->良恶性": "良",
         "主病灶位置": "1左叶", "甲状腺外侵犯": "1无", "临床T分期": "0无",
         "ROI->CLINICAL->TI-RADS": "3类"},
    ]
    xlsx = os.path.join(tmp, "labels.xlsx")
    _hdr = list(rows[0].keys())
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame([["ROI级别"] * len(_hdr), _hdr]).to_excel(
            w, sheet_name="ROI级别", index=False, header=False)
        pd.DataFrame([[r[h] for h in _hdr] for r in rows]).to_excel(
            w, sheet_name="ROI级别", index=False, header=False,
            startrow=2)
    # 影像:pa/pa/pa.nii.gz(动态判据 = 磁盘存在 .nii.gz)
    pa_dir = os.path.join(tmp, "pa", "pa")
    os.makedirs(pa_dir, exist_ok=True)
    vol = np.zeros((SIZE, SIZE, N_FRAMES), np.uint8)
    vol[:, :, :] = 100
    nib.save(nib.Nifti1Image(np.ascontiguousarray(np.transpose(vol, (2, 1, 0))),
                             np.eye(4)), os.path.join(pa_dir, "pa.nii.gz"))
    # 两个 mask 放到 pa/ 下(与 _mask_path_for_roi 同目录搜索规约)
    import shutil
    shutil.copy(m_main, os.path.join(tmp, "pa", "M_1_mask.nii.gz"))
    shutil.copy(m_second, os.path.join(tmp, "pa", "病灶1_1_mask.nii.gz"))

    # 预抽取 pkl(union 版) —— 走真实 extract
    from scripts.data import preextract_dynamic as pex
    cache = os.path.join(tmp, "cache")
    rc = pex.extract(data_root=tmp, xlsx=xlsx, cache_dir=cache, sheet="ROI级别",
                     img_size=SIZE, limit=0, workers=1, union_mask=True)
    print("    extract rc =", rc)
    pkl = os.path.join(cache, "samples.pkl")
    payload = None
    import pickle
    with open(pkl, "rb") as f:
        payload = pickle.load(f)
    dyn = [s for s in payload["samples"] if s["is_dynamic"]]
    print("    pkl 动态样本数 =", len(dyn), "| meta.union_mask =",
          payload["_meta"]["union_mask"])
    if dyn:
        s0 = dyn[0]
        print("    样本 union_mask_paths =", s0.get("union_mask_paths"))
        print("    cache_msk =", os.path.basename(s0["cache_msk"]))
        from src.dataset.official_loader import _read_cached_pair as _rcp
        _, m_cache = _rcp(s0["cache_img"], s0["cache_msk"])
        ds_b = OL.OfficialThyroidDataset([s0], img_size=SIZE)
        m_direct = ds_b._dynamic_union_mask_frame(
            s0["union_mask_paths"], s0["frame_index"])
        print("    [C1] pkl+union=True 前景 =", int((m_cache > 0).sum()),
              "| 直建 union 前景 =", int((m_direct > 0).sum()),
              "| 逐位相等 =", bool(np.array_equal((m_cache > 0), (m_direct > 0))))
    # C2 关闭歧义门的三次调用(union 版 pkl + union_mask=False)
    for tag, um in (("C2 pkl+union=False", False), ("C3 直建 union=True", True),
                    ("C4 直建 union=False", False)):
        pkl_arg = None if tag.startswith("C3") or tag.startswith("C4") else pkl
        try:
            tr, va = OL.build_official_datasets(
                tmp, xlsx, img_size=SIZE, samples_pkl=pkl_arg, union_mask=um)
            ds_all = tr.samples + va.samples
            dsd = [x for x in ds_all if x["is_dynamic"]]
            if dsd:
                d0 = dsd[0]
                from src.dataset.official_loader import _read_cached_pair as _rcp2
                if d0.get("cache_img"):
                    _, mm = _rcp2(d0["cache_img"], d0["cache_msk"])
                else:
                    mm = OL.OfficialThyroidDataset([d0], img_size=SIZE
                                                   )._dynamic_union_mask_frame(
                        d0.get("union_mask_paths") or [d0["mask_path"]],
                        d0["frame_index"])
                print(f"    [{tag}] 放行,前景 =", int((mm > 0).sum()),
                      "| union_paths =", bool(d0.get("union_mask_paths")))
            else:
                print(f"    [{tag}] 放行但无动态样本")
        except Exception as e:  # noqa: BLE001
            print(f"    [{tag}] {type(e).__name__}: {str(e)[:110]}")



if __name__ == "__main__":
    main()

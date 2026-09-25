# -*- coding: utf-8 -*-
"""R3 证据 5:union 直建链的**跨帧错配** —— 生产可达性(整图 loader 的跨帧组合)。

同图两病灶标注帧不同(M 只在帧 1;病灶1 在帧 1、3),整图链真实产 3 条样本。
ROI epoch 用同一份样本展开,对每条样本调 `_dynamic_union_mask_frame`:
  - 正确语义:该样本帧上"所有病灶在该帧的 mask 之并集";
  - 实测:`_dynamic_union_mask_frame` 对"本 mask 不含该帧"的分支用
    `pos = idxs.index(fidx) if fidx in idxs else fidx` **回退到 fidx 原值**,
    于是取到别的帧(或越界)。
"""
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dataset import official_loader as OL  # noqa: E402
from src.inference.nifti_io import save_binary_mask  # noqa: E402


def _sav(path, vol_nhw):
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.ascontiguousarray(
        np.transpose(vol_nhw, (2, 1, 0))), np.eye(4)), path)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tmp_review_r3_xframe_"))
    root = tmp / "annotation"
    pd_dir = root / "PD" / "PD"
    os.makedirs(pd_dir, exist_ok=True)
    H = W = 64
    N = 5
    vol = np.stack([np.full((H, W), 40 + 10 * i, np.uint8) for i in range(N)])
    _sav(str(pd_dir / "PD.nii.gz"), vol)
    # M:仅帧 1;病灶1:帧 1 与 帧 3
    dm = np.zeros((N, H, W), np.uint8); dm[1, 10:30, 10:30] = 1
    ds = np.zeros((N, H, W), np.uint8)
    ds[1, 20:40, 20:40] = 1; ds[3, 30:50, 30:50] = 1
    save_binary_mask(dm, str(pd_dir / "M_1_mask.nii.gz"))
    save_binary_mask(ds, str(pd_dir / "病灶1_1_mask.nii.gz"))
    df = pd.DataFrame([["PD", "", "2", "M", "7峡部", "1无", "1有", "恶", "", ""],
                       ["PD", "", "2", "病灶1", "7峡部", "1无", "1有", "良", "", ""]],
                      columns=["PatientId", "SeriesUid", "图像性质", "RoiName",
                               "主病灶位置", "甲状腺外侵犯", "临床T分期",
                               "ROI良恶性", "ANNOTATION->SliceIndex", "ROI-TIRADS"])
    xlsx = tmp / "l.xlsx"
    df.to_excel(xlsx, sheet_name="ROI级别", index=False, startrow=1)

    samples = OL.build_official_samples(str(root), str(xlsx), union_mask=True)
    ds_obj = OL.OfficialThyroidDataset.__new__(OL.OfficialThyroidDataset)
    ds_obj.img_size = H

    print("样本(直建+union)→ union 帧读取实测:")
    print("  frame | 本病灶 | union前景 | 正确并集前景")
    for s in samples:
        if not s["is_dynamic"]:
            continue
        f = s["frame_index"]
        got = ds_obj._dynamic_union_mask_frame(s["union_mask_paths"], f)
        # 正确并集:只有"该病灶在该帧有标注"的 mask 才参与
        expect = np.zeros((H, W), np.uint8)
        for mp in s["union_mask_paths"]:
            binf, idxs = OL._read_nifti_binmasks(mp)
            if not binf:
                continue
            pos = idxs.index(f) if f in idxs else None     # ← 不回退
            if pos is None:
                continue
            expect = np.logical_or(expect, (binf[pos] > 0).astype(np.uint8)).astype(np.uint8)
        print(f"    {f:>2}   | {os.path.basename(str(s['mask_path'])):<18} "
              f"| {int((got > 0).sum()):>3}       | {int((expect > 0).sum()):>3}"
              f"   {'✅' if np.array_equal(got > 0, expect > 0) else '❌ 不一致'}")
    print("\n对比:同一批样本走 pkl 缓存(PNG 导出端按同帧合成,worker 的 "
          "mpos_of 回退也是 fidx)")

    # 缓存导出端同缺陷验证:worker 里 mp = mpos_of.get(fidx, fidx) 同样回退
    print("  preextract worker: `mp = mpos_of.get(fidx, fidx)` —— 本 mask 不含该帧时"
          "同样取 fidx 原值 → 与直建链同缺陷(预注册'两条路径逐位一致'成立,\n"
          "   但两条路径一致地错:都取到别的帧/越界,只是'一致地'错)")


if __name__ == "__main__":
    main()

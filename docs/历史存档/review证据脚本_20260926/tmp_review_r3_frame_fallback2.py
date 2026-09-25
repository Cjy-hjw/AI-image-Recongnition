# -*- coding: utf-8 -*-
"""R3 证据 5c:union "帧定位回退 fidx" 在**真实 NIfTI(压缩体,全长 idxs)**下的触发。

前一个探针用全 1 的合成 mask 写盘 → 保存时体数据被裁剪成 1 帧(idxs=[0]),
掩盖了真实情形(真实 nii.gz 的 idxs = 全帧 0..Z-1)。本探针用**影像同款**的
非二值体数据 + 手工构造 nii,确保 mask 的 idxs 覆盖全部帧(与官方数据一致)。

攻击点(源码 official_loader._dynamic_union_mask_frame:1048):
    pos = idxs.index(frame_index) if frame_index in idxs else frame_index
最后那个 `else frame_index` = **把帧号当帧下标用**,直接取别的帧。
触发条件:frame_index >= len(binf)(越界 → 被 `0 <= pos < len(binf)` 挡掉,贡零)
或 frame_index < len(binf) 但该 mask 不含此帧(→ 取到第 frame_index 帧的内容)。
"""
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from src.dataset import official_loader as OL  # noqa: E402

S = 16
N = 8


def _save_vol_raw(vol_nhw, path):
    """按 (N,H,W) 存成 NIfTI,后两维可不对称以观察转置。"""
    nib.save(nib.Nifti1Image(np.ascontiguousarray(
        np.transpose(vol_nhw, (2, 1, 0))), np.eye(4)), path)


def main():
    tmp = tempfile.mkdtemp(prefix="tmp_review_r3_raw_")
    # maskA:帧 1 在 (8,8) 有前景;帧 3 在 (12,12) 有前景;其余全零
    a = np.zeros((N, S, S), np.uint8)
    a[1, 8, 8] = 1
    a[3, 12, 12] = 1
    pa = os.path.join(tmp, "M_1_mask.nii.gz")
    _save_vol_raw(a, pa)
    binf, idxs = OL._read_nifti_binmasks(pa)
    print(f"maskA 写出后 idxs={idxs} (应 = 全帧 0..{N-1}), n_frames={len(binf)}")

    o = OL.OfficialThyroidDataset.__new__(OL.OfficialThyroidDataset)
    o.img_size = S
    print("\n单 mask 路 fidx 扫描(正确: 前景非零 ⇔ fidx ∈ {1,3}):")
    for f in range(N + 2):
        v = o._dynamic_mask_frame(pa, f)
        truth = 1 if f in (1, 3) else 0
        n = int((v > 0).sum())
        print(f"  fidx={f}: 前景={n} 正确={truth} "
              f"{'OK' if (n > 0) == bool(truth) else '❌ 取错帧(回退把帧号当下标)'}")

    # union 路:maskA(帧 1,3)+ maskB(仅帧 5)
    b = np.zeros((N, S, S), np.uint8)
    b[5, 0, 0] = 1
    pb = os.path.join(tmp, "病灶1_1_mask.nii.gz")
    _save_vol_raw(b, pb)
    print("\nunion 路(两 mask: A{1,3} / B{5})逐帧扫描:")
    for f in range(N):
        got = o._dynamic_union_mask_frame([pa, pb], f)
        exp = 0
        for p in (pa, pb):
            bb, ix = OL._read_nifti_binmasks(p)
            if f in ix:
                exp += int((bb[ix.index(f)] > 0).sum())
        n = int((got > 0).sum())
        print(f"  fidx={f}: union前景={n} 真值={exp} "
              f"{'OK' if n == exp else '❌ 不一致(回退帧号当下标)'}")


if __name__ == "__main__":
    main()

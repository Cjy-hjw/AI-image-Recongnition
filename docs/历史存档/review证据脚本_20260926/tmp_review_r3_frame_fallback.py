# -*- coding: utf-8 -*-
"""R3 证据 5b:union "帧定位回退 fidx" 的**生产可达性**对照实验。

严格控制变量:同图两病灶,标注帧**完全不重叠**(M 仅帧 1 / 病灶1 仅帧 3)。
直建链(样本 frame_index ∈ 本病灶非零帧)实测无错配;
关键对照组 = **推理侧调用形态**(逐 mask 遍历,帧号取自"另一条链路"),
即 `frame_index ∉ 该 mask 的 idxs` 的情形必须真的发生才会触发回退。
本脚本枚举 帧号 × mask 组合,找出真正触发"取错帧"的条件。
"""
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from src.dataset import official_loader as OL  # noqa: E402
from src.inference.nifti_io import save_binary_mask  # noqa: E402


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tmp_review_r3_reach_"))
    H = W = 16
    N = 5
    # mask A:仅帧 1 有前景(像素 (8,8));mask B:仅帧 3(像素 (0,0))
    a = np.zeros((N, H, W), np.uint8); a[1, 8, 8] = 1
    b = np.zeros((N, H, W), np.uint8); b[3, 0, 0] = 1
    pa = str(tmp / "M_1_mask.nii.gz"); pb = str(tmp / "病灶1_1_mask.nii.gz")
    save_binary_mask(a, pa); save_binary_mask(b, pb)
    # 另一张:全帧非零(模拟"影像帧集"的 idxs = 全帧)
    img = np.full((N, H, W), 90, np.uint8)
    pi = str(tmp / "img.nii.gz")
    save_binary_mask(np.ones((N, H, W), np.uint8), pi)

    o = OL.OfficialThyroidDataset.__new__(OL.OfficialThyroidDataset)
    o.img_size = H
    paths = [pa, pb]

    print("枚举:frame_index × [mask 的 idxs] → union 结果")
    print("  fidx | maskA.idxs | maskB.idxs | union前景 | 真值(该帧并集) | 判定")
    for f in (1, 3):
        ba, ia = OL._read_nifti_binmasks(pa)
        bb, ib = OL._read_nifti_binmasks(pb)
        got = o._dynamic_union_mask_frame(paths, f)
        exp = np.zeros((H, W), np.uint8)
        for p, ix, bb_ in ((pa, ia, ba), (pb, ib, bb)):
            if f in ix:
                exp = np.logical_or(exp, (bb_[ix.index(f)] > 0).astype(np.uint8)).astype(np.uint8)
        ok = np.array_equal(got > 0, exp > 0)
        print(f"    {f}  | {ia}      | {ib}      | {int((got>0).sum())}         |"
              f" {int((exp>0).sum())}              | {'OK' if ok else 'MISMATCH'}")

    # 触发条件:frame_index 不在该 mask 的 idxs 里 → 回退 fidx 当下标
    print("\n回退分支触发条件:fidx ∉ mask 的 idxs")
    print("  单 mask 路(maskA,idx=[1]) × fidx=3 →",
          int((o._dynamic_mask_frame(pa, 3) > 0).sum()),
          "前景像素(真值应 0; idxs.index 回退 → binf[3] = 帧 3 内容 = 全零,"
          "此处恰好 0,但若帧 3 有别的病灶就串味)")
    # 让 maskA 的帧 3 也有前景,再问 fidx=3(本该用帧 3)与 fidx=0(不在 idxs)
    a2 = np.zeros((N, H, W), np.uint8); a2[1, 8, 8] = 1; a2[3, 12, 12] = 1
    pa2 = str(tmp / "M2_1_mask.nii.gz")
    save_binary_mask(a2, pa2)
    for f in (0, 1, 2, 3, 4):
        v = o._dynamic_mask_frame(pa2, f)
        print(f"    maskA2(非零帧 1,3, idx={sorted(set([1,3]))}) fidx={f} → "
              f"前景={int((v > 0).sum())} (正确: f∈{{1,3}} 才有前景)")


if __name__ == "__main__":
    main()

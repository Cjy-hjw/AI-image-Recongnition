# -*- coding: utf-8 -*-
"""R3 证据 5(定稿):union 帧定位 = **帧号当数组下标** —— 用真实动态体长(1039 帧)
和真实标注非零帧位置(400/500/600)复刻,证明这条口径在**生产数据上系统性错帧**。

生产调用链(preextract_dynamic._process_one_file / _dynamic_union_mask_frame 同式):
    pos_of  = {frame_no: list_index}     ← uniform_sample_indices(n, max_frames)
    ...
    ip = pos_of.get(fidx, fidx)          ← 命中→位置;不命中→**退化为 fidx 本身(=帧号)**
    frames[ip] / binf[mp]                ← 当"位置"用

结论:凡 `fidx != pos_of[fidx]`(即抽样步长 ≠ 1)的帧,都会读**错帧**;
越界时(典型:真实体 1039 帧 > max_frames)该样本被记 failure → 动态样本被剔出 pkl。
"""
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.inference.nifti_io import save_binary_mask  # noqa: E402
from src.inference.video_aggregate import uniform_sample_indices  # noqa: E402
from src.dataset.official_loader import _read_nifti_binmasks  # noqa: E402

N_FRAMES = 1039          # 真实动态体量级(官方最长 750 帧,1039 是常见长例)
MARKED = (400, 500, 600)  # mask 非零帧(= 主病灶标注帧,定稿 §4.2 项6)
S = 16


def main():
    a = np.zeros((N_FRAMES, S, S), np.uint8)
    for f in MARKED:
        a[f, 8, 8] = 1
    p = os.path.join(tempfile.mkdtemp(prefix="tmp_review_r3_pos_"), "M_1_mask.nii.gz")
    save_binary_mask(a, p)
    binf, idxs = _read_nifti_binmasks(p)
    print(f"体: {N_FRAMES} 帧 | idxs = {idxs[0]}..{idxs[-1]} (len={len(idxs)})")
    print(f"mask 非零帧 = {MARKED}")

    for max_frames in (32, -1):
        pos_of = {f: i for i, f in enumerate(
            uniform_sample_indices(N_FRAMES, max_frames))}
        print(f"\n── max_frames={max_frames} (抽样点数 {len(pos_of)}) ──")
        print("  fidx | pos_of[fidx] | 回退 mp | binf[mp] 有前景? | 正确?")
        for fidx in MARKED:
            real_pos = pos_of.get(fidx)
            mp = pos_of.get(fidx, fidx)          # ← 生产写法:pos_of.get(fidx, fidx)
            in_range = 0 <= mp < len(binf)
            hit = bool(binf[mp].any()) if in_range else None
            if real_pos is None:
                verdict = "本帧未被抽中(该项应被跳过,不是取错帧)"
            elif not in_range:
                verdict = f"❌ 越界 → preextract 记 FRM failure,样本被剔"
            elif mp != real_pos:
                verdict = f"❌ 读的是下标 {mp}(帧 {idxs[mp] if mp < len(idxs) else '?'}),不是帧 {fidx}"
            else:
                verdict = "✅"
            print(f"  {fidx:>4} | {str(real_pos):>5}        | {mp:>5}   | "
                  f"{str(hit):>5}           | {verdict}")

    # 对照:正确写法(pos 命中才取,否则跳过)
    pos_of = {f: i for i, f in enumerate(uniform_sample_indices(N_FRAMES, 32))}
    print("\n正确写法(不回退,未抽中即跳过)在同一输入上的结果:")
    for fidx in (0, 7, 16, 400, 500, 600):
        real = pos_of.get(fidx)
        if real is None:
            print(f"  fidx={fidx:>4}: 未抽中 → 跳过(不监督)✅")
        else:
            print(f"  fidx={fidx:>4}: 取 binf[{real}](帧 {idxs[real]}),"
                  f"有前景={bool(binf[real].any())}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""R3 证据 4:ROI 链与整图链在同一输入上的 **病灶集合差异**(duplicate suppression ≠ GT 病灶数)。

复刻 tests/test_official_loader.py::mini_root 的 fixture 式构造(dynamic 患者 PD,
M@帧1 / 病灶1@帧1,3),跑**真实** build(roi_meta=True / 默认)并对比:
  - 整图链样本数(per 病灶 per 非零帧)
  - roi_lesions 长度(per 图,按 lesion_id 去重)
  - 真值:病灶 2 个 × 各自非零帧 = 1 + 2 = 3 条
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


def _save_image_nifti_3d(path, vol_nhw):
    import nibabel as nib
    arr = np.ascontiguousarray(np.transpose(vol_nhw, (2, 1, 0)))
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)


def main():
    for tag, dynamic_mask_frames in (("A: 次病灶标注帧 ⊆ 主病灶帧", None),
                                     ("B: 次病灶有独立标注帧", [1, 3])):
        tmp = Path(tempfile.mkdtemp(prefix="tmp_review_r3_dup_"))
        root = tmp / "annotation"
        pd_dir = root / "PD" / "PD"
        os.makedirs(pd_dir, exist_ok=True)
        H = W = 64
        vol = np.stack([np.full((H, W), 40 + 10 * i, np.uint8) for i in range(5)])
        _save_image_nifti_3d(str(pd_dir / "PD.nii.gz"), vol)
        dm = np.zeros((5, H, W), np.uint8); dm[1, 10:30, 10:30] = 1
        save_binary_mask(dm, str(pd_dir / "M_1_mask.nii.gz"))
        ds = np.zeros((5, H, W), np.uint8)
        if dynamic_mask_frames is None:
            ds[1, 20:40, 20:40] = 1
        else:
            ds[1, 20:40, 20:40] = 1
            ds[3, 30:50, 30:50] = 1
        save_binary_mask(ds, str(pd_dir / "病灶1_1_mask.nii.gz"))

        # 影像与 mask 同目录(动态 = {pid}/{pid}/)
        _save_image_nifti_3d(str(pd_dir / "PD.nii.gz"), vol)

        rows = [["PD", "", "2", "M", "7峡部", "1无", "1有", "恶", "", ""],
                ["PD", "", "2", "病灶1", "7峡部", "1无", "1有", "良", "", ""]]
        df = pd.DataFrame(rows, columns=[
            "PatientId", "SeriesUid", "图像性质", "RoiName", "主病灶位置",
            "甲状腺外侵犯", "临床T分期", "ROI良恶性", "ANNOTATION->SliceIndex",
            "ROI-TIRADS"])
        xlsx = tmp / "l.xlsx"
        df.to_excel(xlsx, sheet_name="ROI级别", index=False, startrow=1)

        s_plain = OL.build_official_samples(str(root), str(xlsx))
        s_roi = OL.build_official_samples(str(root), str(xlsx), roi_meta=True)
        print(f"\n[{tag}]")
        print(f"  整图样本 {len(s_plain)} 条:",
              [(s["frame_index"], os.path.basename(str(s["mask_path"]))) for s in s_plain])
        print(f"  roi_meta 样本 {len(s_roi)} 条(应同数,只加键)")
        for s in s_roi:
            les = s.get("roi_lesions") or []
            print(f"    frame_index={s['frame_index']} 本病灶="
                  f"{os.path.basename(str(s['mask_path']))} "
                  f"→ 该样本 frame_indices={[l['frame_indices'] for l in les]}")
        # roi_lesions 是图级去重列表:整图样本数与 roi_lesions 长度无对应关系
        les0 = (s_roi[0].get("roi_lesions") or [])
        print(f"  → 图级去重后 roi_lesions 长度 = {len(les0)} "
              f"(per-图 唯一病灶数),整图样本数 = {len(s_roi)}")
        print("  ⚠️ 校验 '样本数 == 病灶数' 是无效判据 —— 两者本就不同维度")


if __name__ == "__main__":
    main()

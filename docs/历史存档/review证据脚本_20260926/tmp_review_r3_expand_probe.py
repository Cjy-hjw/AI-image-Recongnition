# -*- coding: utf-8 -*-
"""R3 证据 2:真实 build(roi_meta=True)→ expand_roi_samples 展开语义 + 元数据完整度。

复刻 tests/test_official_loader.py::mini_root 的 fixture 构造方式(df.to_excel
startrow=1 = 双行表头),动态患者 PD 带两个病灶(M 非零帧 [1] / 病灶1 非零帧 [1,3])。
"""
import os
import sys
import tempfile
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.nifti_io import save_binary_mask  # noqa: E402


def _save_image_nifti_3d(path, vol_nhw):
    import nibabel as nib
    arr = np.ascontiguousarray(np.transpose(vol_nhw, (2, 1, 0)))
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tmp_review_r3g_"))
    root = tmp / "annotation"
    H = W = 64
    # 动态患者 PD:5 帧双病灶
    pd_dir = root / "PD" / "PD"
    os.makedirs(pd_dir, exist_ok=True)
    vol = np.stack([np.full((H, W), 40 + 10 * i, np.uint8) for i in range(5)])
    _save_image_nifti_3d(str(pd_dir / "PD.nii.gz"), vol)
    dm = np.zeros((5, H, W), np.uint8); dm[1, 10:30, 10:30] = 1
    save_binary_mask(dm, str(pd_dir / "M_1_mask.nii.gz"))
    ds = np.zeros((5, H, W), np.uint8)
    ds[1, 20:40, 20:40] = 1; ds[3, 30:50, 30:50] = 1
    save_binary_mask(ds, str(pd_dir / "病灶1_1_mask.nii.gz"))

    rows = [
        ["PD", "", "2", "M", "7峡部", "1无", "1有", "恶", "", ""],
        ["PD", "", "2", "病灶1", "7峡部", "1无", "1有", "良", "", ""],
    ]
    df = pd.DataFrame(rows, columns=[
        "PatientId", "SeriesUid", "图像性质", "RoiName", "主病灶位置",
        "甲状腺外侵犯", "临床T分期", "ROI良恶性", "ANNOTATION->SliceIndex",
        "ROI-TIRADS"])
    xlsx = tmp / "labels_union.xlsx"
    df.to_excel(xlsx, sheet_name="ROI级别", index=False, startrow=1)

    from src.dataset import official_loader as OL
    from src.dataset.roi_dataset import expand_roi_samples

    samples = OL.build_official_samples(str(root), str(xlsx), roi_meta=True)
    dyn = [s for s in samples if s["is_dynamic"]]
    print(f"[1] 整图样本 {len(samples)} 动态 {len(dyn)}")
    for s in dyn:
        les = s.get("roi_lesions") or []
        print(f"    frame_index={s['frame_index']} 本病灶={os.path.basename(str(s['mask_path']))} "
              f"roi_lesions={len(les)}")
        for le in les:
            print(f"        {os.path.basename(str(le['lesion_id']))} bbox={le['bbox_xyxy']} "
                  f"frame_indices={le['frame_indices']} "
                  f"mask={os.path.basename(str(le['mask_path']))} "
                  f"is_main={le.get('is_main')} mal={le.get('malignancy')}/"
                  f"{le.get('valid_malignancy')} comp={le.get('composition')}/"
                  f"{le.get('valid_composition')} has_instance_idx={'instance_idx' in le}")

    ex = expand_roi_samples(samples)
    print(f"\n[2] expand → {len(ex)} 条病灶级样本(整图动态 {len(dyn)} 条)")
    c = Counter((os.path.basename(str(r["lesion_id"])), r["frame_index"]) for r in ex)
    print("    (lesion, frame) 计数:", dict(c))
    print("    instance_idx in expanded:", any("instance_idx" in r for r in ex))
    print("    roi_lesions：展开源是否残留:", any("roi_lesions" in r for r in ex))
    for r in ex:
        print(f"      {os.path.basename(str(r['lesion_id']))} f={r['frame_index']} "
              f"bbox={r['bbox_xyxy']} mask={os.path.basename(str(r['mask_path']))} "
              f"is_main={r.get('is_main')} mal={r.get('malignancy')}")
    # 同(lesion,frame)去重是否与"整图样本共享 roi_lesions"配合正确:
    # 期望 M@1 + 病灶1@1 + 病灶1@3 = 3 条
    print("\n[3] 期望 = M@帧1 / 病灶1@帧1 / 病灶1@帧3 = 3 条;"
          f"实际 {len(ex)} 条；去重键唯一数 {len(set(c))}")


if __name__ == "__main__":
    main()

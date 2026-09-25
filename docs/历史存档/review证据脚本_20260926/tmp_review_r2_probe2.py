# -*- coding: utf-8 -*-
"""R2 review probe (part 2): real data-chain label semantics for the ROI concept head.

Tests whether the concept supervision is traceable to a REAL official column /
observed encoding, or rests on unverified inference (rules: inferred facts must
not enter hard-fail / hard-consumption paths).

Run: D:\\program\\Python\\anaconda\\envs\\Demo_D\\python.exe tmp_review_r2_probe2.py
"""
import inspect
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import src.dataset.official_loader as L
from src.dataset import roi_dataset as RD

print("=== A. concept key plumbing (column -> label -> meta -> loss) ===")
print("official_loader._FIELD_ALIASES concept entries:",
      {k: L._FIELD_ALIASES[k] for k in ("composition", "echogenicity", "shape", "margin")})
print("_roi_lesion_meta uses _build_labels:", "_build_labels" in
      inspect.getsource(L._roi_lesion_meta))
src = inspect.getsource(L.build_official_samples)
i = src.find("_meta_memo")
print(src[i:i + 260])

print("\n=== B. concept columns are NOT in the clinical '-warn' / diagnosis list ===")
print("_CONCEPT_TASKS:", L._CONCEPT_TASKS)
print("_CLINICAL_LOGICAL_FIELDS:", L._CLINICAL_LOGICAL_FIELDS)
print("concepts covered by column-hit warning:",
      any(t in L._CLINICAL_LOGICAL_FIELDS for t in L._CONCEPT_TASKS))

print("\n=== C. mock xlsx with official concept columns -> are they hit? ===")
# 模拟 build 的列命中判定(纯字符串层,不碰影像)
cols = ["PatientId", "RoiName", "SeriesUid", "ROI->CLINICAL->TI-RADS",
        "成分", "回声", "形状", "边缘", "ROI->CLINICAL->良恶性"]
cm = L._resolve_columns(cols)
print("colmap concepts:", {k: cm.get(k) for k in ("composition", "echogenicity", "shape", "margin")})

print("\n=== D. _parse_concept on representative cell values ===")
for v in ["1囊性", "2实性", "3混合", "实性", "囊性", "低回声", "1", 2, "2.0", "无", ""]:
    try:
        print(f"  {v!r:12} -> {L._parse_concept(v)}")
    except Exception as e:
        print(f"  {v!r:12} -> EXC {type(e).__name__}: {e}")

print("\n=== E. class-count evidence in repo for concept tasks ===")
import subprocess
g = subprocess.run(["git", "grep", "-n", "-l", "-E", "class_counts|class_counts_", "--",
                    "configs", "scripts", "src"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(g.stdout.strip()[:2000] or "(none)")


print("\n=== F. real end-to-end: build_official_samples -> RoiDataset -> loss ===")
# 造一个最小官方目录结构(静态: pid/series/*.dcm + mask), xlsx 双行表头
import tempfile, shutil
root = tempfile.mkdtemp(prefix="tmp_review_r2_")
pid = "a" * 32
series = "1.2.3.4"
d = os.path.join(root, pid, series)
os.makedirs(d, exist_ok=True)
try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = generate_uid()
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.Rows, ds.Columns = 32, 32
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = (np.arange(32 * 32, dtype=np.uint16) * 30).tobytes()
    ds.save_as(os.path.join(d, "img.dcm"), write_like_original=False)
    dcm_ok = True
except Exception as e:
    dcm_ok = False
    print("  pydicom mock dcm failed:", e)

if dcm_ok:
    # mask: 静态 3D 单帧 (X,Y,Z) ; save via our own writer to keep the convention
    from src.inference.nifti_io import save_binary_mask
    m = np.zeros((32, 32, 1), np.uint8)
    m[8:20, 10:24, 0] = 1
    save_binary_mask(m, os.path.join(d, "M_1_mask.nii.gz"))

    xlsx = os.path.join(root, "ann.xlsx")
    # 双行表头: 第一行分组标题, 第二行真实列名 (loader header=1)
    header_rows = [["PATIENT", "ANNOTATION", "ROI->CLINICAL", "", "", ""],
                   ["PatientId", "RoiName", "SeriesUid", "成分", "回声", "形状"]]
    data = [[pid, "M", series, "1囊性", "2低回声", "1纵横比<1"]]
    df = pd.DataFrame(header_rows + data)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as _w:
        df.to_excel(_w, index=False, header=False, sheet_name="ROI级别")

    samples = L.build_official_samples(root, xlsx, roi_meta=True)
    print(f"  samples built: {len(samples)}")
    les = samples[0].get("roi_lesions") if samples else []
    if les:
        keys = ["composition", "valid_composition", "echogenicity", "valid_echogenicity",
                "shape", "valid_shape", "margin", "valid_margin"]
        print("  roi_lesions[0] concept keys:",
              {k: les[0].get(k, "<MISSING>") for k in keys})
        print("  roi_lesions[0] all keys:", sorted(les[0].keys()))
        # RoiDataset 展开 + __getitem__ -> meta
        roi_s = RD.expand_roi_samples(samples)
        print(f"  expand_roi_samples: {len(roi_s)}")
        ds_ = RD.RoiDataset(roi_s, roi_size=32)
        img, meta = ds_[0]
        print("  RoiDataset meta concept keys:",
              {k: meta.get(k, "<MISSING>") for k in keys})
        labs = RD.roi_labels_from_meta([meta])
        print("  bridged labels:", {k: labs[0].get(k, "<MISSING>") for k in
                                    ["gt_composition", "valid_composition", "gt_shape", "valid_shape"]})
        import torch as T
        from src.models.losses.components.concepts_roi import compute_roi_concept_losses
        head = nn.Linear(4, 4)
        concepts = {"logits_comp": torch.randn(1, 4, requires_grad=True),
                    "logits_echo": torch.randn(1, 4, requires_grad=True),
                    "logits_shape": torch.randn(1, 2, requires_grad=True),
                    "logits_margin": torch.randn(1, 4, requires_grad=True),
                    "logits_foci": torch.randn(1, 4, requires_grad=True)}
        t, dd = compute_roi_concept_losses({"roi_concepts": concepts}, labs,
                                          lambda_concept=1.0)
        print("  concept loss on real chain:", float(t),
              "| n_valid comp/shape:", dd["concept_n_valid_comp"], dd["concept_n_valid_shape"])
        # 全 0 valid(概念列缺失)形态
        labs0 = [{k: (0 if k.startswith("gt_") else 0) for k in labs[0]}]
        for k in ("valid_composition", "valid_echogenicity", "valid_shape", "valid_margin"):
            labs0[0][k] = 0
        _, dd_all0 = compute_roi_concept_losses({"roi_concepts": concepts}, labs0,
                                                lambda_concept=1.0)
        print("  ep-mean if concept cols ALL MISSING (valid-composition==0):",
              dd_all0["concept_n_valid_comp"])
        print("  WARN: concept columns are NOT in _CLINICAL_LOGICAL_FIELDS -> no"
              " warning; per-batch n_valid_* IS logged in JSONL")
    else:
        print("  NO roi_lesions (collector produced nothing)")
shutil.rmtree(root, ignore_errors=True)

print("\nDONE probe2")

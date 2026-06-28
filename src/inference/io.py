"""Image I/O with DICOM PHI anonymization (HIPAA Safe Harbor fields)."""

import os
import cv2
import numpy as np

# HIPAA Safe Harbor 相关的 DICOM PHI 标签 → 匿名化目标值。
# 覆盖患者身份、就诊、机构、设备及操作者标识等可重识别字段。
_PHI_FIELDS = {
    "PatientName": "ANONYMIZED",
    "PatientID": "ANONYMIZED",
    "PatientBirthDate": "00000101",
    "PatientSex": "",
    "PatientAge": "",
    "PatientAddress": "ANONYMIZED",
    "PatientTelephoneNumbers": "",
    "OtherPatientIDs": "",
    "ReferringPhysicianName": "ANONYMIZED",
    "PerformingPhysicianName": "ANONYMIZED",
    "OperatorsName": "ANONYMIZED",
    "InstitutionName": "ANONYMIZED",
    "InstitutionAddress": "ANONYMIZED",
    "StationName": "ANONYMIZED",
    "DeviceSerialNumber": "ANONYMIZED",
    "AccessionNumber": "ANONYMIZED",
    "StudyID": "ANONYMIZED",
    "StudyDate": "00000101",
    "SeriesDate": "00000101",
    "AcquisitionDate": "00000101",
    "ContentDate": "00000101",
}

try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
    """单帧 min-max 归一化到 8-bit 灰度(统一静/动态图 DICOM 的位深处理)。"""
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return ((arr - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """统一转单通道灰度 uint8。

    彩色多普勒等 3 通道帧若不转灰度,送入 preprocess_image 会 unsqueeze 成
    5 维张量 (B,1,H,W,3) 触发模型前向 `too many values to unpack` 崩溃。
    静(io)/动(video_io)两条路径共用本函数,口径一致。
    """
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
    if frame.dtype != np.uint8:
        frame = _rescale_to_uint8(frame)
    return frame


def load_input_image(path: str, output_dir: str):
    """Load a B-mode ultrasound image (PNG/JPG/DCM).

    For DICOM inputs, anonymizes HIPAA Safe Harbor PHI fields (see _PHI_FIELDS)
    and saves the sanitized copy.

    Returns:
        (img_np, is_dicom, error_msg) — img_np is None on failure.
    """
    is_dicom = path.lower().endswith((".dcm", ".dicom"))

    if is_dicom:
        if not HAS_PYDICOM:
            return None, True, "DICOM image provided but pydicom is not installed."
        try:
            print("Reading PACS DICOM metadata and pixel data...")
            ds = pydicom.dcmread(path)

            # 统一经 _to_gray:彩色多普勒单帧 DICOM 的 pixel_array 为 (H,W,3),
            # 必须转灰度,否则下游 preprocess_image 产出 5 维张量令模型前向崩溃。
            # _to_gray 内部对非 uint8(如 16-bit)走 min-max 归一化,等价原 rescale。
            img_np = _to_gray(ds.pixel_array)

            # PHI metadata anonymization (HIPAA Safe Harbor) — 仅覆盖已存在的标签
            for tag, anon_value in _PHI_FIELDS.items():
                if tag in ds:
                    setattr(ds, tag, anon_value)

            # Save anonymized dicom to output
            anon_dicom_path = os.path.join(output_dir, "anonymized_image.dcm")
            ds.save_as(anon_dicom_path)
            print(f"--> Saved anonymized DICOM to {anon_dicom_path}")
        except Exception as e:
            return None, True, f"Error reading/anonymizing DICOM file: {e}"
    else:
        img_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img_np is None:
        return None, is_dicom, f"Could not read image from {path}"

    return img_np, is_dicom, None

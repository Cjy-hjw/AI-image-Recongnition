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

            # Extract grayscale pixel array
            pixel_array = ds.pixel_array.astype(np.float32)
            # Rescale to 8-bit [0, 255] grayscale
            img_np = (
                (pixel_array - pixel_array.min())
                / (pixel_array.max() - pixel_array.min() + 1e-8)
                * 255.0
            ).astype(np.uint8)

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

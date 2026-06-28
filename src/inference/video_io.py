"""动态图/视频(cine)抽帧:MP4/AVI/MOV via OpenCV,多帧 DICOM via pydicom。

返回灰度帧列表,供单帧模型逐帧推理。复用 `io.py` 的 PHI 匿名化字段
(`_PHI_FIELDS`)对 cine DICOM 脱敏。采样下标由 `video_aggregate.uniform_sample_indices`
统一计算,保证与聚合逻辑一致。
"""

import os
import cv2
import numpy as np

from src.inference.io import _PHI_FIELDS, HAS_PYDICOM
from src.inference.video_aggregate import uniform_sample_indices

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
_DICOM_EXTS = (".dcm", ".dicom")


def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
    """单帧 min-max 归一化到 8-bit 灰度(对齐 io.load_input_image 的 DICOM 处理)。"""
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return ((arr - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """统一转单通道灰度 uint8。"""
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
    if frame.dtype != np.uint8:
        frame = _rescale_to_uint8(frame)
    return frame


def load_video_frames(path: str, output_dir: str, max_frames: int = 32):
    """读取视频/cine DICOM,均匀抽至多 max_frames 帧。

    Returns:
        (frames, frame_indices, is_dicom, error_msg)
        frames        — list[np.ndarray] 灰度 uint8,失败为 None
        frame_indices — 每帧对应的原始帧号(用于关键帧定位/报告)
        is_dicom      — 输入是否 DICOM
        error_msg     — 失败原因,成功为 None
    """
    lower = path.lower()
    is_dicom = lower.endswith(_DICOM_EXTS)

    if is_dicom:
        return _load_dicom_cine(path, output_dir, max_frames)
    if lower.endswith(_VIDEO_EXTS):
        return _load_cv_video(path, max_frames)
    return None, [], False, (
        f"Unsupported video format: {os.path.basename(path)} "
        f"(supported: {_VIDEO_EXTS + _DICOM_EXTS})"
    )


def _load_cv_video(path: str, max_frames: int):
    """OpenCV 解码视频。先读全部帧再均匀采样(超声 cine 通常帧数有限)。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, [], False, f"Could not open video: {path}"
    raw = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw.append(frame)
    cap.release()
    if not raw:
        return None, [], False, f"No frames decoded from video: {path}"
    idxs = uniform_sample_indices(len(raw), max_frames)
    frames = [_to_gray(raw[i]) for i in idxs]
    return frames, idxs, False, None


def _load_dicom_cine(path: str, output_dir: str, max_frames: int):
    """多帧 DICOM(cine)。pixel_array 3D=(frames,H,W);2D 视作单帧。

    复用 _PHI_FIELDS 做 HIPAA 脱敏并存匿名化副本(与 io.load_input_image 一致)。
    """
    if not HAS_PYDICOM:
        return None, [], True, "DICOM cine provided but pydicom is not installed."
    try:
        import pydicom

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array
        if arr.ndim == 2:
            arr = arr[None, ...]  # 单帧 → (1,H,W)
        n_total = arr.shape[0]
        idxs = uniform_sample_indices(n_total, max_frames)
        # 经 _to_gray 统一兜底:彩色/多普勒 cine 的帧为 (H,W,3),需转灰度并对齐视频路径,
        # 否则 3 通道帧喂入 preprocess_image 会让模型前向形状报错。
        frames = [_to_gray(arr[i]) for i in idxs]

        for tag, anon in _PHI_FIELDS.items():
            if tag in ds:
                setattr(ds, tag, anon)
        anon_path = os.path.join(output_dir, "anonymized_cine.dcm")
        ds.save_as(anon_path)
        print(f"--> Saved anonymized DICOM cine to {anon_path}")
        return frames, idxs, True, None
    except Exception as e:  # noqa: BLE001 — 守卫契约,失败返回错误信息
        return None, [], True, f"Error reading/anonymizing DICOM cine: {e}"

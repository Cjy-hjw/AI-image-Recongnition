"""动态图/视频(cine)抽帧:MP4/AVI/MOV via OpenCV,多帧 DICOM via pydicom。

返回灰度帧列表,供单帧模型逐帧推理。复用 `io.py` 的 PHI 匿名化函数
(`_anonymize_phi`,标量匿名 + 序列删除)对 cine DICOM 脱敏。采样下标由
`video_aggregate.uniform_sample_indices`
统一计算,保证与聚合逻辑一致。
"""

import os
import cv2
import numpy as np

from src.inference.io import _anonymize_phi, HAS_PYDICOM, _rescale_to_uint8, _to_gray
from src.inference.video_aggregate import uniform_sample_indices

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
_DICOM_EXTS = (".dcm", ".dicom")

# 通道/位深归一化工具(_rescale_to_uint8 / _to_gray)已下沉至 io.py,静/动两路径共用,
# 此处 re-import 保持向后兼容(test_video.py 等仍从 video_io 命名空间引用)。


def load_video_frames(path: str, output_dir: str, max_frames: int = 32):
    """读取视频/cine DICOM,均匀抽至多 max_frames 帧。

    Returns:
        (frames, frame_indices, is_dicom, error_msg, n_total_frames)
        frames          — list[np.ndarray] 灰度 uint8,失败为 None
        frame_indices   — 每帧对应的原始帧号(用于关键帧定位/报告)
        is_dicom        — 输入是否 DICOM
        error_msg       — 失败原因,成功为 None
        n_total_frames  — 原始视频/体数据的**总帧数**(非抽样帧数);
                          供 scatter_frames_to_full_length 把抽样帧 mask 回填
                          成全帧序列 3D mask(规范 §五.2)。失败分支返回 0。
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
    ), 0


def _load_cv_video(path: str, max_frames: int):
    """OpenCV 解码视频，seek 采样避免全量读入 OOM。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, [], False, f"Could not open video: {path}", 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # 某些编码器帧数未知，回退全量读入（超声 cine 通常极短）
        raw = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw.append(frame)
        cap.release()
        if not raw:
            return None, [], False, f"No frames decoded from video: {path}", 0
        idxs = uniform_sample_indices(len(raw), max_frames)
        frames = [_to_gray(raw[i], channel_order="bgr") for i in idxs]
        return frames, idxs, False, None, len(raw)
    idxs = uniform_sample_indices(total, max_frames)
    frames = []
    decoded_idxs = []
    for idx in idxs:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)):
            continue  # codec 不支持 seek → 跳过该帧，不污染其他帧
        ok, frame = cap.read()
        if ok:
            frames.append(_to_gray(frame, channel_order="bgr"))
            decoded_idxs.append(int(idx))
    cap.release()
    if not frames:
        return None, [], False, f"No frames decoded from video: {path}", 0
    return frames, decoded_idxs, False, None, total


def _load_dicom_cine(path: str, output_dir: str, max_frames: int):
    """多帧 DICOM(cine)。pixel_array 3D=(frames,H,W);2D 视作单帧。

    复用 _PHI_FIELDS 做 HIPAA 脱敏并存匿名化副本(与 io.load_input_image 一致)。
    """
    if not HAS_PYDICOM:
        return None, [], True, "DICOM cine provided but pydicom is not installed.", 0
    try:
        import pydicom

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array
        if arr.ndim == 2:
            arr = arr[None, ...]  # 单帧 → (1,H,W)
        n_total = arr.shape[0]
        idxs = uniform_sample_indices(n_total, max_frames)
        # 经 _to_gray 统一兜底:彩色/多普勒 cine 帧为 (H,W,3),需转灰度对齐视频路径,
        # 否则 3 通道帧喂入 preprocess_image 会让模型前向形状报错。DICOM 像素为 RGB 序
        # (默认 channel_order="rgb"),与 OpenCV 视频帧的 BGR 区分。
        frames = [_to_gray(arr[i], channel_order="rgb") for i in idxs]

        # PHI 匿名化:与 io.load_input_image 共用 _anonymize_phi(标量匿名+序列删除),
        # 单一事实源,静/动口径一致(标签清单变更只需改 io.py 一处)。
        _anonymize_phi(ds)
        anon_path = os.path.join(output_dir, "anonymized_cine.dcm")
        ds.save_as(anon_path)
        print(f"--> Saved anonymized DICOM cine to {anon_path}")
        return frames, idxs, True, None, n_total
    except Exception as e:  # noqa: BLE001 — 守卫契约,失败返回错误信息
        return None, [], True, f"Error reading/anonymizing DICOM cine: {e}", 0

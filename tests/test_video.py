"""动态图/视频(cine)推理管线测试。

纯逻辑(抽帧采样 + 帧级聚合)详尽覆盖;视频解码做可跳过的往返测试
(编解码器缺失时 skip,不让 CI 因平台 codec 假失败)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.inference.video_aggregate import (
    uniform_sample_indices,
    aggregate_frame_predictions,
)
from src.inference.video_io import load_video_frames, _to_gray, _rescale_to_uint8


# ═════════════════════════════════════════════════════════════════════════════
# 抽帧采样:uniform_sample_indices
# ═════════════════════════════════════════════════════════════════════════════

class TestUniformSample:
    def test_fewer_than_max_returns_all(self):
        assert uniform_sample_indices(5, 32) == [0, 1, 2, 3, 4]

    def test_equal_returns_all(self):
        assert uniform_sample_indices(8, 8) == list(range(8))

    def test_downsample_includes_endpoints(self):
        idx = uniform_sample_indices(100, 10)
        assert idx[0] == 0 and idx[-1] == 99
        assert len(idx) == 10
        assert idx == sorted(idx) and len(set(idx)) == len(idx)  # 升序去重

    def test_max_frames_nonpositive_returns_all(self):
        assert uniform_sample_indices(50, 0) == list(range(50))
        assert uniform_sample_indices(50, -1) == list(range(50))

    def test_empty(self):
        assert uniform_sample_indices(0, 32) == []

    def test_no_index_out_of_range(self):
        for n in (3, 17, 64, 257):
            for m in (1, 5, 16, 32):
                idx = uniform_sample_indices(n, m)
                assert all(0 <= i < n for i in idx)
                assert len(idx) <= min(n, m) if m > 0 else len(idx) == n


# ═════════════════════════════════════════════════════════════════════════════
# 帧级聚合:aggregate_frame_predictions
# ═════════════════════════════════════════════════════════════════════════════

class TestAggregate:
    def test_empty_safe(self):
        agg = aggregate_frame_predictions([])
        assert agg["n_frames"] == 0
        assert agg["keyframe_index"] == -1
        assert agg["final_is_malignant"] == 0

    def test_max_drives_decision(self):
        # 均值<0.5 但有一帧>0.5 → 最终判恶性(癌症检测宁可召回)
        frames = [{"mal_prob": 0.1}, {"mal_prob": 0.2}, {"mal_prob": 0.92}]
        agg = aggregate_frame_predictions(frames)
        assert agg["final_is_malignant"] == 1
        assert abs(agg["mal_prob_max"] - 0.92) < 1e-9
        assert agg["mal_prob_mean"] < 0.5

    def test_keyframe_is_argmax(self):
        frames = [
            {"mal_prob": 0.3, "frame_index": 10},
            {"mal_prob": 0.8, "frame_index": 20},
            {"mal_prob": 0.5, "frame_index": 30},
        ]
        agg = aggregate_frame_predictions(frames)
        assert agg["keyframe_index"] == 1
        assert agg["keyframe_frame_index"] == 20

    def test_all_benign(self):
        frames = [{"mal_prob": 0.1}, {"mal_prob": 0.2}]
        agg = aggregate_frame_predictions(frames)
        assert agg["final_is_malignant"] == 0

    def test_keyframe_frame_index_fallback(self):
        # 无 frame_index 时回退为列表位置
        agg = aggregate_frame_predictions([{"mal_prob": 0.1}, {"mal_prob": 0.9}])
        assert agg["keyframe_frame_index"] == 1


# ═════════════════════════════════════════════════════════════════════════════
# 灰度转换工具
# ═════════════════════════════════════════════════════════════════════════════

class TestGrayUtils:
    def test_bgr_to_gray(self):
        bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        g = _to_gray(bgr)
        assert g.shape == (64, 64) and g.dtype == np.uint8

    def test_already_gray_passthrough(self):
        g0 = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        assert _to_gray(g0).shape == (32, 32)

    def test_rescale_float_to_uint8(self):
        arr = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)
        out = _rescale_to_uint8(arr)
        assert out.dtype == np.uint8
        assert out.min() == 0 and out.max() == 255


# ═════════════════════════════════════════════════════════════════════════════
# 单图 DICOM 彩色帧回归:彩色多普勒单帧 (H,W,3) 必须经 _to_gray 收敛为 2D,
# 否则 preprocess_image 产出 5 维张量令模型前向 `too many values to unpack` 崩溃。
# 守 io.py 单图路径与 video_io cine 路径的灰度口径对称性。
#
# 用 monkeypatch 伪造 pydicom 注入,只测「接线」(DICOM 分支是否经 _to_gray),
# 不依赖真实 pydicom 安装(本地无 pydicom 时仍真跑,不 skip;容器内 pydicom
# 在 requirements 中,行为一致)。
# ═════════════════════════════════════════════════════════════════════════════

class TestColorDicomSingleFrame:
    def test_color_single_frame_dicom_collapses_to_gray(self, tmp_path, monkeypatch):
        import src.inference.io as io_mod

        color = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)

        class _FakeDS:
            pixel_array = color

            def __contains__(self, tag):  # PHI 字段一律不存在 → 跳过匿名化赋值
                return False

            def save_as(self, path):  # 落匿名化副本,不真写盘内容
                open(path, "wb").close()

        class _FakePydicom:
            @staticmethod
            def dcmread(path):
                return _FakeDS()

        # 顶层 import 在本地无 pydicom 时未绑定 io_mod.pydicom,raising=False 兜底
        monkeypatch.setattr(io_mod, "pydicom", _FakePydicom, raising=False)
        monkeypatch.setattr(io_mod, "HAS_PYDICOM", True, raising=False)

        dcm_path = tmp_path / "color.dcm"
        dcm_path.write_bytes(b"placeholder")  # 仅需扩展名触发 DICOM 分支

        img, is_dicom, err = io_mod.load_input_image(str(dcm_path), str(tmp_path))
        assert err is None, f"彩色单帧 DICOM 读取失败: {err}"
        assert is_dicom is True
        assert img is not None and img.ndim == 2, "彩色单帧未收敛为 2D 灰度(下游会 5 维崩溃)"
        assert img.dtype == np.uint8


# ═════════════════════════════════════════════════════════════════════════════
# 视频解码往返(codec 缺失则跳过)+ 格式守卫
# ═════════════════════════════════════════════════════════════════════════════

class TestVideoIO:
    def test_unsupported_format_returns_error(self, tmp_path):
        bad = str(tmp_path / "x.txt")
        frames, idx, is_dcm, err = load_video_frames(bad, str(tmp_path))
        assert frames is None and "Unsupported" in err

    def test_video_roundtrip(self, tmp_path):
        import cv2

        vpath = str(tmp_path / "clip.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(vpath, fourcc, 10.0, (64, 64), isColor=True)
        if not writer.isOpened():
            pytest.skip("MJPG codec unavailable on this platform")
        n_written = 12
        for i in range(n_written):
            frame = np.full((64, 64, 3), i * 10, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        if not os.path.exists(vpath) or os.path.getsize(vpath) == 0:
            pytest.skip("video file not produced (codec issue)")

        frames, idx, is_dcm, err = load_video_frames(vpath, str(tmp_path), max_frames=5)
        if frames is None:
            pytest.skip(f"decode unavailable: {err}")
        assert err is None and is_dcm is False
        assert 0 < len(frames) <= 5
        assert all(f.ndim == 2 and f.dtype == np.uint8 for f in frames)
        assert len(idx) == len(frames)


# ═════════════════════════════════════════════════════════════════════════════
# Gatekeeper 去重库注入(视频逐帧复用,省重复播种)
# ═════════════════════════════════════════════════════════════════════════════

class TestDeduplicatorInjection:
    def test_injected_deduplicator_is_reused(self, monkeypatch):
        """传入预建 deduplicator 时,run_gatekeeper_audit 不得再新建。"""
        import src.inference.gatekeeper_runner as gr

        created = {"n": 0}
        real_ctor = gr.Deduplicator

        def counting_ctor(*a, **k):
            created["n"] += 1
            return real_ctor(*a, **k)

        monkeypatch.setattr(gr, "Deduplicator", counting_ctor)

        config = {
            "gatekeeper": {"phash_threshold": 1, "cosine_threshold": 0.995},
            "data": {"train_path": "____nonexistent_train____"},
        }
        dedup = gr.build_seeded_deduplicator(config, device="cpu")
        assert created["n"] == 1  # 仅 build 时构建一次

        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        # 注入复用:不应再新建 Deduplicator(计数仍为 1)
        gr.run_gatekeeper_audit(
            img, config, "cpu", "checkpoints/best_model.pth", ".", deduplicator=dedup)
        assert created["n"] == 1, "注入 deduplicator 时不应重复构建"

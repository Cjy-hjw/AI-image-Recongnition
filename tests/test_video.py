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

    def test_all_channel_shapes_collapse_to_2d(self):
        """杜绝任何通道形态漏成 3D(漏过会令 preprocess_image 5 维崩溃)。"""
        cases = [
            np.zeros((10, 12), np.uint8),            # (H,W)
            np.zeros((10, 12, 1), np.uint8),         # (H,W,1)
            np.zeros((10, 12, 3), np.uint8),         # (H,W,3)
            np.zeros((10, 12, 4), np.uint8),         # (H,W,4) RGBA
            np.zeros((3, 10, 12), np.uint8),         # (3,H,W) channel-first
            np.zeros((10, 12, 5), np.uint8),         # 多光谱兜底
        ]
        for arr in cases:
            out = _to_gray(arr)
            assert out.ndim == 2, f"{arr.shape} 未收敛 2D -> {out.shape}"
            assert out.dtype == np.uint8

    def test_channel_order_rgb_vs_bgr_differ(self):
        """RGB/BGR 加权对调:纯红帧 RGB(R=0.299)≈76,BGR(B权重)≈29,必须不同。"""
        red = np.zeros((8, 8, 3), np.uint8)
        red[:, :, 0] = 255  # 数组首通道置满
        g_rgb = int(_to_gray(red, channel_order="rgb")[0, 0])
        g_bgr = int(_to_gray(red, channel_order="bgr")[0, 0])
        assert g_rgb == 76 and g_bgr == 29, f"通道序未生效: rgb={g_rgb} bgr={g_bgr}"

    def test_16bit_grayscale_rescaled(self):
        """16-bit 灰度 DICOM 经 min-max 归一化到 8-bit,且仍为 2D。"""
        arr16 = (np.random.rand(10, 12) * 4095).astype(np.uint16)
        out = _to_gray(arr16)
        assert out.ndim == 2 and out.dtype == np.uint8


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
        frames, idx, is_dcm, err, n_total = load_video_frames(bad, str(tmp_path))
        assert frames is None and "Unsupported" in err
        assert n_total == 0, "失败分支 n_total_frames 必须为 0"

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

        frames, idx, is_dcm, err, n_total = load_video_frames(vpath, str(tmp_path), max_frames=5)
        if frames is None:
            pytest.skip(f"decode unavailable: {err}")
        assert err is None and is_dcm is False
        assert 0 < len(frames) <= 5
        assert all(f.ndim == 2 and f.dtype == np.uint8 for f in frames)
        assert len(idx) == len(frames)
        # 成功路径 n_total_frames 必须为正(原始视频总帧数,非抽样帧数)
        assert n_total > 0, "成功路径 n_total_frames 应为原始视频总帧数"


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

        # 本测试只关心 deduplicator 复用逻辑,与真实性检测无关——2026-07-10
        # Gate2审查修复后 AuthenticityModule.load_weights 在权重缺失时硬报错
        # (此前静默退化为随机初始化分类),本地/CI 无真实 checkpoints/authenticity_model.pth
        # 会导致内部自建 auth_module 报错。注入一个最小 stub 隔离掉这个无关依赖,
        # 保持测试范围聚焦在"deduplicator 是否被重复构建"这一个问题上。
        class _StubAuthModule:
            def predict(self, img_np):
                return 0, np.array([0.9, 0.05, 0.05], dtype=np.float32)

        # 注入复用:不应再新建 Deduplicator(计数仍为 1)
        gr.run_gatekeeper_audit(
            img, config, "cpu", "checkpoints/best_model.pth", ".",
            deduplicator=dedup, auth_module=_StubAuthModule())
        assert created["n"] == 1, "注入 deduplicator 时不应重复构建"


# ═════════════════════════════════════════════════════════════════════════════
# Gatekeeper 统一软 flag 制度(2026-07-12):authenticity 不再一票否决
# ═════════════════════════════════════════════════════════════════════════════

class TestGatekeeperSoftFlagPolicy:
    """authenticity 分类结果降级为软 flag(仅审计,不驱动 REJECTED)。

    背景:authenticity_trainer 的 phantom/synthetic 训练标签 100% 来自
    synthesis.py 程序化合成代理,真实域零 FPR 验证,argmax 无置信度门槛。
    统一软flag制度要求:任何硬拒绝分支在真实 holdout FPR 验收达标前默认软化。
    """

    @staticmethod
    def _config(train_path="____nonexistent_train____", enforcement_mode="hard"):
        # 本类专测「去重/篡改仍能硬拒绝、且 reason 文案精确」这条语义,故显式设
        # enforcement_mode=hard。**生产默认是 soft**(2026-07-28 赛道七豁免医保安全
        # 考核后硬拒绝成纯负期望),soft 侧行为见 TestEnforcementModeSoftDefault。
        return {
            "gatekeeper": {"phash_threshold": 1, "cosine_threshold": 0.995,
                           "enforcement_mode": enforcement_mode},
            "data": {"train_path": train_path},
        }

    class _StubAuth:
        """按构造参数返回固定 (is_phantom, probs)。"""
        def __init__(self, is_phantom):
            self._is_phantom = is_phantom

        def predict(self, img_np):
            probs = np.zeros(3, dtype=np.float32)
            probs[self._is_phantom] = 0.9
            return self._is_phantom, probs

    class _StubDedup:
        def is_duplicate(self, img_np):
            return False, None

    class _StubTamper:
        def detect(self, img_np):
            return False, np.zeros(img_np.shape[:2], dtype=np.uint8), None

    def _run(self, is_phantom, dedup=None, tamper=None, enforcement_mode="hard"):
        import src.inference.gatekeeper_runner as gr
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        return gr.run_gatekeeper_audit(
            img, self._config(enforcement_mode=enforcement_mode),
            "cpu", "checkpoints/best_model.pth", ".",
            deduplicator=dedup or self._StubDedup(),
            auth_module=self._StubAuth(is_phantom),
            tamper_detector=tamper or self._StubTamper(),
        )

    def test_phantom_classification_does_not_reject(self):
        passed, audit = self._run(is_phantom=1)
        assert passed is True, \
            "authenticity 判 phantom(1) 不应触发 REJECTED(已降级为软flag)"
        # 即便 enforcement_mode=hard,phantom 也不得进入硬拒绝依据
        assert audit["would_reject_in_hard_mode"] is False
        assert audit["is_phantom"] == 1 and audit["flagged"] is True

    def test_synthetic_classification_does_not_reject(self):
        passed, audit = self._run(is_phantom=2)
        assert passed is True, \
            "authenticity 判 synthetic(2) 不应触发 REJECTED(已降级为软flag)"
        assert audit["would_reject_in_hard_mode"] is False
        assert audit["is_gan"] == 1 and audit["flagged"] is True

    def test_duplicate_still_hard_rejects(self):
        class _DupHit:
            def is_duplicate(self, img_np):
                return True, "phash distance=0"
        passed, rejection = self._run(is_phantom=0, dedup=_DupHit())
        assert passed is False and rejection is not None
        assert rejection["status"] == "REJECTED"
        assert rejection["is_duplicate"] == 1
        assert "Duplicate" in rejection["reason"]

    def test_tamper_still_hard_rejects(self):
        class _TamperHit:
            def detect(self, img_np):
                return True, np.zeros(img_np.shape[:2], dtype=np.uint8), [0, 0, 10, 10]
        passed, rejection = self._run(is_phantom=0, tamper=_TamperHit())
        assert passed is False and rejection is not None
        assert rejection["status"] == "REJECTED"
        assert rejection["is_tampered"] == 1

    def test_tamper_reason_falls_back_to_generic_text_without_source_attr(self):
        """2026-07-12 code-review(Angle A)发现:REJECTED reason 曾坍缩成笼统的
        'hard-reject path',丢失具体命中源(CatNet vs BAG)。修复后
        gatekeeper_runner 用 `getattr(tamper_detector, "last_hard_reject_source", None)`
        取来源——本测试锁定向后兼容分支:stub/旧版 tamper_detector 没有该属性时,
        不应报 AttributeError,应静默退化为通用文案。"""
        class _TamperHitNoSourceAttr:
            def detect(self, img_np):
                return True, np.zeros(img_np.shape[:2], dtype=np.uint8), [0, 0, 10, 10]
        passed, rejection = self._run(is_phantom=0, tamper=_TamperHitNoSourceAttr())
        assert passed is False and rejection is not None
        assert "Tampering detected (source unspecified)" in rejection["reason"]

    def test_tamper_reason_labels_catnet_source(self):
        class _TamperHitCatNet:
            last_hard_reject_source = "catnet"
            def detect(self, img_np):
                return True, np.zeros(img_np.shape[:2], dtype=np.uint8), [0, 0, 10, 10]
        passed, rejection = self._run(is_phantom=0, tamper=_TamperHitCatNet())
        assert passed is False and rejection is not None
        assert "CAT-Net deep model" in rejection["reason"]

    def test_tamper_reason_labels_bag_phase_source(self):
        class _TamperHitBag:
            last_hard_reject_source = "bag_phase"
            def detect(self, img_np):
                return True, np.zeros(img_np.shape[:2], dtype=np.uint8), [0, 0, 10, 10]
        passed, rejection = self._run(is_phantom=0, tamper=_TamperHitBag())
        assert passed is False and rejection is not None
        assert "BAG phase inconsistency" in rejection["reason"]

    def test_rejection_json_still_carries_phantom_audit_fields(self):
        """软化后 is_phantom/is_gan 仍作为审计字段保留在 REJECTED 产物里
        (由 duplicate 命中触发 REJECTED,同时 phantom=1,验证字段未被误删)。

        2026-07-12 code-review(Angle B)发现:早期版本只验证字段保留、断言
        phantom 完全不出现在人类可读的 reason 文本里——但这会让"是否恰好还有
        另一路真实触发的软flag信号并发发生"这条审计信息从 reason 里彻底消失
        (机器可读字段虽还在,人工审计/日志grep这类只看reason文本的场景看不到)。
        改为断言 phantom 以明确标注为"soft-flag"(非拒绝驱动原因)的形式出现,
        而不是完全不出现——reason 应能区分"是什么导致了REJECTED"(duplicate)
        与"还观察到了什么但不是拒绝原因"(phantom soft-flag)。
        """
        class _DupHit:
            def is_duplicate(self, img_np):
                return True, "near-duplicate"
        passed, rejection = self._run(is_phantom=1, dedup=_DupHit())
        assert passed is False and rejection is not None
        assert rejection["is_phantom"] == 1
        assert "soft-flag: Physical Phantom" in rejection["reason"], \
            "phantom 应以 soft-flag 标注形式出现在 reason 里(区别于拒绝驱动原因)"
        assert rejection["reason"].startswith("Duplicate image:"), \
            "REJECTED 的真正驱动原因(duplicate)应仍是 reason 的主要/首要表述"


# ═════════════════════════════════════════════════════════════════════════════
# 篡改软 flag 接入 audit(2026-07-29):copy_move/splice/BAG 信号不再只走 stdout
# ═════════════════════════════════════════════════════════════════════════════

class TestSoftFlagsLandInAudit:
    """`CatNetTamperingDetector.last_soft_flags` 必须落进 audit 的 `soft_flags` 键。

    背景(2026-07-29 /code-review):此前生产代码零读取 `last_soft_flags`,
    于是仅被 copy_move/splice/BAG(三者均为软判据,恒不进 is_tampered)命中的图,
    `audit["flagged"]` 为 False、result.json 完全沉默——而 copy_move 在 359 张
    真实图上误报 27.3%,信号量不小却全丢。同时锁死红线:软 flag 只计入
    `flagged`,**绝不**进 `would_reject_in_hard_mode`(升级为拒绝依据会误杀真实影像)。
    """

    @staticmethod
    def _config(enforcement_mode="soft"):
        return {
            "gatekeeper": {"phash_threshold": 1, "cosine_threshold": 0.995,
                           "enforcement_mode": enforcement_mode},
            "data": {"train_path": "____nonexistent_train____"},
        }

    class _StubAuth:
        def predict(self, img_np):
            return 0, np.ones(3, dtype=np.float32) / 3

    class _StubDedup:
        def is_duplicate(self, img_np):
            return False, None

    class _StubTamperWithSoftFlags:
        """detect 返回 is_tampered=False,但 last_soft_flags 有命中(软判据)。"""
        last_hard_reject_source = None

        def __init__(self, flags):
            self.last_soft_flags = flags

        def detect(self, img_np):
            return False, np.zeros(img_np.shape[:2], dtype=np.uint8), None

    def _run(self, flags, mode="soft"):
        import src.inference.gatekeeper_runner as gr
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        return gr.run_gatekeeper_audit(
            img, self._config(mode), "cpu", "checkpoints/best_model.pth", ".",
            deduplicator=self._StubDedup(),
            auth_module=self._StubAuth(),
            tamper_detector=self._StubTamperWithSoftFlags(flags),
        )

    def test_soft_flags_in_audit_and_flagged_but_not_hard_reject(self):
        passed, audit = self._run(
            [("copy_move", [5, 5, 40, 40]), ("splice", [0, 0, 10, 10])])
        assert passed is True, "仅软 flag 命中不得拒绝"
        assert audit["flagged"] is True, \
            "仅软 flag 命中时 flagged 必须为 True——此前为 False,信号全丢"
        assert audit["would_reject_in_hard_mode"] is False, \
            "红线:软 flag 恒不得计入硬拒绝依据(copy_move 真实图误报 27.3%)"
        assert audit["soft_flags"] == [
            {"type": "copy_move", "bbox": [5, 5, 40, 40]},
            {"type": "splice", "bbox": [0, 0, 10, 10]},
        ]
        assert "soft-flag: copy_move" in audit["reason"], \
            "软 flag 必须以标注形式进 reason,供只看文本的日志grep/人工审计"

    def test_soft_flags_do_not_reject_even_in_hard_mode(self):
        """hard 模式下软 flag 仍不得驱动 REJECTED(与 phantom/synthetic 同一政策)。"""
        passed, audit = self._run([("bag_phase", None)], mode="hard")
        assert passed is True and audit["would_reject_in_hard_mode"] is False
        assert audit["soft_flags"] == [{"type": "bag_phase", "bbox": None}]

    def test_no_flags_yields_empty_list_and_unflagged(self):
        passed, audit = self._run([])
        assert passed is True and audit["flagged"] is False
        assert audit["soft_flags"] == []

    def test_collect_soft_flags_backward_compatible_without_attr(self):
        """旧 detector 无 last_soft_flags 属性 → getattr 兜 [],不得 AttributeError。"""
        import src.inference.gatekeeper_runner as gr

        class _LegacyTamper:
            def detect(self, img_np):
                return False, np.zeros(img_np.shape[:2], dtype=np.uint8), None

        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        passed, audit = gr.run_gatekeeper_audit(
            img, self._config(), "cpu", "checkpoints/best_model.pth", ".",
            deduplicator=self._StubDedup(), auth_module=self._StubAuth(),
            tamper_detector=_LegacyTamper())
        assert passed is True and audit["soft_flags"] == []

    def test_collect_soft_flags_normalizes_unexpected_shapes(self):
        """last_soft_flags 若混入畸形元素(非二元组),不得让整条 audit 崩掉。"""
        import src.inference.gatekeeper_runner as gr
        assert gr.collect_soft_flags(None) == []
        assert gr.collect_soft_flags(object()) == []  # 无该属性外的"假值"兜底
        out = gr.collect_soft_flags(type("T", (), {
            "last_soft_flags": [("copy_move", [1, 2, 3, 4]), "weird-string"]})())
        assert out[0] == {"type": "copy_move", "bbox": [1, 2, 3, 4]}
        assert out[1]["type"] == "weird-string" and out[1]["bbox"] is None

    def test_cosine_threshold_malformed_config_uses_delivery_default(self):
        """畸形配置下 cosine_threshold 应落到**交付值**而非更宽松的库默认。

        若缺省传 None,Deduplicator 会取类内推荐值 resnet18=0.95,比交付的 0.995
        宽松——等于畸形配置静默放松去重判据(误报变多,踩宁可漏检不可误杀红线)。
        """
        import src.inference.gatekeeper_runner as gr
        d = gr.build_seeded_deduplicator({}, device="cpu")  # 整段缺失
        assert abs(d.cosine_threshold - 0.995) < 1e-9, \
            f"resnet18 后端的畸形配置默认须为交付值 0.995,实为 {d.cosine_threshold}"

    def test_value_level_malformed_thresholds_degrade_not_crash(self):
        """值级畸形(段是 dict 但单个阈值写歪)须降级交付默认,不得穿透成整例 ERROR。

        2026-07-30 /code-review Angle C:`gatekeeper_cfg()` 只兜「段是不是 dict」,
        兜不住段内**值**写歪。运维手改 yaml 把 `cosine_threshold: "high"`(字符串)、
        `phash_threshold: "x"` 写进交付配置,`float("high")`/`int("x")` 会穿透到
        Deduplicator/int() 抛 ValueError,被入口兜成整例 ERROR = 零诊断——正是软化
        要避免的后果。值非法时降级交付默认,而不是抛异常。
        """
        import src.inference.gatekeeper_runner as gr
        for bad_cos in ("high", ["x"], object()):
            cfg = {"gatekeeper": {"cosine_threshold": bad_cos}}
            d = gr.build_seeded_deduplicator(cfg, device="cpu")  # 不抛即达标
            assert abs(d.cosine_threshold - 0.995) < 1e-9, \
                f"cosine_threshold={bad_cos!r} 应降级交付默认 0.995"
        for bad_ph in ("x", None, [1]):
            cfg = {"gatekeeper": {"phash_threshold": bad_ph}}
            d = gr.build_seeded_deduplicator(cfg, device="cpu")  # 不抛即达标
            assert d.phash_threshold == 1, \
                f"phash_threshold={bad_ph!r} 应降级交付默认 1"
        # 合法标量字符串("0.9")可被 float() 解析,属合法值,不得误降级
        cfg = {"gatekeeper": {"cosine_threshold": "0.9"}}
        d = gr.build_seeded_deduplicator(cfg, device="cpu")
        assert abs(d.cosine_threshold - 0.9) < 1e-9, \
            "可解析的标量字符串是合法值,不得误当畸形降级"

    def test_collect_soft_flags_drops_flag_with_malformed_bbox_elements(self):
        """bbox 元素畸形(含 None/str/NaN)的整条 flag 须丢弃,不得让整例审计崩。

        2026-07-30 /code-review Angle A:外层 try 只兜「非二元组」,二元组里 bbox
        为 [None, 10, 20, 30] 时 `int(None)` 抛 TypeError 会逃逸出函数,被入口兜成
        整例 ERROR——软 flag 是纯审计字段,它的畸形绝不能让整例零诊断。
        """
        import src.inference.gatekeeper_runner as gr
        tamper = type("T", (), {"last_soft_flags": [
            ("copy_move", [None, 10, 20, 30]),      # bbox 元素含 None → 丢整条
            ("splice", ["a", "b", "c", "d"]),        # bbox 元素是 str → 丢整条
            ("bag_phase", [1, 2, 3, 4]),             # 正常 → 保留
        ]})()
        out = gr.collect_soft_flags(tamper)
        assert out == [{"type": "bag_phase", "bbox": [1, 2, 3, 4]}], \
            "bbox 元素畸形的整条 flag 应被丢弃,只保留正常的;不得抛异常"


# ═════════════════════════════════════════════════════════════════════════════
# enforcement_mode:默认 soft — 审计命中不再中断诊断(2026-07-28 豁免后修正)
# ═════════════════════════════════════════════════════════════════════════════

class TestEnforcementModeSoftDefault:
    """赛道七豁免全部医保安全考核后,硬拒绝成纯负期望:gatekeeper 零计分,
    而 REJECTED 会让该病例**不输出任何诊断**,白丢唯一的分数来源(官方明确
    "真实病例被误判为伪造,诊断结果仍参与目标三四五计分",前提是先输出诊断)。
    故默认 `enforcement_mode: soft`——命中只记 flag,诊断照出。
    """

    @staticmethod
    def _run(config_gk, dedup=None, tamper=None, is_phantom=0):
        import src.inference.gatekeeper_runner as gr

        class _StubAuth:
            def predict(self, img_np):
                probs = np.zeros(3, dtype=np.float32)
                probs[is_phantom] = 0.9
                return is_phantom, probs

        class _StubDedup:
            def is_duplicate(self, img_np):
                return False, None

        class _StubTamper:
            def detect(self, img_np):
                return False, np.zeros(img_np.shape[:2], dtype=np.uint8), None

        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        config = {"gatekeeper": config_gk, "data": {"train_path": "____nonexistent____"}}
        return gr.run_gatekeeper_audit(
            img, config, "cpu", "checkpoints/best_model.pth", ".",
            deduplicator=dedup or _StubDedup(),
            auth_module=_StubAuth(),
            tamper_detector=tamper or _StubTamper())

    class _DupHit:
        def is_duplicate(self, img_np):
            return True, "phash distance=0"

    class _TamperHit:
        def detect(self, img_np):
            return True, np.zeros(img_np.shape[:2], dtype=np.uint8), [0, 0, 10, 10]

    def test_duplicate_does_not_reject_by_default(self):
        """配置未写 enforcement_mode → 默认 soft → 命中去重仍 passed=True。"""
        passed, audit = self._run({}, dedup=self._DupHit())
        assert passed is True, "默认 soft 模式下命中去重不得中断诊断"
        assert audit["enforcement_mode"] == "soft"
        assert audit["is_duplicate"] == 1
        assert audit["flagged"] is True
        assert audit["would_reject_in_hard_mode"] is True, \
            "审计事实必须保留:soft 只是不执行拒绝,不是假装没命中"
        assert "status" not in audit, "soft 模式产物不得带 REJECTED 状态"

    def test_tamper_does_not_reject_by_default(self):
        passed, audit = self._run({}, tamper=self._TamperHit())
        assert passed is True and audit["is_tampered"] == 1
        assert audit["would_reject_in_hard_mode"] is True

    def test_hard_mode_still_rejects(self):
        """hard 模式仍保留一票否决能力(未来规则变化/回归可测)。"""
        passed, rejection = self._run(
            {"enforcement_mode": "hard"}, dedup=self._DupHit())
        assert passed is False and rejection["status"] == "REJECTED"

    def test_clean_image_audit_is_not_flagged(self):
        passed, audit = self._run({})
        assert passed is True
        assert audit["flagged"] is False and audit["would_reject_in_hard_mode"] is False
        assert audit["reason"] == ""

    def test_invalid_mode_degrades_to_soft(self):
        """拼写错误不得静默启用硬拒绝——失败方向必须偏向"多输出一份诊断"。"""
        passed, audit = self._run(
            {"enforcement_mode": "HARD_typo"}, dedup=self._DupHit())
        assert passed is True and audit["enforcement_mode"] == "soft"

    def test_hard_mode_is_case_insensitive(self):
        passed, rejection = self._run(
            {"enforcement_mode": " Hard "}, dedup=self._DupHit())
        assert passed is False and rejection["status"] == "REJECTED"

    def test_shipped_config_defaults_to_soft(self):
        """交付配置 configs/thyroid_config.yaml 必须是 soft(防止改回 hard 后忘了)。"""
        import yaml
        from pathlib import Path
        from src.inference.gatekeeper_runner import resolve_enforcement_mode
        # 用 __file__ 定位仓库根,不依赖 cwd(Gate2 审查 P2-4)
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "thyroid_config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert resolve_enforcement_mode(cfg) == "soft"

    def test_malformed_gatekeeper_section_degrades_to_soft(self):
        """`gatekeeper` 段写歪(YAML 少一层缩进 → 标量/列表)不得抛异常。

        抛异常会被入口 except 兜成 ERROR result.json = 整例零诊断,恰是软化要避免
        的后果(Gate2 审查 P2-1)。
        """
        from src.inference.gatekeeper_runner import resolve_enforcement_mode
        for bad in ("hard", ["hard"], 3, True, None):
            assert resolve_enforcement_mode({"gatekeeper": bad}) == "soft", \
                f"gatekeeper={bad!r} 应降级 soft 而非抛异常/误判 hard"
        assert resolve_enforcement_mode({}) == "soft"

    @pytest.mark.parametrize("bad_cfg", [
        {},                                    # 整段缺失
        {"gatekeeper": None},                  # gatekeeper:
        {"gatekeeper": "hard"},                # gatekeeper: hard(少一层缩进)
        {"gatekeeper": ["hard"]},              # 写成列表
    ], ids=["missing", "none", "scalar", "list"])
    def test_malformed_config_does_not_break_builders(self, bad_cfg):
        """降级必须贯穿**整条**读取路径,不能只在解析模式那一处兜底。

        2026-07-29 `/code-review` 实测复现的真实缺口:`resolve_enforcement_mode`
        当时已 isinstance 兜底、正确返回 soft,但紧接着的
        `build_seeded_deduplicator`(KeyError/TypeError)与 `build_tamper_detector`
        (AttributeError)仍直接 `config["gatekeeper"][...]` —— 三种畸形写法都在
        "模式已降级为 soft"**之后**崩,入口兜成 status=ERROR,整例零诊断。
        现统一走 `gatekeeper_cfg()`。
        """
        import src.inference.gatekeeper_runner as gr
        gr.build_seeded_deduplicator(bad_cfg, device="cpu")   # 不抛即达标
        gr.build_tamper_detector(bad_cfg)
        assert gr.gatekeeper_cfg(bad_cfg) == {}

    def test_gatekeeper_cfg_passes_through_valid_dict(self):
        """兜底不得吞掉合法配置——阈值必须照常透传。"""
        import src.inference.gatekeeper_runner as gr
        cfg = {"gatekeeper": {"phash_threshold": 7, "bag_hard_reject": True}}
        assert gr.gatekeeper_cfg(cfg)["phash_threshold"] == 7
        assert gr.build_tamper_detector(cfg).bag_hard_reject is True

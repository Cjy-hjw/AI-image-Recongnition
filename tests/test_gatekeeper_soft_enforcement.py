"""gatekeeper 软化(enforcement_mode)端到端回归:命中审计仍须输出完整诊断。

依据(2026-07-28 官方赛前动员会):
- 赛道七**豁免全部医保安全考核**,总分 = 临床诊断得分 Sclin × 100 —— gatekeeper
  零计分贡献;
- "真实病例被误判为伪造:诊断结果**仍参与**目标三四五计分" —— 前提是我们必须
  先把诊断输出出来。

两条合起来:硬拒绝在本赛道是**纯负期望**。拒对了不加分,误拒一次即白丢该病例
全部临床得分。动态图路径尤甚——旧逻辑逐帧审计、任一帧命中即整段 `return`,
32 帧里一次误报就让整段视频零诊断。

本文件在**入口脚本层**(而非 gatekeeper 内部)锁定修复:审计命中后 result.json
必须仍带完整诊断字段与概率分布,同时如实记录 flag;hard 模式仍能拒绝。
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DIAGNOSIS_FIELDS = (
    "malignancy_probability", "is_malignant", "tirads_points", "tirads_level",
    "risk_percentage", "concept_probabilities", "malignancy_probabilities",
)

# 一份"去重命中 + phantom 软flag"的审计结论(soft 模式下 passed=True)
_FLAGGED_AUDIT = {
    "flagged": True,
    "would_reject_in_hard_mode": True,
    "enforcement_mode": "soft",
    "reason": "Duplicate image: phash distance=0, (soft-flag: Physical Phantom suspected)",
    "is_phantom": 1, "is_gan": 0, "is_duplicate": 1, "is_tampered": 0,
}


def _write_noise_png(path, h=128, w=160):
    import cv2
    cv2.imwrite(str(path), (np.random.rand(h, w) * 255).astype(np.uint8))


def _assert_full_diagnosis(data):
    assert data["status"] == "PASSED", f"审计命中不得中断诊断输出: {data}"
    for field in _DIAGNOSIS_FIELDS:
        assert field in data, f"缺诊断字段 {field}(官方要求每例输出全部维度)"
    assert abs(sum(data["malignancy_probabilities"]) - 1.0) < 1e-5


class TestPredictSoftFlagStillDiagnoses:
    """静态图路径:predict.py 审计命中后仍走完分割/分类/报告/导出。"""

    @staticmethod
    def _run(tmp_path, monkeypatch, audit_result):
        import predict as predict_mod

        img = tmp_path / "in.png"
        _write_noise_png(img)
        out = tmp_path / "out"
        monkeypatch.setattr(
            predict_mod, "run_gatekeeper_audit", lambda *a, **k: audit_result)
        monkeypatch.setattr(sys, "argv", [
            "predict.py", "--input_image", str(img), "--output_dir", str(out),
            "--precision", "fp32",
            # 权重缺失时 build_model 降级随机初始化 + warning(不报错),本测试只验
            # 接线连通与产物 schema,不验精度
            "--weights", "checkpoints/__absent_for_test__.pth",
        ])
        predict_mod.main()
        return json.loads((out / "result.json").read_text(encoding="utf-8"))

    def test_flagged_audit_still_emits_full_diagnosis(self, tmp_path, monkeypatch):
        data = self._run(tmp_path, monkeypatch, (True, _FLAGGED_AUDIT))
        _assert_full_diagnosis(data)
        gk = data["gatekeeper"]
        assert gk["flagged"] is True and gk["is_duplicate"] == 1, \
            "命中事实必须如实记录,不得因为不拒绝就抹掉审计信息"
        assert gk["would_reject_in_hard_mode"] is True
        assert gk["skipped"] is False, \
            "静态图侧也须带 skipped 键,让消费方与动态图共用同一份解析代码"

    def test_clean_audit_records_gatekeeper_section(self, tmp_path, monkeypatch):
        clean = {"flagged": False, "would_reject_in_hard_mode": False,
                 "enforcement_mode": "soft", "reason": "",
                 "is_phantom": 0, "is_gan": 0, "is_duplicate": 0, "is_tampered": 0}
        data = self._run(tmp_path, monkeypatch, (True, clean))
        _assert_full_diagnosis(data)
        assert data["gatekeeper"]["flagged"] is False

    def test_legacy_foci_key_stays_consistent_with_new_export(self, tmp_path, monkeypatch):
        """`foci_probabilities`(旧键,下游依赖)与 `concept_probabilities.foci` 必须同值。

        二者现在同源(都取 `preds["foci_prob"].tolist()`),但若有人只改一侧会静默分叉——
        评测方按新 schema 读、旧下游按旧键读,两边拿到不同数字(Gate2 审查 P2-6)。
        """
        data = self._run(tmp_path, monkeypatch, (True, _FLAGGED_AUDIT))
        assert data["foci_probabilities"] == data["concept_probabilities"]["foci"]
        assert data["malignancy_probability"] == data["malignancy_probabilities"][1], \
            "标量恶性概率必须等于分布的恶性分量"

    def test_hard_mode_rejection_path_intact(self, tmp_path, monkeypatch):
        """hard 模式(passed=False)仍写 REJECTED 且不产出诊断——能力未被删除。"""
        rejection = {**_FLAGGED_AUDIT, "enforcement_mode": "hard", "status": "REJECTED"}
        data = self._run(tmp_path, monkeypatch, (False, rejection))
        assert data["status"] == "REJECTED"
        assert "malignancy_probability" not in data


class TestVideoSoftFlagStillDiagnoses:
    """动态图路径:单帧命中不得再让整段视频零诊断(本轮最危险的那处 return)。"""

    N_FRAMES = 3
    FLAGGED_POS = 1

    @staticmethod
    def _stub_heavy_objects(monkeypatch, video_mod):
        """三个重型 gatekeeper 对象在本地无权重时会硬报错,注入 stub 隔离。"""
        monkeypatch.setattr(video_mod, "build_seeded_deduplicator", lambda *a, **k: object())
        monkeypatch.setattr(video_mod, "build_auth_module", lambda *a, **k: object())
        monkeypatch.setattr(video_mod, "build_tamper_detector", lambda *a, **k: object())

    def _run(self, tmp_path, monkeypatch, audit_for_frame):
        import video_predict as video_mod

        frames = [(np.random.rand(96, 96) * 255).astype(np.uint8)
                  for _ in range(self.N_FRAMES)]
        indices = list(range(self.N_FRAMES))
        monkeypatch.setattr(
            video_mod, "load_video_frames",
            lambda *a, **k: (frames, indices, False, None))
        self._stub_heavy_objects(monkeypatch, video_mod)

        calls = {"n": 0}

        def _fake_audit(*a, **k):
            pos = calls["n"]
            calls["n"] += 1
            return audit_for_frame(pos)

        monkeypatch.setattr(video_mod, "run_gatekeeper_audit", _fake_audit)

        out = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", [
            "video_predict.py", "--input_video", str(tmp_path / "fake.mp4"),
            "--output_dir", str(out), "--precision", "fp32",
            "--weights", "checkpoints/__absent_for_test__.pth",
        ])
        video_mod.main()
        return json.loads((out / "result.json").read_text(encoding="utf-8")), calls

    def test_single_flagged_frame_does_not_kill_whole_video(self, tmp_path, monkeypatch):
        clean = {"flagged": False, "would_reject_in_hard_mode": False,
                 "enforcement_mode": "soft", "reason": "",
                 "is_phantom": 0, "is_gan": 0, "is_duplicate": 0, "is_tampered": 0}

        def per_frame(pos):
            return (True, dict(_FLAGGED_AUDIT) if pos == self.FLAGGED_POS else dict(clean))

        data, calls = self._run(tmp_path, monkeypatch, per_frame)
        _assert_full_diagnosis(data)
        assert calls["n"] == self.N_FRAMES, \
            "命中帧之后的帧仍须继续审计(旧逻辑在此 return,后续帧根本不跑)"
        gk = data["gatekeeper"]
        assert gk["skipped"] is False
        assert gk["n_frames_audited"] == self.N_FRAMES
        assert gk["flagged_frame_indices"] == [self.FLAGGED_POS]
        assert gk["n_frames_flagged"] == 1
        assert gk["any_would_reject_in_hard_mode"] is True
        # 关键帧是否恰好是被 flag 的那帧,必须显式落字段(报告/mask 全出自该帧)
        assert gk["keyframe_flagged"] == (
            data["keyframe_frame_index"] in gk["flagged_frame_indices"])
        assert len(data["per_frame_malignancy"]) == self.N_FRAMES

    def test_hard_mode_first_failing_frame_still_rejects_whole_video(
            self, tmp_path, monkeypatch):
        rejection = {**_FLAGGED_AUDIT, "enforcement_mode": "hard", "status": "REJECTED"}

        def per_frame(pos):
            if pos == self.FLAGGED_POS:
                return (False, dict(rejection))
            return (True, {"flagged": False, "would_reject_in_hard_mode": False,
                           "enforcement_mode": "hard", "reason": "",
                           "is_phantom": 0, "is_gan": 0,
                           "is_duplicate": 0, "is_tampered": 0})

        data, calls = self._run(tmp_path, monkeypatch, per_frame)
        assert data["status"] == "REJECTED"
        assert data["rejected_frame_index"] == self.FLAGGED_POS
        assert calls["n"] == self.FLAGGED_POS + 1, "hard 模式命中即停,不再审计后续帧"

    def test_skip_gatekeeper_marks_audit_as_skipped(self, tmp_path, monkeypatch):
        import video_predict as video_mod

        frames = [(np.random.rand(96, 96) * 255).astype(np.uint8) for _ in range(2)]
        monkeypatch.setattr(
            video_mod, "load_video_frames", lambda *a, **k: (frames, [0, 1], False, None))
        out = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", [
            "video_predict.py", "--input_video", str(tmp_path / "fake.mp4"),
            "--output_dir", str(out), "--precision", "fp32", "--skip_gatekeeper",
            "--weights", "checkpoints/__absent_for_test__.pth",
        ])
        video_mod.main()
        data = json.loads((out / "result.json").read_text(encoding="utf-8"))
        _assert_full_diagnosis(data)
        assert data["gatekeeper"] == {"skipped": True}

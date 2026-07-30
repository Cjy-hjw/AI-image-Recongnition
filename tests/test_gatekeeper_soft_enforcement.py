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


# ═════════════════════════════════════════════════════════════════════════════
# 坏帧隔离(2026-07-29):单帧数值异常不得拖垮整段
# ═════════════════════════════════════════════════════════════════════════════

class TestBadFrameIsolation:
    """`decode_predictions` 的 `_assert_finite` 抛 ValueError 时,只废该帧。

    这是**本轮自引入回归**的回归测试:Phase 1 给全部分类头加了 NaN/Inf 硬校验,
    但 `video_predict.py` 的逐帧循环当时无 try/except,异常直接冒泡到最外层
    `except Exception` → 整段 32 帧全废、status=ERROR。性质与 Phase 2 花大力气修的
    「一帧 gatekeeper 命中拖垮整段」完全相同,只是触发源从"审计拒绝"换成"数值校验"。

    同时锁死修复过程中最容易踩的**静默错位**陷阱:旧实现把 img_tensor 存进以
    **全帧下标**为键的平行 dict `key_tensors`,而 `agg["keyframe_index"]` 是
    **per_frame** 的下标。一旦跳帧,两个下标空间错位,而 `key_tensors[kf_pos]`
    **永远取得到值、不抛 KeyError** —— mask/Grad-CAM 静默出自错帧,JSON 里的
    keyframe_frame_index 却仍是对的,产出自相矛盾且零报错。故修法是把 tensor 存进
    per_frame 元素本身(下标空间只剩一个),而不是"记得同步改两处"。
    """

    @staticmethod
    def _run(tmp_path, monkeypatch, n_frames, bad_positions, mal_probs_by_pos=None,
             spy_keyframe_np=None):
        """跑 video_predict.main(),让 `bad_positions` 里的帧抛 ValueError。

        `mal_probs_by_pos`: {list_pos: p_malignant},用于指定哪一帧最恶性(=关键帧)。
        `spy_keyframe_np`: 传入一个 dict 时,启用 --enable_gradcam 并 stub 掉
            maybe_save_gradcam,把**实际被当作关键帧的 frame_np 指纹**写进
            `spy_keyframe_np["fingerprint"]`——这是唯一直接观测"mask/Grad-CAM 出自
            哪一帧"的途径;只断言 JSON 里的 keyframe_frame_index 对错位陷阱是盲的
            (错位时该字段仍是对的,矛盾恰恰在产物与字段之间)。
        """
        import video_predict as video_mod

        # 每帧填成不同的常数灰度值,作为"这一帧的身份指纹"——被跳过的帧不参与
        # 诊断,而 _fake_diagnose 靠指纹反查自己拿到的是第几帧,无需依赖调用顺序。
        # 0 号帧填 10、1 号填 20 ...,避开 0/255 以免与空白/饱和图混淆。
        frames = [np.full((96, 96), (i + 1) * 10, dtype=np.uint8) for i in range(n_frames)]
        indices = list(range(n_frames))
        monkeypatch.setattr(
            video_mod, "load_video_frames",
            lambda *a, **k: (frames, indices, False, None))

        real_diagnose = video_mod.diagnose_frame
        from src.inference.postprocess import NonFiniteProbabilityError

        def _fake_diagnose(model, frame_np, precision, device, img_size, calibrator=None):
            pos = int(frame_np[0, 0]) // 10 - 1
            if pos in bad_positions:
                raise NonFiniteProbabilityError(
                    f"decode_predictions: malignancy 概率含 NaN/Inf (test frame {pos})")
            preds, img_tensor = real_diagnose(
                model, frame_np, precision, device, img_size, calibrator=calibrator)
            if mal_probs_by_pos is not None and pos in mal_probs_by_pos:
                # 必须复刻生产代码的同源关系 `mal_prob = float(mal_probs[1])`
                # (postprocess.py:78)。若这里各自独立赋值,float32 舍入会让二者
                # 差 1e-8,把测试自己变成假阳性——而"同源"恰是本测试要锁的不变量。
                probs = np.array([1.0 - mal_probs_by_pos[pos], mal_probs_by_pos[pos]],
                                 dtype=np.float32)
                preds["mal_probs"] = probs
                preds["mal_prob"] = float(probs[1])
            return preds, img_tensor

        monkeypatch.setattr(video_mod, "diagnose_frame", _fake_diagnose)

        argv = [
            "video_predict.py", "--input_video", str(tmp_path / "fake.mp4"),
            "--output_dir", str(tmp_path / "out"), "--precision", "fp32",
            "--skip_gatekeeper",
            "--weights", "checkpoints/__absent_for_test__.pth",
        ]
        if spy_keyframe_np is not None:
            # 探关键帧实际取的是哪一帧:Grad-CAM 的 img_np 就是关键帧原图(修复后
            # 与 mask 同源、都出自 per_frame[kf_pos])。stub 掉真实实现,避免走
            # Grad-CAM 的 hook/反向传播,只截获入参。
            argv.append("--enable_gradcam")

            def _spy_gradcam(model, img_tensor, img_np, output_dir, **kw):
                spy_keyframe_np["fingerprint"] = int(img_np[0, 0])
                return 1, "stub.png"

            monkeypatch.setattr(video_mod, "maybe_save_gradcam", _spy_gradcam)

        out = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", argv)
        video_mod.main()
        return json.loads((out / "result.json").read_text(encoding="utf-8"))

    def test_single_bad_frame_does_not_kill_whole_video(self, tmp_path, monkeypatch):
        data = self._run(tmp_path, monkeypatch, n_frames=3, bad_positions={1})
        _assert_full_diagnosis(data)
        assert data["n_frames_sampled"] == 3, "抽样帧数须如实为 3"
        assert data["n_frames_diagnosed"] == 2, "成功诊断 2 帧"
        assert data["n_frames_failed"] == 1
        assert data["failed_frame_indices"] == [1]
        assert len(data["per_frame_malignancy"]) == 2, \
            "per_frame 只含成功帧,不得给坏帧编一份假概率"
        assert data["failed_frames"][0]["frame_index"] == 1
        assert "NaN/Inf" in data["failed_frames"][0]["reason"]

    def test_keyframe_not_misaligned_after_skipping(self, tmp_path, monkeypatch):
        """**核心错位回归**:第 0 帧坏、第 2 帧最恶性 → 关键帧必须是原始帧 2。

        旧的 key_tensors 平行 dict 实现下:per_frame=[帧1, 帧2],最恶性在
        per_frame 位置 1,而 key_tensors 的键是 {1: 帧1, 2: 帧2} —— `key_tensors[1]`
        取到的是**帧1**,不抛任何异常。于是 mask/Grad-CAM 出自帧1,而 JSON 里
        keyframe_frame_index=2。只断言那个字段对它是盲的——本测试直接截获
        mask/Grad-CAM 实际入参的帧指纹,锁死这个静默不一致。
        """
        spy = {}
        data = self._run(
            tmp_path, monkeypatch, n_frames=4, bad_positions={0},
            mal_probs_by_pos={1: 0.10, 2: 0.95, 3: 0.20}, spy_keyframe_np=spy)
        _assert_full_diagnosis(data)
        assert data["keyframe_frame_index"] == 2, \
            "最恶性帧是原始帧 2(跳过帧 0 后它在 per_frame 的位置是 1)"
        assert spy["fingerprint"] == 30, \
            f"mask/Grad-CAM 必须出自原始帧 2(指纹 30),实际出自指纹 " \
            f"{spy['fingerprint']}(=原始帧 {spy['fingerprint']//10 - 1})——平行下标错位"
        assert data["malignancy_probabilities"][1] == data["malignancy_probability"], \
            "关键帧概率分布必须与聚合出的 max 同源——错位时二者会分叉"
        # per_frame_malignancy 的帧号集合必须恰是成功帧,顺序与原始帧号一致
        assert [f["frame_index"] for f in data["per_frame_malignancy"]] == [1, 2, 3]
        # 关键帧报告文本引用的帧号也须一致(报告与 JSON 不能各说一套)
        report = (tmp_path / "out" / "clinical_report.txt").read_text(encoding="utf-8")
        assert "#frame2" in report or "帧 2" in report, \
            f"报告未引用关键帧 2,疑似错位: {report[:400]}"

    def test_all_frames_failing_yields_error_not_crash(self, tmp_path, monkeypatch):
        """全帧数值异常 → status=ERROR + 明细,且**不是** AssertionError/IndexError。

        `aggregate_frame_predictions([])` 自身安全返回 keyframe_index=-1 不抛,
        旧实现靠 `assert kf_pos >= 0` 拦——assert 在 `python -O` 下会被整条剥离,
        且文案对"全帧数值异常"是误导性的。改为显式 if + 落 failed_frames 明细。
        """
        data = self._run(tmp_path, monkeypatch, n_frames=3, bad_positions={0, 1, 2})
        assert data["status"] == "ERROR"
        assert data["n_frames_sampled"] == 3 and data["n_frames_failed"] == 3
        assert len(data["failed_frames"]) == 3
        assert "3" in data["reason"], f"reason 须含帧数便于定位: {data['reason']}"
        for bad_type in ("AssertionError", "IndexError", "KeyError"):
            assert bad_type not in data["reason"], \
                f"全帧失败应走显式守卫,不该是 {bad_type}: {data['reason']}"
        assert "malignancy_probability" not in data, \
            "无任何有效帧时不得编造聚合概率"

    def test_no_failure_keeps_original_field_semantics(self, tmp_path, monkeypatch):
        """无坏帧时新字段退化为原行为(sampled == diagnosed、failed 为空)。"""
        data = self._run(tmp_path, monkeypatch, n_frames=3, bad_positions=set())
        _assert_full_diagnosis(data)
        assert data["n_frames_sampled"] == data["n_frames_diagnosed"] == 3
        assert data["n_frames_failed"] == 0 and data["failed_frame_indices"] == []

    def test_runtime_error_still_propagates_to_error_status(self, tmp_path, monkeypatch):
        """只捕 ValueError:RuntimeError(CUDA OOM/权重维度不符)必须冒泡成 ERROR。

        若逐帧兜底吞掉 RuntimeError,真故障会被伪装成"少了几帧"的 PASSED 结果——
        评测方拿到一份看似正常的诊断,而实际上模型根本没正常跑。
        """
        import video_predict as video_mod

        frames = [np.full((96, 96), 30, dtype=np.uint8) for _ in range(2)]
        monkeypatch.setattr(
            video_mod, "load_video_frames", lambda *a, **k: (frames, [0, 1], False, None))

        def _boom(*a, **k):
            raise RuntimeError("CUDA out of memory (simulated)")

        monkeypatch.setattr(video_mod, "diagnose_frame", _boom)
        out = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", [
            "video_predict.py", "--input_video", str(tmp_path / "fake.mp4"),
            "--output_dir", str(out), "--precision", "fp32", "--skip_gatekeeper",
            "--weights", "checkpoints/__absent_for_test__.pth",
        ])
        video_mod.main()
        data = json.loads((out / "result.json").read_text(encoding="utf-8"))
        assert data["status"] == "ERROR"
        assert "RuntimeError" in data["reason"], \
            "RuntimeError 必须原样落 reason,不得被逐帧 except ValueError 吞掉"

    def test_malformed_frame_shape_skipped_not_kills_video(self, tmp_path, monkeypatch):
        """AssertionError(帧形状非法)也属"这一帧坏",只丢帧不拖垮整段。

        2026-07-30 /code-review Angle C:捕异常从宽口 ValueError 收窄为
        (NonFiniteProbabilityError, AssertionError) 两类。preprocess_image 对非 2D
        灰度帧 `assert img_np.ndim == 2` 抛 AssertionError——DICOM 多帧序列偶发混入
        彩色/畸形帧,这是"该帧数据坏"不是系统故障,穿透会整段 ERROR,正是本轮要
        消灭的"一帧拖垮整段"换个异常类型复活。
        """
        import video_predict as video_mod
        frames = [np.full((96, 96), 10, np.uint8),          # 帧0 正常
                  np.zeros((96, 96, 3), np.uint8),          # 帧1 彩色(3通道→ndim=3)
                  np.full((96, 96), 30, np.uint8)]          # 帧2 正常
        monkeypatch.setattr(
            video_mod, "load_video_frames", lambda *a, **k: (frames, [0, 1, 2], False, None))
        out = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", [
            "video_predict.py", "--input_video", str(tmp_path / "fake.mp4"),
            "--output_dir", str(out), "--precision", "fp32", "--skip_gatekeeper",
            "--weights", "checkpoints/__absent_for_test__.pth",
        ])
        video_mod.main()
        data = json.loads((out / "result.json").read_text(encoding="utf-8"))
        _assert_full_diagnosis(data)
        assert data["n_frames_sampled"] == 3 and data["n_frames_diagnosed"] == 2
        assert data["failed_frame_indices"] == [1], \
            "彩色帧(ndim=3)应被当'这一帧坏'跳过,而非冒泡成整段 ERROR"

    def test_negative_keyframe_index_never_negative_index_silently(self, tmp_path, monkeypatch):
        """聚合返回 keyframe_index=-1 时须显式 ERROR,不得靠 Python 负索引静默取末帧。

        双重守卫:即便全帧失败守卫未来被误改(只 print 不 return),`per_frame[-1]`
        **不抛异常**而静默取最后一帧,mask/Grad-CAM 出自末帧而 JSON 写 -1——与本轮
        修掉的 key_tensors 错位同性质(负索引静默错位)。
        """
        import video_predict as video_mod
        frames = [np.full((96, 96), 10, np.uint8) for _ in range(2)]
        monkeypatch.setattr(
            video_mod, "load_video_frames", lambda *a, **k: (frames, [0, 1], False, None))
        # 强制聚合返回 -1(模拟全帧失败守卫被绕过的极端情形)
        monkeypatch.setattr(
            video_mod, "aggregate_frame_predictions",
            lambda pf: {"n_frames": len(pf), "mal_prob_mean": 0.0, "mal_prob_max": 0.0,
                        "mal_prob_min": 0.0, "keyframe_index": -1,
                        "keyframe_frame_index": -1, "final_is_malignant": 0})
        out = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", [
            "video_predict.py", "--input_video", str(tmp_path / "fake.mp4"),
            "--output_dir", str(out), "--precision", "fp32", "--skip_gatekeeper",
            "--weights", "checkpoints/__absent_for_test__.pth",
        ])
        video_mod.main()
        data = json.loads((out / "result.json").read_text(encoding="utf-8"))
        assert data["status"] == "ERROR"
        assert "keyframe" in data["reason"].lower(), \
            "keyframe_index=-1 须显式报错,不得负索引静默取末帧编出假关键帧诊断"
        assert "malignancy_probability" not in data

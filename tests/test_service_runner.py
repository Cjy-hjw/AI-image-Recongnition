"""`src/service/runner.py` 单测:批量编排逻辑(不真跑模型,重点验 runner 的分派/隔离/复用)。

合成假数据集目录(复用 `test_case_discovery.py` 的 `_make_dcm`/`_save_nifti` 写法),
用 mock 的 `diagnose_*_case`/`write_case_v21`/重型对象(builder/gatekeeper)注入,断言:

- 逐病例都调了 `write_case_v21`,且 `case_dir = {output_dir}/{patient_id}`;
- 单病例失败隔离:故意让一例 diagnose 抛异常,断言其余病例仍正常产出、该例落了
  结构合规的 ERROR 占位 `prediction.json`;
- 模型只 build 一次(patch `build_model` 计数,多病例 + 多次调用 build_service_runner);
- 阴性/空目录数据集不崩(返回空 runner 直跑无输出)。

依赖 pydicom/nibabel(造 DICOM/NIfTI),缺则整文件 skip —— 与 test_case_discovery.py 同口径。
torch 由基础镜像/requirements 保证(测试环境必装,不在此 skip)。
"""

import json
import os

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")  # noqa: F841 — 造 DICOM 必需
nib = pytest.importorskip("nibabel")

import warnings  # noqa: E402 — 抑制合成 DICOM 的 pydicom 弃用警告(与 test_case_discovery 同口径)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydicom")
warnings.filterwarnings("ignore", message="Invalid value for VR UI", category=UserWarning)

from src.service import runner as runner_mod  # noqa: E402


# ── 合成影像 fixture(与 test_case_discovery.py 完全同款,复用其写法)────────────
def _make_dcm(path, patient_id="P001", series_uid=None, rows=8, cols=8):
    """最小可读 MONOCHROME2 8bit DICOM,带 PatientID/SeriesInstanceUID。"""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    if series_uid is None:
        series_uid = generate_uid()

    ds = Dataset()
    ds.Rows, ds.Columns = rows, cols
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((rows, cols), np.uint8).tobytes()
    ds.PatientID = patient_id
    ds.SeriesInstanceUID = series_uid

    fm = FileMetaDataset()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = generate_uid()
    fm.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta = fm
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        ds.save_as(path, write_like_original=False)   # pydicom 2.x
    except TypeError:                                  # pydicom 3.x
        ds.save_as(path, enforce_file_format=True)


def _save_nifti(path, n_frames=4, h=8, w=8):
    """最小 3D NIfTI(.nii.gz),维度约定与 save_binary_mask 一致(写出前转置)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vol = np.zeros((n_frames, h, w), np.uint8)
    data = np.transpose(vol, (2, 1, 0))  # (n,H,W) → (W,H,n)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(data), np.eye(4)), path)


@pytest.fixture(autouse=True)
def _isolate_env_and_cache(monkeypatch, tmp_path):
    """每个用例隔离:清空 runner 模块级缓存 + 屏蔽 AICOMP_* 环境变量。

    - `_RUNNER_CACHE` 是模块级单例,跨用例残留会让「build_model 计数」等断言被污染;
    - 环境变量(AICOMP_WEIGHTS 等)可能被外部 shell 设置,干扰缺省路径解析,统一清掉。
    """
    monkeypatch.delenv("AICOMP_WEIGHTS", raising=False)
    monkeypatch.delenv("AICOMP_CONFIG", raising=False)
    monkeypatch.delenv("AICOMP_PRECISION", raising=False)
    monkeypatch.delenv("AICOMP_MAX_FRAMES", raising=False)
    monkeypatch.delenv("AICOMP_AUDIT_FRAMES", raising=False)
    runner_mod._RUNNER_CACHE.clear()
    yield
    runner_mod._RUNNER_CACHE.clear()


@pytest.fixture
def mock_heavy(monkeypatch):
    """把 runner 的所有重依赖(build_model/gatekeeper 三件/diagnose/write_case_v21)
    全部 mock 掉,只留 runner 自己的编排逻辑。

    Returns:
        dict 收集器:
          build_count        — build_model 调用次数(验「只 build 一次」)
          written            — list[dict],每次 write_case_v21 的实参(case_dir/patient_id/...)
          diagnose_static    — 可调换的静态 diagnose 实现(默认返回最小 case_result)
          diagnose_dynamic   — 同上,动态
          gatekeeper_calls   — run_gatekeeper_audit 调用次数(验 gatekeeper 未被跳过)
    """
    import src.inference.model_runner as model_runner_mod
    import src.inference.case_pipeline as case_pipeline_mod
    import src.inference.gatekeeper_runner as gatekeeper_mod

    box = {
        "build_count": 0,
        "written": [],
        "gatekeeper_calls": 0,
        "diagnose_static": lambda *a, **k: {
            "preds": {"mal_prob": 0.5}, "mask_resized": np.zeros((8, 8), np.uint8),
            "breakdown": {}, "img_tensor": None},
        "diagnose_dynamic": lambda *a, **k: {
            "per_frame": [], "agg": {"keyframe_index": 0, "keyframe_frame_index": 0,
                                     "mal_prob_max": 0.5, "mal_prob_mean": 0.5,
                                     "n_frames": 4, "final_is_malignant": 0},
            "kf_preds": {"mal_prob": 0.5}, "kf_frame_np": np.zeros((8, 8), np.uint8),
            "kf_tensor": None, "failed_frames": [], "breakdown": {},
            "mask_resized": np.zeros((8, 8), np.uint8)},
    }

    def _fake_build_model(config, weights, device):
        box["build_count"] += 1
        return object()  # 假模型句柄

    def _fake_load_config(path):
        return {"environment": {"device": "cpu"}, "data": {"img_size": 512},
                "calibration": {}, "gatekeeper": {}}

    def _fake_run_gatekeeper(*a, **k):
        box["gatekeeper_calls"] += 1
        return True, {"flagged": False, "would_reject_in_hard_mode": False,
                      "enforcement_mode": "soft", "reason": "", "soft_flags": []}

    def _fake_write_case_v21(case_dir, patient_id, subdir_uid, case_result, kind,
                             processing_ms, interpretation=None, roi_ctx=None):
        box["written"].append({
            "case_dir": case_dir, "patient_id": patient_id, "subdir_uid": subdir_uid,
            "kind": kind, "processing_ms": processing_ms, "case_result": case_result,
            "roi_ctx": roi_ctx})

    monkeypatch.setattr(model_runner_mod, "build_model", _fake_build_model)
    monkeypatch.setattr(model_runner_mod, "load_config", _fake_load_config)
    # diagnose/write 是 runner 内部局部 import,打到被调方模块命名空间
    monkeypatch.setattr(case_pipeline_mod, "diagnose_static_case",
                        lambda *a, **k: box["diagnose_static"](*a, **k))
    monkeypatch.setattr(case_pipeline_mod, "diagnose_dynamic_case",
                        lambda *a, **k: box["diagnose_dynamic"](*a, **k))
    monkeypatch.setattr(gatekeeper_mod, "run_gatekeeper_audit", _fake_run_gatekeeper)
    monkeypatch.setattr(gatekeeper_mod, "build_seeded_deduplicator", lambda *a, **k: object())
    monkeypatch.setattr(gatekeeper_mod, "build_auth_module", lambda *a, **k: object())
    monkeypatch.setattr(gatekeeper_mod, "build_tamper_detector", lambda *a, **k: object())
    monkeypatch.setattr(gatekeeper_mod, "resolve_enforcement_mode", lambda cfg: "soft")

    # write_case_v21 来自 A3 的 src.inference.v21_output(已真实落地)。runner 在
    # `_write_via_v21` 内做函数级延迟 `from src.inference.v21_output import
    # write_case_v21`(见 runner.py),故 patch 源模块属性即可——runner 仍拿真模块、
    # 真模块的其它契约不被掩盖,只把写盘 mock 掉(验编排,不真落盘)。
    import src.inference.v21_output as v21_output_mod
    monkeypatch.setattr(v21_output_mod, "write_case_v21", _fake_write_case_v21)

    return box


# ── 测试:逐病例编排 + write_case_v21 调用 ────────────────────────────────────
class TestPerCaseDispatch:
    def test_static_and_dynamic_both_written(self, mock_heavy, tmp_path):
        """静/动各一例,二者都触发 write_case_v21,case_dir = {output}/{patient_id}。"""
        _make_dcm(str(tmp_path / "P_STATIC" / "S1" / "img.dcm"),
                  patient_id="P_STATIC", series_uid="1.2.3.4.5")
        _save_nifti(str(tmp_path / "P_DYN" / "P_DYN" / "P_DYN.nii.gz"))
        out = tmp_path / "out"

        runner = runner_mod.build_service_runner()
        runner(str(tmp_path), str(out))

        assert len(mock_heavy["written"]) == 2
        by_pid = {w["patient_id"]: w for w in mock_heavy["written"]}
        # case_dir 路径契约:{output_dir}/{patient_id}
        assert by_pid["P_STATIC"]["case_dir"] == os.path.join(str(out), "P_STATIC")
        assert by_pid["P_DYN"]["case_dir"] == os.path.join(str(out), "P_DYN")
        # kind 分派正确
        assert by_pid["P_STATIC"]["kind"] == "static"
        assert by_pid["P_DYN"]["kind"] == "dynamic"
        # subdir_uid 透传(静态=SeriesUID,动态=占位 PatientId)
        assert by_pid["P_STATIC"]["subdir_uid"] == "1.2.3.4.5"
        assert by_pid["P_DYN"]["subdir_uid"] == "P_DYN"

    def test_processing_ms_is_nonnegative_int(self, mock_heavy, tmp_path):
        _make_dcm(str(tmp_path / "PX" / "S" / "x.dcm"), patient_id="PX")
        out = tmp_path / "out"
        runner_mod.build_service_runner()(str(tmp_path), str(out))
        ms = mock_heavy["written"][0]["processing_ms"]
        assert isinstance(ms, int) and ms >= 0
        assert isinstance(ms, int) and ms >= 0

    def test_gatekeeper_not_skipped_per_case(self, mock_heavy, tmp_path):
        """批量下 gatekeeper 仍每病例照跑(静态 1 次 + 动态审计帧各 1 次)。"""
        _make_dcm(str(tmp_path / "PS" / "S" / "a.dcm"), patient_id="PS")
        _save_nifti(str(tmp_path / "PD" / "PD" / "PD.nii.gz"), n_frames=3)
        out = tmp_path / "out"
        runner_mod.build_service_runner()(str(tmp_path), str(out))
        # 静态 1 次 + 动态 3 帧(3 帧 ≤ 默认审计上限 3 → 全审) = 4 次 gatekeeper 调用
        assert mock_heavy["gatekeeper_calls"] == 4

    def test_dynamic_audit_frames_sampled_not_all(self, mock_heavy, tmp_path):
        """2026-09-03 提速回归:动态审计只抽至多 AICOMP_AUDIT_FRAMES 帧(默认 3),
        不逐帧审全部抽样帧;诊断仍消费全部 frames(n_frames_sampled 不变)。
        减审依据=soft 模式零分值影响(见 runner._run_dynamic_case 注释)。"""
        # 12 帧 > 默认审计上限 3 → 只审 3 帧(均匀:首/中/尾)
        _save_nifti(str(tmp_path / "PD" / "PD" / "PD.nii.gz"), n_frames=12)
        out = tmp_path / "out"
        runner_mod.build_service_runner()(str(tmp_path), str(out))

        # 只审 3 帧,不是 12 帧(提速生效)
        assert mock_heavy["gatekeeper_calls"] == 3
        dyn = mock_heavy["written"][0]["case_result"]
        ga = dyn["gatekeeper_audit"]
        assert ga["n_frames_audited"] == 3
        assert ga["n_frames_sampled"] == 12, "推理消费的帧数不受减审影响"

    def test_dynamic_audit_frames_env_override_full(self, mock_heavy, tmp_path,
                                                    monkeypatch):
        """AICOMP_AUDIT_FRAMES<=0 恢复全帧审计(旧行为,回退开关)。"""
        _save_nifti(str(tmp_path / "PD" / "PD" / "PD.nii.gz"), n_frames=5)
        monkeypatch.setenv("AICOMP_AUDIT_FRAMES", "0")
        out = tmp_path / "out"
        runner_mod.build_service_runner()(str(tmp_path), str(out))
        assert mock_heavy["gatekeeper_calls"] == 5
        ga = mock_heavy["written"][0]["case_result"]["gatekeeper_audit"]
        assert ga["n_frames_audited"] == 5

    def test_dynamic_audit_frames_env_invalid_falls_back(self, mock_heavy, tmp_path,
                                                         monkeypatch):
        """AICOMP_AUDIT_FRAMES 非法值(非整数)回默认 3,不崩。"""
        _save_nifti(str(tmp_path / "PD" / "PD" / "PD.nii.gz"), n_frames=6)
        monkeypatch.setenv("AICOMP_AUDIT_FRAMES", "not-a-number")
        out = tmp_path / "out"
        runner_mod.build_service_runner()(str(tmp_path), str(out))
        assert mock_heavy["gatekeeper_calls"] == 3



# ── 测试:单病例失败隔离(红线)────────────────────────────────────────────────
class TestPerCaseFailureIsolation:
    def test_one_case_crash_does_not_block_others(self, mock_heavy, tmp_path):
        """故意让一例 diagnose 抛异常:其余病例仍正常产出,该例落 ERROR 占位。"""
        _make_dcm(str(tmp_path / "GOOD1" / "S" / "a.dcm"), patient_id="GOOD1")
        _make_dcm(str(tmp_path / "BAD" / "S" / "b.dcm"), patient_id="BAD")
        _make_dcm(str(tmp_path / "GOOD2" / "S" / "c.dcm"), patient_id="GOOD2")
        out = tmp_path / "out"

        # 2026-09-03 起病例级并行(_run_cases 组间并发)下,「第 N 次调用=某例」
        # 不再成立(执行次序非确定),旧版按 call_n==1 判 BAD 会随机炸掉 GOOD1。
        # 改为按调用方栈帧的 case["patient_id"] 精确判定目标病例。
        from src.service.case_discovery import discover_cases
        assert sorted(c["patient_id"] for c in discover_cases(str(tmp_path))) == [
            "BAD", "GOOD1", "GOOD2",
        ], "discover_cases 未发现全部三例 —— 本测试的病例集合前提失效"

        box_diagnose = mock_heavy["diagnose_static"]
        bad_pids = {"BAD"}

        def _explode_on_bad(*a, **k):
            # 由被调方透传的 case dict 拿 patient_id(诊断签名不含它,故从 a[1]
            # 或 kwargs 里找不可靠;改为闭包内按"是否已知坏例"的调用次序不可行
            # ——并行执行下无稳定次序)。最稳的判据:mock_heavy["diagnose_static"]
            # 的调用方是 _run_static_case(case, ...),case 在调用栈上,通过
            # inspect 从栈帧取 case["patient_id"] 判定。
            import inspect
            for frame_info in inspect.stack():
                loc = frame_info.frame.f_locals
                if "case" in loc and isinstance(loc["case"], dict):
                    if loc["case"].get("patient_id") in bad_pids:
                        raise RuntimeError("simulated diagnose crash")
            return box_diagnose(*a, **k)

        mock_heavy["diagnose_static"] = _explode_on_bad

        runner_mod.build_service_runner()(str(tmp_path), str(out))

        # 其余两例仍正常 write_case_v21
        pids_written = {w["patient_id"] for w in mock_heavy["written"]}
        assert "GOOD1" in pids_written and "GOOD2" in pids_written
        assert "BAD" not in pids_written

        # BAD 例落了结构合规的 ERROR 占位 prediction.json
        bad_pred = out / "BAD" / "prediction.json"
        assert bad_pred.exists(), "失败病例未落 ERROR 占位 prediction.json"
        payload = json.loads(bad_pred.read_text(encoding="utf-8"))
        # 结构合规:必填顶层字段齐全、可过 validate_prediction_json
        from src.inference import prediction_json as pj
        pj.validate_prediction_json(payload)  # 不抛即合规
        # 占位语义:StudyUID=patient_id,Interpretation 写明 ERROR 原因
        assert payload["StudyUID"] == "BAD"
        assert "simulated diagnose crash" in payload["Interpretation"]["Conclusion"]

    def test_gatekeeper_hard_reject_writes_rejected_placeholder(
            self, mock_heavy, tmp_path, monkeypatch):
        """gatekeeper hard 拒绝:该例落 REJECTED 占位(非 ERROR、不走 write_case_v21),
        且不 raise 拖垮整批。"""
        _make_dcm(str(tmp_path / "P_OK" / "S" / "ok.dcm"), patient_id="P_OK")
        _make_dcm(str(tmp_path / "P_REJECTED" / "S" / "bad.dcm"),
                  patient_id="P_REJECTED")
        out = tmp_path / "out"

        import src.inference.gatekeeper_runner as gatekeeper_mod
        import src.inference.io as io_mod
        # P_REJECTED 触发硬拒绝,其余仍 pass。2026-09-03 病例级并行下「第 N 次调用
        # =某例」不再成立,改为按调用方栈帧的 case["patient_id"] 精确判定目标病例
        # (run_gatekeeper_audit 的调用方 _run_static_case 持有 case dict)。
        from src.service.case_discovery import discover_cases
        assert sorted(c["patient_id"] for c in discover_cases(str(tmp_path))) == [
            "P_OK", "P_REJECTED",
        ], "discover_cases 未发现全部两例 —— 本测试的病例集合前提失效"

        import inspect
        _reject_pids = {"P_REJECTED"}

        def _reject_target(*a, **k):
            for frame_info in inspect.stack():
                loc = frame_info.frame.f_locals
                if "case" in loc and isinstance(loc["case"], dict) \
                        and loc["case"].get("patient_id") in _reject_pids:
                    return False, {
                        "flagged": True, "would_reject_in_hard_mode": True,
                        "enforcement_mode": "hard",
                        "reason": "simulated tamper hard-reject",
                        "soft_flags": ["tamper"]}
            return True, {"flagged": False, "would_reject_in_hard_mode": False,
                          "enforcement_mode": "hard", "reason": "", "soft_flags": []}

        monkeypatch.setattr(gatekeeper_mod, "run_gatekeeper_audit", _reject_target)

        runner_mod.build_service_runner()(str(tmp_path), str(out))

        # P_REJECTED 落 REJECTED 占位 prediction.json(非 ERROR、且未走 write_case_v21)
        pids_written = {w["patient_id"] for w in mock_heavy["written"]}
        assert "P_REJECTED" not in pids_written  # rejected 不进 write_case_v21
        rej = out / "P_REJECTED" / "prediction.json"
        assert rej.exists(), "gatekeeper 拒绝病例未落 REJECTED 占位 prediction.json"
        payload = json.loads(rej.read_text(encoding="utf-8"))
        from src.inference import prediction_json as pj
        pj.validate_prediction_json(payload)
        assert payload["StudyUID"] == "P_REJECTED"
        assert "拒绝" in payload["Interpretation"]["Conclusion"]
        assert "simulated tamper hard-reject" in payload["Interpretation"]["Conclusion"]
        # Gate2 finding #1:占位必须**阴性**(空结节),绝不伪造结节——伪造的假结节会被
        # 评测当真实阳性检出,且其 MaskFile 指向从未写盘的 phantom mask 文件。
        assert payload["Prediction"]["Nodules"] == [], "REJECTED 占位不得伪造结节"
        assert payload["Prediction"]["MaskFiles"] == [], "REJECTED 占位不得引用不存在的 mask"

        # P_OK 例正常产出(被 REJECTED 例不影响,失败隔离红线)
        assert "P_OK" in pids_written

    def test_dynamic_auth_skipped_propagated_to_audit(
            self, mock_heavy, tmp_path, monkeypatch):
        """Gate2 P2 回归:动态链路聚合 gatekeeper_audit 的 auth_skipped 须**取自逐帧
        audit**(任一幕 True 即 True),覆盖'权重存在但 predict 在帧上崩溃'这种跳过——
        不能用 `state['auth'] is None` 判据(那只覆盖权重缺失,漏了 predict 崩溃,
        会让聚合层误报 auth_skipped=False 而实际已跳过,误导运维/评测)。"""
        _save_nifti(str(tmp_path / "P_DYN" / "P_DYN" / "P_DYN.nii.gz"))
        out = tmp_path / "out"

        import src.inference.gatekeeper_runner as gatekeeper_mod
        # 逐帧 audit 返回 auth_skipped=True(= predict 崩溃被 catch 的效果),passed=True(soft)。
        # 注意 build_auth_module 被 mock_heavy 设为返回真实 object(非 None),即 state["auth"]
        # 非 None——此时若聚合层仍用 `state["auth"] is None` 会得 False,本测试正是要抓这个。
        def _audit_auth_skipped(*a, **k):
            return True, {"flagged": False, "would_reject_in_hard_mode": False,
                          "enforcement_mode": "soft", "reason": "", "soft_flags": [],
                          "auth_skipped": True}
        monkeypatch.setattr(gatekeeper_mod, "run_gatekeeper_audit", _audit_auth_skipped)

        runner_mod.build_service_runner()(str(tmp_path), str(out))

        dyn = [w for w in mock_heavy["written"] if w["patient_id"] == "P_DYN"][0]
        ga = dyn["case_result"]["gatekeeper_audit"]
        assert ga["auth_skipped"] is True, (
            "动态聚合层须透传逐帧 auth_skipped(predict 崩溃场景),不能用 state['auth'] is None")

    def test_loader_failure_isolated(self, mock_heavy, tmp_path, monkeypatch):
        """输入加载失败(如损坏 NIfTI)也只隔离该例,不拖垮整批。"""
        import src.inference.nifti_io as nifti_mod
        _make_dcm(str(tmp_path / "OKS" / "S" / "ok.dcm"), patient_id="OKS")
        _save_nifti(str(tmp_path / "BROKEN" / "BROKEN" / "BROKEN.nii.gz"))
        out = tmp_path / "out"

        # 强制 load_nifti_frames 返回失败(模拟损坏文件)
        monkeypatch.setattr(
            nifti_mod, "load_nifti_frames",
            lambda *a, **k: (None, [], False, "simulated corrupt NIfTI", 0))

        runner_mod.build_service_runner()(str(tmp_path), str(out))

        pids = {w["patient_id"] for w in mock_heavy["written"]}
        assert "OKS" in pids
        # BROKEN 例落 ERROR 占位
        assert (out / "BROKEN" / "prediction.json").exists()


# ── 测试:模型只 build 一次 ───────────────────────────────────────────────────
class TestModelBuiltOnce:
    def test_build_once_across_many_cases(self, mock_heavy, tmp_path):
        """同一 runner 跑多个病例,build_model 只调一次。"""
        for i in range(3):
            _make_dcm(str(tmp_path / f"CASE{i}" / "S" / f"f{i}.dcm"),
                      patient_id=f"CASE{i}")
        out = tmp_path / "out"
        runner_mod.build_service_runner()(str(tmp_path), str(out))
        assert mock_heavy["build_count"] == 1
        assert len(mock_heavy["written"]) == 3

    def test_build_once_across_multiple_runner_calls(self, mock_heavy, tmp_path):
        """同参数重复调 build_service_runner(模拟多次 /call),走模块级缓存仍只 build 一次。"""
        _make_dcm(str(tmp_path / "C1" / "S" / "x.dcm"), patient_id="C1")
        out = tmp_path / "out"
        r1 = runner_mod.build_service_runner()
        r2 = runner_mod.build_service_runner()  # 同参数 → 命中缓存
        assert r1 is r2  # 缓存复用同一闭包
        r1(str(tmp_path), str(out))
        r2(str(tmp_path), str(out))
        assert mock_heavy["build_count"] == 1


# ── 测试:空/无效数据集不崩 ───────────────────────────────────────────────────
class TestEmptyDataset:
    def test_empty_dir_no_crash(self, mock_heavy, tmp_path):
        out = tmp_path / "out"
        runner_mod.build_service_runner()(str(tmp_path), str(out))
        assert mock_heavy["written"] == []

    def test_nonexistent_path_no_crash(self, mock_heavy, tmp_path):
        out = tmp_path / "out"
        # 不存在的目录:discover_cases 返回 [],runner 不抛
        runner_mod.build_service_runner()(str(tmp_path / "does_not_exist"), str(out))
        assert mock_heavy["written"] == []

    def test_all_non_image_skipped(self, mock_heavy, tmp_path):
        """目录里只有非影像文件 → 0 病例,不崩。"""
        (tmp_path / "labels.json").write_text("{}")
        (tmp_path / "README.md").write_text("notes")
        out = tmp_path / "out"
        runner_mod.build_service_runner()(str(tmp_path), str(out))
        assert mock_heavy["written"] == []

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
                             processing_ms, interpretation=None):
        box["written"].append({
            "case_dir": case_dir, "patient_id": patient_id, "subdir_uid": subdir_uid,
            "kind": kind, "processing_ms": processing_ms})

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

    def test_gatekeeper_not_skipped_per_case(self, mock_heavy, tmp_path):
        """批量下 gatekeeper 仍每病例照跑(静态 1 次 + 动态每帧 1 次)。"""
        _make_dcm(str(tmp_path / "PS" / "S" / "a.dcm"), patient_id="PS")
        _save_nifti(str(tmp_path / "PD" / "PD" / "PD.nii.gz"), n_frames=3)
        out = tmp_path / "out"
        runner_mod.build_service_runner()(str(tmp_path), str(out))
        # 静态 1 次 + 动态 3 帧 = 4 次 gatekeeper 调用
        assert mock_heavy["gatekeeper_calls"] == 4


# ── 测试:单病例失败隔离(红线)────────────────────────────────────────────────
class TestPerCaseFailureIsolation:
    def test_one_case_crash_does_not_block_others(self, mock_heavy, tmp_path):
        """故意让一例 diagnose 抛异常:其余病例仍正常产出,该例落 ERROR 占位。"""
        _make_dcm(str(tmp_path / "GOOD1" / "S" / "a.dcm"), patient_id="GOOD1")
        _make_dcm(str(tmp_path / "BAD" / "S" / "b.dcm"), patient_id="BAD")
        _make_dcm(str(tmp_path / "GOOD2" / "S" / "c.dcm"), patient_id="GOOD2")
        out = tmp_path / "out"

        def _explode_on_bad(*a, **k):
            raise RuntimeError("simulated diagnose crash")

        # 让 BAD 例崩掉。discover_cases 按 os.walk + 同目录文件名排序,GOOD1/GOOD2/
        # BAD 同挂 tmp_path 根下,目录名字典序为 BAD < GOOD1 < GOOD2,故 BAD 是第 1 例。
        box_diagnose = mock_heavy["diagnose_static"]
        call_n = {"n": 0}

        def _selective(*a, **k):
            call_n["n"] += 1
            if call_n["n"] == 1:  # 首例(BAD,字典序最小)
                raise RuntimeError("simulated diagnose crash")
            return box_diagnose(*a, **k)

        mock_heavy["diagnose_static"] = _selective

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

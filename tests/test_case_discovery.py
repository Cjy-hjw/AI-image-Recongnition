"""`src/service/case_discovery.py` 单测。

合成 mini 目录树(几个假 .dcm 用 pydicom 造带 PatientID/SeriesInstanceUID 的
最小 DICOM、几个假 .nii.gz 用 nibabel 造),验证:

- discover_cases 正确识别 static/dynamic、提取 patient_id/subdir_uid、跳过非影像
  文件、空目录返回 [];
- DICOM 头缺失时走路径推断的降级路径也测一个;
- pydicom/nibabel 缺失时 skipif(与 CI/容器一致,本地可能缺)。

依赖 pydicom(造 DICOM)与 nibabel(造 NIfTI),缺任一则整文件 skip —— 与
`tests/test_official_loader.py` / `tests/test_nifti_io.py` 同口径。
"""

import os

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")  # noqa: F841 — 用 importorskip 既跳过测试又让 _make_dcm 可用
nib = pytest.importorskip("nibabel")

import warnings  # noqa: E402 — 抑制合成 DICOM 触发的 pydicom 弃用/值警告

from src.service.case_discovery import (  # noqa: E402
    discover_cases,
    _build_case,
    _build_static_case,
    _build_dynamic_case,
    _fallback_patient_id_from_path,
    _fallback_subdir_uid_from_path,
    _infer_ancestor_name,
    _clean_dicom_value,
)

# 合成 DICOM 用 pydicom.generate_uid() 生成合法 UID(见 _make_dcm),但仍会触发
# pydicom 3.x 的 is_little_endian/is_implicit_VR/write_like_original 弃用警告
# (合成 fixture 为 2.x/3.x 双兼容保留这些旧 API)。与业务逻辑无关,整文件过滤。
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydicom")
warnings.filterwarnings(
    "ignore", message="Invalid value for VR UI", category=UserWarning
)


# ── 合成影像 fixture 生成器 ─────────────────────────────────────────────────
# _make_dcm 复用 `tests/test_official_loader.py` 的最小 MONOCHROME2 8bit DICOM 写法
# (pydicom 2.x/3.x 双兼容);_save_nifti 复用 `tests/test_nifti_io.py` 的维度约定
# (写出前 transpose,使 nib.load 读回形状与输入一致 —— 此处只造文件,不读回,但保持
# 与项目内 NIfTI 写出口径一致,避免引入第三套维度约定)。
def _make_dcm(path, patient_id="P001", series_uid=None,
              rows=8, cols=8, include_phi=True):
    """写最小可读 MONOCHROME2 8bit DICOM,带 PatientID/SeriesInstanceUID。

    Args:
        series_uid: 默认 None → 用 pydicom.generate_uid() 生成合法 UID(避免非
            规范 UID 触发 pydicom UserWarning)。
        include_phi: False 时不写 PatientID/SeriesInstanceUID(测 DICOM 头缺失的
            降级路径)。其余必需 tag 仍写(保证 pydicom.dcmread 不崩)。
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    if series_uid is None:
        series_uid = generate_uid()

    ds = Dataset()
    ds.Rows, ds.Columns = rows, cols
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((rows, cols), np.uint8).tobytes()
    if include_phi:
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
        ds.save_as(path, write_like_original=False)          # pydicom 2.x
    except TypeError:                                        # pydicom 3.x
        ds.save_as(path, enforce_file_format=True)


def _save_nifti(path, n_frames=4, h=8, w=8):
    """写最小 3D NIfTI(.nii.gz)。维度约定与项目 save_binary_mask 一致(写出前转置)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vol = np.zeros((n_frames, h, w), np.uint8)
    data = np.transpose(vol, (2, 1, 0))  # (n,H,W) → (W,H,n) 与 nifti_io 对齐
    nib.save(nib.Nifti1Image(np.ascontiguousarray(data), np.eye(4)), path)


# ── 小工具函数单测 ───────────────────────────────────────────────────────────
class TestInferAncestorName:
    """路径推断小函数:取向上第 N 级父目录名,越界兜底文件 stem。"""

    def test_depth_1_direct_parent(self):
        # 文件直接父目录 'series_dir'
        p = "/data/P1/series_dir/img.dcm"
        assert _infer_ancestor_name(p, 1) == "series_dir"

    def test_depth_2_grandparent(self):
        # 向上 2 级 = 'P1'(训练集静态口径:pid 在第 2 级父)
        p = "/data/P1/series_dir/img.dcm"
        assert _infer_ancestor_name(p, 2) == "P1"

    def test_depth_3_great_grandparent(self):
        p = "/root/P1/series_dir/img.dcm"
        assert _infer_ancestor_name(p, 3) == "root"

    def test_dynamic_layout_patient_at_depth_2(self):
        # 定稿 §1.11 动态口径:{pid}/{pid}/{pid}.nii.gz → pid 在第 2 级父
        p = "/data/P3/P3/P3.nii.gz"
        assert _infer_ancestor_name(p, 2) == "P3"

    def test_undershoot_fallback_to_stem(self):
        # depth=2 但路径浅于 2 级(文件直接在根下)→ 兜底文件 stem(去 .nii.gz)
        p = "/data/lonely.nii.gz"
        # dirname 链:lonely.nii.gz → /data → / → 越界
        assert _infer_ancestor_name(p, 2) == "lonely"

    def test_stem_strips_dcm_double_ext(self):
        # .dcm 是单段后缀,兜底 stem 去 .dcm
        p = "img.dcm"  # 极浅路径,depth=2 必越界
        assert _infer_ancestor_name(p, 2) == "img"

    def test_stem_strips_nii_gz_double_segment(self):
        # .nii.gz 是双段后缀,兜底 stem 须去 .nii.gz(不是只去 .gz 留 .nii)
        p = "case.nii.gz"
        assert _infer_ancestor_name(p, 2) == "case"

    def test_windows_path_separators(self):
        # os.path 跨平台;Windows 分隔符也要正常解析
        p = r"C:\data\P1\series\img.dcm"
        assert _infer_ancestor_name(p, 2) == "P1"

    def test_trailing_sep_no_empty_segment(self):
        # normpath 去尾 sep,避免 a/b/ → ['a','b',''] 的空尾段干扰 depth
        p = "/data/P1/series_dir/img.dcm/"
        assert _infer_ancestor_name(p, 2) == "P1"

    def test_bottomed_out_and_empty_stem_returns_none(self):
        # 祖先链越界(dirname 到 "")且文件 stem 也空(隐藏文件 .nii.gz 无 basename)
        # → None。这是唯一返回 None 的分支:两个兜底层都落空。
        # .nii.gz: normpath=".nii.gz";dirname 链 → "" → "" → "";basename("")="";
        # stem=basename(".nii.gz")=".nii.gz" 去 .nii.gz → "";`stem.strip() or None` → None
        assert _infer_ancestor_name(".nii.gz", 3) is None

    def test_dotted_file_returns_none(self):
        # 纯扩展名文件(如 .dcm)无 stem → 同样 None
        assert _infer_ancestor_name(".dcm", 3) is None


class TestCleanDicomValue:
    """DICOM 字段值清洗:None/空 → None,其余 str 去空白。"""

    def test_none(self):
        assert _clean_dicom_value(None) is None

    def test_empty_string(self):
        assert _clean_dicom_value("") is None

    def test_whitespace_only(self):
        assert _clean_dicom_value("   ") is None

    def test_strips_whitespace(self):
        assert _clean_dicom_value("  P001  ") == "P001"

    def test_passthrough_nonempty(self):
        assert _clean_dicom_value("1.2.3.4") == "1.2.3.4"

    def test_multivalue_str_coerces(self):
        # MultiValue 的 str() 表现与 list 一致,发现层只要一个可用标识
        from pydicom.multival import MultiValue
        mv = MultiValue(str, ["a", "b"])
        assert _clean_dicom_value(mv) == str(mv).strip()


# ── 静态/动态 case 构造单测 ─────────────────────────────────────────────────
class TestBuildStaticCase:
    """静态 .dcm case:优先读 DICOM 头,头缺失走路径推断。"""

    def test_reads_dicom_header(self, tmp_path):
        dcm = tmp_path / "P1" / "S1" / "img.dcm"
        _make_dcm(str(dcm), patient_id="P001", series_uid="1.2.3.4.5")
        case = _build_static_case(str(dcm))
        assert case["kind"] == "static"
        assert case["patient_id"] == "P001"
        assert case["subdir_uid"] == "1.2.3.4.5"
        assert os.path.isabs(case["input_path"])
        assert case["input_path"].endswith("img.dcm")
    def test_dicom_header_missing_falls_back_to_path(self, tmp_path):
        # 不写 PatientID/SeriesInstanceUID → 走路径推断(@云TODO 兜底)
        dcm = tmp_path / "P2" / "S2" / "img.dcm"
        _make_dcm(str(dcm), include_phi=False)
        case = _build_static_case(str(dcm))
        assert case["kind"] == "static"
        # 路径推断:静态 pid 在第 2 级父目录名 = 'P2'
        assert case["patient_id"] == "P2"
        # subdir_uid 取父目录名 = 'S2'
        assert case["subdir_uid"] == "S2"

    def test_pydicom_unavailable_falls_back_to_path(self, tmp_path, monkeypatch):
        # monkeypatch 掉 pydicom import(模拟 CI 基镜像缺 pydicom)→ 走路径推断
        dcm = tmp_path / "P3" / "S3" / "img.dcm"
        _make_dcm(str(dcm), patient_id="SHOULD_NOT_BE_USED",
                  series_uid="1.2.3.4.999")
        import src.service.case_discovery as mod
        # __import__ 在 _build_static_case 内 try except ImportError;用 builtins 禁掉 pydicom
        import builtins
        real_import = builtins.__import__

        def _block_pydicom(name, *args, **kwargs):
            if name == "pydicom":
                raise ImportError("simulated pydicom missing")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", _block_pydicom)
        case = _build_static_case(str(dcm))
        assert case["patient_id"] == "P3"   # 路径推断,非 DICOM 头
        assert case["subdir_uid"] == "S3"
        # 确保 pydicom 真还在(只 monkeypatch 了测试期间)
        assert real_import("pydicom") is not None

    def test_corrupt_dcm_falls_back_to_path(self, tmp_path):
        # .dcm 后缀但内容非合法 DICOM → dcmread 抛 → 降级路径推断
        dcm = tmp_path / "P4" / "S4" / "img.dcm"
        dcm.parent.mkdir(parents=True)
        dcm.write_bytes(b"not a dicom file at all")
        case = _build_static_case(str(dcm))
        assert case["kind"] == "static"
        assert case["patient_id"] == "P4"
        assert case["subdir_uid"] == "S4"


class TestBuildDynamicCase:
    """动态 .nii.gz case:无 DICOM 头,全部从路径推断。"""

    def test_dynamic_layout_patient_at_depth_2(self, tmp_path):
        # 定稿 §1.11 动态口径:{pid}/{pid}/{pid}.nii.gz
        nii = tmp_path / "P3" / "P3" / "P3.nii.gz"
        _save_nifti(str(nii))
        case = _build_dynamic_case(str(nii))
        assert case["kind"] == "dynamic"
        assert case["patient_id"] == "P3"
        # subdir_uid = AccessionNumber(@待官方)→ 暂用 PatientId 占位
        assert case["subdir_uid"] == "P3"

    def test_nii_uncompressed(self, tmp_path):
        # .nii(无 gzip)也是 dynamic
        nii = tmp_path / "P5" / "P5" / "P5.nii"
        _save_nifti(str(nii))
        case = _build_dynamic_case(str(nii))
        assert case["kind"] == "dynamic"
        assert case["patient_id"] == "P5"

    def test_shallow_path_falls_back_to_stem(self, tmp_path):
        # 文件埋深不足 2 级:_DYNAMIC_PATIENT_DEPTH=2 取不到祖父目录,dirname 链
        # 顶到祖先盘符根 → basename('')='' → 越界兜底文件 stem = 'lonely'。
        # tmp_path 是 pytest 临时目录(深多层),要构造"埋深不足"得把文件直接
        # 挂在 tmp_path 下、且 tmp_path 自身名短到 dirname 链能顶到盘符根:
        #   tmp_path/lonely.nii.gz → dirname×2 = 盘符根 '' → 越界兜底 stem。
        # 但 tmp_path 的祖先链足够深(dirname×2 仍在 pytest 临时目录树内),
        # 故用 monkeypatch 把 _DYNAMIC_PATIENT_DEPTH 抬到能逼出越界的值,验证兜底路径。
        nii = tmp_path / "lonely.nii.gz"
        _save_nifti(str(nii))
        import src.service.case_discovery as mod
        # 抬到 20 级:tmp_path 通常埋深 5~8 级,20 级必顶到盘符根 → 越界兜底 stem。
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mod, "_DYNAMIC_PATIENT_DEPTH", 20)
        try:
            case = _build_dynamic_case(str(nii))
        finally:
            monkeypatch.undo()
        # 兜底 stem = 'lonely'(@云TODO:浅层路径在官方数据是否出现未知)
        assert case["patient_id"] == "lonely"
        assert case["subdir_uid"] == "lonely"


# ── _build_case 分派单测 ───────────────────────────────────────────────────
class TestBuildCaseDispatch:
    """扩展名分派:.dcm→static, .nii.gz/.nii→dynamic, 其余→None(跳过)。"""

    def test_dcm_dispatches_static(self, tmp_path):
        dcm = tmp_path / "a.dcm"
        _make_dcm(str(dcm))
        assert _build_case(str(dcm))["kind"] == "static"

    def test_dicom_uppercase_ext(self, tmp_path):
        # .DCM 大写也须判 static(与 io.py _is_dicom 的 endswith lower 一致)
        dcm = tmp_path / "b.DICOM"
        _make_dcm(str(dcm))
        assert _build_case(str(dcm))["kind"] == "static"

    def test_nii_gz_dispatches_dynamic(self, tmp_path):
        nii = tmp_path / "P" / "P" / "a.nii.gz"
        _save_nifti(str(nii))
        assert _build_case(str(nii))["kind"] == "dynamic"

    def test_nii_dispatches_dynamic(self, tmp_path):
        nii = tmp_path / "P" / "P" / "a.nii"
        _save_nifti(str(nii))
        assert _build_case(str(nii))["kind"] == "dynamic"
    def test_non_image_returns_none(self, tmp_path):
        # 非影像文件 → None(跳过)
        for ext in (".json", ".txt", ".md", ".xlsx", ".csv", ".xml", ".html"):
            f = tmp_path / f"note{ext}"
            f.write_text("x")
            assert _build_case(str(f)) is None, f"{ext} 应被跳过"

    def test_png_returns_none(self, tmp_path):
        # .png 非 .dcm/.nii → 跳过(本项目静/动均不接受裸 png 作 V2.1 输入)
        f = tmp_path / "img.png"
        f.write_bytes(b"\x89PNG\r\n")
        assert _build_case(str(f)) is None


# ── discover_cases 端到端 ───────────────────────────────────────────────────
class TestDiscoverCases:
    """discover_cases 端到端:遍历、跳过非影像、空目录、不存在目录。"""

    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert discover_cases(str(tmp_path)) == []

    def test_nonexistent_path_returns_empty_list(self, tmp_path):
        assert discover_cases(str(tmp_path / "does_not_exist")) == []

    def test_file_not_dir_returns_empty_list(self, tmp_path):
        # dataset_path 传文件而非目录 → [](不抛)
        f = tmp_path / "notadir.txt"
        f.write_text("x")
        assert discover_cases(str(f)) == []

    def test_none_path_returns_empty_list(self):
        assert discover_cases(None) == []

    def test_mixed_tree(self, tmp_path):
        # 合成树:2 静态(dcm)+ 1 动态(nii.gz)+ 若干非影像(跳过)
        # P1 静态:DICOM 头带 PatientID/SeriesInstanceUID
        _make_dcm(str(tmp_path / "P1" / "S1" / "img1.dcm"),
                  patient_id="PID001", series_uid="1.2.3.4.5")
        # P2 静态:DICOM 头缺失 → 路径推断
        _make_dcm(str(tmp_path / "P2" / "S2" / "img2.dcm"), include_phi=False)
        # P3 动态:定稿口径 {pid}/{pid}/{pid}.nii.gz
        _save_nifti(str(tmp_path / "P3" / "P3" / "P3.nii.gz"))
        # 非影像文件(应跳过)
        (tmp_path / "P1" / "labels.json").write_text("{}")
        (tmp_path / "README.md").write_text("notes")

        cases = discover_cases(str(tmp_path))
        # 期望:img1.dcm + img2.dcm + P3.nii.gz = 3 个病例
        kinds = [c["kind"] for c in cases]
        assert kinds.count("static") == 2
        assert kinds.count("dynamic") == 1
        # 非影像(.json/.md)被跳过
        assert all("input_path" in c for c in cases)
        assert not any(c["input_path"].endswith((".json", ".md")) for c in cases)

        # P1 静态:DICOM 头取到真值
        p1 = next(c for c in cases if c["patient_id"] == "PID001")
        assert p1["subdir_uid"] == "1.2.3.4.5"
        assert p1["kind"] == "static"

        # P2 静态:头缺失走路径推断(@云TODO 兜底)
        p2 = next(c for c in cases if c["patient_id"] == "P2")
        assert p2["subdir_uid"] == "S2"
        assert p2["kind"] == "static"

        # P3 动态:路径推断 patient_id=P3,subdir_uid 占位=P3(@待官方)
        p3 = [c for c in cases if c["kind"] == "dynamic" and c["patient_id"] == "P3"]
        assert len(p3) == 1
        assert p3[0]["subdir_uid"] == "P3"

    def test_deterministic_order(self, tmp_path):
        # 同目录内按文件名排序;跨目录按 os.walk 顺序(确定性)
        for name in ("c.dcm", "a.dcm", "b.dcm"):
            _make_dcm(str(tmp_path / "P" / "S" / name), patient_id="P",
                      series_uid="1.2.3.4.0")
        cases = discover_cases(str(tmp_path))
        names = [os.path.basename(c["input_path"]) for c in cases]
        assert names == ["a.dcm", "b.dcm", "c.dcm"]

    def test_nested_subdirs_all_discovered(self, tmp_path):
        # 深层嵌套:所有层级的影像都被发现
        _make_dcm(str(tmp_path / "a" / "b" / "c" / "x.dcm"),
                  patient_id="PA", series_uid="1.2.3.4.1")
        _save_nifti(str(tmp_path / "d" / "e" / "f" / "y.nii.gz"))
        cases = discover_cases(str(tmp_path))
        kinds = sorted(c["kind"] for c in cases)
        assert kinds == ["dynamic", "static"]

    def test_input_path_is_absolute(self, tmp_path):
        _make_dcm(str(tmp_path / "P" / "S" / "img.dcm"),
                  patient_id="P", series_uid="1.2.3.4.2")
        cases = discover_cases(str(tmp_path))
        assert len(cases) == 1
        assert os.path.isabs(cases[0]["input_path"])

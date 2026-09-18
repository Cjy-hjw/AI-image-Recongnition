"""detection_audit.py 单测:五机制判据(DR 6.2)+ 边界值 + 聚合。

纯逻辑测试(无 I/O/无 torch),与 test_detection_match.py 同风格。
判据阈值边界(0.10/0.15/0.35/0.45/0.50)用相邻浮点值钉死比较符方向
——nextafter 变异在 seg_only 轮已证严格/非严格比较零区分力。
"""
import math

import numpy as np
import pytest

import scripts.detection_audit as DA



# ── 五机制独立命中 ──────────────────────────────────────────────────────────

def test_mech5_zero_response():
    """机制5:soft_max<0.10 命中;主桶=mech5(最高优先级)。"""
    row = {"soft_valid": True, "soft_max": 0.05, "soft_p95": 0.04,
           "pred_ious": [], "merge_pred_count": 0}
    cls = DA.classify_miss_mechanisms(row)
    assert cls["mechanism_flags"]["mech5"] is True
    assert cls["mechanism_primary"] == "mech5"


def test_mech5_boundary_exactly_010_not_hit():
    """边界:soft_max==0.10 恰好不命中(< 0.10 严格)。相邻值命中。"""
    row = {"soft_valid": True, "soft_max": 0.10, "soft_p95": 0.09,
           "pred_ious": [], "merge_pred_count": 0}
    assert DA.classify_miss_mechanisms(row)["mechanism_flags"]["mech5"] is False
    row2 = dict(row, soft_max=math.nextafter(0.10, 0))
    assert DA.classify_miss_mechanisms(row2)["mechanism_flags"]["mech5"] is True


def test_mech1_threshold_filtered():
    """机制1:p95>=0.35 且无保留候选覆盖。soft_max 高(非零响应)。"""
    row = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
           "pred_ious": [0.05], "kept_flags": [True],
           "merge_pred_count": 0}
    cls = DA.classify_miss_mechanisms(row)
    assert cls["mechanism_flags"]["mech1"] is True
    assert cls["mechanism_primary"] == "mech1"


def test_mech1_not_hit_when_kept_cover_exists():
    """机制1 不命中:有保留候选覆盖——产物在,非阈值问题。"""
    row = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
           "pred_ious": [0.4], "kept_flags": [True],
           "merge_pred_count": 0}
    assert DA.classify_miss_mechanisms(row)["mechanism_flags"]["mech1"] is False


def test_mech1_p95_boundary():
    """边界:p95==0.35 命中(>= 含);相邻值不命中。"""
    base = {"soft_valid": True, "soft_max": 0.6, "pred_ious": [],
            "merge_pred_count": 0}
    assert DA.classify_miss_mechanisms(
        dict(base, soft_p95=0.35))["mechanism_flags"]["mech1"] is True
    assert DA.classify_miss_mechanisms(
        dict(base, soft_p95=math.nextafter(0.35, 0))
    )["mechanism_flags"]["mech1"] is False


def test_mech2_merge():
    """机制2:merge_pred_count>0(该 GT 卷入多 GT 共享预测)。"""
    row = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
           "pred_ious": [0.4], "kept_flags": [True],
           "merge_pred_count": 2}
    cls = DA.classify_miss_mechanisms(row)
    assert cls["mechanism_flags"]["mech2"] is True
    assert cls["mechanism_primary"] == "mech2"


def test_mech2_priority_over_mech1():
    """主桶优先级:mech2 > mech1。同时命中时主桶=mech2。"""
    row = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
           "pred_ious": [], "merge_pred_count": 1}
    cls = DA.classify_miss_mechanisms(row)
    assert cls["mechanism_flags"]["mech1"] is True
    assert cls["mechanism_flags"]["mech2"] is True
    assert cls["mechanism_primary"] == "mech2"


def test_mech3_fragmentation():
    """机制3:>=2 碎块各自 IoU 在 [0.15,0.45]。"""
    row = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
           "pred_ious": [0.2, 0.3], "merge_pred_count": 0}
    cls = DA.classify_miss_mechanisms(row)
    assert cls["mechanism_flags"]["mech3"] is True
    assert cls["mechanism_primary"] == "mech3"


def test_mech3_boundaries():
    """边界:0.15 含;0.45 含;两侧相邻值不命中。"""
    base = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
            "merge_pred_count": 0}
    assert DA.classify_miss_mechanisms(
        dict(base, pred_ious=[0.15, 0.30]))["mechanism_flags"]["mech3"] is True
    assert DA.classify_miss_mechanisms(
        dict(base, pred_ious=[math.nextafter(0.15, 0), 0.30])
    )["mechanism_flags"]["mech3"] is False
    assert DA.classify_miss_mechanisms(
        dict(base, pred_ious=[0.45, 0.30]))["mechanism_flags"]["mech3"] is True
    assert DA.classify_miss_mechanisms(
        dict(base, pred_ious=[math.nextafter(0.45, 1), 0.30])
    )["mechanism_flags"]["mech3"] is False


def test_mech3_single_fragment_not_hit():
    """机制3 不命中:只有 1 个碎块(需要 >=2)。"""
    row = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
           "pred_ious": [0.2], "merge_pred_count": 0}
    assert DA.classify_miss_mechanisms(row)["mechanism_flags"]["mech3"] is False


def test_mech4_drift():
    """机制4:max IoU 在 [0.35,0.5) 且质心距>等效半径。"""
    row = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
           "pred_ious": [0.4], "merge_pred_count": 0,
           "max_iou": 0.4, "centroid_distance": 10.0,
           "equivalent_radius": 5.0}
    cls = DA.classify_miss_mechanisms(row)
    assert cls["mechanism_flags"]["mech4"] is True
    assert cls["mechanism_primary"] == "mech4"


def test_mech4_iou_band_boundaries():
    """边界:0.35 含 / 0.50 不含(0.50 已是匹配线)。"""
    base = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
            "merge_pred_count": 0, "centroid_distance": 10.0,
            "equivalent_radius": 5.0}
    assert DA.classify_miss_mechanisms(
        dict(base, max_iou=0.35))["mechanism_flags"]["mech4"] is True
    assert DA.classify_miss_mechanisms(
        dict(base, max_iou=math.nextafter(0.35, 0))
    )["mechanism_flags"]["mech4"] is False
    assert DA.classify_miss_mechanisms(
        dict(base, max_iou=math.nextafter(0.50, 0))
    )["mechanism_flags"]["mech4"] is True
    assert DA.classify_miss_mechanisms(
        dict(base, max_iou=0.50))["mechanism_flags"]["mech4"] is False


def test_mech4_distance_ratio_boundary():
    """边界:质心距==等效半径 不命中(> 严格)。"""
    row = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
           "pred_ious": [0.4], "merge_pred_count": 0,
           "max_iou": 0.4, "centroid_distance": 5.0,
           "equivalent_radius": 5.0}
    assert DA.classify_miss_mechanisms(row)["mechanism_flags"]["mech4"] is False


def test_mech4_in_band_but_close_centroid_not_hit():
    """IoU 在带内但质心近(<=半径):中心对准的欠分割,非 drift。"""
    row = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
           "pred_ious": [0.4], "merge_pred_count": 0,
           "max_iou": 0.4, "centroid_distance": 2.0,
           "equivalent_radius": 5.0}
    assert DA.classify_miss_mechanisms(row)["mechanism_flags"]["mech4"] is False


# ── 不可观测 / 未分类 ───────────────────────────────────────────────────────

def test_unobservable_when_soft_missing():
    """soft 缺失 → 机制5=None(不伪装零响应)。IoU/质心证据充分时仍可裁
    mech4(软证据缺失不让位于 unobservable);全无命中且缺证据 → unobservable。"""
    row = {"soft_valid": False, "soft_p95": None, "soft_max": None,
           "pred_ious": [0.4], "merge_pred_count": 0,
           "max_iou": 0.4, "centroid_distance": 10.0,
           "equivalent_radius": 5.0}
    cls = DA.classify_miss_mechanisms(row)
    assert cls["mechanism_flags"]["mech5"] is None
    assert cls["mechanism_primary"] == "mech4"      # 质心证据在,可裁
    # 全部不可观测:软概率缺 + 无预测(IoU/质心无候选)
    row2 = {"soft_valid": False, "soft_p95": None, "soft_max": None,
            "pred_ious": [], "merge_pred_count": None,
            "max_iou": 0.0, "centroid_distance": None,
            "equivalent_radius": None}
    cls2 = DA.classify_miss_mechanisms(row2)
    assert cls2["mechanism_primary"] == "unobservable"


def test_unclassified_when_no_mechanism_hits():
    """全证据在但全未命中 → unclassified(不强塞)。"""
    row = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.2,
           "pred_ious": [0.4], "kept_flags": [True],
           "merge_pred_count": 0,
           "max_iou": 0.4, "centroid_distance": 1.0,
           "equivalent_radius": 5.0}
    cls = DA.classify_miss_mechanisms(row)
    assert all(v is False for v in cls["mechanism_flags"].values())
    assert cls["mechanism_primary"] == "unclassified"


# ── collect_gt_probability_stats ────────────────────────────────────────────

def test_prob_stats_basic():
    s = DA.collect_gt_probability_stats([0.1, 0.2, 0.3, 0.9, 1.0])
    assert s["soft_valid"] is True
    assert s["soft_max"] == 1.0
    assert s["soft_mean"] == pytest.approx(0.5)
    assert 0.2 < s["soft_p95"] <= 1.0


def test_prob_stats_empty_is_unobservable():
    """空列表 = 不可观测(soft_valid=False),不是零响应。"""
    s = DA.collect_gt_probability_stats([])
    assert s["soft_valid"] is False
    assert s["soft_max"] is None


def test_prob_stats_nan_invalid():
    """含 NaN → soft_valid=False(不产 NaN 判定)。"""
    s = DA.collect_gt_probability_stats([0.1, float("nan")])
    assert s["soft_valid"] is False


# ── compute_merge_pred_count ────────────────────────────────────────────────

def test_merge_count_pred_overlapping_two_gts():
    """机制2 预计算:一预测与 2 个 GT IoU>0.1 → 两个 GT 各 +1。"""
    iou_m = np.array([[0.4, 0.3]])
    assert DA.compute_merge_pred_count(iou_m, 0) == 1
    assert DA.compute_merge_pred_count(iou_m, 1) == 1


def test_merge_count_pred_only_one_gt():
    """预测只与 1 个 GT 重叠 → merge_count 均 0。"""
    iou_m = np.array([[0.4, 0.05]])
    assert DA.compute_merge_pred_count(iou_m, 0) == 0
    assert DA.compute_merge_pred_count(iou_m, 1) == 0


def test_merge_count_empty_matrix():
    assert DA.compute_merge_pred_count(np.zeros((0, 3)), 0) == 0


# ── compute_centroid_metrics ────────────────────────────────────────────────

def test_centroid_metrics_static():
    """GT 20x20 块 vs 偏移预测:质心距=偏移量,半径=sqrt(area/pi)。"""
    gt = np.zeros((64, 64), np.uint8)
    gt[20:40, 20:40] = 1
    pred = np.zeros((64, 64), np.uint8)
    pred[30:50, 30:50] = 1
    d, r, ni = DA.compute_centroid_metrics(gt, [pred])
    assert d == pytest.approx(np.sqrt(200), abs=1e-9)
    assert r == pytest.approx(np.sqrt(400 / np.pi), abs=1e-9)
    assert ni == 0


def test_centroid_metrics_no_pred():
    """无预测 → (None, radius, -1)。"""
    gt = np.zeros((64, 64), np.uint8)
    gt[20:40, 20:40] = 1
    d, r, ni = DA.compute_centroid_metrics(gt, [])
    assert d is None
    assert r == pytest.approx(np.sqrt(400 / np.pi), abs=1e-9)
    assert ni == -1


def test_centroid_metrics_nearest_index():
    """多预测时 nearest_pred_index = 质心最近的(漂移候选不等于最近的参考)。"""
    gt = np.zeros((64, 64), np.uint8)
    gt[20:40, 20:40] = 1
    far = np.zeros((64, 64), np.uint8); far[0:5, 0:5] = 1
    near = np.zeros((64, 64), np.uint8); near[25:45, 25:45] = 1
    d, r, ni = DA.compute_centroid_metrics(gt, [far, near])
    assert ni == 1


# ── make_gt_key ─────────────────────────────────────────────────────────────

def test_make_gt_key_stable():
    assert DA.make_gt_key("P1", "M") == "P1::M"
    assert DA.make_gt_key("P1", "M") == DA.make_gt_key("P1", "M")
    assert DA.make_gt_key("P1", "M") != DA.make_gt_key("P1", "病灶1")


# ── aggregate_mechanism_counts ──────────────────────────────────────────────

def test_aggregate_union_no_double_count():
    """并集口径:同 GT 命中多机制各记 1(分母=GT 数,禁重复计数)。"""
    rows = [{"soft_valid": True, "soft_max": 0.05, "soft_p95": 0.04,
             "pred_ious": [0.4], "merge_pred_count": 1}]
    agg = DA.aggregate_mechanism_counts(rows)
    assert agg["n_miss"] == 1
    assert agg["union_ratios"]["mech5"] == 1.0
    assert agg["union_ratios"]["mech2"] == 1.0


def test_aggregate_primary_hist():
    rows = [
        {"soft_valid": True, "soft_max": 0.05, "soft_p95": 0.04,
         "pred_ious": [], "merge_pred_count": 0},
        {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.5,
         "pred_ious": [], "merge_pred_count": 1},
    ]
    agg = DA.aggregate_mechanism_counts(rows)
    assert agg["primary_hist"]["mech5"] == 1
    assert agg["primary_hist"]["mech2"] == 1
    assert agg["n_miss"] == 2


def test_aggregate_empty():
    agg = DA.aggregate_mechanism_counts([])
    assert agg["n_miss"] == 0
    assert all(v == 0 for v in agg["primary_hist"].values())


# ── mechanism_audit 接线(diagnose_detection 端到端,复用 mock fixture)──────
# 复制 test_diagnose_detection.py 的 fixture 先例(不跨文件 import)。

pydicom = pytest.importorskip("pydicom")
nib = pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

import scripts.diagnose_detection as DD
from src.inference.nifti_io import save_binary_mask

H2 = W2 = 64


def _make_dcm2(path, arr):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    ds = Dataset()
    ds.Rows, ds.Columns = int(arr.shape[0]), int(arr.shape[1])
    ds.BitsAllocated = 8; ds.BitsStored = 8; ds.HighBit = 7
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = arr.astype(np.uint8).tobytes()
    fm = FileMetaDataset()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = generate_uid()
    fm.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta = fm
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    import os as _os
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    try:
        ds.save_as(path, write_like_original=False)
    except TypeError:
        ds.save_as(path, enforce_file_format=True)


@pytest.fixture
def mech_root(tmp_path):
    """两个静态病例:P1 漏检(零响应)+ P2 匹配(侧证 matched GT 不进分账)。"""
    root = tmp_path / "annotation"
    p1 = root / "MP1" / "MS1"
    _make_dcm2(str(p1 / "img1.dcm"), np.full((H2, W2), 120, np.uint8))
    m1 = np.zeros((H2, W2), np.uint8); m1[20:40, 20:40] = 1
    save_binary_mask(m1, str(p1 / "M_1_mask.nii.gz"))
    p2 = root / "MP2" / "MS2"
    _make_dcm2(str(p2 / "img2.dcm"), np.full((H2, W2), 90, np.uint8))
    m2 = np.zeros((H2, W2), np.uint8); m2[10:30, 10:30] = 1
    save_binary_mask(m2, str(p2 / "M_1_mask.nii.gz"))
    rows = [
        ["MP1", "MS1", "1", "M", "2左中叶", "1无", "0无", "恶", "", "V类"],
        ["MP2", "MS2", "1", "M", "4右上叶", "2甲状腺包膜", "1有", "良", "", "III类"],
    ]
    df = pd.DataFrame(rows, columns=[
        "PatientId", "SeriesUid", "图像性质", "RoiName", "主病灶位置",
        "甲状腺外侵犯", "临床T分期", "ROI良恶性", "ANNOTATION->SliceIndex", "ROI-TIRADS",
    ])
    xlsx = tmp_path / "labels.xlsx"
    df.to_excel(xlsx, sheet_name="ROI级别", index=False, startrow=1)
    return str(root), str(xlsx)


@pytest.fixture
def mech_pipeline(mech_root, tmp_path, monkeypatch):
    """P1=零响应漏检(soft 全 0.05);P2=完美匹配(预测=GT)。"""
    root, xlsx = mech_root
    MODEL_HW2 = (32, 32)

    def fake_static(model, img_np, config, calibrator, precision, device):
        mean = float(np.mean(img_np))
        hw = tuple(img_np.shape[:2])
        soft = np.full(MODEL_HW2, 0.05, np.float32)
        binary = np.zeros(MODEL_HW2, np.uint8)
        sy = MODEL_HW2[0] / hw[0]; sx = MODEL_HW2[1] / hw[1]
        boxes = []
        if mean < 100:      # P2:匹配预测
            boxes = [(10, 30, 10, 30)]
            soft_v = 0.9
        else:               # P1:漏检(低概率+无二值)
            boxes = []
            soft_v = 0.05
        for (y0, y1, x0, x1) in boxes:
            my0, my1 = int(y0 * sy), max(int(y0 * sy) + 1, int(y1 * sy))
            mx0, mx1 = int(x0 * sx), max(int(x0 * sx) + 1, int(x1 * sx))
            soft[my0:my1, mx0:mx1] = soft_v
            binary[my0:my1, mx0:mx1] = 255
        preds = {"pred_mask": soft, "mask_binary": binary,
                 "mal_prob": 0.7, "mal_probs": np.array([0.3, 0.7]),
                 "ete_probs": np.array([0.7, 0.2, 0.1, 0.0]),
                 "location_probs": np.full(7, 1 / 7),
                 "t_probs": np.array([0.4, 0.6]), "t_conf": 0.6,
                 "tirads_probs": np.array([0.1, 0.2, 0.4, 0.2, 0.1])}
        return {"preds": preds}

    monkeypatch.setattr(DD, "diagnose_static_case", fake_static)
    fake_cfg = {"data": {"img_size": 512},
                "environment": {"device": "cpu", "seed": 42},
                "calibration": {"param_file": "none.json"}}
    import src.inference.model_runner as MR
    monkeypatch.setattr(MR, "build_model", lambda c, w, d: object())
    monkeypatch.setattr(MR, "load_config", lambda p: fake_cfg)
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import src.inference.calibration as CAL
    monkeypatch.setattr(
        CAL, "TemperatureScaler",
        type("T", (), {"from_file": staticmethod(lambda p: None)}))

    def fake_build(root, xlsx, **kw):
        from src.dataset import official_loader as O
        samples = O.build_official_samples(root, xlsx)
        return None, type("DS", (), {"samples": samples})()
    monkeypatch.setattr(DD, "build_official_datasets", fake_build)
    return None


class _MArgs:
    def __init__(self, root, xlsx, log_dir):
        self.data_root = root
        self.xlsx = xlsx
        import tempfile
        self.weights = tempfile.mktemp(suffix=".pth")
        open(self.weights, "wb").close()
        self.config = "configs/thyroid_config.yaml"
        self.samples_pkl = None
        self.max_frames = 0
        self.precision = "fp32"
        self.sweep = None
        self.val_frac = 0.9
        self.seed = 42
        self.limit = None
        self.log_dir = str(log_dir)
        self.mechanism_audit = True


def test_mechanism_audit_end_to_end(mech_root, mech_pipeline, tmp_path):
    """开 --mechanism-audit:漏检 GT 进分账(零响应→mech5),matched GT 不进;
    sidecar v2 字段(schema_version/gt_key/soft_*/pred_ious)齐全。"""
    root, xlsx = mech_root
    report = DD.run_diagnosis(_MArgs(root, xlsx, tmp_path))
    ma = report["mechanism_audit"]
    assert ma["static"]["n_miss"] == 1          # P1 漏检,P2 matched 不进
    assert ma["static"]["primary_hist"]["mech5"] == 1
    assert ma["dynamic"]["n_miss"] == 0
    # sidecar v2:读回检查
    import json as _json
    sidecar = report["meta"]["sidecar"]
    rows = [_json.loads(l) for l in open(sidecar, encoding="utf-8")]
    v2_rows = [r for r in rows if r.get("schema_version") == 2]
    assert len(v2_rows) == 2
    p1 = next(r for r in v2_rows if "MP1" in r["case_key"])
    g = p1["gts"][0]
    assert g["gt_key"].startswith("MP1::")
    assert g["soft_valid"] is True
    assert g["soft_max"] < 0.10
    assert g["matched"] is False
    assert "pred_ious" in g and "centroid_distance" in g
    # clinical 标签快照在(任务 B 的标签来源)
    assert g["clinical"]["malignancy"] == 1


def test_mechanism_audit_off_zero_regression(mech_root, mech_pipeline, tmp_path):
    """关 --mechanism-audit:sidecar 行无 v2 字段(schema_version 缺),
    报告 mechanism_audit 段为占位(无漏检行,不崩)。"""
    root, xlsx = mech_root
    args = _MArgs(root, xlsx, tmp_path)
    args.mechanism_audit = False
    report = DD.run_diagnosis(args)
    ma = report["mechanism_audit"]
    assert ma["static"]["n_miss"] == 0        # 无 v2 证据 → 不进分账
    assert ma["n_evidence_missing"] >= 1     # 漏检 GT 计入证据缺失
    import json as _json
    sidecar = report["meta"]["sidecar"]
    rows = [_json.loads(l) for l in open(sidecar, encoding="utf-8")]
    assert all("schema_version" not in r for r in rows)
    # preds/gts v1 字段不受开关影响:matched/max_iou 语义不变
    p2 = next(r for r in rows if "MP2" in r["case_key"])
    assert p2["gts"][0]["matched"] is True


# ── Gate2 修复轮回归(2026-09-18,审查 11 finding 修复钉死)────────────────

def test_p0_1_shape_mismatch_is_anomaly_not_crash(tmp_path):
    """P0-1:GT mask hw 与影像 hw 不一致时,机制采集 ValueError → 单 case
    anomaly,不崩整轮、报告照常落盘(修复前崩点在 try 外=整轮作废)。"""
    # 直接构造 _collect_soft_probs_static 的 shape 冲突
    case_result = {"orig_hw": (64, 64),
                   "preds": {"pred_mask": np.full((32, 32), 0.5, np.float32)}}
    from scripts.diagnose_detection import _collect_soft_probs_static
    gts = [{"mask": np.zeros((32, 32), np.uint8),  # GT 32×32 vs 影像 64×64
            "is_main": True, "area_ratio": 0.1, "annotated_frames": None,
            "patient_id": "P", "lesion_id": "M"}]
    with pytest.raises(ValueError, match="shape"):
        _collect_soft_probs_static(case_result, gts)


def test_p0_1_dynamic_shape_mismatch_raises():
    """P0-1 动态分支:帧 hw 不齐同样 ValueError(调用方捕成 anomaly)。"""
    from scripts.diagnose_detection import _collect_soft_probs_dynamic
    per_frame = [{"frame_index": 0,
                  "preds": {"pred_mask": np.full((16, 16), 0.5, np.float32)},
                  "frame_np": np.zeros((64, 64), np.uint8)}]
    gts = [{"mask": np.zeros((1, 32, 32), np.uint8),
            "annotated_frames": [0]}]
    with pytest.raises(ValueError, match="shape"):
        _collect_soft_probs_dynamic({"per_frame": per_frame}, gts)


def test_p1_3_classification_before_rounding():
    """P1-3:机制分类在 diagnose_one_case 内已用未舍入值算好
    (mechanism_primary 字段随 sidecar 行落盘,不从 round 后副本反推)。"""
    # round(6) 窗口内的边界值:0.3499999996 round 后 0.35 会翻转 mech1
    row = {"soft_valid": True, "soft_max": 0.6,
           "soft_p95": 0.3499999996,   # <0.35(未舍入),round(6) 后 ==0.35
           "pred_ious": [], "merge_pred_count": 0}
    cls = DA.classify_miss_mechanisms(row)
    assert cls["mechanism_flags"]["mech1"] is False   # 未舍入值判:不命中
    # 落盘 round 后反推会得 True——分类必须先行(由 diagnose_one_case 保证,
    # 此处钉死库函数对未舍入值的判定方向)。
    rounded = {**row, "soft_p95": round(row["soft_p95"], 6)}
    cls_r = DA.classify_miss_mechanisms(rounded)
    assert cls_r["mechanism_flags"]["mech1"] is True  # round 确实会翻转


def test_p1_4_out_of_band_max_iou_is_false_not_unobservable():
    """P1-4:max_iou 带外(如 0.0)时 mech4=False(可判定),不再因缺质心
    拉进 unobservable。"""
    row = {"soft_valid": True, "soft_max": 0.6, "soft_p95": 0.2,
           "pred_ious": [], "merge_pred_count": 0,
           "max_iou": 0.0, "centroid_distance": None,
           "equivalent_radius": None}
    cls = DA.classify_miss_mechanisms(row)
    assert cls["mechanism_flags"]["mech4"] is False
    # 全部可判 + 无命中 → unclassified(而非 unobservable)
    assert cls["mechanism_primary"] == "unclassified"


def test_p1_2_anomaly_gt_counts_in_miss_denominator():
    """P1-2:anomaly 行的 GT 并入漏检分母(n_miss),单列 n_anomaly_missed,
    不进机制桶也不消失。再审 P1-A:取值源 = n_gt 字段(6 条 anomaly 路径
    全部在 gts 循环前早退,真实行 gts 恒空;旧行写法 gts=[...] 是真实管线
    产不出的假形状,把 no-op 挡在测试外)。"""
    from scripts.diagnose_detection import _mechanism_pool_summary
    rows = [
        # 正常行:一个漏检 GT(零响应)
        {"case_key": "ok", "is_dynamic": False, "anomaly": None,
         "gts": [{"gt_key": "P1::M", "matched": False, "soft_valid": True,
                  "soft_max": 0.05, "soft_p95": 0.04, "pred_ious": [],
                  "merge_pred_count": 0, "max_iou": 0.0}]},
        # anomaly 行:真实管线形状——gts 空、n_gt=2(推理失败早退)
        {"case_key": "bad", "is_dynamic": False, "n_gt": 2,
         "anomaly": "推理失败: X", "gts": []},
        # GT 读取失败的 anomaly 行:n_gt=0(真不可知,计 0 不猜)
        {"case_key": "badgt", "is_dynamic": False, "n_gt": 0,
         "anomaly": "GT 读取异常: X", "gts": []},
    ]
    ma = _mechanism_pool_summary(rows)
    s = ma["static"]
    assert s["n_miss"] == 3                     # 1 可分类 + 2 anomaly
    assert s["n_anomaly_missed"] == 2
    assert s["primary_hist"]["mech5"] == 1      # 机制桶只含可分类的 1 个


def test_p2_5_evidence_missing_counts_only_missed():
    """P2-5:n_evidence_missing 只计漏检 GT(v1 行 matched GT 不计)。"""
    from scripts.diagnose_detection import _mechanism_pool_summary
    rows = [{"case_key": "v1", "is_dynamic": False, "anomaly": None,
             "gts": [{"matched": True, "is_main": True},      # matched:不计
                     {"matched": False, "is_main": True}]}]   # 漏检:计 1
    ma = _mechanism_pool_summary(rows)
    assert ma["n_evidence_missing"] == 1


def test_p2_7_nan_stats_return_none():
    """P2-7:NaN 软概率 → 数值字段 None(严格 JSON 可写),soft_valid=False。"""
    s = DA.collect_gt_probability_stats([0.1, float("nan")])
    assert s["soft_valid"] is False
    assert s["soft_max"] is None and s["soft_p95"] is None
    import json
    json.dumps({"s": s}, allow_nan=False)       # 不炸


# ── cls_preds 采集(任务 B 恒定人口分类分的接线,2026-09-18)──────────────────

def test_cls_snapshot_static():
    """静态:cls_preds 直读 case_result['preds'](无额外前向)。"""
    from scripts.diagnose_detection import _classification_snapshot
    preds = {"mal_probs": np.array([0.3, 0.7]),
             "ete_probs": np.array([0.7, 0.2, 0.1, 0.0]),
             "location_probs": np.full(7, 1 / 7),
             "t_conf": 0.6,
             "tirads_probs": np.array([0.1, 0.2, 0.4, 0.2, 0.1])}
    snap = _classification_snapshot({"preds": preds}, is_dynamic=False)
    assert snap["mal_probs"] == [0.3, 0.7]
    assert snap["t_conf"] == 0.6
    assert len(snap["location_probs"]) == 7
    assert all(isinstance(x, float) for x in snap["tirads_probs"])


def test_cls_snapshot_dynamic_uses_lesion_level_mean():
    """动态:复用 v21_output._lesion_level_head_preds(per_frame 均值),
    不取关键帧单帧——两侧算法互为约束(与生产 _write_dynamic 同口径)。"""
    from scripts.diagnose_detection import _classification_snapshot
    def fp(ete):
        return {"frame_index": 0,
                "preds": {"ete_probs": np.array(ete),
                          "location_probs": np.full(7, 1 / 7),
                          "t_probs": np.array([0.5, 0.5]), "t_conf": 0.5,
                          "mal_probs": np.array([0.5, 0.5]),
                          "tirads_probs": np.full(5, 0.2)}}
    per_frame = [fp([0.8, 0.1, 0.1, 0.0]), fp([0.2, 0.1, 0.1, 0.6])]
    kf = fp([1.0, 0.0, 0.0, 0.0])
    snap = _classification_snapshot(
        {"per_frame": per_frame, "kf_preds": kf}, is_dynamic=True)
    # 帧均值 [0.5,0.1,0.1,0.3],关键帧是 [1,0,0,0]——必须拿到均值
    assert snap["ete_probs"] == pytest.approx([0.5, 0.1, 0.1, 0.3])


def test_cls_preds_written_in_sidecar_v2(mech_root, mech_pipeline, tmp_path):
    """端到端:--mechanism-audit 行含 cls_preds 五项(任务 B 消费契约)。"""
    root, xlsx = mech_root
    report = DD.run_diagnosis(_MArgs(root, xlsx, tmp_path))
    import json as _json
    sidecar = report["meta"]["sidecar"]
    rows = [_json.loads(l) for l in open(sidecar, encoding="utf-8")]
    v2 = [r for r in rows if r.get("schema_version") == 2]
    assert len(v2) == 2
    for r in v2:
        cp = r["cls_preds"]
        assert set(cp) == {"mal_probs", "ete_probs", "location_probs",
                           "t_conf", "tirads_probs"}
        assert cp["mal_probs"] == [0.3, 0.7]      # fake_static 的固定输出
        assert abs(sum(cp["ete_probs"]) - 1.0) < 1e-6
    # JSON 严格可序列化(全有限)
    _json.dumps({"cp": v2[0]["cls_preds"]}, allow_nan=False)

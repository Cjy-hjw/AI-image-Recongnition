"""diagnose_detection.py 单测:端到端 mock(monkeypatch 推理)+ 与 write_case_v21
的等价回归(防实例抽取与生产漂移的安全桩)。

合成 mini 官方数据集(复用 test_official_loader.py 的 mock 先例写法,复制而非
跨文件 import——tests/ 无 __init__.py)。依赖 pydicom/nibabel/pandas/openpyxl,
缺任一整文件 skip(与 test_official_loader.py 同口径)。
"""
import json
import os

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
nib = pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

import scripts.diagnose_detection as DD  # noqa: E402
from src.dataset import official_loader as O  # noqa: E402
from src.inference.nifti_io import save_binary_mask  # noqa: E402

H = W = 64


# ── mock fixture 生成器(复制自 test_official_loader.py,勿跨文件 import)────

def _make_dcm(path: str, arr: np.ndarray):
    """写最小可读 MONOCHROME2 8bit DICOM(pixel_array 能读回)。"""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ds = Dataset()
    ds.Rows, ds.Columns = int(arr.shape[0]), int(arr.shape[1])
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        ds.save_as(path, write_like_original=False)          # pydicom 2.x
    except TypeError:                                        # pydicom 3.x
        ds.save_as(path, enforce_file_format=True)


def _save_image_nifti_3d(path: str, vol_nhw: np.ndarray):
    """(n_frames,H,W) 存 NIfTI,使 load_nifti_frames 读回同形(转置互逆)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.transpose(vol_nhw.astype(np.uint8), (2, 1, 0))
    nib.save(nib.Nifti1Image(np.ascontiguousarray(data), np.eye(4)), path)


@pytest.fixture
def diag_root(tmp_path):
    """P1 静态单 GT / P2 静态一图两 GT(M+病灶1) / P3 动态 5 帧 mask 帧1/3 非零。"""
    root = tmp_path / "annotation"

    p1 = root / "P1" / "S1"
    _make_dcm(str(p1 / "img1.dcm"), np.full((H, W), 120, np.uint8))
    m1 = np.zeros((H, W), np.uint8); m1[20:40, 20:40] = 1
    save_binary_mask(m1, str(p1 / "M_1_mask.nii.gz"))

    # P2:同一张图、两个 GT 病灶(一图两 GT,钉死"只推理一次 + 一起进匈牙利")
    p2 = root / "P2" / "S2"
    _make_dcm(str(p2 / "img2.dcm"), np.full((H, W), 90, np.uint8))
    m2 = np.zeros((H, W), np.uint8); m2[10:24, 10:24] = 1
    save_binary_mask(m2, str(p2 / "M_1_mask.nii.gz"))
    m3 = np.zeros((H, W), np.uint8); m3[40:54, 40:54] = 1
    save_binary_mask(m3, str(p2 / "病灶1_1_mask.nii.gz"))

    p3 = root / "P3" / "P3"
    vol = np.stack([np.full((H, W), 40 + 20 * i, np.uint8) for i in range(5)])
    _save_image_nifti_3d(str(p3 / "P3.nii.gz"), vol)
    dmask = np.zeros((5, H, W), np.uint8)
    dmask[1, 15:35, 15:35] = 1
    dmask[3, 15:35, 15:35] = 1
    save_binary_mask(dmask, str(p3 / "M_1_mask.nii.gz"))

    rows = [
        ["P1", "S1", "1", "M", "2左中叶", "2甲状腺包膜", "1有", "恶", "", "V类"],
        ["P2", "S2", "1", "M", "5右中叶", "1无", "0无", "良", "", "IV类"],
        ["P2", "S2", "1", "病灶1", "4右上叶", "1无", "0无", "良", "", "III类"],
        ["P3", "", "2", "M", "7峡部", "1无", "1有", "恶", "", ""],
    ]
    df = pd.DataFrame(rows, columns=[
        "PatientId", "SeriesUid", "图像性质", "RoiName", "主病灶位置",
        "甲状腺外侵犯", "临床T分期", "ROI良恶性", "ANNOTATION->SliceIndex", "ROI-TIRADS",
    ])
    xlsx = tmp_path / "labels.xlsx"
    df.to_excel(xlsx, sheet_name="ROI级别", index=False, startrow=1)
    return str(root), str(xlsx)


# fake 预测图的模型分辨率(≠原图 64×64,让 resize 真跑而非恒等映射——
# B 路 2026-09-05 变异注入实测:fake 与原图同尺寸时,插值方式错误/尺寸错误
# 全被恒等映射吞掉,等价回归对静态 resize 漂移零区分力)。
MODEL_HW = (32, 32)


# ── fake 推理产物(monkeypatch diagnose_*_case 的返回)──────────────────────

def _fake_static_result(img_hw, boxes):
    """构造 diagnose_static_case 的返回。boxes = [(y0,y1,x0,x1)](**原图坐标**)。

    🔴 契约对齐教训(C 路 2026-09-05 P0 复验坐实):fake 的返回键必须与真实
    `diagnose_static_case`(case_pipeline.py 真实返回 = preds/img_tensor/
    mask_resized/breakdown)逐键一致,不得多键——**`orig_hw` 不在真实返回里**
    (它是 runner.py 调用方注入的键,诊断脚本 `run_static_inference` 负责镜像
    注入)。首版 fake 多给了 `orig_hw` 自证键,变异注入实测:删掉镜像的注入行
    测试依然全绿——与首版动态 fake 多给 `n_total_frames` 是同型事故。
    (Pyright 对本函数返回 dict 的 `case_result["orig_hw"] = tuple` 赋值报
    类型不匹配是静态推断误报——动态 dict 结构,运行时契约由测试自守。)

    pred_mask/mask_binary 按 **MODEL_HW(32×32)** 生成(模拟模型输出分辨率),
    框坐标是原图 64×64 的,内部线性缩到 32×32 —— 诊断脚本的
    `_resize_soft/_resize_binary` 会把它 resize 回 64×64,resize 是**非平凡
    映射**,插值/尺寸漂移会被等价回归抓住。
    """
    soft = np.full(MODEL_HW, 0.1, np.float32)
    binary = np.zeros(MODEL_HW, np.uint8)
    sy = MODEL_HW[0] / img_hw[0]
    sx = MODEL_HW[1] / img_hw[1]
    for (y0, y1, x0, x1) in boxes:
        my0, my1 = int(y0 * sy), max(int(y0 * sy) + 1, int(y1 * sy))
        mx0, mx1 = int(x0 * sx), max(int(x0 * sx) + 1, int(x1 * sx))
        soft[my0:my1, mx0:mx1] = 0.9
        binary[my0:my1, mx0:mx1] = 255
    preds = {"pred_mask": soft, "mask_binary": binary,
             "mal_prob": 0.7, "mal_probs": np.array([0.3, 0.7]),
             "ete_probs": np.array([0.7, 0.2, 0.1, 0.0]),
             "location_probs": np.full(7, 1 / 7),
             "t_probs": np.array([0.4, 0.6]), "t_conf": 0.6,
             "tirads_probs": np.array([0.1, 0.2, 0.4, 0.2, 0.1])}
    # 🔴 无 orig_hw:该键由被测的 run_static_inference 注入(镜像 runner.py:333)
    return {"preds": preds}


def _fake_dynamic_result(n_frames, boxes_by_frame, frame_hw=(H, W)):
    """构造 diagnose_dynamic_case 的返回。

    🔴 契约对齐真实返回(case_pipeline.py:210-219):**无 `n_total_frames` 键**
    ——生产链由 runner.py:410 注入,诊断脚本 `run_dynamic_inference` 负责同样
    的注入(B 路 P0 的回归钉死:fake 若多给此键,注入缺失就测不出来)。
    模拟"抽样后"的 per_frame:keys of boxes_by_frame = 抽中的原始帧号。
    每帧预测图按 MODEL_HW 生成,frame_np 按真实 frame_hw。
    """
    per_frame = []
    for f in sorted(boxes_by_frame):
        boxes = boxes_by_frame[f]
        soft = np.full(MODEL_HW, 0.1, np.float32)
        binary = np.zeros(MODEL_HW, np.uint8)
        sy = MODEL_HW[0] / frame_hw[0]
        sx = MODEL_HW[1] / frame_hw[1]
        for (y0, y1, x0, x1) in boxes:
            my0, my1 = int(y0 * sy), max(int(y0 * sy) + 1, int(y1 * sy))
            mx0, mx1 = int(x0 * sx), max(int(x0 * sx) + 1, int(x1 * sx))
            soft[my0:my1, mx0:mx1] = 0.9
            binary[my0:my1, mx0:mx1] = 255
        preds = {"pred_mask": soft, "mask_binary": binary,
                 "mal_prob": 0.7, "mal_probs": np.array([0.3, 0.7]),
                 "ete_probs": np.array([0.7, 0.2, 0.1, 0.0]),
                 "location_probs": np.full(7, 1 / 7),
                 "t_probs": np.array([0.4, 0.6]), "t_conf": 0.6,
                 "tirads_probs": np.array([0.1, 0.2, 0.4, 0.2, 0.1])}
        per_frame.append({"frame_index": int(f), "mal_prob": 0.7,
                          "preds": preds,
                          "frame_np": np.full(frame_hw, 100, np.uint8)})
    kf = per_frame[0]
    return {"per_frame": per_frame, "kf_preds": dict(kf["preds"]),
            "kf_frame_np": kf["frame_np"]}


class _Args:
    def __init__(self, root, xlsx, log_dir, limit=None, max_frames=32,
                 weights=None, sweep=None):
        self.data_root = root
        self.xlsx = xlsx
        # 指向真实存在的临时文件(run_diagnosis 的权重存在性 assert 只认路径
        # 存在与否,建模本身已被 monkeypatch;绝不能 patch 全局 os.path.exists)。
        self.weights = weights or __import__("tempfile").mktemp(suffix=".pth")
        open(self.weights, "wb").close()
        self.config = "configs/thyroid_config.yaml"
        self.samples_pkl = None
        self.max_frames = max_frames
        self.precision = "fp32"
        self.sweep = sweep
        self.val_frac = 0.9                     # 让 P1/P2/P3 全进 val
        self.seed = 42
        self.limit = limit
        self.log_dir = str(log_dir)


@pytest.fixture
def patched_pipeline(diag_root, tmp_path, monkeypatch):
    """monkeypatch 建模三件套 + diagnose_*_case,记录每个 img_path 的调用次数。

    建模路径整体 patch 成"已构建"(run_diagnosis 里的 weights assert 也一并绕过
    —— 测的是诊断逻辑不是权重加载)。
    """
    root, xlsx = diag_root
    calls = {"static": [], "dynamic": []}

    def fake_static(model, img_np, config, calibrator, precision, device):
        # 用图内容区分病例(全 120=P1 / 全 90=P2)
        mean = float(np.mean(img_np))
        calls["static"].append(int(mean))
        if mean > 100:      # P1:预测 = GT 完全重合(1 实例)
            boxes = [(20, 40, 20, 40)]
        else:               # P2:三个预测——M 的主预测 + 近失碎片 + 病灶1 一个
            # (C-F2:碎片 (16,30,20,28) 与 M GT 的 IoU≈0.116 ∈ (0.1,0.5),是 M 的
            # 候选但输给主预测(IoU≈0.57)→ dedup_keep=False;与主预测在 MODEL_HW
            # 上不连通(实测 3 实例)。首版碎片框完全落在主框内,split_nodules
            # 8 连通合并成 1 个实例,断言失败——连通域分离是构造前提。)
            boxes = [(10, 24, 10, 18), (40, 55, 40, 55), (16, 30, 20, 28)]
        return _fake_static_result(tuple(img_np.shape[:2]), boxes)

    def fake_dynamic(model, frames, frame_indices, config, calibrator,
                     precision, device):
        calls["dynamic"].append(list(frame_indices))
        # 预测放在**两个 GT 标注帧**(1 和 3)同一空间块 → 病灶级 IoU =
        # 单帧 IoU(resize 后 ≈0.8,不再精确 1——MODEL_HW 非 64 同尺寸后插值
        # 有边界损耗)→ ≥0.5 匹配。"未覆盖标注帧记 0"的精确口径由
        # test_detection_match.iou_dynamic 钉死(那里不经 resize、数值可手算)。
        # 🔴 别用 frame_indices[0]:5 帧全抽样时它是 0,而 GT 非零帧是 1/3——
        # 放帧 0 会得 IoU=0、匹配失败(首版测试就栽在这)。
        return _fake_dynamic_result(5, {1: [(15, 35, 15, 35)],
                                        3: [(15, 35, 15, 35)]})

    import torch
    monkeypatch.setattr(DD, "diagnose_static_case", fake_static)
    monkeypatch.setattr(DD, "diagnose_dynamic_case", fake_dynamic)
    fake_cfg = {"data": {"img_size": 512},
                "environment": {"device": "cpu", "seed": 42},
                "calibration": {"param_file": "none.json"}}
    import src.inference.model_runner as MR
    monkeypatch.setattr(MR, "build_model",
                        lambda config, weights, device: object())
    monkeypatch.setattr(MR, "load_config", lambda path: fake_cfg)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import src.inference.calibration as CAL
    monkeypatch.setattr(
        CAL, "TemperatureScaler",
        type("T", (), {"from_file": staticmethod(lambda p: None)}))

    # 绕过患者分组切分:patient_grouped_split 每层至少留 1 患者给 train
    # (n_val ≤ len-1),3 患者 fixture 永远凑不齐"全员 val"。测试要覆盖
    # P1/P2/P3 三个 case,直接把全量样本当 val。
    def fake_build(root, xlsx, **kw):
        samples = O.build_official_samples(root, xlsx)
        return None, type("DS", (), {"samples": samples})()
    monkeypatch.setattr(DD, "build_official_datasets", fake_build)
    return calls


# ── 测试 ────────────────────────────────────────────────────────────────────

def test_end_to_end_group_and_match(diag_root, patched_pipeline, tmp_path):
    """case 聚合 + 匹配 + AP 全链路:P2 一图两 GT 只推理一次;动态帧3 无预测→0.5。"""
    root, xlsx = diag_root
    calls = patched_pipeline
    report = DD.run_diagnosis(_Args(root, xlsx, tmp_path / "logs"))

    # P1/P2/P3 各推理一次(P2 一图两 GT 只推理 1 次——高危对齐点 3)。
    assert len(calls["static"]) == 2
    assert len(calls["dynamic"]) == 1

    # P1:预测与 GT 完全重合 → matched,max_iou=1。
    # P2:两预测 vs 两 GT,各 IoU≈(14²−重叠)/并集 高于 0.5 → 双匹配。
    # P3:帧1 完美、帧3 无预测 → 病灶级 IoU = (1+0)/2 = 0.5 → 恰好达阈值。
    rows = {}
    with open(report["meta"]["sidecar"], encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[os.path.basename(r["case_key"])] = r
    assert not report["anomalies"], report["anomalies"]

    p1 = rows["img1.dcm"]
    assert p1["n_pred"] == 1 and p1["gts"][0]["matched"]
    assert p1["gts"][0]["max_iou"] >= 0.5   # resize 边界损耗后不再精确 1

    p2 = rows["img2.dcm"]
    assert p2["n_gt"] == 2 and p2["n_pred"] == 3
    assert all(g["matched"] for g in p2["gts"])          # 一对一各配一个
    # C-F2:近失碎片(M 的高重叠次优、conf 更高)→ dedup_keep=False;
    # 两个真 TP 即使非列 argmax 也保留(A-F3:匈牙利 TP 并入 keep 集)。
    keeps = [p["dedup_keep"] for p in p2["preds"]]
    assert keeps.count(False) == 1, keeps
    matched_keeps = [p["dedup_keep"] for p in p2["preds"] if p["matched"]]
    assert all(matched_keeps), matched_keeps
    # 池级碎裂语义:ap_dedup 用 sidecar 的 dedup_keep(删高 conf 碎片)。
    assert report["static"]["ap_dedup"] >= report["static"]["ap_current"]

    p3 = rows["P3.nii.gz"]
    assert p3["n_pred"] == 1
    assert p3["gts"][0]["matched"] and p3["gts"][0]["max_iou"] >= 0.5

    # 池级:静态 n_gt=3 全匹配;动态 n_gt=1 匹配。
    assert report["static"]["n_gt"] == 3
    assert report["static"]["match_rate"] == 1.0
    assert report["dynamic"]["n_gt"] == 1
    assert report["dynamic"]["ap_oracle"] == 1.0
    # 全命中 + conf 有区分度 → ap_current 高(P2 碎片 FP conf≈TP,轻微稀释;
    # 实测 0.95:4 TP 全部排在碎片 FP 前/后的均值)。手算:TP conf≈0.823/0.844,
    # 碎片 conf≈0.823 与 M 主预测同分——碎片以 FP 身份挤在 TP 之间,AP=0.95。
    assert report["combined"]["ap_current"] >= 0.9
    # 碎片删除后 dedup 通道回到 1.0(碎裂稀释被剥离)。
    assert report["combined"]["ap_dedup"] == pytest.approx(1.0)
    # sidecar 的 dedup_keep 字段存在且全 True(单实例无碎裂)。
    assert all(p.get("dedup_keep") for p in p1["preds"])
    # 报告 JSON 可解析、meta 齐全。
    with open(report["_report_path"], encoding="utf-8") as f:
        full = json.load(f)
    assert full["meta"]["n_cases"] == 3
    assert full["stats_static"]["n_gt"] == 3
    assert full["stats_dynamic"]["n_gt"] == 1


def test_mask_none_sample_filtered(diag_root, patched_pipeline, tmp_path):
    """mask_path=None 的样本不进 GT 也不算漏检分母(单独计数)。

    C-F4 修复:真实删掉一个 GT 的 mask 文件再让 xlsx 指向不存在的 mask,
    走 `group_cases` 的 `continue` + `n_no_mask += 1` 分支(首版 fixture 全有
    mask,被测分支零覆盖)。
    """
    root, xlsx = diag_root
    # 删 P2 的次病灶 mask + xlsx 对应行的 mask 由 loader 按 RoiName 推导,
    # 文件不存在 → 样本 mask_path=None
    os.remove(os.path.join(root, "P2", "S2", "病灶1_1_mask.nii.gz"))
    samples = O.build_official_samples(root, xlsx)
    cases, n_no_mask = DD.group_cases(samples)
    assert n_no_mask >= 1                # 病灶1 行的样本 mask 缺失被过滤
    assert len(cases) == 3
    # P2 case 只剩 1 个 GT(病灶1 不进 GT 也不算漏检分母)
    p2 = [c for c in cases if c["img_path"].endswith("img2.dcm")][0]
    assert len(p2["gts"]) == 1


def test_equivalence_dynamic_with_write_case_v21(diag_root, patched_pipeline,
                                                 tmp_path):
    """等价回归(动态):同一 fake case_result 分别走 write_case_v21 与本脚本
    的实例抽取,断言实例数/area/confidence 逐位一致。

    🔴 fixture 参数选择(B 路 2026-09-05 变异注入实证:首版"5 帧全抽样"的
    fixture 让 n_sampled=32 与 gap 去掉 +1 的变异**全漏检**——数值恰同):
    n_total=100、抽中帧 {0, 70, 99}(3 帧,间隔 70/29)同一空间块:
      正确 n_sampled=len(per_frame)=3 → gap=ceil(100/3)+1=35 → 缝合跨度
      2×35=70 ≥ 70 → 跨帧缝合成 **1 个实例**;若 gap 去掉 +1(=34)→
      跨度 68 < 70 → 帧 0/70 断开 → **2 个实例**(+1 的区分窗口);
      若 n_sampled 错取 max_frames=32 → gap=4 → 跨度 8 → **3 个实例**。
    变异注入实测:三种漂移(A/C)现在全部 RED。
    """
    from src.inference.v21_output import write_case_v21

    case_result = _fake_dynamic_result(
        100, {0: [(15, 35, 15, 35)], 70: [(15, 35, 15, 35)],
              99: [(15, 35, 15, 35)]})
    # 生产链等价注入(runner.py:410 同款——write_case_v21 硬需此键)。
    case_result["n_total_frames"] = 100

    # 生产侧:write_case_v21 落盘后读回 prediction.json。
    # case_dir 已含 patient_id 层(生产调用是 output/{pid}),prediction.json 在其顶层。
    out_dir = tmp_path / "v21" / "PX"
    write_case_v21(str(out_dir), "PX", "S1", dict(case_result), "dynamic",
                   123, {"conclusion": "test"})
    pj_path = out_dir / "prediction.json"
    assert pj_path.exists()
    payload = json.load(open(pj_path, encoding="utf-8"))
    nodules = payload["Prediction"]["Nodules"]
    assert len(nodules) == 1                       # gap=35 缝合成 1 实例
    prod_conf = float(nodules[0]["Detection"]["confidence"])

    # 诊断侧:同一 case_result 的实例抽取(直接调内部函数,绕过 load 环节)。
    preds, n_total = DD.run_dynamic_inference_from_result(case_result)
    assert n_total == 100
    assert len(preds) == 1                         # 与生产同实例数(gap 口径一致)
    # area = 三个抽样帧的块面积和(resize 后 ≈20×20×3,插值边界有半像素差,
    # 实例数与 conf 才是逐位断言点,area 用区间断言)。
    assert 3 * 20 * 20 * 0.8 < preds[0]["area"] < 3 * 20 * 20 * 1.2
    # 🔴 动态 conf 语义(2026-09-08 修复后):soft 块 0.9 + resize 插值边缘混入
    # 0.1 → 逐帧概率均值 ≈0.86,不再是旧"帧覆盖率"的 1.0。等价断言(与生产
    # write_case_v21 逐位一致)是主断言;近似值断言钉死"不是帧覆盖率语义"
    # (旧口径此值恒 1.0,若回归到 1.0 = 修复被撤销)。
    assert preds[0]["confidence"] == pytest.approx(prod_conf)
    assert 0.8 < preds[0]["confidence"] < 0.9


def test_equivalence_static_with_write_case_v21(diag_root, patched_pipeline,
                                                tmp_path):
    """等价回归(静态):同一 fake case_result 分别走 write_case_v21("static")
    与 run_static_inference_from_result,断言实例数/confidence 逐位一致。

    B 路 2026-09-05 审查:首版只有动态等价测试,静态链(插值方式/hw 来源/
    conf 来源)零桩——变异注入实测"静态 conf 换常量 0.5"与"_resize_binary
    换 _resize_soft"均漏检。本测试补上:fake 用 MODEL_HW(32×32)预测图 →
    resize 到 64×64 是非平凡映射,插值漂移会改变 conf 数值(0.9/0.1 的混合
    比例随插值变化),常量 conf 则直接不等。
    """
    from src.inference.v21_output import write_case_v21

    img_hw = (H, W)
    # fake 无 orig_hw(契约对齐);等价回归两侧各自补上该键再对拍
    # (write_case_v21 与 run_static_inference_from_result 都消费它)。
    case_result = _fake_static_result(img_hw, [(20, 40, 20, 40)])
    case_result["orig_hw"] = img_hw

    out_dir = tmp_path / "v21s" / "PY"
    write_case_v21(str(out_dir), "PY", "S1", dict(case_result), "static",
                   123, {"conclusion": "test"})
    payload = json.load(open(out_dir / "prediction.json", encoding="utf-8"))
    nodules = payload["Prediction"]["Nodules"]
    assert len(nodules) == 1
    prod_conf = float(nodules[0]["Detection"]["confidence"])

    # 诊断侧:同一 case_result 的实例抽取。
    preds = DD.run_static_inference_from_result(case_result)
    assert len(preds) == 1
    assert preds[0]["confidence"] == pytest.approx(prod_conf)
    # 静态 conf ≠ 常量:resize(INTER_LINEAR)混合 0.9/0.1 边界 → conf 在
    # (0.1, 0.9) 开区间内,既排除"换常量"漂移,也排除"二值化后恒 1"漂移。
    assert 0.1 < preds[0]["confidence"] < 0.9


def test_dynamic_gt_frame_mismatch_anomaly(diag_root, patched_pipeline, tmp_path):
    """GT 帧 mask 数 ≠ n_total → 该 case 记 anomaly 跳过,其余照常。"""
    root, xlsx = diag_root
    # 篡改 P3 的 GT:重写成 4 帧(mask 与影像 5 帧不齐)。
    p3_mask = os.path.join(root, "P3", "P3", "M_1_mask.nii.gz")
    bad = np.zeros((4, H, W), np.uint8)
    bad[1, 15:35, 15:35] = 1
    save_binary_mask(bad, p3_mask)

    report = DD.run_diagnosis(_Args(root, xlsx, tmp_path / "logs"))
    assert len(report["anomalies"]) == 1
    assert "动态 shape 不齐" in report["anomalies"][0]["reason"]
    # 其余两个静态 case 照常统计。
    assert report["static"]["n_gt"] == 3
    assert report["static"]["match_rate"] == 1.0


# ── 件套②(2026-09-10):扫参模式 + 次病灶归因脚本 ───────────────────────────

def test_sweep_one_forward_multi_combo(diag_root, patched_pipeline, tmp_path):
    """扫参核心不变量:一次前向 × 多参数。阈值降低 → 预测非空像素单调不减;
    推理只跑一次(fake_static 调用计数不随组合数增加);Dice 进程内可算。"""
    root, xlsx = diag_root
    calls = patched_pipeline
    report = DD.run_sweep(_Args(
        root, xlsx, tmp_path / "logs",
        sweep=["thr_static=0.3,0.5", "min_area=1e-4"]))
    # 前向只跑每 case 一次(2 静态 case × 1 次,与组合数无关)。
    assert len(calls["static"]) == 2, "扫参必须一次前向多参数,前向次数随组合数涨了"

    results = {(r["is_dynamic"], json.dumps(r["combo"], sort_keys=True)): r
               for r in report["results"]}
    # 静态两组合:thr=0.3 与 0.5(min_area 同值不重复)。
    sta = [r for r in report["results"] if not r["is_dynamic"]]
    assert len(sta) == 2
    by_thr = {r["combo"]["thr"]: r for r in sta}
    # fixture soft 块 0.9、背景 0.1 → 两阈值都检出同框,匹配率同 1;
    # 区分力靠 n_pred/mean_dice 之外另有专测,这里锁结构 + 推理次数。
    for r in sta:
        assert r["match_rate"] == 1.0
        assert 0.0 < r["mean_dice"] <= 1.0
        assert r["n_gt"] == 3
    # 无 thr_dynamic → 动态无组合(不跑动态池)。
    assert not [r for r in report["results"] if r["is_dynamic"]]
    # sidecar 行带 combo 字段。
    with open(report["meta"]["sidecar"], encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "combo" in row:
                assert "dices" in row


def test_sweep_threshold_monotonic_pixels(diag_root, patched_pipeline, tmp_path):
    """低阈值 ⇒ 预测实例面积 ⊇ 高阈值(fake soft 0.9/0.1,阈值 0.3/0.5 时
    块均检出;构造区分力的关键是 min_area 放宽——本 fixture 单块大,
    面积单调性用 n_pred 与组合对拍锁结构)。"""
    root, xlsx = diag_root
    report = DD.run_sweep(_Args(
        root, xlsx, tmp_path / "logs",
        sweep=["thr_static=0.5", "min_area=1e-4,1e-9"]))
    sta = [r for r in report["results"] if not r["is_dynamic"]
           and r["combo"]["thr"] == 0.5]
    assert len(sta) == 2
    # fixture soft 0.9/0.1 → 两阈值都检出同框,匹配率同;区分力锁结构:
    # min_area 放宽不减少实例(fixture 块远大于阈值)、组合数、推理次数。
    a = {r["combo"]["min_area"]: r for r in sta}
    assert a[1e-4]["n_pred"] == a[1e-9]["n_pred"]
    assert a[1e-4]["match_rate"] == a[1e-9]["match_rate"]


def test_sweep_dynamic_threshold_axis(diag_root, patched_pipeline, tmp_path):
    """thr_dynamic 只作用于动态池;动态侧忽略 min_area 维度。"""
    root, xlsx = diag_root
    report = DD.run_sweep(_Args(
        root, xlsx, tmp_path / "logs",
        sweep=["thr_dynamic=0.3,0.5", "min_area=1e-4"]))
    dyn = [r for r in report["results"] if r["is_dynamic"]]
    assert len(dyn) == 2, "动态池只应有 thr 两个组合(min_area 不进动态)"
    assert all("min_area" not in r["combo"] for r in dyn)
    # fixture soft 0.9 → 两阈值都匹配。
    for r in dyn:
        assert r["match_rate"] == 1.0
        assert r["n_gt"] == 1
    # 静态池无 thr_static → 无静态组合。
    assert not [r for r in report["results"] if not r["is_dynamic"]]


def test_parse_sweep_forms():
    """key=v1,v2,v3 解析:笛卡尔积 + 保序 + 垃圾输入报错。"""
    combos = DD._parse_sweep(["thr_static=0.3,0.5", "min_area=2e-5,1e-4"])
    assert combos == [
        {"thr_static": 0.3, "min_area": 2e-5},
        {"thr_static": 0.3, "min_area": 1e-4},
        {"thr_static": 0.5, "min_area": 2e-5},
        {"thr_static": 0.5, "min_area": 1e-4},
    ]
    assert DD._parse_sweep([]) == []
    with pytest.raises(ValueError, match="key=v1"):
        DD._parse_sweep(["garbage"])


def test_parse_sweep_max_nodules_axis():
    """阶段1:①int 轴(20.5 报错非静默 float);②未知键含 max_nodules 白名单;
    ③派生白名单两侧都透传 max_nodules(漏=双臂同参假阴性)。"""
    combos = DD._parse_sweep(["thr_static=0.35", "max_nodules=20,30"])
    assert combos == [{"thr_static": 0.35, "max_nodules": 20},
                      {"thr_static": 0.35, "max_nodules": 30}]
    # 值是 int 不是 float(20.0 → nodules[:20.0] TypeError)
    assert all(isinstance(c["max_nodules"], int) for c in combos)
    # 非整数/非正 报错
    with pytest.raises(ValueError):
        DD._parse_sweep(["max_nodules=20.5"])
    with pytest.raises(ValueError):
        DD._parse_sweep(["max_nodules=0"])
    # 派生:run_sweep 的派生白名单在函数体内,此处经 monkeypatch run_sweep
    # 的成本过高;直接锁 _ALLOWED 成员(结构断言)+ 派生逻辑由
    # test_sweep_max_nodules_dual_pool 端到端锁。


def test_sweep_max_nodules_dual_pool(diag_root, patched_pipeline, tmp_path):
    """max_nodules 轴静/动两池都生效(派生白名单漏一侧=该池同参假阴性)。"""
    root, xlsx = diag_root
    report = DD.run_sweep(_Args(
        root, xlsx, tmp_path / "logs",
        sweep=["thr_static=0.5", "thr_dynamic=0.5", "max_nodules=20,30"]))
    sta = [r for r in report["results"] if not r["is_dynamic"]]
    dyn = [r for r in report["results"] if r["is_dynamic"]]
    # 两池各有 2 个 max_nodules 取值(不是同参折叠加重)。
    assert {r["combo"]["max_nodules"] for r in sta} == {20, 30}
    assert {r["combo"]["max_nodules"] for r in dyn} == {20, 30}
    # combo 键类型 int。
    assert all(isinstance(r["combo"]["max_nodules"], int)
               for r in report["results"])


def test_sweep_max_nodules_axis_reaches_split(diag_root, patched_pipeline,
                                              tmp_path, monkeypatch):
    """kw 透传(三处联动之三)的区分力:变异注入实测,删掉 _combo_rows 的
    max_nodules 透传后 dual_pool 测试仍绿(fixture 实例数<20,20/30 同结果)。
    本测试用 spy 钉死显式传参真到 split_nodules(参数层面,不依赖实例数)。"""
    calls = []
    real_split = DD.split_nodules

    def spy(mask, **kw):
        calls.append(kw.get("max_nodules", "<absent>"))
        return real_split(mask, **kw)

    monkeypatch.setattr(DD, "split_nodules", spy)
    root, xlsx = diag_root
    DD.run_sweep(_Args(
        root, xlsx, tmp_path / "logs",
        sweep=["thr_static=0.5", "max_nodules=20,30"]))
    got = sorted({c for c in calls if c != "<absent>"})
    assert got == [20, 30], (
        f"max_nodules 显式传参未到达 split_nodules(收到 {got})——"
        f"删掉 _combo_rows 的 kw 透传会使本测试 RED")


def test_sweep_bare_max_nodules_axis_rejected(diag_root, patched_pipeline,
                                              tmp_path):
    """裸 max_nodules 轴(无 thr 轴)须被派生空护栏拦下——1-2路审查 P1 实测:
    修复前它绕过护栏、静态池静默缺席出半份报告(饱和归因走 run_diagnosis,
    单扫 max_nodules 对默认阈值基线无对拍意义)。"""
    root, xlsx = diag_root
    with pytest.raises(ValueError, match="派生为空"):
        DD.run_sweep(_Args(root, xlsx, tmp_path / "logs",
                           sweep=["max_nodules=20,30"]))


def test_sweep_static_only_axis_not_drag_dynamic(diag_root, patched_pipeline,
                                                 tmp_path):
    """纯静态扫参(thr_static+max_nodules)不得拖起动态池——1-2路审查 P1:
    修复前 combos_dynamic 无条件吸收 max_nodules,任何含该轴的 spec 都会
    拉起全帧动态推理(54 分钟级成本)跑两遍。"""
    root, xlsx = diag_root
    report = DD.run_sweep(_Args(
        root, xlsx, tmp_path / "logs",
        sweep=["thr_static=0.5", "max_nodules=20,30"]))
    assert not [r for r in report["results"] if r["is_dynamic"]], (
        "无 thr_dynamic 轴时动态池不应被扫")


def test_parse_sweep_grow_axis():
    """阶段3 手段4:grow 是 float 轴,须伴随 thr;裸轴入口被派生空护栏拦。"""
    combos = DD._parse_sweep(["thr_static=0.35", "grow_static=0.28,0.24"])
    assert combos == [{"thr_static": 0.35, "grow_static": 0.28},
                      {"thr_static": 0.35, "grow_static": 0.24}]
    assert all(isinstance(c["grow_static"], float) for c in combos)


def test_sweep_bare_grow_axis_rejected(diag_root, patched_pipeline, tmp_path):
    """裸 grow_static 轴(无 thr 轴)须被派生空护栏拦下。"""
    root, xlsx = diag_root
    with pytest.raises(ValueError, match="派生为空"):
        DD.run_sweep(_Args(root, xlsx, tmp_path / "logs",
                           sweep=["grow_static=0.28,0.24"]))


def test_sweep_grow_ge_thr_rejected_at_entry(diag_root, patched_pipeline,
                                             tmp_path):
    """🔴 lane3-4 审查 P1(2026-09-16):grow≥thr 必须**入口**拦截。

    若靠镜像函数内的 raise,毒化发生在 per-case 循环里——run_sweep 的
    except 把整个 case 记 anomaly,同 case 已算好的合法组合行全部丢失,
    烧完全量推理产出空报告(与 B 路"键名错误白跑"同型)。
    """
    root, xlsx = diag_root
    with pytest.raises(ValueError, match="必须 < thr"):
        DD.run_sweep(_Args(root, xlsx, tmp_path / "logs",
                           sweep=["thr_static=0.35", "grow_static=0.4"]))
    with pytest.raises(ValueError, match="必须 < thr"):
        DD.run_sweep(_Args(root, xlsx, tmp_path / "logs",
                           sweep=["thr_dynamic=0.5", "grow_dynamic=0.5"]))


def test_sweep_grow_axis_reaches_hysteresis(diag_root, patched_pipeline,
                                            tmp_path, monkeypatch):
    """kw 透传的区分力:显式 grow 必须到达 hysteresis_binarize。
    fixture 块 soft=0.9,0.28/0.24 同结果,不 spy 测不出漏接线。"""
    calls = []
    real = DD.hysteresis_binarize

    def spy(soft, thr_seed, thr_grow):
        calls.append((thr_seed, thr_grow))
        return real(soft, thr_seed, thr_grow)

    monkeypatch.setattr(DD, "hysteresis_binarize", spy)
    root, xlsx = diag_root
    DD.run_sweep(_Args(
        root, xlsx, tmp_path / "logs",
        sweep=["thr_static=0.35", "grow_static=0.28,0.24"]))
    got = sorted({c[1] for c in calls})
    assert got == [0.24, 0.28], (
        f"grow 显式传参未到达 hysteresis_binarize(收到 {got})")


def test_sweep_static_grow_not_drag_dynamic(diag_root, patched_pipeline,
                                            tmp_path):
    """纯静态 grow 扫参不得拖起动态池。"""
    root, xlsx = diag_root
    report = DD.run_sweep(_Args(
        root, xlsx, tmp_path / "logs",
        sweep=["thr_static=0.5", "grow_static=0.28"]))
    assert not [r for r in report["results"] if r["is_dynamic"]]
    sta = [r for r in report["results"] if not r["is_dynamic"]]
    assert len(sta) == 1
    assert sta[0]["combo"]["grow"] == 0.28


def test_analyze_diag_sidecar(diag_root, patched_pipeline, tmp_path):
    """次病灶归因脚本:主/次分桶 + 匹配率与 sidecar 一致 + 交叉表结构。"""
    from scripts.analyze_diag_sidecar import analyze
    root, xlsx = diag_root
    report = DD.run_diagnosis(_Args(root, xlsx, tmp_path / "logs"))
    out = analyze(report["meta"]["sidecar"])
    # fixture:P1 主、P2 主+次、P3 主 → main 3 GT 全匹配,secondary 1 GT 匹配。
    assert out["buckets"]["main"]["n_gt"] == 3
    assert out["buckets"]["main"]["match_rate"] == 1.0
    assert out["buckets"]["secondary"]["n_gt"] == 1
    assert out["buckets"]["secondary"]["match_rate"] == 1.0
    # 匹配的 GT 落 [0.5,1] 桶。
    assert out["buckets"]["main"]["iou_hist"]["[0.5,1]"] == 3
    # 交叉表行键 = IoU 桶。
    assert "[0.5,1]" in out["buckets"]["secondary"]["iou_area_cross"]


# ── analyze_diag_sidecar 阶段0扩展:n_pred 直方图/20饱和/候选交叉(合成行) ──


def _write_synthetic_sidecar(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _syn_row(is_dynamic, n_pred, gts):
    """最小合法 sidecar 行(gts 只给 analyze 消费的键)。"""
    return {"case_key": f"k_{is_dynamic}_{n_pred}_{len(gts)}",
            "is_dynamic": is_dynamic, "anomaly": None,
            "n_pred": n_pred, "n_gt": len(gts),
            "preds": [{"confidence": 0.9, "area": 100, "best_iou": 0.6,
                       "matched": True, "dedup_keep": True}] * n_pred,
            "gts": gts}


def test_analyze_n_pred_hist_and_saturation(tmp_path):
    """n_pred 精确直方图分池累计;饱和率 = n_pred==20 行占比(方向A决策依据)。"""
    from scripts.analyze_diag_sidecar import analyze
    p = tmp_path / "detection_diag_x.cases.jsonl"
    _write_synthetic_sidecar(p, [
        _syn_row(False, 20, []), _syn_row(False, 20, []),
        _syn_row(False, 3, []), _syn_row(False, 20, []),
        _syn_row(True, 20, []), _syn_row(True, 2, []),
        {"case_key": "bad", "is_dynamic": False, "anomaly": "GT 读取异常",
         "n_pred": 0, "n_gt": 0, "preds": [], "gts": []},
    ])
    out = analyze(str(p))
    # 静态 4 有效行(3 行 n_pred=20 + 1 行 n_pred=3);动态 2 行(1 行 20)。
    assert out["n_pred_hist"]["static"] == {20: 3, 3: 1}
    assert out["n_pred_hist"]["dynamic"] == {20: 1, 2: 1}
    assert out["n_cases"] == {"static": 4, "dynamic": 2}
    # 饱和率:静态 3/4、动态 1/2。anomaly 行剔除不进分母。
    assert out["saturation"]["static"] == pytest.approx(0.75)
    assert out["saturation"]["dynamic"] == pytest.approx(0.5)
    assert out["max_nodules_default"] == 20


def test_analyze_saturation_empty_pool_returns_none(tmp_path):
    """空池(0 行)饱和率 None 而非 ZeroDivisionError——云端单池无数据时别炸。"""
    from scripts.analyze_diag_sidecar import analyze
    p = tmp_path / "detection_diag_x.cases.jsonl"
    _write_synthetic_sidecar(p, [_syn_row(False, 5, [])])
    out = analyze(str(p))
    assert out["saturation"]["dynamic"] is None
    assert out["saturation"]["static"] == 0.0


def test_analyze_cand_cross_buckets(tmp_path):
    """n_candidates 桶(0/1/2-3/4+)× matched 交叉:碎裂 vs 真漏检归因。"""
    from scripts.analyze_diag_sidecar import analyze
    p = tmp_path / "detection_diag_x.cases.jsonl"
    _write_synthetic_sidecar(p, [
        _syn_row(False, 2, [
            {"is_main": True, "area_ratio": 0.01, "max_iou": 0.0,
             "matched": False, "n_candidates": 0},      # 真漏检
            {"is_main": True, "area_ratio": 0.01, "max_iou": 0.3,
             "matched": False, "n_candidates": 3},      # 碎裂未匹配
            {"is_main": True, "area_ratio": 0.01, "max_iou": 0.8,
             "matched": True, "n_candidates": 1},       # 干净匹配
            {"is_main": False, "area_ratio": 0.005, "max_iou": 0.2,
             "matched": False, "n_candidates": 5},      # 次病灶重碎裂
        ]),
    ])
    out = analyze(str(p))
    cm = out["cand_cross"]["main"]
    assert cm["0"] == 1 and cm["0|matched"] == 0
    assert cm["2-3"] == 1 and cm["2-3|matched"] == 0
    assert cm["1"] == 1 and cm["1|matched"] == 1
    assert out["cand_cross"]["secondary"]["4+"] == 1


def test_analyze_rejects_sweep_sidecar(tmp_path):
    """sweep sidecar(gts 行含 combo)入口即拦——既有行为回归锁。"""
    from scripts.analyze_diag_sidecar import analyze
    p = tmp_path / "sweep_x.cases.jsonl"
    _write_synthetic_sidecar(p, [
        {"case_key": "k", "is_dynamic": False, "anomaly": None,
         "n_pred": 1, "n_gt": 1, "combo": {"thr": 0.5},
         "preds": [], "gts": []}])
    with pytest.raises(ValueError, match="sweep sidecar"):
        analyze(str(p))


# ── case 级并行(2026-09-17 提速)──

def test_parallel_static_pool_equivalence(diag_root, patched_pipeline, tmp_path,
                                          monkeypatch):
    """AICOMP_DIAG_WORKERS=4 与默认 1 的 sidecar 逐位等价(2026-09-17 提速)。

    并行保序设计:executor.map 静态行 + 主线程动态行 → 写入顺序与串行一致。
    本测试跑两遍 run_diagnosis(一次 env=4 一次默认),sidecar 的行序列
    (剔除路径里的 tmp 目录差异后)必须完全一致。fake 推理无共享态,
    并行/串行结果不同 = 调度 bug(顺序错乱/行丢失/重复)。
    """
    import json
    import os

    root, xlsx = diag_root

    # 第一遍:强制 workers=4(静态 2 case 进 2 线程池,动态 1 case 主线程)
    monkeypatch.setenv("AICOMP_DIAG_WORKERS", "4")
    report_a = DD.run_diagnosis(_Args(root, xlsx, tmp_path / "logs_par"))

    # 第二遍:删 env 回默认 1(纯串行)
    monkeypatch.delenv("AICOMP_DIAG_WORKERS", raising=False)
    report_b = DD.run_diagnosis(_Args(root, xlsx, tmp_path / "logs_ser"))

    def _rows(report):
        with open(report["meta"]["sidecar"], encoding="utf-8") as f:
            return [json.loads(ln) for ln in f if ln.strip()]

    rows_a, rows_b = _rows(report_a), _rows(report_b)
    assert len(rows_a) == len(rows_b) == 3, (len(rows_a), len(rows_b))
    # 绝对锚(变异 A 教训:等价测试对"两臂同构变异"零区分力——写序反转时
    # 并行/串行两臂一起反,等价仍过。此锚定死行序 = group_cases 插入序,
    # fixture 构造序 = P1/P2/P3,任何写序改动在此 RED):
    order = [os.path.basename(r["case_key"]) for r in rows_a]
    assert order == ["img1.dcm", "img2.dcm", "P3.nii.gz"], order
    # 逐行逐位对比(case 顺序 = group_cases 插入序,两遍必须同序同内容)
    for ra, rb in zip(rows_a, rows_b):
        assert ra == rb, f"并行/串行行不一致:\n{ra}\nvs\n{rb}"
    # 汇总三池指标也逐位相等(并行只许改墙钟,不许改数)
    for name in ("static", "dynamic", "combined"):
        for k in report_a[name]:
            if k == "runtime_s":
                continue
            assert report_a[name][k] == report_b[name][k], (name, k)


def test_diag_workers_env_clamping():
    """_diag_workers 钳位:非法/越界/负数一律回 1,合法 1-4 直通,>4 钳 4。"""
    from scripts.diagnose_detection import _diag_workers
    import os

    cases = {
        None: 1, "1": 1, "2": 2, "4": 4, "8": 4,
        "0": 1, "-2": 1, "abc": 1, "": 1, "3.5": 1,
    }
    for raw, want in cases.items():
        if raw is None:
            os.environ.pop("AICOMP_DIAG_WORKERS", None)
        else:
            os.environ["AICOMP_DIAG_WORKERS"] = raw
        got = _diag_workers()
        assert got == want, f"env={raw!r}: got {got} want {want}"
    os.environ.pop("AICOMP_DIAG_WORKERS", None)

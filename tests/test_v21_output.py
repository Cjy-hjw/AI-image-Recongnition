"""`src/inference/v21_output.py` 回归测试。

覆盖 V2.1 单病例落盘的核心契约:

- **目录结构**:`{case_dir}/{subdir_uid}/nodule_NNN.nii.gz` 逐结节生成,`prediction.json`
  落在 `{case_dir}/` 下;
- **mask 合规**:NIfTI 尺寸=原图 H×W、值域仅 {0,1}(规范硬约束;非方阵 H≠W 才能区分
  维度转置错误,与 test_nifti_io 同口径);
- **JSON 合规**:`validate_prediction_json` 通过,`MaskFiles` ↔ `Nodules[].MaskFile`
  一一对应;
- **阴性**:空 mask → 空 Nodules 的合规 json,不崩;
- **路径安全**:MaskFile 是相对路径、无 `..`、非绝对(官方评测在 Linux 按 `/` 解析,
  故断言用正斜杠形态);
- **静/动各至少一例**;动态断言 3D mask 的 Z 轴=n_total_frames(全帧回填)。

依赖 nibabel/cv2/scipy,缺则整个模块 skip(与 test_nifti_io/test_nodule_split 同口径)。
合成最小 case_result(不跑真模型):preds 的各字段按 `decode_predictions` 的真实 key
构造,breakdown 只造 `total_points`(`_tirads_class_from_breakdown` 唯一消费的键)。
"""

import json
import os

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("cv2")
pytest.importorskip("scipy")

from src.inference import v21_output  # noqa: E402
from src.inference.prediction_json import (  # noqa: E402
    NUM_LOCATIONS,
    PredictionSchemaError,
    validate_prediction_json,
)

# 非方阵:H=13, W=17。任何尺寸断言都必须能区分 (13,17) 与 (17,13)(转置错误在方阵下隐形)。
_H, _W = 13, 17
# 模型输出尺寸(与 preprocess_image 的 img_size 对齐,不必等于原图)。
_MODEL = 8


def _soft_and_binary_single(blob_model):
    """造模型输出尺寸的软概率图 + 二值图,中央一个连通块。

    `blob_model`: (m, m) 的 0/1,软图在其上叠加 0.8/0.1 形成清晰前景。
    """
    soft = np.full((_MODEL, _MODEL), 0.05, dtype=np.float32)
    soft[blob_model != 0] = 0.9
    binary = (soft > 0.5).astype(np.uint8) * 255
    return soft, binary


def _static_case_result(soft_model, binary_model, orig_hw=(_H, _W),
                        mal_prob=0.72, total_points=5):
    """合成静态 case_result。preds 的 key 严格对齐 `decode_predictions` 返回。"""
    preds = {
        "pred_mask": soft_model,
        "mask_binary": binary_model,
        "comp_idx": 0, "echo_idx": 1, "shape_idx": 0, "margin_idx": 1,
        "comp_prob": np.array([0.6, 0.4]), "echo_prob": np.array([0.3, 0.7]),
        "shape_prob": np.array([0.5, 0.5]), "margin_prob": np.array([0.2, 0.8]),
        "foci_prob": np.array([0.1, 0.9]),
        "mal_prob": float(mal_prob),
        "mal_probs": np.array([1.0 - mal_prob, mal_prob]),
    }
    breakdown = {
        "composition": ("实性", 2), "echogenicity": ("低回声", 2),
        "shape": ("纵横比<1", 0), "margin": ("分叶/不规则", 1),
        "foci": ["点状强回声 (+1分)"],
        "total_points": int(total_points),
        "tr_level": "TR4 (中度可疑 / Moderately suspicious)",
        "risk": "恶性风险 ~9.1%", "action": "...",
    }
    return {
        "preds": preds,
        "img_tensor": np.zeros((1, 1, _MODEL, _MODEL), np.float32),  # 仅占位,不用于尺寸
        "mask_resized": binary_model,
        "breakdown": breakdown,
        "orig_hw": orig_hw,
        "gatekeeper_audit": {"skipped": False},
        "is_dicom": False,
    }


def _dynamic_case_result(n_total=6, sampled_idx=(0, 2, 4), orig_hw=(_H, _W),
                         mal_prob=0.65, total_points=3):
    """合成动态 case_result:key 严格对齐 `diagnose_dynamic_case` 返回 + A2 注入的
    `n_total_frames`/`gatekeeper_audit`。每帧 preds 带软/二值 mask。"""
    per_frame = []
    for i, fi in enumerate(sampled_idx):
        soft = np.full((_MODEL, _MODEL), 0.05, dtype=np.float32)
        # 同一结节在抽样帧上位置微移(模拟跨帧连续出现)。
        r = 2 + i
        soft[r:r + 3, 3:6] = 0.85
        binary = (soft > 0.5).astype(np.uint8) * 255
        per_frame.append({
            "frame_index": int(fi),
            "mal_prob": float(mal_prob),
            "preds": {
                "pred_mask": soft, "mask_binary": binary,
                "comp_idx": 0, "echo_idx": 0, "shape_idx": 0, "margin_idx": 0,
                "foci_prob": np.array([0.1]),
                "mal_prob": float(mal_prob),
                "mal_probs": np.array([1.0 - mal_prob, mal_prob]),
            },
            "frame_np": np.zeros(orig_hw, np.uint8),
        })
    kf = per_frame[0]
    breakdown = {
        "composition": ("实性", 2), "echogenicity": ("等回声", 1),
        "shape": ("纵横比<1", 0), "margin": ("光滑", 0), "foci": ["无 (+0分)"],
        "total_points": int(total_points), "tr_level": "TR3", "risk": "...", "action": "...",
    }
    return {
        "per_frame": per_frame,
        "agg": {"keyframe_index": 0, "n_frames": len(per_frame)},
        "kf_preds": kf["preds"],
        "kf_frame_np": kf["frame_np"],
        "kf_tensor": None,
        "failed_frames": [],
        "breakdown": breakdown,
        "mask_resized": kf["preds"]["mask_binary"],
        "orig_hw": orig_hw,
        "n_total_frames": int(n_total),
        "gatekeeper_audit": {"skipped": False},
    }


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 静态路径
# --------------------------------------------------------------------------- #

class TestStatic:
    def test_dir_structure_and_mask_binary(self, tmp_path):
        """目录结构 + mask NIfTI 尺寸=原图 + 二值 {0,1} + MaskFiles↔Nodules 一一对应。"""
        blob = np.zeros((_MODEL, _MODEL), np.uint8)
        blob[2:5, 3:6] = 1  # 模型分辨率下的一个小块
        soft, binary = _soft_and_binary_single(blob)
        cr = _static_case_result(soft, binary)

        case_dir = str(tmp_path / "P001")
        sub = "SERIES_X"
        v21_output.write_case_v21(case_dir, "P001", sub, cr, "static", 123)

        # prediction.json 落 case_dir 根
        pj_path = os.path.join(case_dir, "prediction.json")
        assert os.path.isfile(pj_path), "prediction.json 必须落在 case_dir 根"
        payload = _read_json(pj_path)
        validate_prediction_json(payload)  # 不抛即合规
        assert payload["StudyUID"] == "P001"
        assert payload["ProcessingTime_ms"] == 123

        nodules = payload["Prediction"]["Nodules"]
        mask_files = payload["Prediction"]["MaskFiles"]
        assert len(nodules) >= 1, "应至少检出一个结节"
        # MaskFiles ↔ Nodules[].MaskFile 一一对应
        assert sorted(mask_files) == sorted(n["MaskFile"] for n in nodules)

        # 每个 mask 文件存在、尺寸=原图 (W,H) header、值域 {0,1}
        for rel in mask_files:
            # 相对路径必须无 ..、非绝对、用正斜杠
            assert ".." not in rel.split("/"), f"MaskFile 含 '..': {rel}"
            assert not os.path.isabs(rel), f"MaskFile 须相对路径: {rel}"
            assert "\\" not in rel, f"MaskFile 须用正斜杠(Linux 评测): {rel}"
            assert rel.startswith(f"{sub}/"), f"MaskFile 须以 subdir_uid 开头: {rel}"
            abs_path = os.path.join(case_dir, rel)
            assert os.path.isfile(abs_path), f"mask 文件未生成: {abs_path}"
            img = nib.load(abs_path)
            assert img.shape == (_W, _H), (
                f"mask header 应为 (W,H)=({_W},{_H}),得 {img.shape}(漏转置会得到 (H,W))")
            data = np.asarray(img.dataobj)
            assert set(np.unique(data).tolist()) <= {0, 1}, (
                f"mask 值域须 {{0,1}},得 {np.unique(data).tolist()}")

    def test_negative_case_empty_nodules(self, tmp_path):
        """阴性(空 mask)→ 空 Nodules + 空 MaskFiles 的合规 json,不崩、不建结节文件。"""
        soft = np.full((_MODEL, _MODEL), 0.05, np.float32)
        binary = np.zeros((_MODEL, _MODEL), np.uint8)
        cr = _static_case_result(soft, binary)

        case_dir = str(tmp_path / "P_NEG")
        sub = "SERIES_N"
        v21_output.write_case_v21(case_dir, "P_NEG", sub, cr, "static", 50)

        payload = _read_json(os.path.join(case_dir, "prediction.json"))
        validate_prediction_json(payload)
        assert payload["Prediction"]["Nodules"] == []
        assert payload["Prediction"]["MaskFiles"] == []
        # 阴性不应产出任何 nodule_*.nii.gz
        sub_dir = os.path.join(case_dir, sub)
        if os.path.isdir(sub_dir):
            assert not [f for f in os.listdir(sub_dir) if f.endswith(".nii.gz")], \
                "阴性病例不应生成结节 mask 文件"

    def test_multiple_nodules_enumerated(self, tmp_path):
        """两个分离连通块 → 两个结节条目,NoduleID 与 mask 文件一一对应且互不重叠。"""
        soft = np.full((_MODEL, _MODEL), 0.05, np.float32)
        soft[1:3, 1:3] = 0.9
        soft[5:7, 5:7] = 0.85
        binary = (soft > 0.5).astype(np.uint8) * 255
        cr = _static_case_result(soft, binary)

        case_dir = str(tmp_path / "P_MULTI")
        v21_output.write_case_v21(case_dir, "P_MULTI", "S", cr, "static", 80)
        payload = _read_json(os.path.join(case_dir, "prediction.json"))
        validate_prediction_json(payload)
        nodules = payload["Prediction"]["Nodules"]
        assert len(nodules) == 2, f"两个分离块应得 2 个结节,得 {len(nodules)}"
        ids = [n["NoduleID"] for n in nodules]
        assert len(set(ids)) == 2, "NoduleID 不得重复"
        # 实例 mask 互不重叠
        masks = [np.transpose(np.asarray(nib.load(
            os.path.join(case_dir, n["MaskFile"])).dataobj)) for n in nodules]
        assert int((masks[0] & masks[1]).sum()) == 0, "实例 mask 不得重叠"

    def test_interpretation_passthrough(self, tmp_path):
        blob = np.zeros((_MODEL, _MODEL), np.uint8)
        blob[2:5, 3:6] = 1
        soft, binary = _soft_and_binary_single(blob)
        cr = _static_case_result(soft, binary)
        interp = {"Conclusion": "恶性可疑", "KeyFindings": ["低回声", "边缘不规则"]}
        case_dir = str(tmp_path / "P_INT")
        v21_output.write_case_v21(case_dir, "P_INT", "S", cr, "static", 60,
                                  interpretation=interp)
        payload = _read_json(os.path.join(case_dir, "prediction.json"))
        assert payload.get("Interpretation") == interp

    def test_missing_orig_hw_raises(self, tmp_path):
        """静态缺 orig_hw 必须报错(不能用 img_tensor 的模型输入尺寸冒充原图)。"""
        blob = np.zeros((_MODEL, _MODEL), np.uint8)
        blob[2:5, 3:6] = 1
        soft, binary = _soft_and_binary_single(blob)
        cr = _static_case_result(soft, binary)
        del cr["orig_hw"]
        with pytest.raises(v21_output.V21OutputError):
            v21_output.write_case_v21(str(tmp_path / "P_X"), "P_X", "S", cr, "static", 10)

    def test_rejected_case_raises(self, tmp_path):
        """rejected 占位 case_result 不应走正常 V2.1 落盘。"""
        cr = {"rejected": True, "reason": "gatekeeper", "preds": None}
        with pytest.raises(v21_output.V21OutputError):
            v21_output.write_case_v21(str(tmp_path / "P_R"), "P_R", "S", cr, "static", 10)

    def test_atomic_write_no_tmp_left(self, tmp_path):
        """原子写完成后不得残留 .tmp 文件。"""
        blob = np.zeros((_MODEL, _MODEL), np.uint8)
        blob[2:5, 3:6] = 1
        soft, binary = _soft_and_binary_single(blob)
        cr = _static_case_result(soft, binary)
        case_dir = str(tmp_path / "P_A")
        v21_output.write_case_v21(case_dir, "P_A", "S", cr, "static", 40)
        leftovers = [f for f in os.listdir(case_dir) if f.endswith(".tmp")]
        assert not leftovers, f"原子写后不得残留 .tmp: {leftovers}"


# --------------------------------------------------------------------------- #
# 动态路径
# --------------------------------------------------------------------------- #

class TestDynamic:
    def test_full_length_3d_mask(self, tmp_path):
        """动态:3D mask 的 Z 轴=n_total_frames(全帧回填),尺寸 (W,H,n_total),二值。"""
        n_total = 6
        cr = _dynamic_case_result(n_total=n_total, sampled_idx=(0, 2, 4))
        case_dir = str(tmp_path / "D001")
        sub = "ACC_Y"
        v21_output.write_case_v21(case_dir, "D001", sub, cr, "dynamic", 200)

        payload = _read_json(os.path.join(case_dir, "prediction.json"))
        validate_prediction_json(payload)
        assert payload["StudyUID"] == "D001"
        assert payload["ProcessingTime_ms"] == 200

        nodules = payload["Prediction"]["Nodules"]
        mask_files = payload["Prediction"]["MaskFiles"]
        assert len(nodules) >= 1, "动态应至少检出一个跨帧结节"
        assert sorted(mask_files) == sorted(n["MaskFile"] for n in nodules)

        for rel in mask_files:
            assert ".." not in rel.split("/")
            assert not os.path.isabs(rel)
            assert "\\" not in rel
            abs_path = os.path.join(case_dir, rel)
            assert os.path.isfile(abs_path)
            img = nib.load(abs_path)
            # 3D header: (W, H, n_total) —— 帧轴落 Z,长度=全帧
            assert img.shape == (_W, _H, n_total), (
                f"动态 mask 应为 (W,H,n_total)=({_W},{_H},{n_total}),得 {img.shape}")
            data = np.asarray(img.dataobj)
            assert set(np.unique(data).tolist()) <= {0, 1}

    def test_negative_dynamic_empty(self, tmp_path):
        """动态阴性:所有帧 mask 全空 → 空 Nodules 合规 json。"""
        n_total = 5
        cr = _dynamic_case_result(n_total=n_total, sampled_idx=(0, 1, 2))
        # 把所有帧的 mask 清空
        for f in cr["per_frame"]:
            f["preds"]["pred_mask"] = np.full((_MODEL, _MODEL), 0.05, np.float32)
            f["preds"]["mask_binary"] = np.zeros((_MODEL, _MODEL), np.uint8)
        case_dir = str(tmp_path / "D_NEG")
        v21_output.write_case_v21(case_dir, "D_NEG", "ACC_N", cr, "dynamic", 90)
        payload = _read_json(os.path.join(case_dir, "prediction.json"))
        validate_prediction_json(payload)
        assert payload["Prediction"]["Nodules"] == []
        assert payload["Prediction"]["MaskFiles"] == []

    def test_missing_n_total_frames_falls_back(self, tmp_path):
        """缺 n_total_frames 时回退 max(frame_indices)+1,不崩,Z 长度=回退值。"""
        cr = _dynamic_case_result(n_total=6, sampled_idx=(0, 2, 4))
        del cr["n_total_frames"]
        case_dir = str(tmp_path / "D_FB")
        v21_output.write_case_v21(case_dir, "D_FB", "ACC_F", cr, "dynamic", 70)
        payload = _read_json(os.path.join(case_dir, "prediction.json"))
        validate_prediction_json(payload)
        rel = payload["Prediction"]["MaskFiles"][0]
        img = nib.load(os.path.join(case_dir, rel))
        # 回退 Z = max(0,2,4)+1 = 5
        assert img.shape[2] == 5, f"回退 Z 应为 max_idx+1=5,得 {img.shape[2]}"


# --------------------------------------------------------------------------- #
# 占位维度契约(训练阶段4未训,结构合规即可)
# --------------------------------------------------------------------------- #

class TestPlaceholderDims:
    def test_ete_location_t_present_and_valid(self, tmp_path):
        """ETE/Location/T 三占位维度必须存在且落在合法编码内,LocationProbs 和为 1。"""
        blob = np.zeros((_MODEL, _MODEL), np.uint8)
        blob[2:5, 3:6] = 1
        soft, binary = _soft_and_binary_single(blob)
        cr = _static_case_result(soft, binary)
        case_dir = str(tmp_path / "P_DIM")
        v21_output.write_case_v21(case_dir, "P_DIM", "S", cr, "static", 30)
        payload = _read_json(os.path.join(case_dir, "prediction.json"))
        n = payload["Prediction"]["Nodules"][0]
        assert n["ExtrathyroidalExtension"] in (1, 2, 3, 4)
        assert 1 <= n["Location"] <= NUM_LOCATIONS
        assert len(n["LocationProbs"]) == NUM_LOCATIONS
        assert abs(sum(n["LocationProbs"]) - 1.0) < 1e-6
        assert n["T_ClinSig"]["T"] in (0, 1)
        assert 0.0 <= n["T_ClinSig"]["T_conf"] <= 1.0
        # 占位下 Malignancy 用真实预测
        assert 0.0 <= n["Malignancy"]["confidence"] <= 1.0
        assert isinstance(n["Malignancy"]["predicted"], bool)

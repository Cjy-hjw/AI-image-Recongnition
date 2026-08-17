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


def _head_probs(ete=None, loc=None, t=None, tirads=None, t_conf=0.3):
    """官方四头概率(#116b)。默认造**非均匀、argmax 非 idx0** 的分布。

    🔴 默认值刻意避开 idx0:接线代码是 `argmax+1`,若某处漏写 `+1`,idx0 的
    one-hot 恰好也能得到"看似正确"的码 1 —— 用 idx0 当默认等于零区分力。
    默认 argmax:ETE→idx1(码2 包膜) / Location→idx4(码5 右中叶) /
    T→idx1(T=1,**不 +1**) / TI-RADS→idx3(码4)。
    """
    return {
        "ete_probs": np.array(ete if ete is not None else [0.1, 0.6, 0.2, 0.1]),
        "location_probs": np.array(
            loc if loc is not None else [0.05, 0.05, 0.1, 0.1, 0.5, 0.1, 0.1]),
        "t_probs": np.array(t if t is not None else [0.3, 0.7]),
        "tirads_probs": np.array(
            tirads if tirads is not None else [0.05, 0.05, 0.1, 0.7, 0.1]),
        "t_conf": float(t_conf),
    }


def _static_case_result(soft_model, binary_model, orig_hw=(_H, _W),
                        mal_prob=0.72, total_points=5, head_probs=None):
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
        **(head_probs if head_probs is not None else _head_probs()),
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
                         mal_prob=0.65, total_points=3, per_frame_heads=None):
    """合成动态 case_result:key 严格对齐 `diagnose_dynamic_case` 返回 + A2 注入的
    `n_total_frames`/`gatekeeper_audit`。每帧 preds 带软/二值 mask。

    `per_frame_heads`: 可选,与 `sampled_idx` 等长的 `_head_probs()` 列表,用于构造
    "各帧四头概率不同"的场景(测病灶级均值)。缺省时各帧同一份默认分布。
    """
    per_frame = []
    for i, fi in enumerate(sampled_idx):
        soft = np.full((_MODEL, _MODEL), 0.05, dtype=np.float32)
        # 同一结节在抽样帧上位置微移(模拟跨帧连续出现)。
        r = 2 + i
        soft[r:r + 3, 3:6] = 0.85
        binary = (soft > 0.5).astype(np.uint8) * 255
        heads = (per_frame_heads[i] if per_frame_heads is not None
                 else _head_probs())
        per_frame.append({
            "frame_index": int(fi),
            "mal_prob": float(mal_prob),
            "preds": {
                "pred_mask": soft, "mask_binary": binary,
                "comp_idx": 0, "echo_idx": 0, "shape_idx": 0, "margin_idx": 0,
                "foci_prob": np.array([0.1]),
                "mal_prob": float(mal_prob),
                "mal_probs": np.array([1.0 - mal_prob, mal_prob]),
                **heads,
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
# 四头真实输出接线契约(backlog #116b,2026-08-17)
#
# 这些测试守的是**编码映射**,不是预测质量。历史上这里是 `TestPlaceholderDims`,
# 只断言"落在合法范围内"——接线后那种断言**仍会绿**,但它守的是已消失的语义
# (占位维度结构合规),对 off-by-one 零区分力。现改为逐项断言精确期望码。
# --------------------------------------------------------------------------- #

class TestOfficialHeadWiring:
    """四头真实概率 → prediction.json 的端到端映射。

    🔴 **必须从 `write_case_v21` 入口测到落盘 JSON**,不能只测
    `build_nodule_entry` 单函数 —— 本项目有过"修复没打到生产路径终点"的事故
    (memory `fix-must-reach-production-path-not-just-function`),且改动三的
    对冲键正是"单函数测试全绿、整链不通"的形态(调用处不透传即静默丢弃)。
    """

    def _write_and_read(self, tmp_path, name, head_probs=None):
        blob = np.zeros((_MODEL, _MODEL), np.uint8)
        blob[2:5, 3:6] = 1
        soft, binary = _soft_and_binary_single(blob)
        cr = _static_case_result(soft, binary, head_probs=head_probs)
        case_dir = str(tmp_path / name)
        v21_output.write_case_v21(case_dir, name, "S", cr, "static", 30)
        payload = _read_json(os.path.join(case_dir, "prediction.json"))
        validate_prediction_json(payload)
        return payload["Prediction"]["Nodules"][0]

    def test_schema_fields_present_and_valid(self, tmp_path):
        """基础合规:四项字段存在、落在合法编码内、LocationProbs 和为 1。"""
        n = self._write_and_read(tmp_path, "P_DIM")
        assert n["ExtrathyroidalExtension"] in (1, 2, 3, 4)
        assert 1 <= n["Location"] <= NUM_LOCATIONS
        assert len(n["LocationProbs"]) == NUM_LOCATIONS
        assert abs(sum(n["LocationProbs"]) - 1.0) < 1e-6
        assert n["T_ClinSig"]["T"] in (0, 1)
        assert 0.0 <= n["T_ClinSig"]["T_conf"] <= 1.0
        assert 0.0 <= n["Malignancy"]["confidence"] <= 1.0
        assert isinstance(n["Malignancy"]["predicted"], bool)

    def test_not_placeholder_values(self, tmp_path):
        """🔴 最可靠的"接上了没"单条判据:LocationProbs 不是均匀分布。

        接线前生产恒输出 ETE=1 / 均匀 LocationProbs / T=0 / T_conf=0.0。
        本测试用默认 fixture(argmax 全部避开 idx0)断言这四项都不再是那组值。
        """
        n = self._write_and_read(tmp_path, "P_REAL")
        uniform = 1.0 / NUM_LOCATIONS
        assert not all(abs(p - uniform) < 1e-6 for p in n["LocationProbs"]), \
            "LocationProbs 仍是均匀分布 —— 四头没接上(与权重质量无关)"
        assert n["ExtrathyroidalExtension"] == 2, "ETE 仍是占位恒 1"
        assert n["Location"] == 5, "Location 仍是占位 argmax(uniform)+1=1"
        assert n["T_ClinSig"]["T"] == 1, "T 仍是占位恒 0"
        assert n["T_ClinSig"]["T_conf"] > 0.0, "T_conf 仍是占位恒 0.0"

    @pytest.mark.parametrize("idx,expected", [(1, 2), (2, 3), (3, 4)])
    def test_ete_code_is_argmax_plus_one(self, tmp_path, idx, expected):
        """ETE:`code = argmax + 1`。参数化**全部避开 idx0** —— 用 idx0 测等于没测
        (漏写 `+1` 时 idx0 恰好也"对",off-by-one 逃逸)。"""
        probs = [0.05] * 4
        probs[idx] = 0.85
        n = self._write_and_read(tmp_path, f"P_ETE{idx}",
                                 head_probs=_head_probs(ete=probs))
        assert n["ExtrathyroidalExtension"] == expected

    @pytest.mark.parametrize("idx,expected", [(1, 2), (3, 4), (4, 5)])
    def test_tirads_class_is_argmax_plus_one(self, tmp_path, idx, expected):
        """TI-RADS:`code = argmax + 1`,且**不再走 ACR 概念积分反推**。

        fixture 的 `total_points=5` 按旧 `_tirads_class_from_breakdown` 会得 4;
        本测试构造的期望码与之不同(idx1→2 / idx4→5),故能区分两条路径。
        """
        probs = [0.05] * 5
        probs[idx] = 0.8
        n = self._write_and_read(tmp_path, f"P_TR{idx}",
                                 head_probs=_head_probs(tirads=probs))
        assert n["TI_RADS_Class"] == expected

    @pytest.mark.parametrize("idx,expected", [(1, 2), (4, 5), (6, 7)])
    def test_location_is_argmax_plus_one(self, tmp_path, idx, expected):
        """Location:`code = argmax + 1`;idx6 → 7(峡部)是上边界。"""
        probs = [0.05] * NUM_LOCATIONS
        probs[idx] = 0.7
        n = self._write_and_read(tmp_path, f"P_LOC{idx}",
                                 head_probs=_head_probs(loc=probs))
        assert n["Location"] == expected
        # 概率数组本身必须原样透传(归一化后 argmax 不变)
        assert n["LocationProbs"].index(max(n["LocationProbs"])) == idx

    def test_t_clinsig_is_argmax_without_plus_one(self, tmp_path):
        """🔴 T 是唯一**不 +1** 的:官方 `T∈{0,1}` 与 0-based 下标同值。

        双向钉子:`[0.3,0.7]`→1 且 `[0.7,0.3]`→0。正向防误写 `+1`
        (会得 2,被 prediction_json 拦成整例 ERROR);反向防 argmax 取反。
        """
        n1 = self._write_and_read(tmp_path, "P_T1",
                                  head_probs=_head_probs(t=[0.3, 0.7]))
        assert n1["T_ClinSig"]["T"] == 1
        n0 = self._write_and_read(tmp_path, "P_T0",
                                  head_probs=_head_probs(t=[0.7, 0.3]))
        assert n0["T_ClinSig"]["T"] == 0

    def test_t_conf_passthrough(self, tmp_path):
        """T_conf 直传 `preds["t_conf"]`(= P(T=1)),不重算、不取 max(t_probs)。"""
        n = self._write_and_read(tmp_path, "P_TC",
                                 head_probs=_head_probs(t=[0.3, 0.7], t_conf=0.42))
        assert abs(n["T_ClinSig"]["T_conf"] - 0.42) < 1e-9


class TestHedgeKeys:
    """#116a 对冲键 `TI_RADS_Probs` / `ETE_Probs` 的三处联通(改动三)。

    🔴 只测 `build_nodule_entry` 单函数会漏掉"调用处没透传"这一步 —— 那正是
    审查抓到的缺陷形态:参数加了、写出逻辑也有,但 `_build_entries_and_save_masks`
    不传,于是键静默消失,而 `validate_prediction_json` 只查缺失不查多余 → 不报错。
    """

    def test_hedge_keys_written_end_to_end(self, tmp_path):
        blob = np.zeros((_MODEL, _MODEL), np.uint8)
        blob[2:5, 3:6] = 1
        soft, binary = _soft_and_binary_single(blob)
        cr = _static_case_result(soft, binary)
        case_dir = str(tmp_path / "P_HEDGE")
        v21_output.write_case_v21(case_dir, "P_HEDGE", "S", cr, "static", 30)
        payload = _read_json(os.path.join(case_dir, "prediction.json"))
        # 含额外键时 validate 仍须放行(官方赛道七无"不得包含额外字段"条款)
        validate_prediction_json(payload)
        n = payload["Prediction"]["Nodules"][0]
        assert "TI_RADS_Probs" in n, "对冲键 TI_RADS_Probs 未写出(调用处漏透传?)"
        assert "ETE_Probs" in n, "对冲键 ETE_Probs 未写出(调用处漏透传?)"
        assert len(n["TI_RADS_Probs"]) == 5
        assert len(n["ETE_Probs"]) == 4
        assert abs(sum(n["TI_RADS_Probs"]) - 1.0) < 1e-6
        assert abs(sum(n["ETE_Probs"]) - 1.0) < 1e-6
        # 与 int 字段自洽:argmax+1 必须等于写出的 int
        assert n["ETE_Probs"].index(max(n["ETE_Probs"])) + 1 == \
            n["ExtrathyroidalExtension"]
        assert n["TI_RADS_Probs"].index(max(n["TI_RADS_Probs"])) + 1 == \
            n["TI_RADS_Class"]

    def test_absent_when_not_provided(self, tmp_path):
        """不传时**不写出该键**(而非写 null)——CLI 调试等旧调用方零污染。"""
        from src.inference import prediction_json as pj
        entry = pj.build_nodule_entry(
            nodule_id="N1", mask_file="S/nodule_001.nii.gz",
            detection_confidence=0.5, malignancy_confidence=0.5, tirads_class=3)
        assert "TI_RADS_Probs" not in entry
        assert "ETE_Probs" not in entry

    @pytest.mark.parametrize("kwargs", [
        {"tirads_probs": [0.5, 0.5]},          # 应为 5 维
        {"ete_probs": [0.2, 0.2, 0.2, 0.2, 0.2]},  # 应为 4 维
    ])
    def test_wrong_length_rejected(self, kwargs):
        """长度错 = 上游取错了头,当场炸而非静默写出自相矛盾的 JSON。"""
        from src.inference import prediction_json as pj
        with pytest.raises(PredictionSchemaError):
            pj.build_nodule_entry(
                nodule_id="N1", mask_file="S/nodule_001.nii.gz",
                detection_confidence=0.5, malignancy_confidence=0.5,
                tirads_class=3, **kwargs)


class TestDynamicLesionLevelMean:
    """动态路径:四头按**病灶级均值**汇总(官方 :215),不是取关键帧。"""

    def test_mean_differs_from_keyframe(self, tmp_path):
        """🔴 构造成"三帧均值的 argmax ≠ 关键帧的 argmax",否则测试零区分力
        (沿用关键帧也能过)。

        关键帧 = per_frame[0](fixture 的 keyframe_index=0),其 ETE argmax=idx0(码1);
        三帧均值 = ([0.7,0.1,0.1,0.1] + [0.1,0.8,0.05,0.05] + [0.1,0.8,0.05,0.05])/3
                 = [0.30, 0.5667, 0.0667, 0.0667] → argmax=idx1(码2)。
        """
        heads = [
            _head_probs(ete=[0.7, 0.1, 0.1, 0.1]),
            _head_probs(ete=[0.1, 0.8, 0.05, 0.05]),
            _head_probs(ete=[0.1, 0.8, 0.05, 0.05]),
        ]
        cr = _dynamic_case_result(per_frame_heads=heads)
        # 前提自检:关键帧单独看是码 1,与期望的均值结果 2 不同
        assert int(np.argmax(cr["kf_preds"]["ete_probs"])) + 1 == 1
        case_dir = str(tmp_path / "D_MEAN")
        v21_output.write_case_v21(case_dir, "D_MEAN", "ACC", cr, "dynamic", 50)
        payload = _read_json(os.path.join(case_dir, "prediction.json"))
        validate_prediction_json(payload)
        n = payload["Prediction"]["Nodules"][0]
        assert n["ExtrathyroidalExtension"] == 2, \
            "动态四头应取全帧均值,拿到的是关键帧值"
        # 对冲键写出的也必须是均值本身
        assert abs(n["ETE_Probs"][1] - (0.1 + 0.8 + 0.8) / 3) < 1e-9

    def test_t_conf_is_mean(self, tmp_path):
        heads = [_head_probs(t_conf=c) for c in (0.2, 0.5, 0.8)]
        cr = _dynamic_case_result(per_frame_heads=heads)
        case_dir = str(tmp_path / "D_TCONF")
        v21_output.write_case_v21(case_dir, "D_TCONF", "ACC", cr, "dynamic", 50)
        n = _read_json(os.path.join(case_dir,
                                    "prediction.json"))["Prediction"]["Nodules"][0]
        assert abs(n["T_ClinSig"]["T_conf"] - 0.5) < 1e-9

    def test_empty_per_frame_falls_back_to_keyframe(self):
        """空 per_frame 不得除零。上游 `diagnose_dynamic_case` 已保证非空
        (为空时抛 DynamicCaseError),但此处不赌上游契约。"""
        kf = _head_probs()
        out = v21_output._lesion_level_head_preds([], kf)
        assert out is kf

    def test_mean_denominator_excludes_failed_frames(self, tmp_path):
        """🔴 钉住"`per_frame` 只含成功帧"这个前提。

        `case_pipeline.py:144-163` 里异常帧走 `failed_frames.append` + `continue`,
        **在 append 进 per_frame 之前** → 均值分母天然只算成功帧。若将来改成
        "失败帧也进 per_frame",分母就错了且**静默**。此处构造 2 成功帧 +
        `failed_frames` 非空,断言均值只按 2 帧算。
        """
        heads = [_head_probs(t_conf=0.2), _head_probs(t_conf=0.6)]
        cr = _dynamic_case_result(sampled_idx=(0, 2), per_frame_heads=heads)
        cr["failed_frames"] = [{"frame_index": 4, "reason": "NaN"}]
        case_dir = str(tmp_path / "D_FAIL")
        v21_output.write_case_v21(case_dir, "D_FAIL", "ACC", cr, "dynamic", 50)
        n = _read_json(os.path.join(case_dir,
                                    "prediction.json"))["Prediction"]["Nodules"][0]
        # (0.2+0.6)/2 = 0.4;若把失败帧也算进分母(/3)会得 0.2667
        assert abs(n["T_ClinSig"]["T_conf"] - 0.4) < 1e-9

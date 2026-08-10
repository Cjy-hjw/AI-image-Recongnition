"""`src/inference/prediction_json.py` 回归测试。

两类断言最关键:

1. **`TestLocationOrdering`** — `LOCATION_LABELS` 的顺序是训练标签与输出编码之间的
   唯一契约。顺序写错会让概率分布与类别标签系统性错位,而 JSON 完全合法、指标只是
   "偏低",本地永远查不出根因。

2. **`TestNumpyScalarSerialization`** — 模型输出全是 numpy 标量。`numpy.bool_` 不是
   Python `bool`,`json.dump` 遇到直接抛 TypeError ⇒ **整份 prediction.json 写不出去**,
   比任何单字段错误都严重。
"""

import json

import numpy as np
import pytest

from src.inference.prediction_json import (
    ETE_CAPSULE,
    ETE_CODES,
    ETE_NONE,
    ETE_SOFT_TISSUE,
    ETE_VESSEL,
    LOCATION_LABELS,
    NUM_LOCATIONS,
    PredictionSchemaError,
    build_nodule_entry,
    build_prediction_json,
    ete_label_to_code,
    location_label_to_index,
    normalize_location_probs,
    uniform_location_probs,
    validate_prediction_json,
)


def _entry(nid="N1", mask="n1.nii.gz", **kw):
    kw.setdefault("detection_confidence", 0.9)
    kw.setdefault("malignancy_confidence", 0.7)
    kw.setdefault("tirads_class", 4)
    return build_nodule_entry(nid, mask, **kw)


class TestSchemaShape:
    def test_full_payload_passes_validation(self):
        p = build_prediction_json(
            "STU001", [_entry()], 1234,
            {"Conclusion": "结论", "KeyFindings": ["发现1"]})
        validate_prediction_json(p)  # 不抛即通过
        assert p["StudyUID"] == "STU001"
        assert p["ProcessingTime_ms"] == 1234
        assert p["Prediction"]["MaskFiles"] == ["n1.nii.gz"]

    def test_negative_case_is_empty_array_not_error(self):
        """规范:阴性病例 Nodules 为空数组。空是合法输出,不是错误。"""
        p = build_prediction_json("S", [], 10)
        assert p["Prediction"]["Nodules"] == []
        assert p["Prediction"]["MaskFiles"] == []
        validate_prediction_json(p)

    def test_interpretation_omitted_when_none(self):
        """Interpretation 是"推荐"非必填,不传时不应写出 null 值键。"""
        assert "Interpretation" not in build_prediction_json("S", [], 1)

    def test_maskfiles_mirrors_nodule_maskfile(self):
        p = build_prediction_json(
            "S", [_entry("N1", "a.nii.gz"), _entry("N2", "b.nii.gz")], 1)
        assert p["Prediction"]["MaskFiles"] == ["a.nii.gz", "b.nii.gz"]


class TestLocationOrdering:
    """LOCATION_LABELS 顺序即 `Location` 1..7 的定义,是单一权威。"""

    def test_exact_order_from_official_xlsx_cells(self):
        """顺序取自实测官方 xlsx 单元格(`2左中叶`/`5右中叶`),按解剖习惯上→中→下;
        《数据格式说明》的文字列举次序被实测证伪(照它写会错 4/7 类),不得据文档重排。"""
        assert LOCATION_LABELS == (
            "左上叶", "左中叶", "左下叶", "右上叶", "右中叶", "右下叶", "峡部")
        assert NUM_LOCATIONS == 7

    def test_label_to_index_is_zero_based(self):
        assert location_label_to_index("左上叶") == 0
        assert location_label_to_index("峡部") == 6

    @pytest.mark.parametrize("label", ["无（阴性）", "无(阴性)", "NA/UNK", None, "乱码"])
    def test_null_labels_return_none_not_zero(self, label):
        """返回 None 而非 0:把未知样本当"左上叶"训练会污染整个位置分布。"""
        assert location_label_to_index(label) is None

    def test_location_defaults_to_argmax_plus_one(self):
        probs = [0.05, 0.05, 0.6, 0.1, 0.1, 0.05, 0.05]
        assert _entry(location_probs=probs)["Location"] == 3  # argmax=2 → 1-based 3

    def test_explicit_location_overrides_argmax(self):
        """允许调用方用先验覆盖 argmax(如结合解剖侧别信息)。"""
        probs = [0.6] + [0.4 / 6] * 6
        assert _entry(location_probs=probs, location=5)["Location"] == 5


class TestLocationProbs:
    def test_sum_is_exactly_one(self):
        """规范明写"和为 1"。0.9999999 会被严格解析器拒绝。"""
        assert sum(_entry()["LocationProbs"]) == 1.0

    def test_softmax_residual_renormalized(self):
        raw = np.exp([2.0, 1.0, 0.5, 0.3, 0.1, 0.05, 0.02]).astype(np.float32)
        raw = (raw / raw.sum()).tolist()
        out = _entry(location_probs=raw)["LocationProbs"]
        assert sum(out) == 1.0
        assert out.index(max(out)) == 0, "重归一化不得改变 argmax"

    def test_uniform_placeholder_argmax_is_first_not_last(self):
        """残差若并入末位,均匀分布的 argmax 恒落在"峡部"(解剖上最小、结节发生率
        最低的分区)—— 占位输出系统性报峡部是明确有偏的错误信号。"""
        u = uniform_location_probs()
        assert sum(u) == 1.0
        assert u.index(max(u)) == 0
        assert _entry()["Location"] == 1

    def test_unnormalized_input_is_normalized(self):
        out = normalize_location_probs([2, 2, 2, 2, 2, 2, 2])
        assert sum(out) == 1.0
        assert all(abs(v - 1 / 7) < 1e-9 for v in out)

    @pytest.mark.parametrize(
        "probs,desc",
        [
            ([1 / 6] * 6, "长度6"),
            ([1 / 8] * 8, "长度8"),
            ([-0.1] + [0.5] * 6, "含负值"),
            ([float("nan")] + [0.1] * 6, "含NaN"),
            ([float("inf")] + [0.1] * 6, "含Inf"),
            ([0.0] * 7, "全零无法归一化"),
        ],
    )
    def test_invalid_probs_rejected(self, probs, desc):
        with pytest.raises(PredictionSchemaError):
            _entry(location_probs=probs)


class TestEteMapping:
    """xlsx 训练标签(6类) → 输出编码(4值)的归并。"""

    @pytest.mark.parametrize(
        "label,code",
        [
            ("甲状腺包膜", ETE_CAPSULE),
            ("毗邻软组织", ETE_SOFT_TISSUE),
            ("血管", ETE_VESSEL),
            ("无", ETE_NONE),
        ],
    )
    def test_known_labels(self, label, code):
        assert ete_label_to_code(label) == code

    @pytest.mark.parametrize("label", ["无病灶（阴性）", "无病灶(阴性)", "NA/UNK", None, "乱码"])
    def test_unmappable_falls_back_to_none_conservatively(self, label):
        """输出侧只有 4 个合法值。未知按最保守("无侵犯")处置 —— 把未知报成有侵犯
        会虚增分期。"""
        assert ete_label_to_code(label) == ETE_NONE

    def test_whitespace_tolerated(self):
        """xlsx 单元格常带前后空格。"""
        assert ete_label_to_code("  甲状腺包膜  ") == ETE_CAPSULE


class TestMalignancyConsistency:
    def test_predicted_derived_from_confidence(self):
        """predicted 由 confidence 与阈值比较得出,不单独传入 —— 两者分开传会出现
        `predicted=true` 而 `confidence=0.2` 这类自相矛盾的输出。"""
        assert _entry(malignancy_confidence=0.7)["Malignancy"]["predicted"] is True
        assert _entry(malignancy_confidence=0.3)["Malignancy"]["predicted"] is False

    def test_threshold_boundary_is_inclusive(self):
        assert _entry(malignancy_confidence=0.5)["Malignancy"]["predicted"] is True

    def test_custom_threshold(self):
        e = _entry(malignancy_confidence=0.4, malignancy_threshold=0.3)
        assert e["Malignancy"]["predicted"] is True


class TestFieldGuards:
    @pytest.mark.parametrize("v", [0, 6, -1])
    def test_tirads_out_of_range(self, v):
        with pytest.raises(PredictionSchemaError, match="TI_RADS"):
            _entry(tirads_class=v)

    @pytest.mark.parametrize("v", [0, 5, -1])
    def test_ete_out_of_range(self, v):
        with pytest.raises(PredictionSchemaError, match="Extrathyroidal"):
            _entry(ete_code=v)

    @pytest.mark.parametrize("v", [8, -1])
    def test_location_out_of_range(self, v):
        with pytest.raises(PredictionSchemaError, match="Location"):
            _entry(location=v)

    def test_location_zero_allowed(self):
        """规范:"1~7,仅无结节时可为 0"。"""
        assert _entry(location=0)["Location"] == 0

    @pytest.mark.parametrize("v", [2, -1])
    def test_t_clinsig_out_of_range(self, v):
        with pytest.raises(PredictionSchemaError, match="T_ClinSig"):
            _entry(t_clin_sig=v)

    def test_confidence_clamped_not_rejected(self):
        """置信度轻微越界(fp16 舍入可产生 1.0000001)钳位即可,不必整例失败。"""
        assert _entry(detection_confidence=1.2)["Detection"]["confidence"] == 1.0
        assert _entry(detection_confidence=-0.1)["Detection"]["confidence"] == 0.0

    def test_non_finite_confidence_rejected(self):
        with pytest.raises(PredictionSchemaError):
            _entry(detection_confidence=float("nan"))


class TestDuplicateKeys:
    """NoduleID / MaskFile 都是评测索引结节的键,重复会让后者覆盖前者、静默丢结节。"""

    def test_duplicate_nodule_id(self):
        with pytest.raises(PredictionSchemaError, match="NoduleID 重复"):
            build_prediction_json("S", [_entry("N1", "a"), _entry("N1", "b")], 1)

    def test_duplicate_mask_file(self):
        with pytest.raises(PredictionSchemaError, match="MaskFile 重复"):
            build_prediction_json("S", [_entry("N1", "a"), _entry("N2", "a")], 1)


class TestNumpyScalarSerialization:
    """模型输出全是 numpy 标量;numpy.bool_ 不可 JSON 序列化 ⇒ 整份文件写不出去。"""

    def test_numpy_inputs_produce_serializable_json(self):
        e = build_nodule_entry(
            "N1", "a.nii.gz",
            detection_confidence=np.float32(0.9),
            malignancy_confidence=np.float64(0.73),
            tirads_class=np.int64(4),
            ete_code=np.int32(2),
            t_clin_sig=np.int8(1),
            t_confidence=np.float16(0.6),
            location_probs=np.full(7, 1 / 7, np.float32),
        )
        json.dumps(build_prediction_json("S", [e], np.int64(100)))  # 不抛即通过

    def test_predicted_is_python_bool(self):
        e = _entry(malignancy_confidence=np.float32(0.9))
        assert type(e["Malignancy"]["predicted"]) is bool


class TestValidateStandalone:
    """validate_ 作为发布闸门,须能拦住手工拼装/跨版本反序列化来的畸形 payload。"""

    def test_missing_top_level_field(self):
        with pytest.raises(PredictionSchemaError, match="StudyUID"):
            validate_prediction_json({"ProcessingTime_ms": 1, "Prediction": {}})

    def test_missing_nodule_field(self):
        p = build_prediction_json("S", [_entry()], 1)
        del p["Prediction"]["Nodules"][0]["TI_RADS_Class"]
        with pytest.raises(PredictionSchemaError, match="TI_RADS_Class"):
            validate_prediction_json(p)

    def test_maskfiles_out_of_sync_detected(self):
        """两边对不上 = 该结节的 mask 或属性丢失,评测关联不上。"""
        p = build_prediction_json("S", [_entry("N1", "a.nii.gz")], 1)
        p["Prediction"]["MaskFiles"] = ["other.nii.gz"]
        with pytest.raises(PredictionSchemaError, match="不一致"):
            validate_prediction_json(p)

    def test_probs_sum_drift_detected(self):
        p = build_prediction_json("S", [_entry()], 1)
        p["Prediction"]["Nodules"][0]["LocationProbs"] = [0.1] * 7
        with pytest.raises(PredictionSchemaError, match="和不为 1"):
            validate_prediction_json(p)

    def test_numpy_bool_predicted_detected(self):
        p = build_prediction_json("S", [_entry()], 1)
        p["Prediction"]["Nodules"][0]["Malignancy"]["predicted"] = np.bool_(True)
        with pytest.raises(PredictionSchemaError, match="bool"):
            validate_prediction_json(p)

    def test_processing_time_must_be_int(self):
        p = build_prediction_json("S", [], 1)
        p["ProcessingTime_ms"] = 1.5
        with pytest.raises(PredictionSchemaError, match="int"):
            validate_prediction_json(p)


class TestEteCodesConstant:
    def test_codes_match_official_encoding(self):
        """1 无 / 2 包膜 / 3 软组织 / 4 血管(规范 §五.4)。"""
        assert ETE_CODES == (1, 2, 3, 4)
        assert (ETE_NONE, ETE_CAPSULE, ETE_SOFT_TISSUE, ETE_VESSEL) == (1, 2, 3, 4)

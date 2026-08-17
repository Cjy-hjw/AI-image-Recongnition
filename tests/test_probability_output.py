"""概率分布输出契约测试(官方硬约束,不达标直接计 0 分)。

依据:2026-07-28 官方赛前动员会《五大核心评审目标与属性分类》——
"**强制输出功率**:所有分类/分级/分型任务必须输出功率分布或置信度,否则计0分"。

此前 `decode_predictions` 对 composition/echogenicity/shape/margin 只取 `argmax`,
softmax 分布在 decode 层就被丢弃,`result.json` 里四个 TI-RADS 概念**完全没有概率**
(模型 `cbm_head.py` 内部其实已算出,属导出遗漏而非能力缺失)。本文件锁定修复后的
契约:概率必须落盘、和为 1、类别顺序与报告映射表一致、静态图/动态图 schema 相同。
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference.postprocess import build_probability_export, decode_predictions
from src.utils.report_generator import (
    COMPOSITION_MAP, ECHOGENICITY_MAP, FOCI_MAP, MARGIN_MAP, SHAPE_MAP,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 概念名 → (report_generator 映射表, decode_predictions 概率键, 索引键)
_SOFTMAX_HEADS = {
    "composition": (COMPOSITION_MAP, "comp_prob", "comp_idx"),
    "echogenicity": (ECHOGENICITY_MAP, "echo_prob", "echo_idx"),
    "shape": (SHAPE_MAP, "shape_prob", "shape_idx"),
    "margin": (MARGIN_MAP, "margin_prob", "margin_idx"),
}


def _fake_outputs(seed: int = 0) -> dict:
    """各头 logits,形状与 MultiTaskThyroidModel 输出对齐。

    ⚠️ 官方四头(`ete_probs`/`location_probs`/`t_probs`/`tirads_probs`)在真实模型里
    **输出的已经是 softmax 概率**(各头内部自己算,见 `heads/*.py`),不是 logits ——
    `decode_predictions` 对它们不再二次 softmax,故此处 fixture 也必须造归一化分布。
    """
    torch.manual_seed(seed)
    return {
        "mask": torch.randn(1, 1, 8, 8),
        "composition": torch.randn(1, len(COMPOSITION_MAP)),
        "echogenicity": torch.randn(1, len(ECHOGENICITY_MAP)),
        "shape": torch.randn(1, len(SHAPE_MAP)),
        "margin": torch.randn(1, len(MARGIN_MAP)),
        "foci": torch.randn(1, len(FOCI_MAP)),
        "malignancy": torch.randn(1, 2),
        # 官方四头(#116b):已 softmax 的概率
        "ete_probs": torch.softmax(torch.randn(1, 4), dim=1),
        "location_probs": torch.softmax(torch.randn(1, 7), dim=1),
        "t_probs": torch.softmax(torch.randn(1, 2), dim=1),
        "tirads_probs": torch.softmax(torch.randn(1, 5), dim=1),
        "t_conf": torch.rand(1),
    }


class TestDecodeEmitsDistributions:
    """decode_predictions 必须同时给出 argmax 索引与完整概率分布。"""

    @pytest.mark.parametrize("head", sorted(_SOFTMAX_HEADS))
    def test_softmax_head_distribution_shape_and_sum(self, head):
        map_, prob_key, _ = _SOFTMAX_HEADS[head]
        preds = decode_predictions(_fake_outputs())
        prob = preds[prob_key]
        assert prob.shape == (len(map_),), \
            f"{head} 概率维度须与 report_generator 映射表类别数一致"
        assert np.isclose(prob.sum(), 1.0, atol=1e-5), f"{head} softmax 分布须和为 1"
        assert np.all(prob >= 0.0)

    @pytest.mark.parametrize("head", sorted(_SOFTMAX_HEADS))
    def test_index_matches_argmax_of_distribution(self, head):
        """`*_idx` 与概率分布必须自洽——否则报告文字与概率会各说各话。"""
        _, prob_key, idx_key = _SOFTMAX_HEADS[head]
        for seed in range(8):
            preds = decode_predictions(_fake_outputs(seed))
            assert preds[idx_key] == int(np.argmax(preds[prob_key]))

    def test_malignancy_two_class_distribution(self):
        preds = decode_predictions(_fake_outputs())
        probs = preds["mal_probs"]
        assert probs.shape == (2,), "良恶性须输出完整二类分布 [p_benign, p_malignant]"
        assert np.isclose(probs.sum(), 1.0, atol=1e-6)
        assert probs[1] == pytest.approx(preds["mal_prob"]), \
            "标量 mal_prob 必须等于分布的恶性分量(同源,不得两条计算路径)"

    def test_foci_is_multilabel_not_a_distribution(self):
        """foci 是多标签 sigmoid,各元素独立、**不**求和为 1(易被误当分布)。"""
        preds = decode_predictions(_fake_outputs())
        foci = preds["foci_prob"]
        assert foci.shape == (len(FOCI_MAP),)
        assert np.all((foci >= 0.0) & (foci <= 1.0))

    def test_legacy_index_fields_are_preserved(self):
        """旧字段不得删除:build_tirads_breakdown / 报告生成仍依赖 `*_idx`。"""
        preds = decode_predictions(_fake_outputs())
        for key in ("comp_idx", "echo_idx", "shape_idx", "margin_idx",
                    "mal_prob", "foci_prob", "pred_mask", "mask_binary"):
            assert key in preds

    def test_temperature_scaling_applies_to_distributions(self):
        """概率必须走与 argmax 同一条温度校准通路,而不是绕开校准直接 softmax。"""
        from src.inference.calibration import TemperatureScaler

        base = decode_predictions(_fake_outputs(seed=3), calibrator=None)
        warm = decode_predictions(
            _fake_outputs(seed=3), calibrator=TemperatureScaler({"composition": 4.0}))
        n = len(COMPOSITION_MAP)
        # T>1 → 分布向均匀分布收缩(最大概率被压低),但 argmax 不变
        assert warm["comp_prob"].max() < base["comp_prob"].max()
        assert warm["comp_idx"] == base["comp_idx"]
        assert np.isclose(warm["comp_prob"].sum(), 1.0, atol=1e-5)
        assert warm["comp_prob"].max() > 1.0 / n  # 收缩但未塌成均匀


    def test_fp16_logits_promoted_to_fp32_before_softmax(self):
        """半精度 logits 必须先 `.float()` 再校准/softmax。

        AMP fp16 下 `logits/T` 的温度除法在半精度有可观舍入损失。若有人删掉
        `_softmax_head` 里的 `.float()`,纯 fp32 的测试全绿也抓不到——故显式喂
        `.half()` 输入对拍(Gate2 审查 P2-6)。CPU 上 fp16 softmax 本身也可能不受支持,
        这条同时守住"不会因半精度输入崩溃"。
        """
        fp32 = _fake_outputs(seed=11)
        fp16 = {k: v.half() for k, v in _fake_outputs(seed=11).items()}
        base = decode_predictions(fp32)
        half = decode_predictions(fp16)
        for key in ("comp_prob", "echo_prob", "shape_prob", "margin_prob", "mal_probs"):
            np.testing.assert_allclose(base[key], half[key], atol=1e-3)
            assert half[key].dtype == np.float32, f"{key} 必须以 fp32 导出"
        assert np.isclose(half["mal_probs"].sum(), 1.0, atol=1e-5)


class TestNonFiniteProbabilityIsRejected:
    """概率含 NaN/Inf 必须硬报错,不得导出"看起来正常"的无效分布。

    2026-07-29 `/code-review` 实测复现的两个后果:
    ① `torch.argmax` 对全 NaN 向量**静默返回 0** —— logits=[0.2, nan, 1.5, -0.3]
       真实最大值在 idx=2,却报 idx=0,报告打印错误的 TI-RADS 分项;
    ② `json.dumps` 写出字面量 `NaN`,**非标准 JSON**,严格解析器(Go/Rust/Jackson
       默认)拒绝整个 result.json —— 该样本零输出,比预测错更严重。
    与 `report_generator.generate_report` 对 malignancy_prob 的既定处置同范式,
    但覆盖面扩到全部分类头(generate_report 只校验恶性标量)。
    """

    @pytest.mark.parametrize("head,bad_logits", [
        ("composition", [[0.2, float("nan"), 1.5, -0.3]]),
        ("echogenicity", [[float("inf"), 0.1, 0.2, 0.3]]),
        ("shape", [[float("nan"), 0.5]]),
        ("margin", [[0.1, 0.2, float("nan"), 0.4]]),
        ("foci", [[float("nan"), 0.0, 0.0, 0.0]]),
        ("malignancy", [[float("inf"), 0.3]]),
    ])
    def test_non_finite_logits_raise(self, head, bad_logits):
        outputs = _fake_outputs()
        outputs[head] = torch.tensor(bad_logits)
        with pytest.raises(ValueError, match="NaN/Inf"):
            decode_predictions(outputs)

    @pytest.mark.parametrize("head", ["composition", "margin", "foci", "malignancy"])
    def test_negative_infinity_logit_is_legal(self, head):
        """`-inf` **不该**被拦:softmax/sigmoid 后是有限值(exp(-inf)=0)。

        数学含义就是"该类概率为 0",分布仍和为 1、仍可 JSON 序列化。拦它会把合法
        的极端置信输出误杀成 ERROR、白丢该样本诊断——本轮写这条断言时第一版测试
        就误把 `-inf` 当非法,被这条反向用例纠正。校验的是**输出**是否有限,
        不是输入 logits 是否有限,二者不等价。
        """
        outputs = _fake_outputs()
        n = outputs[head].shape[1]
        logits = [0.1] * n
        logits[0] = float("-inf")
        outputs[head] = torch.tensor([logits])
        preds = decode_predictions(outputs)  # 不抛即达标
        key = {"composition": "comp_prob", "margin": "margin_prob",
               "foci": "foci_prob", "malignancy": "mal_probs"}[head]
        assert np.all(np.isfinite(preds[key]))
        assert preds[key][0] == pytest.approx(0.0), "-inf 对应类的概率应为 0"

    def test_argmax_would_have_silently_returned_zero(self):
        """锁定"为什么必须拦"——证明不拦时 argmax 真的给出错误索引。"""
        nan_prob = torch.softmax(torch.tensor([[0.2, float("nan"), 1.5, -0.3]]), dim=1)[0]
        assert int(torch.argmax(nan_prob).item()) == 0, \
            "前提变了:若 argmax 不再对 NaN 返回 0,本拦截的理由需重新评估"
        assert 2 == int(torch.argmax(torch.tensor([0.2, -1e9, 1.5, -0.3])).item()), \
            "真实最大 logit 在 idx=2,与 NaN 下的 0 不符——这就是静默错判"

    def test_error_message_names_the_offending_head(self):
        outputs = _fake_outputs()
        outputs["margin"] = torch.tensor([[float("nan"), 0.1, 0.2, 0.3]])
        with pytest.raises(ValueError, match="margin"):
            decode_predictions(outputs)

    @pytest.mark.parametrize("head", [
        "ete_probs", "location_probs", "t_probs", "tirads_probs", "t_conf",
    ])
    def test_official_heads_non_finite_raise(self, head):
        """🔴 官方四头(#116b)同样硬拦 NaN,不得静默产出 argmax=0。

        不拦的两个后果对四头**更严重**:① argmax 对全 NaN 静默返回 0 →
        `+1` 后 ETE 报"1 无侵犯"、TI-RADS 报"1 类"、Location 报"1 左上叶" ——
        全是合法编码,`validate_prediction_json` 放行,评测静默失分,
        本地肉眼完全看不出;② NaN 写进 `LocationProbs`/`T_conf` 会让整份
        prediction.json 变成非标准 JSON,严格解析器拒收 → 该病例零分。
        """
        outputs = _fake_outputs()
        shape = outputs[head].shape
        outputs[head] = torch.full(shape, float("nan"))
        with pytest.raises(ValueError, match="NaN/Inf"):
            decode_predictions(outputs)

    @pytest.mark.parametrize("head", [
        "ete_probs", "location_probs", "t_probs", "tirads_probs", "t_conf",
    ])
    def test_official_head_missing_key_raises_not_silently_defaults(self, head):
        """🔴 四头键缺失必须 KeyError,**不得**兜底成占位。

        四头是 `MultiTaskThyroidModel.forward` 的无条件输出。若缺席说明加载的是
        不含四头结构的旧模型(如生产默认权重 `thyroidxl_merged_v2.pth`)——此时
        静默兜底会让 prediction.json 写出确定性占位值,目标四 40% 三项 +
        目标三 TI-RADS 全部失分且本地看不出来(JSON 完全合法)。宁可当场炸。
        """
        outputs = _fake_outputs()
        del outputs[head]
        with pytest.raises(KeyError):
            decode_predictions(outputs)

    def test_official_heads_not_re_softmaxed(self):
        """🔴 四头概率**原样透传**,不得二次 softmax。

        各头已在模型内部做过 fp32 softmax(`heads/*.py`),这里重算会把分布压平
        (softmax(softmax(x)) ≠ softmax(x))—— argmax 不变故不会报错,但概率值全错,
        而"必须输出概率分布"正是官方计分项。用一个**尖锐**分布对拍:若被二次
        softmax,峰值会从 0.9 塌到约 0.35。
        """
        outputs = _fake_outputs()
        sharp = [0.9, 0.04, 0.03, 0.03]
        outputs["ete_probs"] = torch.tensor([sharp])
        preds = decode_predictions(outputs)
        np.testing.assert_allclose(preds["ete_probs"], sharp, atol=1e-6)

    def test_official_heads_not_temperature_scaled(self):
        """四头不在 `CALIBRATED_TASKS` 内 → 温度缩放对其**无效**(伪精度)。"""
        from src.inference.calibration import TemperatureScaler
        outputs = _fake_outputs(seed=7)
        base = decode_predictions(outputs, calibrator=None)
        # 给一个极端温度;若四头误走 calibrator,分布会被显著软化
        cal = TemperatureScaler({"malignancy": 5.0})
        scaled = decode_predictions(_fake_outputs(seed=7), calibrator=cal)
        for key in ("ete_probs", "location_probs", "t_probs", "tirads_probs"):
            np.testing.assert_allclose(base[key], scaled[key], atol=1e-9)

    def test_raises_dedicated_subclass_for_frame_isolation(self):
        """必须抛 `NonFiniteProbabilityError`(`ValueError` 子类),不能只是基类 ValueError。

        视频逐帧循环靠这个**专属类型**区分「概率无效可丢帧」与「其它 ValueError
        (cv2/形状故障)须冒泡」——若退化成基类,cv2.error 之外的普通 ValueError
        会被误当可丢帧吞掉,故障被伪装成"少几帧"的 PASSED。
        """
        from src.inference.postprocess import NonFiniteProbabilityError
        outputs = _fake_outputs()
        outputs["malignancy"] = torch.tensor([[float("nan"), 0.3]])
        with pytest.raises(NonFiniteProbabilityError):
            decode_predictions(outputs)
        # 同时仍是 ValueError(向后兼容旧 `pytest.raises(ValueError)` 契约)
        assert issubclass(NonFiniteProbabilityError, ValueError)

    def test_mask_nan_is_also_rejected(self):
        """mask 的 NaN 不得静默产出全零分割图(2026-07-30 /code-review Angle C)。

        此前只拦六个分类头,mask 路径裸奔:NaN logits 经 sigmoid 得 NaN,
        `(NaN > 0.5)` 恒 False → **全零 mask**,而 status=PASSED、报告正常——
        评测/人工只看 JSON 察觉不到分割已废,比报错更隐蔽。
        """
        from src.inference.postprocess import NonFiniteProbabilityError
        outputs = _fake_outputs()
        outputs["mask"] = torch.full((1, 1, 8, 8), float("nan"))
        with pytest.raises(NonFiniteProbabilityError, match="mask"):
            decode_predictions(outputs)


class TestProbabilityExportContract:
    """build_probability_export = 两个推理入口共用的导出层,schema 唯一真源。"""

    @staticmethod
    def _export(seed=0):
        return build_probability_export(decode_predictions(_fake_outputs(seed)))

    def test_export_is_json_serializable(self):
        """numpy float32 不可直接 json.dumps——必须 .tolist() 成原生 float。"""
        payload = self._export()
        text = json.dumps(payload)  # 不抛异常即达标
        assert json.loads(text) == payload

    def test_all_four_concepts_plus_foci_present(self):
        probs = self._export()["concept_probabilities"]
        assert set(probs) == {"composition", "echogenicity", "shape", "margin", "foci"}

    def test_class_counts_match_report_maps(self):
        """类别顺序即索引顺序,长度须与 report_generator 映射表严格一致。"""
        probs = self._export()["concept_probabilities"]
        for name, (map_, _, _) in _SOFTMAX_HEADS.items():
            assert len(probs[name]) == len(map_)
        assert len(probs["foci"]) == len(FOCI_MAP)

    def test_class_counts_match_shipped_config(self):
        """交付配置的 num_classes_* 与导出维度必须一致(改配置忘改映射表会被这条抓到)。"""
        import yaml
        with open(_REPO_ROOT / "configs" / "thyroid_config.yaml", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)["model"]
        probs = self._export()["concept_probabilities"]
        for name in ("composition", "echogenicity", "shape", "margin", "foci"):
            assert len(probs[name]) == model_cfg[f"num_classes_{name}"]
        assert len(self._export()["malignancy_probabilities"]) == \
            model_cfg["num_classes_malignancy"]

    def test_predicted_index_agrees_with_exported_distribution(self):
        payload = self._export(seed=5)
        for name, prob in payload["concept_probabilities"].items():
            if name == "foci":
                continue  # 多标签,无单一 argmax 语义
            assert payload["concept_predicted_index"][name] == int(np.argmax(prob))

    def test_malignancy_distribution_sums_to_one(self):
        probs = self._export()["malignancy_probabilities"]
        assert len(probs) == 2 and abs(sum(probs) - 1.0) < 1e-6


class TestEntryPointsShareOneSchema:
    """静态图与动态图 result.json 的概率 schema 必须同源。

    源码级断言:两个入口都必须调用 `build_probability_export`。若有人在其中一侧
    手写概率字段(哪怕当下数值正确),两条链路的 schema 会随后续改动悄悄分叉——
    官方要求每一例影像都输出全部维度,静态/动态不一致等于其中一条链路漏项。
    """

    @staticmethod
    def _calls_export(filename):
        tree = ast.parse((_REPO_ROOT / filename).read_text(encoding="utf-8"))
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_probability_export"
            for node in ast.walk(tree)
        )

    @pytest.mark.parametrize("entry", ["predict.py", "video_predict.py"])
    def test_entry_point_uses_shared_exporter(self, entry):
        assert self._calls_export(entry), \
            f"{entry} 必须复用 build_probability_export,禁止各自手写概率字段"

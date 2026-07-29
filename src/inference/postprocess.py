"""Decode model outputs and compute TI-RADS breakdown."""

import numpy as np
import torch

from src.utils.report_generator import calculate_tirads_points


def decode_predictions(outputs: dict, calibrator=None) -> dict:
    """Extract all prediction fields from raw model outputs.

    Args:
        outputs: 原始模型输出(各头 logits)。
        calibrator: 可选 `TemperatureScaler`。每个分类头在 softmax/sigmoid **之前**
            先经 `calibrator.apply(task, logits)` 做温度缩放。为 None 或全 T=1.0 时
            **无任何行为变化**(逐位等同无校准)。argmax 预测类别不受 T 影响。

    Returns dict with keys:
        pred_mask, mask_binary, comp_idx, echo_idx, shape_idx, margin_idx,
        comp_prob, echo_prob, shape_prob, margin_prob, foci_prob,
        mal_prob, mal_probs

    `*_prob` 为各头**完整概率分布**(softmax 头和为 1;foci 为多标签 sigmoid,不求和),
    `mal_probs` 为良恶性二类分布 `[p_benign, p_malignant]`。官方硬约束"所有分类/分级/
    分型任务必须输出功率分布或置信度,否则计0分"(赛前动员会 2026-07-28),概率必须
    随 `*_idx` 一并导出到 result.json,不得只留 argmax。
    """
    def _cal(task, logits):
        return calibrator.apply(task, logits) if calibrator is not None else logits

    def _softmax_head(task, logits):
        """→ (完整概率向量 np.ndarray, argmax 索引 int)。

        argmax 取自概率向量而非 logits:softmax 严格单调,二者等价,少一次前向。
        """
        prob = torch.softmax(_cal(task, logits.float()), dim=1)[0]
        return prob.cpu().numpy(), int(torch.argmax(prob).item())

    # mask 为分割,不走温度缩放(分割校准是另一套,本期不碰)。
    pred_mask = torch.sigmoid(outputs["mask"].float())[0, 0].cpu().numpy()
    mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255

    # argmax 对温度缩放不变(T>0 单调),保留校准以与概率刻度一致、便于上层取置信度。
    # 各头统一 .float():AMP fp16 下温度除法 logits/T 在半精度有舍入损失,先升 fp32 再缩放/softmax,
    # 与 malignancy 的精度路径对齐(reviewer 建议),不改 argmax 结果。
    comp_prob, comp_idx = _softmax_head("composition", outputs["composition"])
    echo_prob, echo_idx = _softmax_head("echogenicity", outputs["echogenicity"])
    shape_prob, shape_idx = _softmax_head("shape", outputs["shape"])
    margin_prob, margin_idx = _softmax_head("margin", outputs["margin"])
    foci_prob = torch.sigmoid(_cal("foci", outputs["foci"].float()))[0].cpu().numpy()
    mal_probs = torch.softmax(_cal("malignancy", outputs["malignancy"].float()), dim=1)[0].cpu().numpy()
    mal_prob = float(mal_probs[1])

    return {
        "pred_mask": pred_mask,
        "mask_binary": mask_binary,
        "comp_idx": comp_idx,
        "echo_idx": echo_idx,
        "shape_idx": shape_idx,
        "margin_idx": margin_idx,
        "comp_prob": comp_prob,
        "echo_prob": echo_prob,
        "shape_prob": shape_prob,
        "margin_prob": margin_prob,
        "foci_prob": foci_prob,
        "mal_prob": mal_prob,
        "mal_probs": mal_probs,
    }


def build_tirads_breakdown(comp_idx, echo_idx, shape_idx, margin_idx, foci_prob):
    """Compute TI-RADS point breakdown using calculate_tirads_points."""
    return calculate_tirads_points(
        comp_idx, echo_idx, shape_idx, margin_idx, foci_prob
    )


def build_probability_export(preds: dict) -> dict:
    """把 `decode_predictions` 的概率字段整成 JSON 可序列化的导出结构。

    官方硬约束(赛前动员会 2026-07-28):"所有分类/分级/分型任务必须输出功率分布或
    置信度,否则计0分"。静态图(`predict.py`)与动态图(`video_predict.py`)共用本函数,
    保证两条链路的 result.json schema 完全一致——新增维度只需改这一处。

    类别顺序即索引顺序,与 `src/utils/report_generator.py` 的 `COMPOSITION_MAP` /
    `ECHOGENICITY_MAP` / `SHAPE_MAP` / `MARGIN_MAP` / `FOCI_MAP` 的键严格对应。
    `composition`/`echogenicity`/`shape`/`margin` 为 softmax 分布(和为 1);
    `foci` 是**多标签 sigmoid**,各元素独立、不求和。
    """
    return {
        "concept_probabilities": {
            "composition": preds["comp_prob"].tolist(),
            "echogenicity": preds["echo_prob"].tolist(),
            "shape": preds["shape_prob"].tolist(),
            "margin": preds["margin_prob"].tolist(),
            "foci": preds["foci_prob"].tolist(),
        },
        "concept_predicted_index": {
            "composition": preds["comp_idx"],
            "echogenicity": preds["echo_idx"],
            "shape": preds["shape_idx"],
            "margin": preds["margin_idx"],
        },
        "malignancy_probabilities": preds["mal_probs"].tolist(),
    }
